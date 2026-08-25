from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import ASVspoofDataset, collate_fn, split_paths
from .features import LogPowerSTFT
from .loss import ASoftmaxLoss
from .metrics import bonafide_scores, eer
from .model import LCNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_loader(dataset: ASVspoofDataset, batch_size: int, workers: int, balanced: bool, shuffle: bool) -> DataLoader:
    sampler = None
    if balanced:
        labels = np.asarray(dataset.labels)
        counts = np.bincount(labels, minlength=2)
        weights = np.asarray([1.0 / max(counts[label], 1) for label in labels], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


@torch.no_grad()
def evaluate(model, frontend, loader, device) -> dict[str, float]:
    model.eval()
    frontend.eval()
    all_scores, all_labels = [], []
    correct = total = 0
    for batch in tqdm(loader, desc="dev", leave=False):
        audio = batch["audio"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        output = model(frontend(audio))
        logits = output["logits"]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
        all_scores.append(bonafide_scores(logits).cpu())
        all_labels.append(labels.cpu())
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    return {"accuracy": correct / total, "eer": eer(scores, labels)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FFT-LCNN on ASVspoof 2019 LA")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to the ASVspoof2019 LA root")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="asvspoof-fft-lcnn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_audio, train_protocol = split_paths(args.data_root, "train")
    dev_audio, dev_protocol = split_paths(args.data_root, "dev")
    train_set = ASVspoofDataset(train_audio, train_protocol)
    dev_set = ASVspoofDataset(dev_audio, dev_protocol)
    train_loader = make_loader(train_set, args.batch_size, args.workers, args.balanced_sampling, True)
    dev_loader = make_loader(dev_set, args.batch_size, args.workers, False, False)

    frontend = LogPowerSTFT().to(device)
    model = LCNN().to(device)
    criterion = ASoftmaxLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run = None
    if args.wandb:
        import wandb
        wandb_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run = wandb.init(project=args.wandb_project, config=wandb_config)

    best_eer = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        frontend.train()
        criterion.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")
        for batch in progress:
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(frontend(audio))
            loss = criterion(output["logits"], output["margin_logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * labels.numel()
            seen += labels.numel()
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        metrics = evaluate(model, frontend, dev_loader, device)
        train_loss = running_loss / seen
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"dev_accuracy={metrics['accuracy']:.4%} dev_eer={metrics['eer']:.4%}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "frontend": frontend.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        torch.save(state, args.output_dir / "latest.pt")

        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            stale_epochs = 0
            torch.save(state, args.output_dir / "best.pt")
            (args.output_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            stale_epochs += 1

        if run is not None:
            run.log({"epoch": epoch, "train/loss": train_loss, "dev/accuracy": metrics["accuracy"], "dev/eer": metrics["eer"]})

        if stale_epochs >= args.early_stop:
            print(f"Early stopping after {stale_epochs} epochs without EER improvement")
            break

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
