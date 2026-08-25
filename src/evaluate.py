from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ASVspoofDataset, collate_fn, split_paths
from .features import LogPowerSTFT
from .metrics import bonafide_scores, eer
from .model import LCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FFT-LCNN on ASVspoof 2019 LA")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "eval"], default="eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scores", type=Path, default=Path("results/scores.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    audio_dir, protocol_path = split_paths(args.data_root, args.split)
    dataset = ASVspoofDataset(audio_dir, protocol_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    frontend = LogPowerSTFT().to(device)
    model = LCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "frontend" in checkpoint:
        frontend.load_state_dict(checkpoint["frontend"])
    model.eval()
    frontend.eval()

    rows = []
    all_scores, all_labels = [], []
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            logits = model(frontend(audio))["logits"]
            scores = bonafide_scores(logits)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
            for file_id, score, label in zip(batch["file_id"], scores.cpu().tolist(), labels.cpu().tolist()):
                rows.append({"id": file_id, "score": score, "label": label})

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = {"accuracy": correct / total, "eer": eer(scores, labels)}
    print(json.dumps(metrics, indent=2))

    args.scores.parent.mkdir(parents=True, exist_ok=True)
    with args.scores.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "score", "label"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
