from __future__ import annotations

import argparse
from pathlib import Path
import sys

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import repeat_or_crop  # noqa: E402
from src.features import LogPowerSTFT  # noqa: E402
from src.metrics import bonafide_scores  # noqa: E402
from src.model import LCNN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one audio file as bonafide or spoof")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    samples, _ = sf.read(args.audio, dtype="float32", always_2d=False)
    audio = torch.as_tensor(samples, dtype=torch.float32)
    if audio.ndim == 2:
        audio = audio.mean(dim=1)
    audio = repeat_or_crop(audio.unsqueeze(0)).unsqueeze(0).to(device)

    frontend = LogPowerSTFT().to(device).eval()
    model = LCNN().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "frontend" in checkpoint:
        frontend.load_state_dict(checkpoint["frontend"])

    with torch.no_grad():
        logits = model(frontend(audio))["logits"]
        score = float(bonafide_scores(logits).item())
        label = "bonafide" if int(logits.argmax(dim=1).item()) == 1 else "spoof"
    print(f"prediction={label} score={score:.6f}")


if __name__ == "__main__":
    main()
