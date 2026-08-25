from __future__ import annotations

from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
from torch.utils.data import Dataset

TARGET_NUM_SAMPLES = 1724 + 128 * (600 - 1)
LABEL_TO_ID = {"spoof": 0, "bonafide": 1}


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 LA dataset backed by an official protocol file."""

    def __init__(self, audio_dir: str | Path, protocol_path: str | Path):
        self.audio_dir = Path(audio_dir)
        self.protocol_path = Path(protocol_path)
        self.records = self._read_protocol()

    def _read_protocol(self) -> list[dict]:
        records: list[dict] = []
        with self.protocol_path.open("r", encoding="utf-8") as file:
            for line in file:
                values = line.strip().split()
                if not values:
                    continue
                if len(values) < 5:
                    raise ValueError(f"Malformed protocol line: {line!r}")
                key = values[4]
                if key not in LABEL_TO_ID:
                    raise ValueError(f"Unknown ASVspoof label: {key}")
                file_id = values[1]
                records.append(
                    {
                        "speaker_id": values[0],
                        "file_id": file_id,
                        "system_id": values[3],
                        "key": key,
                        "path": self.audio_dir / f"{file_id}.flac",
                        "label": LABEL_TO_ID[key],
                    }
                )
        return records

    @property
    def labels(self) -> list[int]:
        return [record["label"] for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        audio, _ = sf.read(record["path"], dtype="float32", always_2d=False)
        audio = torch.as_tensor(audio, dtype=torch.float32)
        if audio.ndim == 2:
            audio = audio.mean(dim=1)
        return {
            "audio": audio.unsqueeze(0),
            "label": record["label"],
            "file_id": record["file_id"],
            "key": record["key"],
            "path": str(record["path"]),
        }


def repeat_or_crop(audio: torch.Tensor, target_length: int = TARGET_NUM_SAMPLES) -> torch.Tensor:
    """Repeat short utterances and crop long utterances to a fixed length."""
    if audio.ndim != 2 or audio.shape[0] != 1:
        raise ValueError(f"Expected mono audio with shape [1, T], got {tuple(audio.shape)}")
    current_length = audio.shape[-1]
    if current_length == 0:
        raise ValueError("Empty audio file")
    repeats = (target_length + current_length - 1) // current_length
    return audio.repeat(1, repeats)[..., :target_length]


def collate_fn(items: Iterable[dict]) -> dict:
    items = list(items)
    audio = torch.stack([repeat_or_crop(item["audio"]) for item in items], dim=0)
    labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
    return {
        "audio": audio,
        "labels": labels,
        "file_id": [item["file_id"] for item in items],
        "key": [item["key"] for item in items],
        "path": [item["path"] for item in items],
    }


def split_paths(root: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(root)
    split = split.lower()
    if split not in {"train", "dev", "eval"}:
        raise ValueError(f"Unknown split: {split}")

    audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
    protocol_name = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }[split]
    protocol_path = root / "ASVspoof2019_LA_cm_protocols" / protocol_name
    return audio_dir, protocol_path
