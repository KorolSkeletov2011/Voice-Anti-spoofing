# Voice Anti-Spoofing — FFT-LCNN

PyTorch system for detecting **bonafide vs. spoofed speech** on the **ASVspoof 2019 Logical Access (LA)** benchmark. Audio is converted to a fixed-size log-power STFT representation and classified with a Light CNN (LCNN) using Max-Feature-Map activations and an angular-margin (A-Softmax) head.

**Original Kaggle experiment:** https://www.kaggle.com/code/codeinecrazy/voice-anti-spoofing-system

## Results

Recorded result for the selected development checkpoint:

| Split | Accuracy | EER |
|---|---:|---:|
| ASVspoof 2019 LA dev | 94.19% | 2.43% |

The repository does not claim an evaluation-set metric unless it is reproduced with `src.evaluate`.

## Pipeline

```text
audio (.flac)
   ↓
repeat / crop to a fixed ~4.9 s window
   ↓
Blackman STFT (n_fft=1724, hop=128)
   ↓
log-power spectrogram
   ↓
Light CNN + Max-Feature-Map
   ↓
80-d embedding + A-Softmax head
   ↓
spoof / bonafide
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── results/
│   └── metrics.json
├── notebooks/
│   └── README.md
├── scripts/
│   └── predict.py
└── src/
    ├── data.py
    ├── features.py
    ├── loss.py
    ├── metrics.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

## Dataset

The code expects the **ASVspoof 2019 LA** directory as `--data-root`:

```text
ASVspoof2019_LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

On Kaggle, the experiment used the public ASVspoof 2019 dataset mounted under `/kaggle/input/...`.

## Installation

```bash
git clone https://github.com/KorolSkeletov2011/Voice-Anti-spoofing.git
cd Voice-Anti-spoofing

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python -m src.train \
  --data-root /path/to/ASVspoof2019_LA \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-5 \
  --wandb
```

Training uses Adam, A-Softmax, gradient clipping, balanced sampling by default, development EER for model selection, early stopping, and saves:

```text
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/best_metrics.json
```

Disable balanced sampling if needed:

```bash
python -m src.train --data-root /path/to/ASVspoof2019_LA --no-balanced-sampling
```

## Evaluation

```bash
python -m src.evaluate \
  --data-root /path/to/ASVspoof2019_LA \
  --checkpoint checkpoints/best.pt \
  --split eval \
  --scores results/eval_scores.csv
```

The score used for EER is `logit_bonafide - logit_spoof`.

## Single-file inference

```bash
python scripts/predict.py sample.flac --checkpoint checkpoints/best.pt
```

## Notes

- The dataset and model checkpoints are intentionally excluded from Git.
- W&B logging is optional (`--wandb`). No API keys are stored in the repository.
- The fixed input geometry matches the original FFT-LCNN experiment: 600 STFT frames feed the LCNN and produce the `32 × 53 × 37` tensor expected by the embedding layer.

## Credits

The project uses ideas and scaffold code from the PyTorch Project Template and the modular FFT-LCNN implementation at `Scesss/asvspoof-fft-lcnn`. The retained MIT license covers code derived from that scaffold.
