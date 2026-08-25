import torch
from torch import nn


class LogPowerSTFT(nn.Module):
    """Blackman-window log-power spectrogram used by the FFT-LCNN model."""

    def __init__(self, n_fft: int = 1724, hop_length: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.blackman_window(n_fft))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.squeeze(1)
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        return torch.log(spectrum.abs().pow(2) + 1e-6).unsqueeze(1)
