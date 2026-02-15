
import typing
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

try:
    from audiotools import AudioSignal
    from audiotools import STFTParams
except ImportError:
    print("Warning: audiotools not found. MelSpectrogramLoss will fail if initialized.")

class L1Loss(nn.L1Loss):
    """L1 Loss between AudioSignals."""
    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x, y):
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)

class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.
    
    Ported from descript-audio-codec/dac/nn/loss.py
    """

    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal, sample_rate: int = 44100):
        loss = 0.0
        # If input is Tensor, wrap in AudioSignal
        if isinstance(x, torch.Tensor):
            x = AudioSignal(x, sample_rate=sample_rate)
        if isinstance(y, torch.Tensor):
            y = AudioSignal(y, sample_rate=sample_rate)

        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            try:
                x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
                y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            except Exception as e:
                # Fallback if audiotools fails or kwargs mismatch
                print(f"Error in mel_spectrogram: {e}")
                raise e

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss
