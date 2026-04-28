"""AUDIO_FEATURES: analyzed audio signal for per-frame curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class AudioFeatures:
    waveform: torch.Tensor       # (S,) mono float32 in [-1,1]
    sr: int
    stft_mag: torch.Tensor       # (F, T_stft)
    onsets: torch.Tensor         # (T_stft,) onset envelope
    bpm: float
    centroid: torch.Tensor       # (T_stft,)
    rms: torch.Tensor            # (T_stft,)
    hop_length: int

    def __post_init__(self) -> None:
        assert self.waveform.ndim == 1
        assert self.stft_mag.ndim == 2
        assert self.sr > 0

    @property
    def duration_seconds(self) -> float:
        return float(self.waveform.shape[0]) / float(self.sr)

    def to(self, device) -> "AudioFeatures":
        return AudioFeatures(
            waveform=self.waveform.to(device),
            sr=self.sr,
            stft_mag=self.stft_mag.to(device),
            onsets=self.onsets.to(device),
            bpm=self.bpm,
            centroid=self.centroid.to(device),
            rms=self.rms.to(device),
            hop_length=self.hop_length,
        )
