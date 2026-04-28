"""FFT_TENSOR: complex spectrum stored as magnitude+phase for portability."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FFTTensor:
    magnitude: torch.Tensor   # (B, C, H, W) real >= 0
    phase: torch.Tensor       # (B, C, H, W) in [-pi, pi]
    spatial_h: int
    spatial_w: int
    centered: bool = True     # True if fftshift'd

    def __post_init__(self) -> None:
        assert self.magnitude.shape == self.phase.shape
        assert self.magnitude.ndim == 4

    @property
    def shape(self):
        return self.magnitude.shape

    def as_complex(self) -> torch.Tensor:
        return torch.polar(self.magnitude, self.phase)

    @classmethod
    def from_complex(cls, c: torch.Tensor, centered: bool = True) -> "FFTTensor":
        assert c.is_complex()
        return cls(
            magnitude=c.abs(),
            phase=c.angle(),
            spatial_h=c.shape[-2],
            spatial_w=c.shape[-1],
            centered=centered,
        )

    def to(self, device) -> "FFTTensor":
        return FFTTensor(
            magnitude=self.magnitude.to(device),
            phase=self.phase.to(device),
            spatial_h=self.spatial_h,
            spatial_w=self.spatial_w,
            centered=self.centered,
        )
