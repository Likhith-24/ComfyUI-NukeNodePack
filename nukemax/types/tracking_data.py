"""TRACKING_DATA: per-frame point tracks with velocity and confidence."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrackingData:
    coords: torch.Tensor       # (T, K, 2)
    velocity: torch.Tensor     # (T, K, 2)
    confidence: torch.Tensor   # (T, K)
    canvas_h: int
    canvas_w: int

    def __post_init__(self) -> None:
        assert self.coords.ndim == 3 and self.coords.shape[-1] == 2
        T, K, _ = self.coords.shape
        assert self.velocity.shape == (T, K, 2)
        assert self.confidence.shape == (T, K)

    @property
    def T(self) -> int:
        return self.coords.shape[0]

    @property
    def K(self) -> int:
        return self.coords.shape[1]

    def to(self, device) -> "TrackingData":
        return TrackingData(
            coords=self.coords.to(device),
            velocity=self.velocity.to(device),
            confidence=self.confidence.to(device),
            canvas_h=self.canvas_h,
            canvas_w=self.canvas_w,
        )
