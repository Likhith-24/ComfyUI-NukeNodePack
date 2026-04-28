"""FLOW_FIELD: dense optical flow with forward/backward + occlusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class FlowField:
    # (T-1, 2, H, W) forward flow (frame i -> i+1), pixel units, channel order (dx, dy)
    flow_fwd: torch.Tensor
    # (T-1, 2, H, W) backward flow (frame i+1 -> i)
    flow_bwd: torch.Tensor
    # (T-1, 1, H, W) occlusion mask in [0,1] for forward direction (1 = occluded)
    occlusion_fwd: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        assert self.flow_fwd.ndim == 4 and self.flow_fwd.shape[1] == 2
        assert self.flow_bwd.shape == self.flow_fwd.shape
        if self.occlusion_fwd is not None:
            T, _, H, W = self.flow_fwd.shape
            assert self.occlusion_fwd.shape == (T, 1, H, W)

    @property
    def T_pairs(self) -> int:
        return self.flow_fwd.shape[0]

    @property
    def hw(self) -> tuple[int, int]:
        return self.flow_fwd.shape[-2], self.flow_fwd.shape[-1]

    def to(self, device) -> "FlowField":
        return FlowField(
            flow_fwd=self.flow_fwd.to(device),
            flow_bwd=self.flow_bwd.to(device),
            occlusion_fwd=None if self.occlusion_fwd is None else self.occlusion_fwd.to(device),
        )
