"""LIGHT_PROBE: HDR equirectangular environment map."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LightProbe:
    env_map: torch.Tensor   # (B, 3, H, W) linear HDR, equirectangular layout
    exposure: float = 0.0   # log2 stops, applied at sample time

    def __post_init__(self) -> None:
        assert self.env_map.ndim == 4 and self.env_map.shape[1] == 3

    def to(self, device) -> "LightProbe":
        return LightProbe(env_map=self.env_map.to(device), exposure=self.exposure)
