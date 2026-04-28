"""MATERIAL_SET: PBR decomposition of an image batch."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaterialSet:
    albedo: torch.Tensor      # (B, 3, H, W) linear RGB in [0,1]
    normal: torch.Tensor      # (B, 3, H, W) tangent-space, components in [-1,1]
    depth: torch.Tensor       # (B, 1, H, W) metric or normalized [0,1] depth
    roughness: torch.Tensor   # (B, 1, H, W) in [0,1]

    def __post_init__(self) -> None:
        B, _, H, W = self.albedo.shape
        assert self.normal.shape == (B, 3, H, W)
        assert self.depth.shape == (B, 1, H, W)
        assert self.roughness.shape == (B, 1, H, W)

    def to(self, device) -> "MaterialSet":
        return MaterialSet(
            albedo=self.albedo.to(device),
            normal=self.normal.to(device),
            depth=self.depth.to(device),
            roughness=self.roughness.to(device),
        )
