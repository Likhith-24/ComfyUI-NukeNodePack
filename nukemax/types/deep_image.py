"""Deep image data type — Nuke-style multi-sample-per-pixel deep data.

Unlike a regular IMAGE (B,H,W,C) which has a single colour+alpha sample
per pixel, a deep image stores *multiple* samples per pixel, each with
its own depth, colour and coverage. This enables true volumetric
compositing (holdouts, deep merges, fog cards) without flat alpha mats.

Storage strategy (memory-friendly, vectorised):
    samples_z      : (B,K,H,W)        — depth per sample (front-to-back if sorted)
    samples_rgba   : (B,K,H,W,4)      — premultiplied colour + coverage
    sample_count   : (B,H,W) int      — number of valid samples (<=K)

K is a fixed upper bound per batch. Unused slots are zeroed and ignored
via `sample_count`. This trades a little RAM for fully tensorised math.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DeepImage:
    samples_z: torch.Tensor          # (B,K,H,W) float
    samples_rgba: torch.Tensor       # (B,K,H,W,4) float, premultiplied
    sample_count: torch.Tensor       # (B,H,W) int32
    name: str = ""

    @property
    def K(self) -> int:
        return int(self.samples_z.shape[1])

    @property
    def shape_bhw(self):
        b, _, h, w = self.samples_z.shape
        return b, h, w

    @classmethod
    def from_image_depth(cls, image_bhwc: torch.Tensor, depth_bhw: torch.Tensor,
                         alpha: torch.Tensor | None = None) -> "DeepImage":
        """Build a single-sample DeepImage from a flat IMAGE + depth map."""
        B, H, W, C = image_bhwc.shape
        rgba = torch.zeros(B, 1, H, W, 4, device=image_bhwc.device, dtype=image_bhwc.dtype)
        rgb = image_bhwc[..., :3]
        if alpha is None:
            a = image_bhwc[..., 3:4] if C == 4 else torch.ones(B, H, W, 1, device=image_bhwc.device, dtype=image_bhwc.dtype)
        else:
            a = alpha.view(B, H, W, 1) if alpha.dim() == 3 else alpha
        # premultiply
        rgba[:, 0, :, :, :3] = rgb * a
        rgba[:, 0, :, :, 3:4] = a
        z = depth_bhw.view(B, 1, H, W).to(image_bhwc.dtype)
        cnt = (a.squeeze(-1) > 1e-5).to(torch.int32)
        return cls(samples_z=z, samples_rgba=rgba, sample_count=cnt)

    def flatten_over(self) -> torch.Tensor:
        """Front-to-back over-composite of all samples -> (B,H,W,4) IMAGE."""
        B, K, H, W = self.samples_z.shape
        # Sort each pixel's samples by depth (front first).
        z = self.samples_z
        order = torch.argsort(z, dim=1)
        idx = order.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        rgba_sorted = torch.gather(self.samples_rgba, 1, idx)
        # Mask out unused slots with sample_count.
        k_idx = torch.arange(K, device=z.device).view(1, K, 1, 1)
        valid = (k_idx < self.sample_count.unsqueeze(1)).unsqueeze(-1).float()
        rgba_sorted = rgba_sorted * valid
        # Iterative front-to-back: out = sum_i ( rgba_i * prod_{j<i}(1-a_j) )
        out = torch.zeros(B, H, W, 4, device=z.device, dtype=z.dtype)
        trans = torch.ones(B, H, W, 1, device=z.device, dtype=z.dtype)
        for k in range(K):
            samp = rgba_sorted[:, k]                     # (B,H,W,4)
            out = out + samp * trans
            trans = trans * (1.0 - samp[..., 3:4]).clamp(0, 1)
        return out.clamp(0, 1)

    def to_image_depth(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten and return (IMAGE, MASK-depth)."""
        flat = self.flatten_over()
        # Front depth (smallest valid z).
        z = self.samples_z.clone()
        K = z.shape[1]
        k_idx = torch.arange(K, device=z.device).view(1, K, 1, 1)
        invalid = k_idx >= self.sample_count.unsqueeze(1)
        z = z.masked_fill(invalid, float("inf"))
        front = z.min(dim=1).values
        front = torch.where(torch.isinf(front), torch.zeros_like(front), front)
        return flat, front
