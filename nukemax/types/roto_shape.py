"""ROTO_SHAPE: animated cubic bezier spline(s).

Tensors carry per-frame keyframes. Frame dimension `T` may be 1 for a
static shape; nodes are expected to broadcast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class RotoShape:
    # (T, N, 2) — control point xy in pixel space
    points: torch.Tensor
    # (T, N, 2) — incoming bezier handle xy (relative to point or absolute? -> absolute)
    handles_in: torch.Tensor
    # (T, N, 2) — outgoing bezier handle
    handles_out: torch.Tensor
    # (T, N) — per-vertex feather radius in pixels
    feather: torch.Tensor
    # (H, W) image-space extent for rasterization
    canvas_h: int
    canvas_w: int
    closed: bool = True
    # Optional name / id for multi-shape compositions in the future.
    name: str = ""

    def __post_init__(self) -> None:
        # Validate shapes consistently.
        assert self.points.ndim == 3 and self.points.shape[-1] == 2, "points must be (T,N,2)"
        T, N, _ = self.points.shape
        for tname, t in (
            ("handles_in", self.handles_in),
            ("handles_out", self.handles_out),
        ):
            assert t.shape == (T, N, 2), f"{tname} must be (T,N,2)"
        assert self.feather.shape == (T, N), "feather must be (T,N)"
        assert self.canvas_h > 0 and self.canvas_w > 0

    @property
    def T(self) -> int:
        return self.points.shape[0]

    @property
    def N(self) -> int:
        return self.points.shape[1]

    def to(self, device: torch.device | str) -> "RotoShape":
        return RotoShape(
            points=self.points.to(device),
            handles_in=self.handles_in.to(device),
            handles_out=self.handles_out.to(device),
            feather=self.feather.to(device),
            canvas_h=self.canvas_h,
            canvas_w=self.canvas_w,
            closed=self.closed,
            name=self.name,
        )

    @classmethod
    def from_polygon(
        cls,
        polygon_xy: torch.Tensor,
        canvas_hw: tuple[int, int],
        feather: float = 0.0,
        closed: bool = True,
    ) -> "RotoShape":
        """Build a roto shape from a flat polygon (N,2). Bezier handles
        collapse onto the points (i.e. the curve is piecewise linear).
        """
        if polygon_xy.ndim == 2:
            polygon_xy = polygon_xy.unsqueeze(0)  # (1,N,2)
        T, N, _ = polygon_xy.shape
        return cls(
            points=polygon_xy.float(),
            handles_in=polygon_xy.float().clone(),
            handles_out=polygon_xy.float().clone(),
            feather=torch.full((T, N), float(feather)),
            canvas_h=int(canvas_hw[0]),
            canvas_w=int(canvas_hw[1]),
            closed=closed,
        )
