"""Optical flow warping primitives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _make_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0)  # (2,H,W)


def backward_warp(img: torch.Tensor, flow_dst_to_src: torch.Tensor) -> torch.Tensor:
    """Warp `img` (frame at source) to dest using `flow_dst_to_src` which
    maps each dest pixel to its source pixel. Bilinear `grid_sample`.

    img: (B,C,H,W); flow: (B,2,H,W) in pixel units (dx,dy).
    """
    B, C, H, W = img.shape
    grid = _make_grid(H, W, img.device, img.dtype).unsqueeze(0).expand(B, -1, -1, -1)
    src = grid + flow_dst_to_src
    # Normalize to [-1,1]
    nx = src[:, 0] / max(W - 1, 1) * 2 - 1
    ny = src[:, 1] / max(H - 1, 1) * 2 - 1
    g = torch.stack([nx, ny], dim=-1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)


def forward_warp(img: torch.Tensor, flow_src_to_dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Splatting forward warp.

    For each source pixel, scatter its value to the (subpixel) destination
    using bilinear weights. Returns (warped, weight) so callers can divide
    or fall back where weight is zero (occlusion holes).
    """
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype
    grid = _make_grid(H, W, device, dtype).unsqueeze(0).expand(B, -1, -1, -1)
    dst = grid + flow_src_to_dst  # (B,2,H,W)
    x = dst[:, 0]
    y = dst[:, 1]
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = (x - x0.to(dtype)).clamp(0, 1)
    wy1 = (y - y0.to(dtype)).clamp(0, 1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    out = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    wsum = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    def _splat(xs: torch.Tensor, ys: torch.Tensor, w: torch.Tensor) -> None:
        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs_c = xs.clamp(0, W - 1)
        ys_c = ys.clamp(0, H - 1)
        flat_idx = (ys_c * W + xs_c)  # (B,H,W)
        v = valid.to(dtype).unsqueeze(1)  # (B,1,H,W)
        weighted = img * w.unsqueeze(1) * v
        # Scatter add per batch into flat HW
        idx_exp = flat_idx.unsqueeze(1).expand(-1, C, -1, -1).reshape(B, C, -1)
        out.view(B, C, -1).scatter_add_(2, idx_exp, weighted.reshape(B, C, -1))
        idx_w = flat_idx.unsqueeze(1).reshape(B, 1, -1)
        wsum.view(B, 1, -1).scatter_add_(2, idx_w, (w * v.squeeze(1)).reshape(B, 1, -1))

    _splat(x0, y0, wx0 * wy0)
    _splat(x1, y0, wx1 * wy0)
    _splat(x0, y1, wx0 * wy1)
    _splat(x1, y1, wx1 * wy1)
    return out, wsum


def occlusion_from_consistency(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    threshold: float = 1.5,
) -> torch.Tensor:
    """Forward-backward consistency check. Returns (B,1,H,W) occlusion in {0,1}.
    A pixel is occluded if forward-then-backward displacement exceeds
    `threshold` pixels.
    """
    # Backward-warp the backward flow into the forward frame.
    bwd_at_fwd = backward_warp(flow_bwd, flow_fwd)
    diff = (flow_fwd + bwd_at_fwd).norm(dim=1, keepdim=True)  # ideally 0
    return (diff > threshold).to(flow_fwd.dtype)
