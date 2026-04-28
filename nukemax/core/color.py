"""Color-space math.

Matrices hardcoded as torch tensor constants. All conversions assume
linear unless explicitly stated. ComfyUI's image convention is sRGB-encoded
in [0,1], so `srgb_to_linear`/`linear_to_srgb` are the typical entry/exit.
"""
from __future__ import annotations

import torch

# BT.709 (sRGB primaries) RGB -> YCbCr (full-range, JFIF style).
BT709_RGB_TO_YCBCR = torch.tensor(
    [
        [0.2126, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5000],
        [0.5000, -0.4542, -0.0458],
    ],
    dtype=torch.float32,
)

BT2020_RGB_TO_YCBCR = torch.tensor(
    [
        [0.2627, 0.6780, 0.0593],
        [-0.13963, -0.36037, 0.5000],
        [0.5000, -0.45979, -0.04021],
    ],
    dtype=torch.float32,
)


def to_bchw(img: torch.Tensor) -> torch.Tensor:
    """ComfyUI (B,H,W,C) -> (B,C,H,W). Idempotent on already-BCHW."""
    if img.ndim == 4 and img.shape[-1] in (1, 3, 4) and img.shape[1] not in (1, 3, 4):
        return img.permute(0, 3, 1, 2).contiguous()
    if img.ndim == 4:
        return img
    if img.ndim == 3:  # (H,W,C)
        return img.permute(2, 0, 1).unsqueeze(0).contiguous()
    raise ValueError(f"Unrecognized image shape {tuple(img.shape)}")


def to_bhwc(img: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B,H,W,C)."""
    if img.ndim == 4 and img.shape[1] in (1, 3, 4):
        return img.permute(0, 2, 3, 1).contiguous()
    return img


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    lin_lo = x / 12.92
    lin_hi = ((x + a) / (1 + a)).clamp_min(0).pow(2.4)
    return torch.where(x <= 0.04045, lin_lo, lin_hi)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    lo = 12.92 * x
    hi = (1 + a) * x.clamp_min(0).pow(1 / 2.4) - a
    return torch.where(x <= 0.0031308, lo, hi)


def rgb_to_ycbcr(rgb: torch.Tensor, matrix: torch.Tensor = BT709_RGB_TO_YCBCR) -> torch.Tensor:
    """rgb: (B,3,H,W) in [0,1]. Returns (B,3,H,W) Y in [0,1], Cb/Cr in [-0.5,0.5]."""
    B, C, H, W = rgb.shape
    assert C == 3
    m = matrix.to(rgb.device, rgb.dtype)
    flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)
    out = flat @ m.T
    return out.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()


def luminance(rgb: torch.Tensor, matrix: torch.Tensor = BT709_RGB_TO_YCBCR) -> torch.Tensor:
    """Returns (B,1,H,W) luminance."""
    w = matrix[0].to(rgb.device, rgb.dtype).view(1, 3, 1, 1)
    return (rgb * w).sum(dim=1, keepdim=True)
