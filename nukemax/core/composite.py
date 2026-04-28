"""Porter-Duff alpha compositing.

PREMULTIPLIED ALPHA CONTRACT
---------------------------
All operators here take *premultiplied* RGBA. ComfyUI conventionally
deals in straight (non-premultiplied) RGB + a separate MASK; use
`to_premul` / `from_premul` at the boundary.

Conventions:
  - rgba shape: (B, 4, H, W) with channel order (R, G, B, A)
  - all operators return (B, 4, H, W) premultiplied
"""
from __future__ import annotations

import torch


def to_premul(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """rgb: (B,3,H,W); alpha: (B,1,H,W). Returns (B,4,H,W) premultiplied."""
    return torch.cat([rgb * alpha, alpha], dim=1)


def from_premul(rgba: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    a = rgba[:, 3:4]
    rgb = rgba[:, :3] / a.clamp_min(eps)
    return rgb, a


def over(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Out = Src + Dst * (1 - SrcA). Both premultiplied."""
    sa = src[:, 3:4]
    return src + dst * (1.0 - sa)


def in_op(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    da = dst[:, 3:4]
    return src * da


def out_op(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    da = dst[:, 3:4]
    return src * (1.0 - da)


def atop(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    sa = src[:, 3:4]
    da = dst[:, 3:4]
    return src * da + dst * (1.0 - sa)


def xor(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    sa = src[:, 3:4]
    da = dst[:, 3:4]
    return src * (1.0 - da) + dst * (1.0 - sa)


def merge_over_straight(
    src_rgb: torch.Tensor,
    src_alpha: torch.Tensor,
    dst_rgb: torch.Tensor,
    dst_alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: straight-RGB Porter-Duff Over, returns (rgb, alpha)."""
    if dst_alpha is None:
        dst_alpha = torch.ones_like(src_alpha)
    s = to_premul(src_rgb, src_alpha)
    d = to_premul(dst_rgb, dst_alpha)
    out = over(s, d)
    return from_premul(out)
