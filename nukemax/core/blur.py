"""Blur kernels — separable Gaussian, integral-image box, flow motion blur."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel_1d(sigma: float, dtype=torch.float32, device=None, truncate: float = 4.0) -> torch.Tensor:
    sigma = max(float(sigma), 1e-8)
    radius = max(int(math.ceil(truncate * sigma)), 1)
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k


def gaussian_blur(img: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """Separable Gaussian on (B,C,H,W). Two 1D conv passes — never a 2D kernel.
    Reflect padding to avoid darkening at borders. Kernel radius is
    automatically clamped to ``min(H, W) - 1`` so blurring tiny images
    never crashes (Nuke parity: Blur node works at any resolution).
    """
    if sigma <= 0:
        return img
    B, C, H, W = img.shape
    max_pad = min(H, W) - 1
    if max_pad < 1:
        return img
    sigma_eff = max(float(sigma), 1e-8)
    radius = max(int(math.ceil(truncate * sigma_eff)), 1)
    radius = min(radius, max_pad)
    x = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-0.5 * (x / sigma_eff) ** 2)
    k = k / k.sum()
    pad = radius
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, k.shape[0])
    ky = k.view(1, 1, -1, 1).expand(C, 1, k.shape[0], 1)
    x = F.pad(img, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, ky, groups=C)
    return x


def box_blur(img: torch.Tensor, radius: int) -> torch.Tensor:
    """Integral-image box blur. radius in pixels; window = 2r+1."""
    if radius <= 0:
        return img
    B, C, H, W = img.shape
    # Integral image with zero-prepend so we can index r and r+1 cleanly.
    I = F.pad(img.cumsum(-2).cumsum(-1), (1, 0, 1, 0))  # (B,C,H+1,W+1)
    r = radius
    y0 = torch.arange(H, device=img.device).clamp(min=0)
    y1 = (y0 + r + 1).clamp(max=H)
    y0c = (y0 - r).clamp(min=0)
    x0 = torch.arange(W, device=img.device).clamp(min=0)
    x1 = (x0 + r + 1).clamp(max=W)
    x0c = (x0 - r).clamp(min=0)
    # Gather corners
    A = I[..., y1[:, None], x1[None, :]]
    B_ = I[..., y0c[:, None], x1[None, :]]
    Cc = I[..., y1[:, None], x0c[None, :]]
    D = I[..., y0c[:, None], x0c[None, :]]
    s = A - B_ - Cc + D
    area = ((y1 - y0c)[:, None] * (x1 - x0c)[None, :]).to(img.dtype)
    return s / area


def directional_blur(img: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, samples: int = 16) -> torch.Tensor:
    """Per-pixel directional motion blur driven by a 2D vector field.

    img: (B,C,H,W); dx,dy: (B,1,H,W) pixel displacements over the frame.
    Samples uniformly along [-0.5, 0.5] of the displacement vector.
    """
    B, C, H, W = img.shape
    device = img.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=img.dtype),
        torch.arange(W, device=device, dtype=img.dtype),
        indexing="ij",
    )
    # Normalize to grid_sample [-1,1] coords
    out = torch.zeros_like(img)
    for k in range(samples):
        t = (k / max(samples - 1, 1)) - 0.5
        sx = (xx + t * dx[:, 0]) / max(W - 1, 1) * 2 - 1
        sy = (yy + t * dy[:, 0]) / max(H - 1, 1) * 2 - 1
        grid = torch.stack([sx, sy], dim=-1)
        if grid.shape[0] != B:
            grid = grid.expand(B, -1, -1, -1)
        out = out + F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    return out / samples
