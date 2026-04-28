"""FFT primitives operating on (B,C,H,W) tensors."""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..types import FFTTensor


def analyze(img: torch.Tensor, centered: bool = True) -> FFTTensor:
    """img: (B,C,H,W) real -> FFTTensor."""
    F = torch.fft.fft2(img, norm="ortho")
    if centered:
        F = torch.fft.fftshift(F, dim=(-2, -1))
    return FFTTensor.from_complex(F, centered=centered)


def synthesize(ft: FFTTensor) -> torch.Tensor:
    F = ft.as_complex()
    if ft.centered:
        F = torch.fft.ifftshift(F, dim=(-2, -1))
    img = torch.fft.ifft2(F, norm="ortho")
    return img.real


def radial_frequency_grid(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Returns (h,w) radial frequency in cycles/pixel, centered."""
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, d=1.0)).to(device=device, dtype=dtype)
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, d=1.0)).to(device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(yy * yy + xx * xx)


def band_filter(
    ft: FFTTensor,
    low_cycles_per_px: float = 0.0,
    high_cycles_per_px: float = 0.5,
    softness: float = 0.0,
) -> FFTTensor:
    """Smooth band-pass in the frequency domain.

    `softness` adds a half-cosine roll-off of that width on each side.
    """
    assert ft.centered, "band_filter assumes centered FFT"
    H, W = ft.spatial_h, ft.spatial_w
    r = radial_frequency_grid(H, W, device=ft.magnitude.device, dtype=ft.magnitude.dtype)
    if softness <= 0:
        mask = ((r >= low_cycles_per_px) & (r <= high_cycles_per_px)).to(ft.magnitude.dtype)
    else:
        # Smooth edges with raised-cosine
        def _rc(x, edge, soft, rising):
            t = ((x - edge) / max(soft, 1e-9)).clamp(-1, 1)
            return 0.5 - 0.5 * torch.cos(math.pi * (t * 0.5 + 0.5)) if rising else (1.0 - (0.5 - 0.5 * torch.cos(math.pi * (t * 0.5 + 0.5))))
        rise = _rc(r, low_cycles_per_px, softness, rising=True)
        fall = _rc(r, high_cycles_per_px, softness, rising=False)
        mask = (rise * fall).clamp(0, 1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return FFTTensor(magnitude=ft.magnitude * mask, phase=ft.phase, spatial_h=H, spatial_w=W, centered=True)


def ring_power_spectrum(ft: FFTTensor, n_bins: int = 64) -> torch.Tensor:
    """Average |F|^2 in concentric rings. Returns (B, C, n_bins)."""
    assert ft.centered
    H, W = ft.spatial_h, ft.spatial_w
    r = radial_frequency_grid(H, W, device=ft.magnitude.device, dtype=ft.magnitude.dtype)
    r_max = float(r.max())
    bin_idx = (r / max(r_max, 1e-9) * (n_bins - 1)).clamp(0, n_bins - 1).round().long()
    power = ft.magnitude.pow(2)
    B, C, _, _ = power.shape
    out = torch.zeros(B, C, n_bins, device=power.device, dtype=power.dtype)
    counts = torch.zeros(n_bins, device=power.device, dtype=power.dtype)
    flat_bin = bin_idx.flatten()
    counts.scatter_add_(0, flat_bin, torch.ones_like(flat_bin, dtype=power.dtype))
    flat_pow = power.flatten(2)  # (B,C,H*W)
    for b in range(B):
        for c in range(C):
            out[b, c].scatter_add_(0, flat_bin, flat_pow[b, c])
    return out / counts.clamp_min(1).unsqueeze(0).unsqueeze(0)


def match_ring_spectrum(noise: torch.Tensor, reference: torch.Tensor, n_bins: int = 64) -> torch.Tensor:
    """Reshape `noise`'s ring-averaged magnitude to match `reference`'s,
    preserving `noise`'s phase. Both inputs (B,C,H,W) real.
    """
    assert noise.shape == reference.shape
    nf = analyze(noise)
    rf = analyze(reference)
    n_pow = ring_power_spectrum(nf, n_bins)        # (B,C,n_bins)
    r_pow = ring_power_spectrum(rf, n_bins)
    H, W = noise.shape[-2:]
    rgrid = radial_frequency_grid(H, W, device=noise.device, dtype=noise.dtype)
    r_max = float(rgrid.max())
    bin_idx = (rgrid / max(r_max, 1e-9) * (n_bins - 1)).clamp(0, n_bins - 1).round().long()
    scale = (r_pow.clamp_min(1e-12) / n_pow.clamp_min(1e-12)).sqrt()  # (B,C,n_bins)
    # Map per-pixel scale via gather
    B, C = noise.shape[:2]
    flat_bin = bin_idx.flatten()
    scale_map = scale[:, :, flat_bin].view(B, C, H, W)
    new_mag = nf.magnitude * scale_map
    matched = FFTTensor(magnitude=new_mag, phase=nf.phase, spatial_h=H, spatial_w=W, centered=True)
    return synthesize(matched)
