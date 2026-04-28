"""Lambertian + Blinn-Phong shading on tensor maps."""
from __future__ import annotations

import math
from typing import Iterable

import torch

from ..types import LightRig, Light


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=1, keepdim=True).clamp_min(eps)


def shade_lambert_phong(
    albedo: torch.Tensor,     # (B,3,H,W) linear
    normal: torch.Tensor,     # (B,3,H,W) in [-1,1]
    depth: torch.Tensor,      # (B,1,H,W)
    rig: LightRig,
    roughness: torch.Tensor | None = None,  # (B,1,H,W) in [0,1]
    view_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
    fov_deg: float = 50.0,
) -> torch.Tensor:
    """Returns (B,3,H,W) linear RGB.

    Reconstructs world-ish positions from depth + a pinhole camera with
    the given vertical FOV. This is approximate (the depth model's units
    are not necessarily metric) but consistent for relighting.
    """
    B, _, H, W = albedo.shape
    device, dtype = albedo.device, albedo.dtype
    n = _normalize(normal)
    # Camera pinhole: x,y in image plane, z = depth.
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    z = depth.clamp_min(1e-4)
    x = xx.unsqueeze(0).unsqueeze(0) * z / f
    y = yy.unsqueeze(0).unsqueeze(0) * z / f
    pos = torch.cat([x, y, z], dim=1)  # (B,3,H,W)
    v = torch.tensor(view_dir, device=device, dtype=dtype).view(1, 3, 1, 1).expand_as(pos)
    v = _normalize(v)
    out = torch.zeros_like(albedo)
    # Ambient
    amb = torch.tensor(rig.ambient, device=device, dtype=dtype).view(1, 3, 1, 1)
    out = out + albedo * amb
    if roughness is None:
        roughness = torch.full((B, 1, H, W), 0.5, device=device, dtype=dtype)
    spec_power = (2.0 / (roughness.clamp(0.05, 1.0) ** 2 + 1e-4))  # GGX-ish
    for L in rig.lights:
        col = torch.tensor(L.color, device=device, dtype=dtype).view(1, 3, 1, 1) * float(L.intensity)
        if L.type == "directional":
            ldir = -torch.tensor(L.direction, device=device, dtype=dtype).view(1, 3, 1, 1)
            ldir = _normalize(ldir.expand_as(pos))
            atten = 1.0
        else:
            lpos = torch.tensor(L.position, device=device, dtype=dtype).view(1, 3, 1, 1)
            d = lpos - pos
            dist = d.norm(dim=1, keepdim=True).clamp_min(1e-4)
            ldir = d / dist
            atten = 1.0 / (dist ** float(L.falloff)).clamp_min(1e-6)
        ndotl = (n * ldir).sum(dim=1, keepdim=True).clamp_min(0)
        diffuse = albedo * col * ndotl * atten
        h = _normalize(ldir + v)
        ndoth = (n * h).sum(dim=1, keepdim=True).clamp_min(0)
        specular = col * (ndoth ** spec_power) * (1.0 - roughness) * atten
        out = out + diffuse + specular
    return out


def equirect_sample(env_map: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """env_map: (B,3,H,W) equirectangular; dirs: (B,3,H,W) unit world dirs.
    Returns (B,3,H,W) sampled radiance.
    """
    import torch.nn.functional as F
    x, y, z = dirs.unbind(dim=1)
    theta = torch.atan2(x, z)                 # azimuth in [-pi,pi]
    phi = torch.asin(y.clamp(-1, 1))           # elevation in [-pi/2,pi/2]
    u = theta / math.pi                         # [-1,1]
    v = -phi / (math.pi * 0.5)                  # [-1,1] (top of image = +y)
    grid = torch.stack([u, v], dim=-1)
    return F.grid_sample(env_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
