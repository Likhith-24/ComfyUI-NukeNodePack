"""Cubic Bezier evaluation + signed-distance rasterization."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def eval_cubic_bezier(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Each input: (..., 2). Returns (..., n_samples, 2)."""
    t = torch.linspace(0, 1, n_samples, device=p0.device, dtype=p0.dtype)
    t = t.view(*([1] * (p0.ndim - 1)), n_samples, 1)
    omt = 1 - t
    p0e = p0.unsqueeze(-2)
    p1e = p1.unsqueeze(-2)
    p2e = p2.unsqueeze(-2)
    p3e = p3.unsqueeze(-2)
    return (omt ** 3) * p0e + 3 * (omt ** 2) * t * p1e + 3 * omt * (t ** 2) * p2e + (t ** 3) * p3e


def shape_to_polyline(
    points: torch.Tensor,         # (T,N,2)
    handles_in: torch.Tensor,     # (T,N,2)
    handles_out: torch.Tensor,    # (T,N,2)
    closed: bool,
    samples_per_segment: int = 16,
) -> torch.Tensor:
    """Returns polyline (T, N*samples_per_segment, 2)."""
    T, N, _ = points.shape
    if N < 2:
        return points.clone()
    next_idx = torch.arange(N, device=points.device)
    next_idx = (next_idx + 1) % N if closed else (next_idx + 1).clamp(max=N - 1)
    p0 = points
    p3 = points[:, next_idx, :]
    p1 = handles_out
    p2 = handles_in[:, next_idx, :]
    segs = eval_cubic_bezier(p0, p1, p2, p3, samples_per_segment)  # (T,N,S,2)
    if not closed:
        segs = segs[:, :-1]
    return segs.reshape(T, -1, 2)


def rasterize_polygon_sdf(
    polyline: torch.Tensor,    # (T, P, 2)
    H: int,
    W: int,
    feather: torch.Tensor | float = 0.0,  # (T,) or scalar pixels
    closed: bool = True,
) -> torch.Tensor:
    """Rasterize a polygon to a soft mask via signed-distance + raycast.

    Returns (T, H, W) in [0,1]. Inside = 1, feathered band around edge.
    Vectorized across all polygon edges.
    """
    T, P, _ = polyline.shape
    device = polyline.device
    dtype = polyline.dtype
    # Build edges (T, E, 2, 2): start/end points per segment.
    if closed:
        next_pts = polyline[:, torch.arange(P, device=device).roll(-1)]
        edges = torch.stack([polyline, next_pts], dim=2)  # (T,P,2,2)
    else:
        edges = torch.stack([polyline[:, :-1], polyline[:, 1:]], dim=2)
    E = edges.shape[1]

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    pix = torch.stack([xx, yy], dim=-1)  # (H,W,2)
    pix_flat = pix.reshape(-1, 2)        # (HW, 2)

    a = edges[:, :, 0, :]  # (T,E,2)
    b = edges[:, :, 1, :]
    ab = b - a              # (T,E,2)
    ab_len2 = (ab * ab).sum(-1).clamp_min(1e-12)  # (T,E)

    # For memory: process pixels in chunks across H rows.
    out = torch.empty(T, H, W, device=device, dtype=dtype)
    chunk_rows = max(1, min(H, 65536 // max(W, 1) // max(E, 1)))
    for t in range(T):
        a_t = a[t]            # (E,2)
        b_t = b[t]
        ab_t = ab[t]
        ab_len2_t = ab_len2[t]
        for r0 in range(0, H, chunk_rows):
            r1 = min(H, r0 + chunk_rows)
            P_chunk = pix[r0:r1].reshape(-1, 2)        # (M,2)
            M = P_chunk.shape[0]
            ap = P_chunk.unsqueeze(1) - a_t.unsqueeze(0)    # (M,E,2)
            tparam = (ap * ab_t.unsqueeze(0)).sum(-1) / ab_len2_t.unsqueeze(0)
            tparam = tparam.clamp(0, 1)
            closest = a_t.unsqueeze(0) + tparam.unsqueeze(-1) * ab_t.unsqueeze(0)  # (M,E,2)
            d2 = ((P_chunk.unsqueeze(1) - closest) ** 2).sum(-1)                    # (M,E)
            min_d = d2.min(dim=1).values.sqrt()                                     # (M,)

            # Even-odd raycast for inside/outside via horizontal ray to +x.
            ay = a_t[:, 1].unsqueeze(0)
            by = b_t[:, 1].unsqueeze(0)
            ax = a_t[:, 0].unsqueeze(0)
            bx = b_t[:, 0].unsqueeze(0)
            py = P_chunk[:, 1].unsqueeze(1)
            px = P_chunk[:, 0].unsqueeze(1)
            cond_y = (ay > py) != (by > py)
            # x of edge at py: ax + (py-ay)*(bx-ax)/(by-ay)
            denom = (by - ay)
            denom_safe = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
            x_at = ax + (py - ay) * (bx - ax) / denom_safe
            cross = cond_y & (px < x_at)
            inside = (cross.sum(dim=1) % 2 == 1)

            f = float(feather) if not isinstance(feather, torch.Tensor) else float(feather[t].item())
            if not closed:
                # Open polylines (Nuke parity): no fill, only a soft stroke
                # band of width ``f`` along the polyline. With f=0 the mask
                # is empty.
                if f <= 0:
                    m = torch.zeros_like(min_d)
                else:
                    m = (1.0 - (min_d / f).clamp(0, 1))
            elif f <= 0:
                m = inside.to(dtype)
            else:
                # Soft band of width f around the edge.
                signed = torch.where(inside, min_d, -min_d)
                m = (signed / f + 0.5).clamp(0, 1)
            out[t, r0:r1] = m.view(r1 - r0, W)
    return out
