"""Phase 6 (Edges) proofs."""
from __future__ import annotations

import torch

from nukemax.nodes.edges import MatteDensityAdjust, SubPixelEdgeDetect


def test_matte_density_passthrough_on_opaque():
    m = torch.ones(1, 8, 8)
    out = MatteDensityAdjust().execute(m, gamma=0.4, contrast=2.0, edge_lo=0.01, edge_hi=0.99)[0]
    assert torch.allclose(out, m, atol=1e-6)


def test_matte_density_passthrough_on_transparent():
    m = torch.zeros(1, 8, 8)
    out = MatteDensityAdjust().execute(m, gamma=0.4, contrast=2.0, edge_lo=0.01, edge_hi=0.99)[0]
    assert torch.allclose(out, m, atol=1e-6)


def test_subpixel_edge_detect_finds_step():
    img = torch.zeros(1, 32, 32, 3)
    img[:, :, 16:, :] = 1.0
    edges, td = SubPixelEdgeDetect().execute(img, top_k=8)
    # Strongest edges should sit on column 15/16 boundary.
    xs = td.coords[0, :, 0]
    assert ((xs >= 14) & (xs <= 17)).float().mean().item() > 0.5
