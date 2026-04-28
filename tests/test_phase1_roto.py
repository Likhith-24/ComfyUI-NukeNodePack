"""Phase 1 (Roto) proofs."""
from __future__ import annotations

import math

import torch

from nukemax.nodes.roto import (
    RotoShapeRenderer,
    RotoShapeToDiffusionGuidance,
    RotoShapeToAITracker,
)
from nukemax.types import RotoShape, FlowField


def _make_circle_shape(N=64, r=20.0, cx=32.0, cy=32.0, H=64, W=64):
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    pts = torch.stack([cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=-1)
    return RotoShape.from_polygon(pts, (H, W), feather=0.0, closed=True)


def test_renderer_circle_area():
    s = _make_circle_shape()
    out = RotoShapeRenderer().execute(s, samples_per_segment=2, feather_override=0.0)[0]
    area = out.sum().item()
    assert abs(area - math.pi * 20 * 20) / (math.pi * 20 * 20) < 0.02


def test_diffusion_guidance_outputs_shapes():
    s = _make_circle_shape()
    hard, soft, lat, prompts = RotoShapeToDiffusionGuidance().execute(
        s, soft_radius_px=8.0, latent_downscale=8, samples_per_segment=4,
    )
    assert hard.shape == (1, 64, 64)
    assert soft.shape == (1, 64, 64)
    assert lat.shape == (1, 8, 8)
    assert isinstance(prompts, str) and "boxes" in prompts


def test_tracker_with_zero_flow_returns_static():
    s = _make_circle_shape()
    frames = torch.zeros(3, 64, 64, 3)
    flow = FlowField(flow_fwd=torch.zeros(2, 2, 64, 64), flow_bwd=torch.zeros(2, 2, 64, 64))
    animated, td = RotoShapeToAITracker().execute(s, frames, flow=flow)
    assert torch.allclose(animated.points[0], animated.points[-1], atol=1e-5)
    assert td.coords.shape == (3, s.N, 2)
