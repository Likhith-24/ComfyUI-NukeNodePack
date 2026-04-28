"""Phase 3 (Relight) proofs."""
from __future__ import annotations

import json

import torch

from nukemax.nodes.relight import (
    LightProbeEstimator,
    LightRigBuilder,
    MaterialDecomposerHeuristic,
    ThreePointRelight,
)
from nukemax.types import LightProbe, LightRig, MaterialSet


def _gradient_image(h=24, w=24):
    """A simple non-uniform image so heuristics produce non-trivial materials."""
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
    )
    img = torch.stack([xx, yy, (xx + yy) * 0.5], dim=-1).clamp(0, 1)
    return img.unsqueeze(0)  # BHWC


def test_heuristic_decomposer_produces_unit_normals():
    img = _gradient_image()
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    assert isinstance(ms, MaterialSet)
    # Albedo BCHW with right shape
    assert ms.albedo.shape == (1, 3, 24, 24)
    # Normals must be unit length per pixel
    n_norm = ms.normal.norm(dim=1)
    assert torch.allclose(n_norm, torch.ones_like(n_norm), atol=1e-4)
    # Depth in valid range
    assert ms.depth.min().item() >= 0.0499
    assert ms.depth.max().item() <= 1.0001


def test_light_rig_builder_default_three_lights():
    rig = LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0]
    assert isinstance(rig, LightRig)
    assert len(rig.lights) == 3
    assert rig.lights[0].intensity == 1.0
    assert rig.lights[1].intensity == 0.4
    assert rig.lights[2].intensity == 0.6
    assert rig.ambient == (0.05, 0.05, 0.05)


def test_light_rig_builder_parses_json_state():
    state = json.dumps(
        {
            "lights": [
                {
                    "direction": [0.0, 0.0, -1.0],
                    "color": [1.0, 0.5, 0.25],
                    "intensity": 2.5,
                    "type": "directional",
                }
            ],
            "ambient": 0.1,
        }
    )
    rig = LightRigBuilder().execute(state, 1.0, 0.4, 0.6, 0.05)[0]
    assert len(rig.lights) == 1
    assert rig.lights[0].intensity == 2.5
    assert rig.lights[0].color == (1.0, 0.5, 0.25)
    assert rig.ambient == (0.1, 0.1, 0.1)


def test_light_rig_builder_handles_invalid_json():
    rig = LightRigBuilder().execute("{not json", 1.0, 0.4, 0.6, 0.05)[0]
    # Falls back to default 3-point rig
    assert len(rig.lights) == 3


def test_three_point_relight_produces_image_in_range():
    img = _gradient_image()
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    rig = LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0]
    out = ThreePointRelight().execute(ms, rig, 50.0, True)[0]
    # BHWC, same spatial dims, values in [0,1]
    assert out.shape == (1, 24, 24, 3)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    # Not all zero (something was lit)
    assert out.mean().item() > 0.01


def test_light_probe_estimator_shape_and_finite():
    img = _gradient_image()
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    probe = LightProbeEstimator().execute(img, ms, 64, 128)[0]
    assert isinstance(probe, LightProbe)
    assert probe.env_map.shape == (1, 3, 64, 128)
    assert torch.isfinite(probe.env_map).all()
