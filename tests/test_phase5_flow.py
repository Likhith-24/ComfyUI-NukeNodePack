"""Phase 5 (Flow) proofs."""
from __future__ import annotations

import math

import torch

from nukemax.core import flow as nflow
from nukemax.nodes.flow import CleanPlateMerge, FlowBackwardWarp, FlowOcclusionMask
from nukemax.types import FlowField


def test_backward_warp_node_identity():
    img = torch.rand(2, 32, 32, 3)
    flow = FlowField(flow_fwd=torch.zeros(1, 2, 32, 32), flow_bwd=torch.zeros(1, 2, 32, 32))
    out = FlowBackwardWarp().execute(img, flow, "forward")[0]
    diff = (img - out).abs()
    assert diff[..., 1:-1, 1:-1, :].max().item() < 1e-4


def test_occlusion_zero_for_consistent_flow():
    flow = FlowField(flow_fwd=torch.zeros(1, 2, 16, 16), flow_bwd=torch.zeros(1, 2, 16, 16))
    occ = FlowOcclusionMask().execute(flow)[0]
    assert occ.max().item() < 1e-6


def test_clean_plate_merge_zero_flow_passthrough():
    foot = torch.full((2, 16, 16, 3), 0.3)
    plate = torch.full((1, 16, 16, 3), 0.7)
    mask = torch.zeros(2, 16, 16)
    flow = FlowField(flow_fwd=torch.zeros(1, 2, 16, 16), flow_bwd=torch.zeros(1, 2, 16, 16))
    out = CleanPlateMerge().execute(foot, plate, mask, flow, feather_px=0.0)[0]
    # With zero mask the footage should win.
    assert torch.allclose(out, foot, atol=1e-4)
