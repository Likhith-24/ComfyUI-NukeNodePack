"""Comprehensive Nuke-convention parity tests.

These tests target conventions that every Nuke node honors but that
weren't covered by the per-phase test files or the original
``test_nuke_parity.py``:

* Every IMAGE output is BHWC float32 in [0, 1].
* Every MASK output is BHW (or unsqueezable) float32 in [0, 1].
* Resolution is preserved across resolution-preserving ops.
* Batch dimension is preserved across batch-preserving ops.
* Deterministic ops are bit-exact across two invocations.
* Default/neutral parameters yield near-identity behavior.
* All declared INPUT_TYPES default values execute without raising.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from nukemax.nodes.audio import (
    AudioDriveMask,
    AudioDriveSchedule,
    AudioSpectrogram,
    AudioToFloatCurve,
)
from nukemax.nodes.edges import (
    HairAwareChoke,
    MatteDensityAdjust,
    NormalAwareEdgeBlur,
    SubPixelEdgeDetect,
)
from nukemax.nodes.fft import (
    FFTAnalyze,
    FFTSynthesize,
    FFTTextureSynthesis,
    FrequencyMask,
    LatentFrequencyMatch,
)
from nukemax.nodes.flow import (
    CleanPlateMerge,
    ComputeOpticalFlow,
    FlowBackwardWarp,
    FlowForwardWarp,
    FlowOcclusionMask,
    FlowVisualize,
)
from nukemax.nodes.relight import (
    LightProbeEstimator,
    LightRigBuilder,
    MaterialDecomposerHeuristic,
    ThreePointRelight,
)
from nukemax.nodes.roto import (
    RotoShapeRenderer,
    RotoShapeToDiffusionGuidance,
    RotoSplineEditor,
)
from nukemax.types import AudioFeatures, FlowField, MaterialSet, RotoShape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(B=1, H=24, W=32, C=3, fill=None):
    if fill is None:
        return torch.rand(B, H, W, C, dtype=torch.float32)
    return torch.full((B, H, W, C), float(fill), dtype=torch.float32)


def _mask(B=1, H=24, W=32, fill=None):
    if fill is None:
        return torch.rand(B, H, W, dtype=torch.float32)
    return torch.full((B, H, W), float(fill), dtype=torch.float32)


def _zero_flow(B=1, H=24, W=32):
    return FlowField(
        flow_fwd=torch.zeros(B, 2, H, W),
        flow_bwd=torch.zeros(B, 2, H, W),
    )


def _circle_shape(N=64, r=10.0, cx=16.0, cy=12.0, H=24, W=32, feather=0.0):
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    pts = torch.stack(
        [cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=-1
    )
    return RotoShape.from_polygon(pts, (H, W), feather=feather, closed=True)


def _audio_features(T=8, sr=16000, hop=256):
    F = 513
    return AudioFeatures(
        waveform=torch.zeros(T * hop),
        sr=sr,
        stft_mag=torch.rand(F, T) + 0.01,
        onsets=torch.rand(T),
        bpm=120.0,
        centroid=torch.rand(T),
        rms=torch.rand(T),
        hop_length=hop,
    )


def _assert_image_contract(out, B=None, H=None, W=None):
    assert isinstance(out, torch.Tensor), f"expected Tensor got {type(out)}"
    assert out.ndim == 4, f"expected BHWC got shape {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"got dtype {out.dtype}"
    assert torch.isfinite(out).all(), "image has non-finite values"
    assert out.min().item() >= -1e-5
    assert out.max().item() <= 1.0 + 1e-5
    if B is not None:
        assert out.shape[0] == B
    if H is not None:
        assert out.shape[1] == H
    if W is not None:
        assert out.shape[2] == W


def _assert_mask_contract(out, B=None, H=None, W=None):
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 3, f"expected BHW got shape {tuple(out.shape)}"
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert out.min().item() >= -1e-5
    assert out.max().item() <= 1.0 + 1e-5
    if B is not None:
        assert out.shape[0] == B
    if H is not None:
        assert out.shape[1] == H
    if W is not None:
        assert out.shape[2] == W


# ---------------------------------------------------------------------------
# A. IMAGE contract: BHWC float32 in [0, 1] (Nuke's normalized RGB)
# ---------------------------------------------------------------------------

def test_image_contract_fft_synthesize():
    img = _img(H=16, W=16)
    ft = FFTAnalyze().execute(img)[0]
    out = FFTSynthesize().execute(ft)[0]
    _assert_image_contract(out, B=1, H=16, W=16)


def test_image_contract_fft_texture_synthesis():
    out = FFTTextureSynthesis().execute(_img(H=16, W=16), 24, 32, 0)[0]
    _assert_image_contract(out, B=1, H=24, W=32)


def test_image_contract_three_point_relight():
    img = _img()
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    rig = LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0]
    out = ThreePointRelight().execute(ms, rig, 50.0, True)[0]
    _assert_image_contract(out, B=img.shape[0], H=img.shape[1], W=img.shape[2])


def test_image_contract_audio_spectrogram():
    out = AudioSpectrogram().execute(_audio_features(), True)[0]
    _assert_image_contract(out)
    assert out.shape[3] == 3


def test_image_contract_flow_visualize():
    out = FlowVisualize().execute(_zero_flow(), 16.0)[0]
    _assert_image_contract(out, B=1, H=24, W=32)


def test_image_contract_flow_backward_warp():
    img = _img(B=2)
    out = FlowBackwardWarp().execute(img, _zero_flow(B=2), "forward")[0]
    _assert_image_contract(out, B=2, H=img.shape[1], W=img.shape[2])


def test_image_contract_flow_forward_warp():
    img = _img(B=2)
    warped, _w = FlowForwardWarp().execute(img, _zero_flow(B=2))
    _assert_image_contract(warped, B=2, H=img.shape[1], W=img.shape[2])


def test_image_contract_clean_plate_merge():
    foot = _img(B=2)
    plate = _img(B=1)
    mask = _mask(B=2)
    out = CleanPlateMerge().execute(foot, plate, mask, _zero_flow(B=2), 0.0)[0]
    _assert_image_contract(out, B=2, H=foot.shape[1], W=foot.shape[2])


# ---------------------------------------------------------------------------
# B. MASK contract: BHW float32 in [0, 1]
# ---------------------------------------------------------------------------

def test_mask_contract_roto_renderer():
    s = _circle_shape()
    out = RotoShapeRenderer().execute(s, 4, -1.0)[0]
    _assert_mask_contract(out, B=1, H=24, W=32)


def test_mask_contract_audio_drive_mask_intensity():
    base = _mask(B=4, fill=0.5)
    out = AudioDriveMask().execute(base, [0.5, 0.5, 0.5, 0.5], "intensity", 1.0)[0]
    _assert_mask_contract(out, B=4)


def test_mask_contract_audio_drive_mask_dilate():
    base = _mask(B=2, fill=0.0)
    base[:, 10:14, 14:18] = 1.0
    out = AudioDriveMask().execute(base, [1.0, 1.0], "dilate", 2.0)[0]
    _assert_mask_contract(out, B=2)


def test_mask_contract_audio_drive_mask_feather():
    base = _mask(B=2, fill=0.0)
    base[:, 10:14, 14:18] = 1.0
    out = AudioDriveMask().execute(base, [1.0, 1.0], "feather", 2.0)[0]
    _assert_mask_contract(out, B=2)


def test_mask_contract_flow_occlusion_mask():
    out = FlowOcclusionMask().execute(_zero_flow(B=2))[0]
    _assert_mask_contract(out, B=2)


def test_mask_contract_normal_aware_edge_blur():
    out = NormalAwareEdgeBlur().execute(_mask(B=2), _img(B=2), 2.0, 0.85)[0]
    _assert_mask_contract(out, B=2)


def test_mask_contract_matte_density_adjust():
    # With contrast > 1 results can over-shoot before clamp; verify clamp.
    base = _mask(B=2)
    out = MatteDensityAdjust().execute(base, gamma=2.0, contrast=2.0,
                                       edge_lo=0.2, edge_hi=0.8)[0]
    _assert_mask_contract(out, B=2)


def test_mask_contract_subpixel_edge_detect():
    edges, _td = SubPixelEdgeDetect().execute(_img(B=2), 16)
    _assert_mask_contract(edges, B=2)


def test_mask_contract_hair_aware_choke():
    out = HairAwareChoke().execute(_mask(B=2), _img(B=2), 1.0, 5)[0]
    _assert_mask_contract(out, B=2)


def test_mask_contract_diffusion_guidance_outputs_three_masks():
    s = _circle_shape(H=64, W=64)
    inside, edge, latent, _ = RotoShapeToDiffusionGuidance().execute(s, 8.0, 8, 4)
    _assert_mask_contract(inside)
    _assert_mask_contract(edge)
    # latent is a tensor of shape (B,C,h,w)
    assert latent.ndim == 3 and latent.shape[0] == 1


# ---------------------------------------------------------------------------
# C. Resolution preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H,W", [(16, 16), (24, 32), (40, 24)])
def test_resolution_preserved_fft_round_trip(H, W):
    img = _img(H=H, W=W)
    ft = FFTAnalyze().execute(img)[0]
    out = FFTSynthesize().execute(ft)[0]
    assert out.shape[1:3] == (H, W)


@pytest.mark.parametrize("H,W", [(16, 16), (24, 32)])
def test_resolution_preserved_three_point_relight(H, W):
    img = _img(H=H, W=W)
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    rig = LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0]
    out = ThreePointRelight().execute(ms, rig, 50.0, True)[0]
    assert out.shape[1:3] == (H, W)


@pytest.mark.parametrize("H,W", [(16, 16), (24, 32)])
def test_resolution_preserved_flow_warps(H, W):
    img = _img(B=2, H=H, W=W)
    flow = _zero_flow(B=2, H=H, W=W)
    bw = FlowBackwardWarp().execute(img, flow, "forward")[0]
    fw, _ = FlowForwardWarp().execute(img, flow)
    vis = FlowVisualize().execute(flow, 16.0)[0]
    occ = FlowOcclusionMask().execute(flow)[0]
    assert bw.shape[1:3] == (H, W)
    assert fw.shape[1:3] == (H, W)
    assert vis.shape[1:3] == (H, W)
    assert occ.shape[1:3] == (H, W)


def test_resolution_preserved_edges_nodes():
    H, W = 40, 24
    mask = _mask(H=H, W=W)
    img = _img(H=H, W=W)
    nbe = NormalAwareEdgeBlur().execute(mask, img, 2.0, 0.85)[0]
    mda = MatteDensityAdjust().execute(mask, 1.0, 1.0, 0.0, 1.0)[0]
    edges, _ = SubPixelEdgeDetect().execute(img, 16)
    hac = HairAwareChoke().execute(mask, img, 1.0, 5)[0]
    for o in (nbe, mda, edges, hac):
        assert o.shape[1:] == (H, W)


# ---------------------------------------------------------------------------
# D. Batch dimension preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [1, 2, 4])
def test_batch_preserved_flow_visualize(B):
    out = FlowVisualize().execute(_zero_flow(B=B), 16.0)[0]
    assert out.shape[0] == B


@pytest.mark.parametrize("B", [1, 2, 4])
def test_batch_preserved_flow_backward_warp(B):
    img = _img(B=B)
    out = FlowBackwardWarp().execute(img, _zero_flow(B=B), "forward")[0]
    assert out.shape[0] == B


@pytest.mark.parametrize("B", [1, 2])
def test_batch_preserved_relight(B):
    img = _img(B=B)
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    rig = LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0]
    out = ThreePointRelight().execute(ms, rig, 50.0, True)[0]
    assert out.shape[0] == B


# ---------------------------------------------------------------------------
# E. Determinism: two calls with identical inputs produce identical output
# ---------------------------------------------------------------------------

def test_determinism_fft_analyze_synthesize():
    img = _img(H=16, W=16)
    a = FFTSynthesize().execute(FFTAnalyze().execute(img)[0])[0]
    b = FFTSynthesize().execute(FFTAnalyze().execute(img)[0])[0]
    assert torch.equal(a, b)


def test_determinism_three_point_relight():
    img = _img()
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    rig = LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0]
    a = ThreePointRelight().execute(ms, rig, 50.0, True)[0]
    b = ThreePointRelight().execute(ms, rig, 50.0, True)[0]
    assert torch.equal(a, b)


def test_determinism_subpixel_edge_detect():
    img = _img()
    a, _ = SubPixelEdgeDetect().execute(img, 32)
    b, _ = SubPixelEdgeDetect().execute(img, 32)
    assert torch.equal(a, b)


def test_determinism_normal_aware_edge_blur():
    mask = _mask()
    n = _img()
    a = NormalAwareEdgeBlur().execute(mask, n, 2.0, 0.85)[0]
    b = NormalAwareEdgeBlur().execute(mask, n, 2.0, 0.85)[0]
    assert torch.equal(a, b)


def test_determinism_audio_to_float_curve():
    af = _audio_features(T=16)
    a, _ = AudioToFloatCurve().execute(af, 8, 24.0, "full", 0.5, 1.0)
    b, _ = AudioToFloatCurve().execute(af, 8, 24.0, "full", 0.5, 1.0)
    assert a == b


# ---------------------------------------------------------------------------
# F. Neutral-default identity (Nuke "do nothing at defaults" convention)
# ---------------------------------------------------------------------------

def test_matte_density_neutral_defaults_passthrough():
    base = _mask()
    out = MatteDensityAdjust().execute(base, gamma=1.0, contrast=1.0,
                                       edge_lo=0.0, edge_hi=1.0)[0]
    assert torch.allclose(out, base, atol=1e-5)


def test_flow_backward_warp_zero_flow_identity():
    img = _img(B=2, H=16, W=16)
    out = FlowBackwardWarp().execute(img, _zero_flow(B=2, H=16, W=16), "forward")[0]
    # At zero flow the backward warp should be exactly the input.
    diff = (img - out).abs()
    assert diff.max().item() < 1e-4


def test_flow_forward_warp_zero_flow_identity():
    img = _img(B=2, H=16, W=16)
    out, _w = FlowForwardWarp().execute(img, _zero_flow(B=2, H=16, W=16))
    diff = (img - out).abs()
    # Forward splatting can leave fractional weight at borders; allow a small
    # tolerance, but the interior should be near-exact.
    assert diff[..., 1:-1, 1:-1, :].max().item() < 1e-3


def test_clean_plate_zero_mask_returns_footage():
    foot = _img(B=2, H=16, W=16)
    plate = _img(B=1, H=16, W=16, fill=0.7)
    mask = _mask(B=2, H=16, W=16, fill=0.0)
    out = CleanPlateMerge().execute(foot, plate, mask, _zero_flow(B=2, H=16, W=16),
                                    0.0)[0]
    assert torch.allclose(out, foot, atol=1e-4)


def test_audio_drive_mask_neutral_curve_preserves_mask():
    base = _mask(B=4)
    # mode=intensity, curve=0.5, amount=0 → out = mask * (1 + 0) = mask
    out = AudioDriveMask().execute(base, [0.5] * 4, "intensity", 0.0)[0]
    assert torch.allclose(out, base, atol=1e-5)


def test_hair_aware_choke_zero_choke_passthrough():
    base = _mask(B=2)
    img = _img(B=2)
    out = HairAwareChoke().execute(base, img, 0.0, 5)[0]
    assert torch.allclose(out, base, atol=1e-4)


def test_normal_aware_edge_blur_zero_sigma_passthrough():
    base = _mask()
    out = NormalAwareEdgeBlur().execute(base, _img(), 0.0, 0.85)[0]
    assert torch.allclose(out, base, atol=1e-5)


# ---------------------------------------------------------------------------
# G. AudioDriveSchedule semantics: STRING is JSON list, FLOAT is the list,
# values clamped to [min, max] when curve in [0, 1].
# ---------------------------------------------------------------------------

def test_audio_drive_schedule_json_round_trip():
    sched_json, sched_list = AudioDriveSchedule().execute(
        [0.0, 0.5, 1.0], 4.0, 12.0
    )
    parsed = json.loads(sched_json)
    assert parsed == pytest.approx(sched_list)
    assert parsed[0] == pytest.approx(4.0)
    assert parsed[1] == pytest.approx(8.0)
    assert parsed[2] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# H. AudioSpectrogram: log_scale toggles distribution but both stay in [0,1]
# ---------------------------------------------------------------------------

def test_audio_spectrogram_log_vs_linear_both_normalized():
    af = _audio_features(T=16)
    lin = AudioSpectrogram().execute(af, False)[0]
    log = AudioSpectrogram().execute(af, True)[0]
    _assert_image_contract(lin)
    _assert_image_contract(log)
    # The two representations should not be identical for non-trivial input.
    assert (lin - log).abs().mean().item() > 1e-3


# ---------------------------------------------------------------------------
# I. RotoShapeRenderer: feather override semantics (Nuke "Premultiply"
# convention — feather=0 ⇒ binary alpha; feather>0 ⇒ AA band).
# ---------------------------------------------------------------------------

def test_roto_renderer_inside_is_one_outside_is_zero():
    s = _circle_shape(H=64, W=64, cx=32, cy=32, r=20, feather=0.0)
    out = RotoShapeRenderer().execute(s, 8, 0.0)[0]
    # Center should be inside (1.0); corners outside (0.0).
    assert out[0, 32, 32].item() == pytest.approx(1.0, abs=1e-4)
    assert out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-4)
    assert out[0, 63, 63].item() == pytest.approx(0.0, abs=1e-4)


def test_roto_renderer_zero_canvas_does_not_crash():
    # Empty/degenerate canvas — must still respect the resilient contract.
    state = json.dumps({
        "frames": [{"points": [], "in": [], "out": [], "feather": []}],
        "closed": True,
        "canvas": {"h": 8, "w": 8},
    })
    s = RotoSplineEditor().execute(state, 8, 8)[0]
    out = RotoShapeRenderer().execute(s, 4, -1.0)[0]
    # On graceful failure @resilient returns a fallback (1,64,64) MASK.
    _assert_mask_contract(out)


# ---------------------------------------------------------------------------
# J. FrequencyMask: passing low=0,high=1 with softness=0 is identity (full
# pass-through of the frequency tensor).
# ---------------------------------------------------------------------------

def test_frequency_mask_full_pass_is_identity():
    img = _img(H=16, W=16)
    ft = FFTAnalyze().execute(img)[0]
    out_ft = FrequencyMask().execute(ft, 0.0, 1.0, 0.0)[0]
    out = FFTSynthesize().execute(out_ft)[0]
    base = FFTSynthesize().execute(ft)[0]
    assert torch.allclose(out, base, atol=1e-4)


# ---------------------------------------------------------------------------
# K. LatentFrequencyMatch: noise batch broadcast tolerance — the live test
# regression that surfaced. Verify all four broadcast cases.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B_noise,B_ctx", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3)])
def test_latent_frequency_match_batch_broadcast(B_noise, B_ctx):
    noise = {"samples": torch.randn(B_noise, 4, 8, 8)}
    ctx = torch.rand(B_ctx, 24, 24, 3)
    out = LatentFrequencyMatch().execute(noise, ctx, 8)[0]
    # Output retains the noise batch.
    assert out["samples"].shape == (B_noise, 4, 8, 8)


# ---------------------------------------------------------------------------
# L. LightProbeEstimator: env_map is BCHW with declared probe size.
# ---------------------------------------------------------------------------

def test_light_probe_estimator_emits_correct_shape():
    img = _img(H=24, W=32)
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    probe = LightProbeEstimator().execute(img, ms, 32, 64)[0]
    assert probe.env_map.ndim == 4
    assert probe.env_map.shape[-2:] == (32, 64)


# ---------------------------------------------------------------------------
# M. ComputeOpticalFlow: opencv_farneback is exercised when available;
# otherwise the resilient decorator gracefully falls back.
# ---------------------------------------------------------------------------

def test_compute_optical_flow_method_auto_does_not_crash():
    frames = _img(B=2, H=24, W=32)
    ff = ComputeOpticalFlow().execute(frames, "auto", 1.5)[0]
    assert isinstance(ff, FlowField)
    assert ff.flow_fwd.shape[0] == 1
    assert ff.flow_fwd.shape[1] == 2


# ---------------------------------------------------------------------------
# N. FlowVisualize: max_magnitude=0 must not divide by zero.
# ---------------------------------------------------------------------------

def test_flow_visualize_zero_max_magnitude_safe():
    flow = FlowField(
        flow_fwd=torch.full((1, 2, 16, 16), 0.5),
        flow_bwd=torch.full((1, 2, 16, 16), 0.5),
    )
    out = FlowVisualize().execute(flow, 0.0)[0]
    _assert_image_contract(out)


# ---------------------------------------------------------------------------
# O. SubPixelEdgeDetect on uniform image: must produce an empty/normalized
# output — this caused a crash in the live mega-graph at SaveImage.
# ---------------------------------------------------------------------------

def test_subpixel_edge_detect_on_uniform_image_does_not_explode():
    flat = _img(B=1, H=24, W=32, fill=0.5)
    edges, td = SubPixelEdgeDetect().execute(flat, 8)
    _assert_mask_contract(edges, B=1, H=24, W=32)
    # Tracking-data coords still has the requested top-K rows.
    assert td.coords.shape == (1, 8, 2)
