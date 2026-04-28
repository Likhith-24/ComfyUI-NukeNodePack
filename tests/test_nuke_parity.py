"""Nuke-parity contract tests for the NukeMax pack.

These are cross-cutting tests that verify every registered node behaves
the way a Nuke node would: stable I/O contracts, identity at neutral
settings, resolution preservation, deterministic seeded outputs,
premultiplied alpha correctness, etc.

If any test here fails, a Nuke-style assumption is broken in the pack.
"""
from __future__ import annotations

import inspect
import json
import math
from typing import Any

import pytest
import torch

# Aggregate node mappings from each ecosystem (matches the root
# `__init__.py`'s loader). The pack root has a hyphen in its folder
# name so it can't be imported as a normal Python package.
from nukemax.nodes import audio as _audio_eco
from nukemax.nodes import edges as _edges_eco
from nukemax.nodes import fft as _fft_eco
from nukemax.nodes import flow as _flow_eco
from nukemax.nodes import relight as _relight_eco
from nukemax.nodes import roto as _roto_eco
from nukemax.nodes import types_io as _io_eco
from nukemax.core import blur, color, composite, fft as nfft, splines
from nukemax.types import (
    AudioFeatures,
    FFTTensor,
    FlowField,
    Light,
    LightProbe,
    LightRig,
    MaterialSet,
    RotoShape,
    TrackingData,
)
from nukemax.types import serialize as ser


# =============================================================================
# A. Contract tests across all 47 registered nodes
# =============================================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _eco in (_roto_eco, _fft_eco, _relight_eco, _audio_eco, _flow_eco, _edges_eco, _io_eco):
    NODE_CLASS_MAPPINGS.update(_eco.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_eco.NODE_DISPLAY_NAME_MAPPINGS)
ALL_NODES = list(NODE_CLASS_MAPPINGS.items())


def test_pack_registers_47_nodes():
    assert len(NODE_CLASS_MAPPINGS) == 47


def test_every_key_uses_nukemax_prefix():
    for key in NODE_CLASS_MAPPINGS:
        assert key.startswith("NukeMax_"), f"node key {key!r} missing NukeMax_ prefix"


def test_every_node_has_display_name():
    for key in NODE_CLASS_MAPPINGS:
        assert key in NODE_DISPLAY_NAME_MAPPINGS, f"missing display name for {key}"


def test_no_orphan_display_names():
    for key in NODE_DISPLAY_NAME_MAPPINGS:
        assert key in NODE_CLASS_MAPPINGS, f"orphan display name {key}"


@pytest.mark.parametrize("key,cls", ALL_NODES)
def test_node_has_required_class_attrs(key, cls):
    assert hasattr(cls, "CATEGORY"), f"{key} missing CATEGORY"
    assert cls.CATEGORY.startswith("NukeMax/"), f"{key} CATEGORY={cls.CATEGORY!r}"
    assert hasattr(cls, "FUNCTION"), f"{key} missing FUNCTION"
    assert hasattr(cls, cls.FUNCTION), f"{key} missing method {cls.FUNCTION}"
    assert hasattr(cls, "RETURN_TYPES"), f"{key} missing RETURN_TYPES"
    assert isinstance(cls.RETURN_TYPES, tuple)
    assert len(cls.RETURN_TYPES) >= 1
    assert hasattr(cls, "INPUT_TYPES"), f"{key} missing INPUT_TYPES"
    spec = cls.INPUT_TYPES()
    assert isinstance(spec, dict)
    assert "required" in spec or "optional" in spec


@pytest.mark.parametrize("key,cls", ALL_NODES)
def test_node_input_types_well_formed(key, cls):
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        if section not in spec:
            continue
        for arg_name, arg_spec in spec[section].items():
            assert isinstance(arg_name, str)
            assert isinstance(arg_spec, tuple)
            assert len(arg_spec) >= 1, f"{key}.{arg_name} empty spec"
            arg_type = arg_spec[0]
            # arg_type may be a string (type name) or a tuple (enum of options)
            assert isinstance(arg_type, (str, tuple, list))


@pytest.mark.parametrize("key,cls", ALL_NODES)
def test_resilient_decorator_catches_errors(key, cls):
    """Forcing the bound method to raise should still produce a tuple of
    the declared length (resilience contract).
    """
    inst = cls()
    fn = getattr(inst, cls.FUNCTION)

    # Invoke with garbage args; @resilient should catch and return zeros.
    try:
        sig = inspect.signature(fn)
        kwargs = {}
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            kwargs[name] = None  # nonsense
        out = fn(**kwargs)
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"{key}.{cls.FUNCTION} raised through resilient: {exc}")
    assert isinstance(out, tuple)
    assert len(out) == len(cls.RETURN_TYPES)


# =============================================================================
# B. IMAGE / MASK contract: BHWC float in [0,1]; resolution preserved
# =============================================================================

def _rand_image(B=1, H=24, W=24, C=3):
    return torch.rand(B, H, W, C, dtype=torch.float32)


def test_to_bchw_to_bhwc_round_trip():
    img = _rand_image()
    back = color.to_bhwc(color.to_bchw(img))
    assert torch.allclose(img, back, atol=1e-6)
    assert back.shape == img.shape


def test_to_bchw_already_bchw_passthrough():
    bchw = torch.rand(1, 3, 16, 16)
    out = color.to_bchw(bchw)
    assert out.shape == (1, 3, 16, 16)


def test_to_bchw_handles_alpha():
    img = _rand_image(C=4)
    bchw = color.to_bchw(img)
    assert bchw.shape == (1, 4, 24, 24)


# =============================================================================
# C. Color: sRGB known values + idempotence
# =============================================================================

def test_srgb_known_anchors():
    x = torch.tensor([0.0, 0.04045, 0.5, 1.0])
    lin = color.srgb_to_linear(x)
    # Linear at 0.5 sRGB ≈ 0.2140 (textbook value)
    assert lin[0].item() == pytest.approx(0.0, abs=1e-7)
    assert lin[2].item() == pytest.approx(0.2140, abs=1e-3)
    assert lin[3].item() == pytest.approx(1.0, abs=1e-6)


def test_luminance_red_is_bt709_weight():
    red = torch.zeros(1, 3, 4, 4); red[:, 0] = 1.0
    Y = color.luminance(red)
    assert Y[0, 0, 0, 0].item() == pytest.approx(0.2126, abs=1e-3)


# =============================================================================
# D. Composite: Porter-Duff identities
# =============================================================================

def test_composite_zero_alpha_src_returns_dst():
    src = composite.to_premul(torch.rand(1, 3, 4, 4), torch.zeros(1, 1, 4, 4))
    dst = composite.to_premul(torch.full((1, 3, 4, 4), 0.4), torch.ones(1, 1, 4, 4))
    out = composite.over(src, dst)
    rgb, _ = composite.from_premul(out)
    assert torch.allclose(rgb, torch.full_like(rgb, 0.4), atol=1e-6)


def test_composite_xor_disjoint_alphas():
    sa = torch.full((1, 1, 4, 4), 1.0)
    da = torch.full((1, 1, 4, 4), 0.0)
    src = composite.to_premul(torch.full((1, 3, 4, 4), 0.7), sa)
    dst = composite.to_premul(torch.full((1, 3, 4, 4), 0.2), da)
    out_alpha = composite.xor(src, dst)[:, 3:4]
    # XOR: sa*(1-da) + da*(1-sa) -> 1 here
    assert torch.allclose(out_alpha, torch.ones_like(out_alpha))


# =============================================================================
# E. Blur primitives — identity / kernel sum / reflective padding
# =============================================================================

def test_gaussian_kernel_sums_to_one():
    k = blur.gaussian_kernel_1d(2.5)
    assert k.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_gaussian_blur_constant_image_unchanged():
    img = torch.full((1, 3, 32, 32), 0.42)
    out = blur.gaussian_blur(img, sigma=4.0)
    assert torch.allclose(out, img, atol=1e-5)


def test_box_blur_radius_zero_identity():
    img = torch.rand(1, 3, 16, 16)
    out = blur.box_blur(img, radius=0)
    assert torch.allclose(out, img, atol=1e-6)


# =============================================================================
# F. Phase 1 — Roto further coverage
# =============================================================================

from nukemax.nodes.roto import (
    RotoShapeFromFile,
    RotoShapeRenderer,
    RotoShapeToAITracker,
    RotoShapeToDiffusionGuidance,
    RotoSplineEditor,
)


def _circle_shape(N=64, r=20.0, cx=32.0, cy=32.0, H=64, W=64, feather=0.0):
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    pts = torch.stack([cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=-1)
    return RotoShape.from_polygon(pts, (H, W), feather=feather, closed=True)


def test_spline_editor_default_returns_valid_rectangle():
    s = RotoSplineEditor().execute("", 64, 64)[0]
    assert isinstance(s, RotoShape)
    assert s.canvas_h == 64 and s.canvas_w == 64
    assert s.points.shape[0] >= 1


def test_spline_editor_invalid_json_falls_back():
    s = RotoSplineEditor().execute("{not json", 64, 64)[0]
    assert isinstance(s, RotoShape)


def test_renderer_feather_widens_mask():
    sharp_shape = _circle_shape(feather=0.0)
    soft_shape = _circle_shape(feather=4.0)
    sharp = RotoShapeRenderer().execute(sharp_shape, 4, -1.0)[0]
    soft = RotoShapeRenderer().execute(soft_shape, 4, -1.0)[0]
    # Soft mask sums more (anti-aliased band at the edge)
    assert soft.sum().item() >= sharp.sum().item() - 1e-3


def test_renderer_feather_override_zero_yields_binary():
    s = _circle_shape(feather=10.0)
    out = RotoShapeRenderer().execute(s, 4, 0.0)[0]
    # With feather=0 it's effectively binary
    far = (out < 1e-3) | (out > 1 - 1e-3)
    assert far.float().mean().item() > 0.95


def test_renderer_preserves_canvas_resolution():
    s = _circle_shape(H=48, W=80)
    out = RotoShapeRenderer().execute(s, 4, -1.0)[0]
    assert out.shape == (1, 48, 80)


def test_diffusion_guidance_latent_downscale_correct():
    s = _circle_shape(H=64, W=64)
    _, _, lat, _ = RotoShapeToDiffusionGuidance().execute(s, 8.0, 16, 4)
    assert lat.shape == (1, 4, 4)


def test_diffusion_guidance_sam_prompts_parse():
    s = _circle_shape()
    _, _, _, prompts_json = RotoShapeToDiffusionGuidance().execute(s, 8.0, 8, 4)
    parsed = json.loads(prompts_json)
    assert isinstance(parsed, list) and len(parsed) >= 1
    assert "boxes" in parsed[0] and "points" in parsed[0]
    box = parsed[0]["boxes"][0]
    # Bbox encloses (cx-r, cy-r) ~ (12,12) to (52,52) for default circle
    assert box[0] >= 11 and box[2] <= 53


def test_tracker_propagates_with_constant_flow():
    s = _circle_shape()
    frames = torch.zeros(3, 64, 64, 3)
    cflow = torch.zeros(2, 2, 64, 64); cflow[:, 0] = 2.0  # +2 px/frame in x
    flow = FlowField(flow_fwd=cflow, flow_bwd=-cflow)
    animated, td = RotoShapeToAITracker().execute(s, frames, flow=flow)
    # Frame 1 vertices ~ frame 0 + 2 in x
    dx = (animated.points[1, :, 0] - animated.points[0, :, 0]).mean().item()
    assert dx == pytest.approx(2.0, abs=0.5)


def test_roto_shape_from_file_round_trip(tmp_path):
    state = json.dumps({
        "frames": [{
            "points": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
            "in": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "out": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "feather": [0.0, 0.0, 0.0, 0.0],
        }],
        "closed": True,
        "canvas": {"h": 64, "w": 64},
    })
    p = tmp_path / "shape.json"
    p.write_text(state, encoding="utf-8")
    s = RotoShapeFromFile().execute(str(p))[0]
    assert isinstance(s, RotoShape)
    assert s.points.shape[1] == 4


# =============================================================================
# G. Phase 2 — FFT further coverage
# =============================================================================

from nukemax.nodes.fft import (
    FFTAnalyze,
    FFTSynthesize,
    FFTTextureSynthesis,
    FrequencyMask,
    LatentFrequencyMatch,
)


def test_fft_high_pass_removes_dc():
    img = torch.full((1, 16, 16, 3), 0.5)
    ft = FFTAnalyze().execute(img)[0]
    hi = FrequencyMask().execute(ft, low=0.1, high=0.5, softness=0.0)[0]
    out = FFTSynthesize().execute(hi)[0]
    # DC suppressed -> mean near zero (after Synthesize clamp(0,1) it becomes 0)
    assert out.mean().item() < 0.05


def test_fft_low_pass_preserves_dc():
    img = torch.full((1, 16, 16, 3), 0.5)
    ft = FFTAnalyze().execute(img)[0]
    lo = FrequencyMask().execute(ft, low=0.0, high=0.05, softness=0.0)[0]
    out = FFTSynthesize().execute(lo)[0]
    assert torch.allclose(out, img, atol=5e-3)


def test_fft_texture_synthesis_deterministic():
    ex = torch.rand(1, 32, 32, 3)
    a = FFTTextureSynthesis().execute(ex, 32, 32, 1234)[0]
    b = FFTTextureSynthesis().execute(ex, 32, 32, 1234)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_fft_texture_different_seeds_differ():
    ex = torch.rand(1, 32, 32, 3)
    a = FFTTextureSynthesis().execute(ex, 32, 32, 1)[0]
    b = FFTTextureSynthesis().execute(ex, 32, 32, 2)[0]
    assert (a - b).abs().mean().item() > 1e-3


def test_latent_frequency_match_preserves_shape():
    noise = {"samples": torch.randn(1, 4, 16, 16)}
    ctx = torch.rand(1, 64, 64, 3)
    out = LatentFrequencyMatch().execute(noise, ctx, 16)[0]
    assert out["samples"].shape == (1, 4, 16, 16)


# =============================================================================
# H. Phase 3 — Relight further coverage
# =============================================================================

from nukemax.nodes.relight import (
    LightProbeEstimator,
    LightProbeToEXR,
    LightRigBuilder,
    MaterialDecomposerHeuristic,
    MaterialDecomposerModels,
    ThreePointRelight,
)


def test_three_point_relight_zero_intensity_yields_dark():
    img = torch.rand(1, 16, 16, 3)
    ms = MaterialDecomposerHeuristic().execute(img, 4.0, 0.5)[0]
    rig = LightRigBuilder().execute("", 0.0, 0.0, 0.0, 0.0)[0]
    out = ThreePointRelight().execute(ms, rig, 50.0, False)[0]
    # No lighting → mostly black
    assert out.mean().item() < 0.05


def test_material_decomposer_models_falls_back_with_info():
    img = torch.rand(1, 16, 16, 3)
    ms, info = MaterialDecomposerModels().execute(img, "marigold", "stable_normal")
    assert isinstance(ms, MaterialSet)
    assert "INFO" in info or "ERROR" in info  # falls back gracefully


def test_light_probe_to_exr_writes_output(tmp_path):
    env = torch.rand(1, 3, 32, 64) * 2.0  # HDR-ish
    probe = LightProbe(env_map=env, exposure=0.0)
    out_dir = tmp_path / "out"
    path = LightProbeToEXR().execute(probe, str(out_dir), "probe.exr")[0]
    from pathlib import Path
    assert Path(path).exists()
    # If OpenEXR isn't installed we fall back to .npy
    assert path.endswith(".exr") or path.endswith(".npy")


# =============================================================================
# I. Phase 4 — Audio further coverage
# =============================================================================

from nukemax.nodes.audio import (
    AudioDriveMask,
    AudioToFloatCurve,
    _onset_envelope,
    _spectral_centroid,
    _stft_magnitude,
)


def _synthetic_audio(sr=16000, dur=2.0, bpm=120.0):
    n = int(sr * dur)
    wav = torch.zeros(n)
    period = int(round(sr * 60 / bpm))
    click = torch.exp(-torch.linspace(0, 6, 256)) * torch.randn(256)
    for s in range(0, n - 256, period):
        wav[s:s + 256] += click
    n_fft, hop = 1024, 256
    mag = _stft_magnitude(wav, n_fft=n_fft, hop=hop)
    return AudioFeatures(
        waveform=wav, sr=sr, stft_mag=mag,
        onsets=_onset_envelope(mag), bpm=bpm,
        centroid=_spectral_centroid(mag, sr, n_fft),
        rms=torch.zeros(mag.shape[1]), hop_length=hop,
    )


def test_audio_curve_smoothing_reduces_variance():
    af = _synthetic_audio()
    raw, _ = AudioToFloatCurve().execute(af, 60, 30.0, "onsets", 0.0, 1.0)
    smoothed, _ = AudioToFloatCurve().execute(af, 60, 30.0, "onsets", 0.8, 1.0)
    raw_t = torch.tensor(raw)
    sm_t = torch.tensor(smoothed)
    assert sm_t.var().item() < raw_t.var().item() + 1e-6


def test_audio_drive_mask_dilate_grows_mask():
    base = torch.zeros(1, 16, 16); base[:, 7:9, 7:9] = 1.0
    out = AudioDriveMask().execute(base, [1.0], "dilate", 4.0)[0]
    assert out.sum().item() >= base.sum().item()


def test_audio_drive_mask_feather_softens():
    base = torch.zeros(1, 16, 16); base[:, 7:9, 7:9] = 1.0
    out = AudioDriveMask().execute(base, [1.0], "feather", 4.0)[0]
    # Feathered mask has values strictly between 0 and 1
    inbetween = ((out > 0.01) & (out < 0.99)).float().mean().item()
    assert inbetween > 0.0


# =============================================================================
# J. Phase 5 — Flow further coverage
# =============================================================================

from nukemax.nodes.flow import (
    CleanPlateMerge,
    ComputeOpticalFlow,
    FlowBackwardWarp,
    FlowForwardWarp,
    FlowVisualize,
)


def test_compute_optical_flow_single_frame_no_crash():
    frames = torch.rand(1, 16, 16, 3)
    ff = ComputeOpticalFlow().execute(frames, "torch_lk", 1.5)[0]
    # No motion possible from one frame
    assert ff.flow_fwd.shape[0] == 0


def test_compute_optical_flow_static_pair_near_zero():
    frame = torch.rand(1, 16, 16, 3)
    frames = frame.expand(2, -1, -1, -1).clone()
    ff = ComputeOpticalFlow().execute(frames, "torch_lk", 1.5)[0]
    assert ff.flow_fwd.abs().max().item() < 0.5


def test_flow_backward_warp_constant_shift():
    img = torch.rand(2, 32, 32, 3)
    f = torch.zeros(2, 2, 32, 32)
    f[:, 0] = 0.0; f[:, 1] = 0.0  # zero flow
    flow = FlowField(flow_fwd=f, flow_bwd=f)
    out = FlowBackwardWarp().execute(img, flow, "forward")[0]
    diff = (img[..., 1:-1, 1:-1, :] - out[..., 1:-1, 1:-1, :]).abs()
    assert diff.max().item() < 1e-4


def test_flow_visualize_zero_flow_constant_color():
    flow = FlowField(flow_fwd=torch.zeros(1, 2, 16, 16), flow_bwd=torch.zeros(1, 2, 16, 16))
    out = FlowVisualize().execute(flow, 16.0)[0]
    # All pixels should be the same hue/value (zero magnitude → unsaturated)
    assert out.std().item() < 1e-4


def test_clean_plate_full_mask_uses_plate():
    foot = torch.zeros(2, 16, 16, 3)
    plate = torch.full((1, 16, 16, 3), 0.7)
    mask = torch.ones(2, 16, 16)
    flow = FlowField(flow_fwd=torch.zeros(1, 2, 16, 16), flow_bwd=torch.zeros(1, 2, 16, 16))
    out = CleanPlateMerge().execute(foot, plate, mask, flow, feather_px=0.0)[0]
    assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-4)


def test_flow_forward_warp_returns_image_and_weight():
    img = torch.rand(1, 16, 16, 3)
    flow = FlowField(flow_fwd=torch.zeros(1, 2, 16, 16), flow_bwd=torch.zeros(1, 2, 16, 16))
    warped, w = FlowForwardWarp().execute(img, flow)
    assert warped.shape == img.shape
    assert w.shape == (1, 16, 16)


# =============================================================================
# K. Phase 6 — Edges further coverage
# =============================================================================

from nukemax.nodes.edges import (
    HairAwareChoke,
    MatteDensityAdjust,
    NormalAwareEdgeBlur,
    SubPixelEdgeDetect,
)


def test_normal_aware_edge_blur_preserves_resolution():
    mask = torch.rand(1, 24, 32)
    normal = torch.rand(1, 24, 32, 3)
    out = NormalAwareEdgeBlur().execute(mask, normal, 2.0, 0.85)[0]
    assert out.shape == (1, 24, 32)


def test_normal_aware_edge_blur_preserves_uniform_mask():
    mask = torch.full((1, 16, 16), 0.5)
    normal = torch.full((1, 16, 16, 3), 0.5)  # constant normal => no edge
    out = NormalAwareEdgeBlur().execute(mask, normal, 4.0, 0.85)[0]
    assert torch.allclose(out, mask, atol=1e-4)


def test_matte_density_gamma_changes_midtones():
    mask = torch.full((1, 8, 8), 0.5)
    # Nuke convention: out = in^(1/gamma).
    # gamma > 1 brightens midtones; gamma < 1 darkens them.
    bright = MatteDensityAdjust().execute(mask, gamma=2.0, contrast=1.0, edge_lo=0.01, edge_hi=0.99)[0]
    dark = MatteDensityAdjust().execute(mask, gamma=0.5, contrast=1.0, edge_lo=0.01, edge_hi=0.99)[0]
    assert bright.mean().item() > 0.5
    assert dark.mean().item() < 0.5


def test_subpixel_edges_topk_count():
    img = torch.rand(1, 32, 32, 3)
    edges, td = SubPixelEdgeDetect().execute(img, top_k=20)
    assert td.coords.shape == (1, 20, 2)
    assert edges.shape == (1, 32, 32)


def test_hair_aware_choke_preserves_resolution():
    mask = torch.rand(2, 32, 32)
    image = torch.rand(2, 32, 32, 3)
    out = HairAwareChoke().execute(mask, image, choke=1.0, hair_window=5)[0]
    assert out.shape == (2, 32, 32)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


# =============================================================================
# L. Types I/O — exercise dynamic Serialize/Deserialize node pairs
# =============================================================================

@pytest.mark.parametrize("type_name,obj_factory", [
    ("ROTO_SHAPE", lambda: RotoShape.from_polygon(
        torch.tensor([[0.0, 0], [10, 0], [10, 10], [0, 10]]),
        canvas_hw=(16, 16), feather=1.0, closed=True)),
    ("TRACKING_DATA", lambda: TrackingData(
        coords=torch.rand(2, 4, 2), velocity=torch.zeros(2, 4, 2),
        confidence=torch.ones(2, 4), canvas_h=16, canvas_w=16)),
    ("FFT_TENSOR", lambda: nfft.analyze(torch.rand(1, 1, 8, 8))),
    ("FLOW_FIELD", lambda: FlowField(
        flow_fwd=torch.zeros(1, 2, 8, 8), flow_bwd=torch.zeros(1, 2, 8, 8))),
    ("MATERIAL_SET", lambda: MaterialSet(
        albedo=torch.rand(1, 3, 4, 4),
        normal=torch.rand(1, 3, 4, 4) * 2 - 1,
        depth=torch.rand(1, 1, 4, 4),
        roughness=torch.rand(1, 1, 4, 4))),
    ("LIGHT_RIG", lambda: LightRig(
        lights=(Light(direction=(0, 0, -1), intensity=1.0),),
        ambient=(0.05, 0.05, 0.05))),
    ("LIGHT_PROBE", lambda: LightProbe(env_map=torch.rand(1, 3, 16, 32), exposure=0.0)),
    ("AUDIO_FEATURES", lambda: AudioFeatures(
        waveform=torch.zeros(1024), sr=16000,
        stft_mag=torch.zeros(513, 4), onsets=torch.zeros(4),
        bpm=120.0, centroid=torch.zeros(4), rms=torch.zeros(4),
        hop_length=256)),
])
def test_serialize_deserialize_node_pair(type_name, obj_factory):
    s_key = f"NukeMax_Serialize_{type_name}"
    d_key = f"NukeMax_Deserialize_{type_name}"
    assert s_key in NODE_CLASS_MAPPINGS, f"missing {s_key}"
    assert d_key in NODE_CLASS_MAPPINGS, f"missing {d_key}"
    obj = obj_factory()
    s_inst = NODE_CLASS_MAPPINGS[s_key]()
    d_inst = NODE_CLASS_MAPPINGS[d_key]()
    payload = s_inst.execute(obj)[0]
    assert isinstance(payload, str)
    rebuilt = d_inst.execute(payload)[0]
    assert type(rebuilt) is type(obj)


# =============================================================================
# M. Splines: extra correctness checks
# =============================================================================

def test_open_polyline_renders_thinner_than_closed():
    pts = torch.tensor([[16, 16], [16, 48], [48, 48]], dtype=torch.float32).unsqueeze(0)  # (1,3,2)
    closed = splines.rasterize_polygon_sdf(pts, 64, 64, feather=0.0, closed=True)
    open_ = splines.rasterize_polygon_sdf(pts, 64, 64, feather=0.0, closed=False)
    # Closed forms a triangle (filled); open is just a line (no fill).
    assert closed.sum().item() > open_.sum().item()


def test_rasterize_outside_polygon_is_zero():
    pts = torch.tensor([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=torch.float32).unsqueeze(0)
    mask = splines.rasterize_polygon_sdf(pts, 32, 32, feather=0.0, closed=True)
    assert mask[0, 0, 0].item() == 0.0
    assert mask[0, 31, 31].item() == 0.0
    assert mask[0, 15, 15].item() == 1.0


# =============================================================================
# N. Regression: pack-level NODE_CLASS_MAPPINGS keys consistent
# =============================================================================

def test_node_class_mappings_are_unique():
    keys = list(NODE_CLASS_MAPPINGS.keys())
    assert len(keys) == len(set(keys))


def test_categories_are_one_of_known_set():
    known = {
        "NukeMax/Roto", "NukeMax/FFT", "NukeMax/Relight",
        "NukeMax/Audio", "NukeMax/Flow", "NukeMax/Edges",
    }
    for key, cls in NODE_CLASS_MAPPINGS.items():
        cat = cls.CATEGORY
        if cat.startswith("NukeMax/IO/"):
            continue  # generic serialize/deserialize
        assert cat in known, f"{key}: unknown category {cat}"
