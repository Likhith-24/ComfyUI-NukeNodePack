"""Phase 0 proofs: types, math primitives, resilience."""
from __future__ import annotations

import math

import pytest
import torch

from nukemax.core import blur, color, composite, fft as nfft, flow, splines
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
from nukemax.utils.resilience import resilient


# -------------------- Types: round trip --------------------

def _make_roto():
    return RotoShape.from_polygon(
        torch.tensor([[10.0, 10], [50, 10], [50, 50], [10, 50]]),
        canvas_hw=(64, 64),
        feather=2.0,
        closed=True,
    )


def test_roto_serialize_round_trip():
    r = _make_roto()
    d = ser.serialize(r)
    r2 = ser.deserialize(d)
    assert torch.allclose(r.points, r2.points)
    assert torch.allclose(r.feather, r2.feather)
    assert r.canvas_h == r2.canvas_h and r.closed == r2.closed


def test_tracking_data_round_trip():
    td = TrackingData(
        coords=torch.zeros(4, 3, 2),
        velocity=torch.zeros(4, 3, 2),
        confidence=torch.ones(4, 3),
        canvas_h=64, canvas_w=64,
    )
    d = ser.serialize(td)
    td2 = ser.deserialize(d)
    assert torch.equal(td.coords, td2.coords)


def test_fft_tensor_round_trip():
    img = torch.rand(1, 3, 16, 16)
    ft = nfft.analyze(img)
    d = ser.serialize(ft)
    ft2 = ser.deserialize(d)
    assert torch.allclose(ft.magnitude, ft2.magnitude, atol=1e-6)


def test_material_set_round_trip():
    ms = MaterialSet(
        albedo=torch.rand(1, 3, 8, 8),
        normal=torch.rand(1, 3, 8, 8) * 2 - 1,
        depth=torch.rand(1, 1, 8, 8),
        roughness=torch.rand(1, 1, 8, 8),
    )
    ms2 = ser.deserialize(ser.serialize(ms))
    assert torch.allclose(ms.albedo, ms2.albedo)


def test_light_rig_round_trip():
    rig = LightRig(
        lights=(Light(color=(1.0, 0.5, 0.2), type="directional"),),
        ambient=(0.1, 0.1, 0.1),
    )
    rig2 = ser.deserialize(ser.serialize(rig))
    assert rig2.lights[0].color == (1.0, 0.5, 0.2)
    assert rig2.ambient == (0.1, 0.1, 0.1)


def test_flow_field_round_trip():
    ff = FlowField(
        flow_fwd=torch.zeros(1, 2, 8, 8),
        flow_bwd=torch.zeros(1, 2, 8, 8),
    )
    ff2 = ser.deserialize(ser.serialize(ff))
    assert torch.equal(ff.flow_fwd, ff2.flow_fwd)


# -------------------- Color: sRGB round trip --------------------

def test_srgb_round_trip():
    x = torch.linspace(0, 1, 257).view(1, 1, 1, -1).expand(1, 3, 1, -1).clone()
    y = color.linear_to_srgb(color.srgb_to_linear(x))
    assert torch.allclose(x, y, atol=1e-6)


def test_luminance_white_is_one():
    white = torch.ones(1, 3, 4, 4)
    Y = color.luminance(white)
    assert torch.allclose(Y, torch.ones(1, 1, 4, 4), atol=1e-6)


# -------------------- Blur: separable Gaussian vs scipy reference --------------------

def test_separable_gaussian_matches_scipy():
    scipy = pytest.importorskip("scipy.ndimage")
    img = torch.rand(1, 1, 64, 64)
    sigma = 3.0
    out = blur.gaussian_blur(img, sigma)
    ref = torch.from_numpy(
        scipy.gaussian_filter(img[0, 0].numpy(), sigma=sigma, mode="reflect", truncate=4.0)
    ).view(1, 1, 64, 64)
    # Reasonable agreement; reflect padding semantics and float32 vs float64
    # roundoff differ slightly. Strict interior, loose tolerance.
    diff = (out - ref).abs()
    assert diff[..., 16:-16, 16:-16].max().item() < 1e-2


def test_box_blur_constant_image():
    img = torch.full((1, 3, 32, 32), 0.7)
    out = blur.box_blur(img, radius=3)
    assert torch.allclose(out, img, atol=1e-6)


# -------------------- FFT: blur equivalence --------------------

def test_fft_gaussian_matches_spatial():
    img = torch.rand(1, 1, 64, 64)
    sigma = 2.5
    spatial = blur.gaussian_blur(img, sigma)
    # FFT-domain Gaussian: multiply spectrum by Gaussian of width 1/(2*pi*sigma)
    ft = nfft.analyze(img)
    H, W = 64, 64
    fy = torch.fft.fftshift(torch.fft.fftfreq(H))
    fx = torch.fft.fftshift(torch.fft.fftfreq(W))
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    g = torch.exp(-2 * (math.pi ** 2) * (sigma ** 2) * (yy * yy + xx * xx))
    ft_blurred = FFTTensor(
        magnitude=ft.magnitude * g.unsqueeze(0).unsqueeze(0),
        phase=ft.phase, spatial_h=H, spatial_w=W, centered=True,
    )
    fft_out = nfft.synthesize(ft_blurred)
    # Crop borders (different boundary handling).
    diff = (spatial - fft_out).abs()
    assert diff[..., 8:-8, 8:-8].max().item() < 5e-3


# -------------------- Porter-Duff Over correctness --------------------

def test_porter_duff_over_premul_half_alpha():
    # Black src @ 0.5 over white dst -> 0.5 gray.
    src = composite.to_premul(torch.zeros(1, 3, 4, 4), torch.full((1, 1, 4, 4), 0.5))
    dst = composite.to_premul(torch.ones(1, 3, 4, 4), torch.ones(1, 1, 4, 4))
    out = composite.over(src, dst)
    rgb, a = composite.from_premul(out)
    assert torch.allclose(rgb, torch.full_like(rgb, 0.5), atol=1e-6)
    assert torch.allclose(a, torch.ones_like(a), atol=1e-6)


def test_porter_duff_over_full_alpha_is_src():
    src = composite.to_premul(torch.full((1, 3, 4, 4), 0.7), torch.ones(1, 1, 4, 4))
    dst = composite.to_premul(torch.full((1, 3, 4, 4), 0.2), torch.ones(1, 1, 4, 4))
    out = composite.over(src, dst)
    rgb, a = composite.from_premul(out)
    assert torch.allclose(rgb, torch.full_like(rgb, 0.7), atol=1e-6)


# -------------------- Flow: identity warp --------------------

def test_identity_flow_backward_warp():
    img = torch.rand(1, 3, 32, 32)
    f = torch.zeros(1, 2, 32, 32)
    out = flow.backward_warp(img, f)
    psnr = 10 * math.log10(1.0 / max(((img - out) ** 2).mean().item(), 1e-12))
    assert psnr > 60


def test_identity_flow_forward_warp():
    img = torch.rand(1, 3, 32, 32)
    f = torch.zeros(1, 2, 32, 32)
    out, w = flow.forward_warp(img, f)
    out_norm = out / w.clamp_min(1e-6)
    diff = (img - out_norm).abs()
    assert diff.max().item() < 1e-5


# -------------------- Splines: circle area --------------------

def test_polygon_circle_area_within_one_percent():
    # Approximate a circle with 64-gon and rasterize.
    N = 128
    r = 20.0
    cx, cy = 32.0, 32.0
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    pts = torch.stack([cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=-1).unsqueeze(0)  # (1,N,2)
    mask = splines.rasterize_polygon_sdf(pts, H=64, W=64, feather=0.0, closed=True)
    area = mask.sum().item()
    expected = math.pi * r * r
    assert abs(area - expected) / expected < 0.01


# -------------------- Resilience decorator --------------------

def test_resilient_returns_passthrough_on_error():
    @resilient
    class _Boom:
        FUNCTION = "execute"
        RETURN_TYPES = ("IMAGE", "MASK", "STRING")
        RETURN_NAMES = ("image", "mask", "info")

        def execute(self):
            raise RuntimeError("boom")

    out = _Boom().execute()
    assert isinstance(out, tuple) and len(out) == 3
    assert out[0].shape[-1] == 3  # IMAGE in BHWC
    assert out[2].startswith("ERROR:")
