"""Phase 2 (FFT) proofs."""
from __future__ import annotations

import math

import torch

from nukemax.core import fft as nfft
from nukemax.core.fft import match_ring_spectrum, ring_power_spectrum
from nukemax.nodes.fft import FFTAnalyze, FFTSynthesize, FrequencyMask


def test_fft_round_trip():
    img = torch.rand(1, 16, 16, 3)
    ft = FFTAnalyze().execute(img)[0]
    out = FFTSynthesize().execute(ft)[0]
    assert torch.allclose(img, out, atol=1e-4)


def test_band_filter_zero_low_passes_dc():
    img = torch.full((1, 16, 16, 3), 0.5)
    ft = FFTAnalyze().execute(img)[0]
    filt = FrequencyMask().execute(ft, low=0.0, high=0.5, softness=0.0)[0]
    out = FFTSynthesize().execute(filt)[0]
    assert torch.allclose(out, img, atol=1e-3)


def test_match_ring_spectrum_brings_curves_close():
    torch.manual_seed(0)
    noise = torch.randn(1, 1, 32, 32)
    ref = torch.randn(1, 1, 32, 32) * 0.1 + torch.linspace(0, 1, 32).view(1, 1, 32, 1).expand(-1, -1, -1, 32)
    matched = match_ring_spectrum(noise, ref, n_bins=16)
    # Compare ring spectra
    nf = nfft.analyze(matched); rf = nfft.analyze(ref)
    ns = ring_power_spectrum(nf, 16); rs = ring_power_spectrum(rf, 16)
    ratio = (ns.clamp_min(1e-8) / rs.clamp_min(1e-8)).log().abs()
    # Most bins should be within ~0.5 in log power space (corresponds to amplitude factor ~1.6x).
    assert ratio.mean().item() < 0.5
