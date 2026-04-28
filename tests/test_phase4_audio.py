"""Phase 4 (Audio) proofs.

We synthesize tone bursts with NumPy/Torch instead of relying on file I/O,
so the tests run without soundfile/librosa or audio files on disk.
"""
from __future__ import annotations

import json
import math
import wave
import struct
from pathlib import Path

import torch

from nukemax.nodes.audio import (
    AudioDriveMask,
    AudioDriveSchedule,
    AudioLoadAnalyze,
    AudioSpectrogram,
    AudioToFloatCurve,
    _bpm_estimate,
    _onset_envelope,
    _spectral_centroid,
    _stft_magnitude,
)
from nukemax.types import AudioFeatures


def _click_track(sr: int = 22050, duration: float = 4.0, bpm: float = 120.0) -> torch.Tensor:
    """Click train at the given BPM. Each beat is a short decaying impulse."""
    n = int(sr * duration)
    wav = torch.zeros(n)
    period = int(round(sr * 60.0 / bpm))
    click_len = 256
    env = torch.exp(-torch.linspace(0, 6, click_len))
    noise = torch.randn(click_len) * env
    for start in range(0, n - click_len, period):
        wav[start:start + click_len] += noise
    return wav


def _synthetic_audio_features(sr: int = 22050, duration: float = 4.0, bpm: float = 120.0) -> AudioFeatures:
    wav = _click_track(sr=sr, duration=duration, bpm=bpm)
    n_fft, hop = 1024, 256
    mag = _stft_magnitude(wav, n_fft=n_fft, hop=hop)
    onset = _onset_envelope(mag)
    centroid = _spectral_centroid(mag, sr, n_fft)
    rms = torch.zeros_like(centroid)
    return AudioFeatures(
        waveform=wav, sr=sr, stft_mag=mag, onsets=onset,
        bpm=float(_bpm_estimate(onset, sr, hop)),
        centroid=centroid, rms=rms, hop_length=hop,
    )


def test_stft_magnitude_shape():
    wav = torch.randn(22050)
    mag = _stft_magnitude(wav, n_fft=1024, hop=256)
    assert mag.ndim == 2
    assert mag.shape[0] == 1024 // 2 + 1  # F bins
    assert mag.shape[1] > 0
    assert (mag >= 0).all()


def test_bpm_estimate_recovers_120():
    af = _synthetic_audio_features(bpm=120.0)
    # Allow some slack — autocorrelation BPM has lag-quantization error
    assert 100.0 <= af.bpm <= 140.0


def test_audio_to_float_curve_length_and_range():
    af = _synthetic_audio_features()
    curve, viz = AudioToFloatCurve().execute(af, frame_count=60, fps=30.0,
                                              band="onsets", smoothing=0.0, gain=1.0)
    assert isinstance(curve, list)
    assert len(curve) == 60
    assert all(0.0 <= v <= 1.0 for v in curve)
    # Visualization mask is (1, 32, 60)
    assert viz.shape == (1, 32, 60)


def test_audio_to_float_curve_bands_differ():
    af = _synthetic_audio_features()
    bass, _ = AudioToFloatCurve().execute(af, 60, 30.0, "bass", 0.0, 1.0)
    treble, _ = AudioToFloatCurve().execute(af, 60, 30.0, "treble", 0.0, 1.0)
    # Different bands shouldn't yield identical curves
    assert bass != treble


def test_audio_drive_mask_intensity_mode():
    mask = torch.full((4, 8, 8), 0.5)
    curve = [0.0, 0.5, 1.0, 0.25]
    out = AudioDriveMask().execute(mask, curve, "intensity", 1.0)[0]
    assert out.shape == (4, 8, 8)
    # Frame 1 (curve=0.5) is unchanged; frame 0 darker, frame 2 brighter
    assert out[0].mean().item() <= 0.5 + 1e-5
    assert out[2].mean().item() >= 0.5 - 1e-5
    assert out[2].mean().item() > out[0].mean().item()


def test_audio_drive_schedule_interpolates_min_max():
    curve = [0.0, 0.5, 1.0]
    sched_json, sched = AudioDriveSchedule().execute(curve, 4.0, 12.0)
    parsed = json.loads(sched_json)
    assert len(parsed) == 3
    assert math.isclose(parsed[0], 4.0, abs_tol=1e-6)
    assert math.isclose(parsed[1], 8.0, abs_tol=1e-6)
    assert math.isclose(parsed[2], 12.0, abs_tol=1e-6)
    assert sched == parsed


def test_audio_spectrogram_returns_image():
    af = _synthetic_audio_features()
    img = AudioSpectrogram().execute(af, True)[0]
    # BHWC, 3 channels, finite, in [0,1]
    assert img.ndim == 4
    assert img.shape[-1] == 3
    assert torch.isfinite(img).all()
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_audio_load_analyze_round_trip(tmp_path: Path):
    """Write a small PCM WAV via stdlib `wave`, then run AudioLoadAnalyze."""
    sr = 16000
    duration = 2.0
    bpm = 120.0
    wav = _click_track(sr=sr, duration=duration, bpm=bpm)
    pcm = (wav.clamp(-1, 1) * 32767).short().numpy().tobytes()
    out = tmp_path / "click.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    af, est_bpm, info = AudioLoadAnalyze().execute(str(out), 1024, 256)
    assert isinstance(af, AudioFeatures)
    assert af.sr == sr
    assert af.duration_seconds == duration
    assert "sr=16000" in info
    # BPM detection should land in plausible range
    assert 80.0 <= est_bpm <= 180.0
