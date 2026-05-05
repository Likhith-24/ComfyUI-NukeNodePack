"""Audio-Reactive Engine — Concept 4."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ...types import AudioFeatures
from ...utils.resilience import resilient


from ... import _interrupt_check as _IC
from ... import _progress as _PB
def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    p = Path(path)
    try:
        import soundfile as sf  # type: ignore
        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
        wav_t = torch.from_numpy(wav)
        if wav_t.ndim > 1:
            wav_t = wav_t.mean(dim=-1)
        return wav_t.float(), int(sr)
    except ImportError:
        pass
    try:
        import librosa  # type: ignore
        import numpy as np
        wav, sr = librosa.load(str(p), sr=None, mono=True)
        return torch.from_numpy(np.asarray(wav, dtype="float32")), int(sr)
    except ImportError:
        pass
    # Last resort: stdlib wave (PCM only)
    import wave
    import numpy as np
    with wave.open(str(p), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    arr = np.frombuffer(raw, dtype=dtype).astype("float32")
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    arr = arr / float(np.iinfo(dtype).max)
    return torch.from_numpy(arr), int(sr)


def _stft_magnitude(wav: torch.Tensor, n_fft: int = 2048, hop: int = 512) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window,
                      center=True, return_complex=True)
    return spec.abs()  # (F, T_stft)


def _onset_envelope(stft_mag: torch.Tensor) -> torch.Tensor:
    # Spectral flux: positive part of frame-to-frame magnitude diff, summed over freq.
    diff = (stft_mag[:, 1:] - stft_mag[:, :-1]).clamp_min(0)
    flux = diff.sum(dim=0)
    # Pad to original T
    flux = F.pad(flux.unsqueeze(0).unsqueeze(0), (1, 0), mode="replicate").squeeze()
    # Normalize
    return flux / flux.max().clamp_min(1e-6)


def _bpm_estimate(onset: torch.Tensor, sr: int, hop: int) -> float:
    # Autocorrelation of onset envelope, find peak in plausible BPM range.
    if onset.numel() < 32:
        return 0.0
    o = onset - onset.mean()
    n = o.shape[0]
    fft = torch.fft.rfft(o, n=2 * n)
    ac = torch.fft.irfft(fft * fft.conj(), n=2 * n)[:n].real
    # Lag in frames -> seconds -> BPM
    # Plausible range 50..200 BPM
    min_bpm, max_bpm = 50.0, 200.0
    frame_sec = hop / float(sr)
    min_lag = int(60.0 / max_bpm / frame_sec)
    max_lag = int(60.0 / min_bpm / frame_sec)
    if max_lag <= min_lag or max_lag >= n:
        return 0.0
    seg = ac[min_lag:max_lag]
    lag = int(seg.argmax().item()) + min_lag
    return 60.0 / (lag * frame_sec)


def _spectral_centroid(stft_mag: torch.Tensor, sr: int, n_fft: int) -> torch.Tensor:
    freqs = torch.linspace(0, sr / 2, stft_mag.shape[0], device=stft_mag.device, dtype=stft_mag.dtype)
    weighted = (stft_mag * freqs.unsqueeze(1)).sum(dim=0)
    total = stft_mag.sum(dim=0).clamp_min(1e-9)
    return weighted / total


@resilient
class AudioLoadAnalyze:
    DESCRIPTION = "Load an audio file and compute STFT magnitude, onset envelope, BPM, spectral centroid and RMS as AUDIO_FEATURES."
    CATEGORY = "NukeMax/Audio"
    FUNCTION = "execute"
    RETURN_TYPES = ("AUDIO_FEATURES", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "bpm", "info")
    OUTPUT_TOOLTIPS = ("Bundle of audio features (waveform, STFT, onsets, centroid, RMS).", "Estimated tempo in beats per minute.", "Human-readable summary string.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "tooltip": "Filesystem path to the audio file (wav/flac/mp3 if librosa is available)."}),
                "n_fft": ("INT", {"default": 2048, "min": 256, "max": 16384, "tooltip": "FFT window size used for the STFT."}),
                "hop_length": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": "STFT hop length in samples."}),
            },
        }

    def execute(self, path, n_fft, hop_length):
        wav, sr = _load_audio(path)
        mag = _stft_magnitude(wav, n_fft=n_fft, hop=hop_length)
        onset = _onset_envelope(mag)
        bpm = _bpm_estimate(onset, sr, hop_length)
        centroid = _spectral_centroid(mag, sr, n_fft)
        rms = (wav.unfold(0, n_fft, hop_length).pow(2).mean(dim=-1).sqrt()) if wav.shape[0] >= n_fft else torch.zeros_like(centroid)
        # Trim/pad rms to match centroid length
        T = centroid.shape[0]
        if rms.shape[0] < T:
            rms = F.pad(rms, (0, T - rms.shape[0]))
        else:
            rms = rms[:T]
        af = AudioFeatures(
            waveform=wav, sr=sr, stft_mag=mag, onsets=onset, bpm=float(bpm),
            centroid=centroid, rms=rms, hop_length=hop_length,
        )
        info = f"sr={sr} dur={af.duration_seconds:.2f}s bpm={bpm:.1f} stft={tuple(mag.shape)}"
        return (af, float(bpm), info)


def _resample_curve(curve: torch.Tensor, n_target: int) -> torch.Tensor:
    if curve.numel() == n_target:
        return curve
    return F.interpolate(curve.view(1, 1, -1), size=n_target, mode="linear", align_corners=False).view(-1)


@resilient
class AudioToFloatCurve:
    DESCRIPTION = "Convert an audio feature band (bass/mid/treble/onsets/centroid/full) into a per-frame float curve and 1D viz mask."
    CATEGORY = "NukeMax/Audio"
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOAT", "MASK")
    RETURN_NAMES = ("curve", "curve_image_1d")
    OUTPUT_TOOLTIPS = ("Per-frame float curve normalized to [0,1].", "1D mask visualization of the curve over time.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_FEATURES", {"tooltip": "Audio feature bundle from Audio Load & Analyze."}),
                "frame_count": ("INT", {"default": 60, "min": 1, "max": 100000, "tooltip": "Length of the output curve in frames."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "tooltip": "Target frames-per-second for time-aligning the curve."}),
                "band": (("full", "bass", "mid", "treble", "onsets", "centroid"), {"tooltip": "Which audio feature band to extract the curve from."}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Temporal Gaussian smoothing of the curve (0=none)."}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 32.0, "tooltip": "Multiplier applied to the curve before clamping."}),
            },
        }

    def execute(self, audio: AudioFeatures, frame_count, fps, band, smoothing, gain):
        mag = audio.stft_mag
        F_bin = mag.shape[0]
        if band == "bass":
            sl = mag[: F_bin // 8].mean(dim=0)
        elif band == "mid":
            sl = mag[F_bin // 8: F_bin // 2].mean(dim=0)
        elif band == "treble":
            sl = mag[F_bin // 2:].mean(dim=0)
        elif band == "onsets":
            sl = audio.onsets
        elif band == "centroid":
            sl = audio.centroid
        else:
            sl = mag.mean(dim=0)
        # Normalize
        sl = sl / sl.max().clamp_min(1e-6)
        # Resample to frame_count.
        # Actually account for fps vs audio duration.
        target_frames = max(1, int(round(audio.duration_seconds * fps)))
        target_frames = min(target_frames, frame_count)
        curve = _resample_curve(sl, target_frames)
        # Pad / clip to requested frame_count
        if curve.shape[0] < frame_count:
            curve = F.pad(curve, (0, frame_count - curve.shape[0]))
        else:
            curve = curve[:frame_count]
        if smoothing > 0:
            sigma = smoothing * 8.0
            radius = max(int(sigma * 3), 1)
            x = torch.arange(-radius, radius + 1, dtype=curve.dtype)
            k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
            curve = F.conv1d(curve.view(1, 1, -1), k.view(1, 1, -1), padding=radius).view(-1)
        curve = (curve * gain).clamp(0, 1)
        # Visualization mask
        viz = curve.view(1, 1, -1).expand(1, 32, -1).clone()
        return (curve.tolist(), viz)


@resilient
class AudioDriveMask:
    """Modulate a per-frame mask's intensity / dilation by the curve."""
    DESCRIPTION = "Modulate a per-frame mask by an audio-driven float curve via intensity scaling, dilation, or feathering."
    CATEGORY = "NukeMax/Audio"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Mask modulated frame-by-frame by the curve.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Per-frame mask batch to modulate."}),
                "curve": ("FLOAT", {"forceInput": True, "tooltip": "Per-frame float curve (e.g. from Audio To Float Curve)."}),
                "mode": (("intensity", "dilate", "feather"), {"tooltip": "How the curve modulates the mask: brightness, morphological dilation, or Gaussian feather."}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 32.0, "tooltip": "Strength of the modulation."}),
            },
        }

    def execute(self, mask, curve, mode, amount):
        m = mask if mask.ndim == 4 else mask.unsqueeze(1)
        T = m.shape[0]
        c = curve if isinstance(curve, list) else [float(curve)]
        if len(c) < T:
            c = c + [c[-1]] * (T - len(c))
        c_t = torch.tensor(c[:T], dtype=m.dtype, device=m.device).view(T, 1, 1, 1)
        if mode == "intensity":
            out = (m * (1.0 + (c_t - 0.5) * 2 * amount)).clamp(0, 1)
        elif mode == "dilate":
            # Dilate by morphological max with a kernel size driven per-frame.
            # Force odd kernels so output spatial size matches input.
            out = m.clone()
            for t in _PB.track(range(T), T, "Audio"):
                _IC.check()
                k = max(1, int(round(float(c_t[t]) * amount * 8)))
                if k % 2 == 0:
                    k += 1
                if k > 1:
                    out[t:t + 1] = F.max_pool2d(m[t:t + 1], kernel_size=k, stride=1, padding=k // 2)
        else:  # feather
            from ...core import blur as nblur
            out = m.clone()
            for t in _PB.track(range(T), T, "Audio"):
                _IC.check()
                sigma = float(c_t[t]) * amount * 4
                if sigma > 0.01:
                    out[t:t + 1] = nblur.gaussian_blur(m[t:t + 1], sigma)
        return (out.squeeze(1).clamp(0, 1),)


@resilient
class AudioDriveSchedule:
    """Convert an audio curve to a per-frame schedule list usable by
    samplers that accept a CFG/denoise schedule. Output: STRING JSON.
    """
    DESCRIPTION = "Map an audio float curve to a per-frame value schedule (e.g. CFG/denoise) and emit it as JSON plus the scaled curve."
    CATEGORY = "NukeMax/Audio"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("schedule_json", "curve")
    OUTPUT_TOOLTIPS = ("JSON-encoded list of per-frame schedule values.", "Per-frame schedule values as a float list.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curve": ("FLOAT", {"forceInput": True, "tooltip": "Per-frame float curve to remap."}),
                "min_value": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "tooltip": "Schedule value when the curve is at 0."}),
                "max_value": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "tooltip": "Schedule value when the curve is at 1."}),
            },
        }

    def execute(self, curve, min_value, max_value):
        import json
        c = curve if isinstance(curve, list) else [float(curve)]
        scheduled = [min_value + (max_value - min_value) * float(v) for v in c]
        return (json.dumps(scheduled), scheduled)


@resilient
class AudioSpectrogram:
    DESCRIPTION = "Render an STFT magnitude spectrogram from AUDIO_FEATURES as an RGB image, optionally on a log magnitude scale."
    CATEGORY = "NukeMax/Audio"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrogram",)
    OUTPUT_TOOLTIPS = ("Greyscale RGB spectrogram image (frequency on Y, time on X).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_FEATURES", {"tooltip": "Audio feature bundle to visualize."}),
                "log_scale": ("BOOLEAN", {"default": True, "tooltip": "If true, display log10 magnitude; otherwise linear magnitude."}),
            },
        }

    def execute(self, audio, log_scale):
        m = audio.stft_mag.clone()
        if log_scale:
            m = (m + 1e-6).log10()
            m = (m - m.min()) / (m.max() - m.min()).clamp_min(1e-6)
        else:
            m = m / m.max().clamp_min(1e-6)
        # (F, T) -> render top->bottom by flipping freq axis
        m = m.flip(0).unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
        rgb = m.expand(-1, 3, -1, -1).squeeze(0).permute(1, 2, 0).unsqueeze(0)
        return (rgb,)


NODE_CLASS_MAPPINGS = {
    "NukeMax_AudioLoadAnalyze": AudioLoadAnalyze,
    "NukeMax_AudioToFloatCurve": AudioToFloatCurve,
    "NukeMax_AudioDriveMask": AudioDriveMask,
    "NukeMax_AudioDriveSchedule": AudioDriveSchedule,
    "NukeMax_AudioSpectrogram": AudioSpectrogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_AudioLoadAnalyze": "Audio Load & Analyze",
    "NukeMax_AudioToFloatCurve": "Audio → Float Curve",
    "NukeMax_AudioDriveMask": "Audio Drive Mask",
    "NukeMax_AudioDriveSchedule": "Audio Drive Schedule",
    "NukeMax_AudioSpectrogram": "Audio Spectrogram",
}
