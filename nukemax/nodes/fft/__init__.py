"""Frequency-Guided Generation — Concept 2."""
from __future__ import annotations

import json

import torch

from ...core import fft as nfft
from ...core.color import to_bchw, to_bhwc
from ...types import FFTTensor
from ...utils.resilience import resilient


@resilient
class FFTAnalyze:
    DESCRIPTION = "Compute the centered 2D FFT of an image batch and return its magnitude/phase as an FFT_TENSOR."
    CATEGORY = "NukeMax/FFT"
    FUNCTION = "execute"
    RETURN_TYPES = ("FFT_TENSOR",)
    RETURN_NAMES = ("fft",)
    OUTPUT_TOOLTIPS = ("FFT tensor bundle (magnitude, phase, original spatial size).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"tooltip": "Image batch to transform into the frequency domain."})}}

    def execute(self, image):
        return (nfft.analyze(to_bchw(image)),)


@resilient
class FFTSynthesize:
    DESCRIPTION = "Inverse-FFT an FFT_TENSOR back into an image batch."
    CATEGORY = "NukeMax/FFT"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Reconstructed image clamped to [0,1].",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"fft": ("FFT_TENSOR", {"tooltip": "FFT tensor bundle to inverse-transform."})}}

    def execute(self, fft):
        out = nfft.synthesize(fft)
        return (to_bhwc(out.clamp(0, 1)),)


@resilient
class FrequencyMask:
    DESCRIPTION = "Apply a soft radial band-pass filter to an FFT_TENSOR, keeping frequencies between low and high."
    CATEGORY = "NukeMax/FFT"
    FUNCTION = "execute"
    RETURN_TYPES = ("FFT_TENSOR",)
    RETURN_NAMES = ("fft",)
    OUTPUT_TOOLTIPS = ("Band-pass filtered FFT tensor bundle.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fft": ("FFT_TENSOR", {"tooltip": "FFT tensor bundle to filter."}),
                "low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.005, "tooltip": "Inner radius of the pass band as a fraction of Nyquist."}),
                "high": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.5, "step": 0.005, "tooltip": "Outer radius of the pass band as a fraction of Nyquist."}),
                "softness": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005, "tooltip": "Edge softness of the band-pass transition."}),
            },
        }

    def execute(self, fft, low, high, softness):
        return (nfft.band_filter(fft, low, high, softness),)


@resilient
class LatentFrequencyMatch:
    """Reshape a noise tensor's ring-averaged spectrum to match a context
    image's, preserving phase. Output: corrected LATENT.

    Input ``latent_noise`` may be either a LATENT dict or an IMAGE.
    Context is an IMAGE at the same spatial size.
    """
    DESCRIPTION = "Reshape a noise latent's ring-averaged frequency spectrum to match a context image while preserving phase."
    CATEGORY = "NukeMax/FFT"
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = ("Latent whose spectrum is matched to the context image.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_latent": ("LATENT", {"tooltip": "Source noise latent to be spectrum-corrected."}),
                "context_image": ("IMAGE", {"tooltip": "Reference image whose ring-averaged spectrum is the target."}),
                "n_bins": ("INT", {"default": 64, "min": 8, "max": 256, "tooltip": "Number of radial bins used to estimate the ring spectrum."}),
            },
        }

    def execute(self, noise_latent, context_image, n_bins):
        samples = noise_latent["samples"]  # (B,4,h,w) typically
        B, C, h, w = samples.shape
        ctx = to_bchw(context_image)
        # Resample context to latent resolution and replicate channels via avg.
        import torch.nn.functional as F
        ctx_lat = F.interpolate(ctx, size=(h, w), mode="area")
        # Match per-channel: replicate context grayscale to C channels
        ctx_gray = ctx_lat.mean(dim=1, keepdim=True).expand(-1, C, -1, -1)
        if ctx_gray.shape[0] != B:
            if ctx_gray.shape[0] == 1:
                ctx_gray = ctx_gray.expand(B, -1, -1, -1)
            elif B == 1:
                ctx_gray = ctx_gray.mean(dim=0, keepdim=True)
            elif ctx_gray.shape[0] > B:
                ctx_gray = ctx_gray[:B]
            else:
                # tile to cover B then trim
                reps = (B + ctx_gray.shape[0] - 1) // ctx_gray.shape[0]
                ctx_gray = ctx_gray.repeat(reps, 1, 1, 1)[:B]
        matched = nfft.match_ring_spectrum(samples, ctx_gray, n_bins=n_bins)
        return ({"samples": matched},)


@resilient
class FFTTextureSynthesis:
    """Naive seamless tile via spectrum cloning + random phase."""
    DESCRIPTION = "Synthesize a same-spectrum texture from an exemplar by cloning its magnitude and randomizing phase."
    CATEGORY = "NukeMax/FFT"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Synthesized texture image at the requested resolution.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exemplar": ("IMAGE", {"tooltip": "Reference texture image whose spectrum is cloned."}),
                "out_height": ("INT", {"default": 512, "min": 16, "max": 4096, "tooltip": "Output texture height in pixels."}),
                "out_width": ("INT", {"default": 512, "min": 16, "max": 4096, "tooltip": "Output texture width in pixels."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Random seed driving the synthesized phase."}),
            },
        }

    def execute(self, exemplar, out_height, out_width, seed):
        import torch.nn.functional as F
        ex = to_bchw(exemplar)
        ex_r = F.interpolate(ex, size=(out_height, out_width), mode="bilinear", align_corners=False)
        ft = nfft.analyze(ex_r)
        g = torch.Generator(device=ex.device).manual_seed(int(seed))
        new_phase = (torch.rand(ft.phase.shape, generator=g, device=ex.device) - 0.5) * 6.283185
        new_ft = FFTTensor(magnitude=ft.magnitude, phase=new_phase,
                           spatial_h=out_height, spatial_w=out_width, centered=True)
        out = nfft.synthesize(new_ft)
        # Renormalize to original mean/std to keep tonal range.
        out = (out - out.mean()) / out.std().clamp_min(1e-6) * ex_r.std() + ex_r.mean()
        return (to_bhwc(out.clamp(0, 1)),)


NODE_CLASS_MAPPINGS = {
    "NukeMax_FFTAnalyze": FFTAnalyze,
    "NukeMax_FFTSynthesize": FFTSynthesize,
    "NukeMax_FrequencyMask": FrequencyMask,
    "NukeMax_LatentFrequencyMatch": LatentFrequencyMatch,
    "NukeMax_FFTTextureSynthesis": FFTTextureSynthesis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_FFTAnalyze": "FFT Analyze",
    "NukeMax_FFTSynthesize": "FFT Synthesize",
    "NukeMax_FrequencyMask": "Frequency Mask",
    "NukeMax_LatentFrequencyMatch": "Latent Frequency Match",
    "NukeMax_FFTTextureSynthesis": "FFT Texture Synthesis",
}
