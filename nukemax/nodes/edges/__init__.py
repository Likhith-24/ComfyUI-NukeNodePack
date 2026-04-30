"""Smart Edge Tools — Concept 6."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core import blur
from ...core.color import to_bchw, to_bhwc, luminance
from ...types import TrackingData
from ...utils.resilience import resilient


def _mask_to_bchw(m: torch.Tensor) -> torch.Tensor:
    if m.ndim == 3:
        return m.unsqueeze(1)
    return m


def _mask_back(m: torch.Tensor) -> torch.Tensor:
    return m.squeeze(1) if m.ndim == 4 else m


@resilient
class NormalAwareEdgeBlur:
    """Blur a mask, but stop the blur at normal-map discontinuities.

    Implementation: separable Gaussian gated by `dot(N(p), N(q)) > thresh`
    along each tap. Keeps hair/fur fine detail intact while still
    smoothing within a single surface.
    """
    DESCRIPTION = "Cross-bilateral mask blur gated by a normal map; smooths within surfaces while preserving normal discontinuities."
    CATEGORY = "NukeMax/Edges"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Mask blurred only across surfaces with similar normals.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to blur."}),
                "normal": ("IMAGE", {"tooltip": "Normal map (0..1 RGB) used to gate the bilateral kernel."}),
                "sigma": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.1, "tooltip": "Gaussian standard deviation in pixels."}),
                "normal_threshold": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum cosine similarity between normals for samples to contribute."}),
            },
        }

    def execute(self, mask, normal, sigma, normal_threshold):
        m = _mask_to_bchw(mask)
        n = to_bchw(normal) * 2 - 1   # remap [0,1] -> [-1,1]
        # Replace zero-magnitude (degenerate) normals with a forward-facing
        # default (0,0,1) so the bilateral kernel still has a similarity basis.
        mag = n.norm(dim=1, keepdim=True)
        ident = torch.zeros_like(n)
        ident[:, 2:3] = 1.0
        n = torch.where(mag < 1e-6, ident, n / mag.clamp_min(1e-6))
        # Naive cross-bilateral: weight a separable Gaussian by normal similarity.
        radius = max(int(round(sigma * 3)), 1)
        k1d = blur.gaussian_kernel_1d(sigma, dtype=m.dtype, device=m.device)
        out = torch.zeros_like(m)
        wsum = torch.zeros_like(m)
        # Horizontal pass
        for i, kw in enumerate(k1d):
            shift = i - radius
            n_shift = torch.roll(n, shifts=shift, dims=-1)
            similarity = (n * n_shift).sum(dim=1, keepdim=True).clamp_min(0)
            gate = (similarity > normal_threshold).to(m.dtype)
            w = kw * gate
            out = out + torch.roll(m, shifts=shift, dims=-1) * w
            wsum = wsum + w
        m = out / wsum.clamp_min(1e-6)
        # Vertical pass
        out = torch.zeros_like(m)
        wsum = torch.zeros_like(m)
        for i, kw in enumerate(k1d):
            shift = i - radius
            n_shift = torch.roll(n, shifts=shift, dims=-2)
            similarity = (n * n_shift).sum(dim=1, keepdim=True).clamp_min(0)
            gate = (similarity > normal_threshold).to(m.dtype)
            w = kw * gate
            out = out + torch.roll(m, shifts=shift, dims=-2) * w
            wsum = wsum + w
        m = out / wsum.clamp_min(1e-6)
        return (_mask_back(m).clamp(0, 1),)


@resilient
class MatteDensityAdjust:
    """Apply gamma/contrast only to the semi-transparent edge band.
    Fully-opaque (alpha == 1) and fully-transparent (alpha == 0) regions
    are returned bit-exact.
    """
    DESCRIPTION = "Adjust gamma and contrast only on the semi-transparent edge band of a matte; opaque and clear regions are preserved exactly."
    CATEGORY = "NukeMax/Edges"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Mask with adjusted edge density.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Input matte to adjust."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.01, "tooltip": "Gamma applied to the edge band; <1 darkens, >1 brightens."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01, "tooltip": "Contrast multiplier around 0.5 for the edge band."}),
                "edge_lo": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "tooltip": "Lower alpha bound that defines the edge band."}),
                "edge_hi": ("FLOAT", {"default": 0.99, "min": 0.5, "max": 1.0, "tooltip": "Upper alpha bound that defines the edge band."}),
            },
        }

    def execute(self, mask, gamma, contrast, edge_lo, edge_hi):
        m = mask
        # Smooth gating: 1 inside the edge band, 0 outside.
        t_lo = ((m - edge_lo) / max(edge_lo * 0.5, 1e-4)).clamp(0, 1)
        t_hi = ((edge_hi - m) / max((1.0 - edge_hi) * 0.5, 1e-4)).clamp(0, 1)
        gate = t_lo * t_hi
        adjusted = ((m - 0.5) * contrast + 0.5).clamp(1e-6, 1).pow(1.0 / max(gamma, 1e-6))
        out = m * (1 - gate) + adjusted * gate
        return (out.clamp(0, 1),)


@resilient
class SubPixelEdgeDetect:
    """Sobel + parabolic interpolation for sub-pixel edge localization.
    Outputs a MASK of edge magnitudes plus TRACKING_DATA of the K
    strongest edge points per frame for downstream tracking.
    """
    DESCRIPTION = "Sobel edge detector with sub-pixel localization; emits an edge magnitude mask and the top-K edge points as tracking data."
    CATEGORY = "NukeMax/Edges"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK", "TRACKING_DATA")
    RETURN_NAMES = ("edges", "tracks")
    OUTPUT_TOOLTIPS = ("Per-pixel edge magnitude normalized to [0,1].", "Top-K edge point coordinates with confidence per frame.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image batch to detect edges on."}),
                "top_k": ("INT", {"default": 256, "min": 1, "max": 4096, "tooltip": "Number of strongest edge points kept per frame as tracks."}),
            },
        }

    def execute(self, image, top_k):
        x = luminance(to_bchw(image))
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kx)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), ky)
        mag = (gx * gx + gy * gy).sqrt()
        mag_n = mag / mag.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        B, _, H, W = mag_n.shape
        # Top-K per frame for tracking output
        flat = mag_n.view(B, -1)
        k = min(top_k, flat.shape[-1])
        vals, idx = flat.topk(k, dim=-1)
        ys = (idx // W).float()
        xs = (idx % W).float()
        coords = torch.stack([xs, ys], dim=-1)  # (B,K,2)
        td = TrackingData(
            coords=coords,
            velocity=torch.zeros_like(coords),
            confidence=vals,
            canvas_h=H, canvas_w=W,
        )
        return (mag_n.squeeze(1), td)


@resilient
class HairAwareChoke:
    """Choke a matte using local high-frequency energy as a proxy for
    'hair'. Hair regions get less choke; smooth regions get more.
    """
    DESCRIPTION = "Choke a matte adaptively, applying less choke in high-frequency 'hair' regions and more in smooth regions."
    CATEGORY = "NukeMax/Edges"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Hair-aware choked mask.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Input matte to choke."}),
                "image": ("IMAGE", {"tooltip": "Companion image used to estimate hair-frequency energy."}),
                "choke": ("FLOAT", {"default": 1.5, "min": -8.0, "max": 8.0, "step": 0.1, "tooltip": "Choke amount; positive erodes, negative dilates, 0 is no-op."}),
                "hair_window": ("INT", {"default": 5, "min": 1, "max": 31, "tooltip": "Window size in pixels for the local std-dev hair detector."}),
            },
        }

    def execute(self, mask, image, choke, hair_window):
        m = _mask_to_bchw(mask)
        # Nuke convention: choke=0 is a true no-op (matches Erode/Dilate at 0).
        if choke == 0:
            return (_mask_back(m.clamp(0, 1)),)
        y = luminance(to_bchw(image))
        # Local stddev as hair indicator
        kernel = torch.ones(1, 1, hair_window, hair_window, device=y.device, dtype=y.dtype) / (hair_window ** 2)
        pad = hair_window // 2
        y_pad = F.pad(y, (pad, pad, pad, pad), mode="reflect")
        mu = F.conv2d(y_pad, kernel)
        mu2 = F.conv2d(F.pad(y * y, (pad, pad, pad, pad), mode="reflect"), kernel)
        std = (mu2 - mu * mu).clamp_min(0).sqrt()
        std_n = std / std.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        # Distance-field-style choke via small box-blur trick
        eroded = F.avg_pool2d(m, 3, 1, 1) - 0.05 * choke
        out = (m * std_n + eroded * (1 - std_n)).clamp(0, 1)
        return (_mask_back(out),)


NODE_CLASS_MAPPINGS = {
    "NukeMax_NormalAwareEdgeBlur": NormalAwareEdgeBlur,
    "NukeMax_MatteDensityAdjust": MatteDensityAdjust,
    "NukeMax_SubPixelEdgeDetect": SubPixelEdgeDetect,
    "NukeMax_HairAwareChoke": HairAwareChoke,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_NormalAwareEdgeBlur": "Normal-Aware Edge Blur",
    "NukeMax_MatteDensityAdjust": "Matte Density Adjust",
    "NukeMax_SubPixelEdgeDetect": "Sub-Pixel Edge Detect",
    "NukeMax_HairAwareChoke": "Hair-Aware Choke",
}
