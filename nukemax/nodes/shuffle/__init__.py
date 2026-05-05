"""Shuffle — Nuke-style channel routing.

Lets you remap any source channel (R,G,B,A,Lum,Const0,Const1) into any
destination channel of the output. Equivalent to Nuke's `Shuffle` /
`ShuffleCopy`. Works on regular IMAGEs and on LATENT tensors.
"""
from __future__ import annotations

import torch

from ...utils.resilience import resilient


_CH_OPTS = ("R", "G", "B", "A", "Lum", "0", "1", "Inv_R", "Inv_G", "Inv_B", "Inv_A")


def _pick(image_bhwc: torch.Tensor, code: str) -> torch.Tensor:
    B, H, W, C = image_bhwc.shape
    R = image_bhwc[..., 0]
    G = image_bhwc[..., 1] if C > 1 else R
    Bch = image_bhwc[..., 2] if C > 2 else R
    A = image_bhwc[..., 3] if C > 3 else torch.ones_like(R)
    if code == "R": return R
    if code == "G": return G
    if code == "B": return Bch
    if code == "A": return A
    if code == "Lum":
        return 0.2126 * R + 0.7152 * G + 0.0722 * Bch
    if code == "0": return torch.zeros_like(R)
    if code == "1": return torch.ones_like(R)
    if code == "Inv_R": return 1.0 - R
    if code == "Inv_G": return 1.0 - G
    if code == "Inv_B": return 1.0 - Bch
    if code == "Inv_A": return 1.0 - A
    return R


@resilient
class ShuffleImage:
    DESCRIPTION = "Remap RGBA channels Nuke-style. Each output channel is picked from any source channel or a constant."
    CATEGORY = "NukeMax/Channel"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "out_R": (_CH_OPTS, {"default": "R"}),
                "out_G": (_CH_OPTS, {"default": "G"}),
                "out_B": (_CH_OPTS, {"default": "B"}),
                "out_A": (_CH_OPTS, {"default": "A"}),
            },
        }

    def execute(self, image, out_R, out_G, out_B, out_A):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        r = _pick(image, out_R)
        g = _pick(image, out_G)
        b = _pick(image, out_B)
        a = _pick(image, out_A)
        out = torch.stack([r, g, b, a], dim=-1).clamp(0, 1)
        # IMAGE output: drop alpha if you want; keep RGB and pass alpha as MASK.
        return (out[..., :3].contiguous(), a)


@resilient
class ShuffleLatent:
    DESCRIPTION = "Remap latent channels by index. Use this to swap or zero out specific latent dimensions."
    CATEGORY = "NukeMax/Channel"
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {}),
                "mapping": ("STRING", {
                    "default": "0,1,2,3",
                    "tooltip": "Comma-separated source-channel indices (or 'z' for zero, 'o' for one) for each output channel.",
                }),
            },
        }

    def execute(self, latent, mapping):
        z = latent["samples"] if isinstance(latent, dict) else latent
        B, C, H, W = z.shape
        codes = [s.strip().lower() for s in mapping.split(",")]
        out = []
        for code in codes:
            if code == "z":
                out.append(torch.zeros(B, 1, H, W, device=z.device, dtype=z.dtype))
            elif code == "o":
                out.append(torch.ones(B, 1, H, W, device=z.device, dtype=z.dtype))
            else:
                idx = max(0, min(C - 1, int(code)))
                out.append(z[:, idx:idx + 1])
        new_z = torch.cat(out, dim=1)
        return ({"samples": new_z},)


NODE_CLASS_MAPPINGS = {
    "NukeMax_ShuffleImage": ShuffleImage,
    "NukeMax_ShuffleLatent": ShuffleLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_ShuffleImage": "Shuffle (NukeMax)",
    "NukeMax_ShuffleLatent": "Shuffle Latent (NukeMax)",
}
