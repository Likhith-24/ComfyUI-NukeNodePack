"""Deep image compositing — Nuke-style multi-sample-per-pixel compositing.

Nodes:
  - DeepFromImage         : IMAGE + depth(MASK) -> DEEP_IMAGE   (single sample/pixel)
  - DeepMerge             : combine two DEEP_IMAGE streams (sample-wise concat + sort)
  - DeepHoldout           : front-most opaque sample masks all behind it
  - DeepFlatten           : DEEP_IMAGE -> IMAGE + depth (front-to-back over)
  - DeepRecolor           : remap colour of samples deeper / shallower than a Z threshold
"""
from __future__ import annotations

import torch

from ...types import DeepImage
from ...utils.resilience import resilient


@resilient
class DeepFromImage:
    DESCRIPTION = "Build a single-sample DEEP_IMAGE from an IMAGE and a depth MASK (or alpha)."
    CATEGORY = "NukeMax/Deep"
    FUNCTION = "execute"
    RETURN_TYPES = ("DEEP_IMAGE",)
    RETURN_NAMES = ("deep",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "RGB(A) source image batch."}),
                "depth": ("MASK", {"tooltip": "Per-pixel depth (lower = closer)."}),
            },
            "optional": {
                "alpha": ("MASK", {"tooltip": "Optional explicit alpha mask; otherwise IMAGE alpha is used."}),
            },
        }

    def execute(self, image, depth, alpha=None):
        if depth.dim() == 4:
            depth = depth.squeeze(-1)
        if alpha is not None and alpha.dim() == 4:
            alpha = alpha.squeeze(-1)
        deep = DeepImage.from_image_depth(image, depth, alpha)
        return (deep,)


@resilient
class DeepMerge:
    DESCRIPTION = "Merge two DEEP_IMAGE streams sample-wise. Output keeps up to max_samples per pixel, sorted front-to-back."
    CATEGORY = "NukeMax/Deep"
    FUNCTION = "execute"
    RETURN_TYPES = ("DEEP_IMAGE",)
    RETURN_NAMES = ("deep",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("DEEP_IMAGE", {}),
                "b": ("DEEP_IMAGE", {}),
                "max_samples": ("INT", {"default": 8, "min": 1, "max": 64,
                                        "tooltip": "Cap on samples kept per pixel after merge."}),
            },
        }

    def execute(self, a: DeepImage, b: DeepImage, max_samples: int):
        Ba, Ka, Ha, Wa = a.samples_z.shape
        Bb, Kb, Hb, Wb = b.samples_z.shape
        assert (Ba, Ha, Wa) == (Bb, Hb, Wb), "Deep merge requires matching (B,H,W)."
        K = Ka + Kb
        z = torch.cat([a.samples_z, b.samples_z], dim=1)
        rgba = torch.cat([a.samples_rgba, b.samples_rgba], dim=1)
        # Mask invalid slots to +inf so they sort to the back.
        k_a = torch.arange(Ka, device=z.device).view(1, Ka, 1, 1)
        k_b = torch.arange(Kb, device=z.device).view(1, Kb, 1, 1)
        valid_a = (k_a < a.sample_count.unsqueeze(1))
        valid_b = (k_b < b.sample_count.unsqueeze(1))
        valid = torch.cat([valid_a, valid_b], dim=1)
        z_sort = z.masked_fill(~valid, float("inf"))
        order = torch.argsort(z_sort, dim=1)
        z_sorted = torch.gather(z, 1, order)
        rgba_sorted = torch.gather(rgba, 1, order.unsqueeze(-1).expand(-1, -1, -1, -1, 4))
        valid_sorted = torch.gather(valid, 1, order)
        Kc = min(K, int(max_samples))
        z_out = z_sorted[:, :Kc]
        rgba_out = rgba_sorted[:, :Kc]
        valid_out = valid_sorted[:, :Kc]
        cnt = valid_out.sum(dim=1).to(torch.int32)
        # Zero out colour in invalid slots.
        rgba_out = rgba_out * valid_out.unsqueeze(-1).float()
        z_out = z_out.masked_fill(~valid_out, 0.0)
        return (DeepImage(samples_z=z_out, samples_rgba=rgba_out, sample_count=cnt),)


@resilient
class DeepHoldout:
    DESCRIPTION = "Use the front-most opaque sample of `holdout` to mask all `subject` samples behind it (true deep holdout)."
    CATEGORY = "NukeMax/Deep"
    FUNCTION = "execute"
    RETURN_TYPES = ("DEEP_IMAGE",)
    RETURN_NAMES = ("deep",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": ("DEEP_IMAGE", {}),
                "holdout": ("DEEP_IMAGE", {}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def execute(self, subject: DeepImage, holdout: DeepImage, alpha_threshold: float):
        # Pick front-most holdout sample whose alpha exceeds threshold.
        z_h = holdout.samples_z.clone()
        a_h = holdout.samples_rgba[..., 3]
        K = z_h.shape[1]
        k_idx = torch.arange(K, device=z_h.device).view(1, K, 1, 1)
        invalid = (k_idx >= holdout.sample_count.unsqueeze(1)) | (a_h < alpha_threshold)
        z_h = z_h.masked_fill(invalid, float("inf"))
        front_z = z_h.min(dim=1).values  # (B,H,W)
        # Subject samples with z >= front_z get killed (alpha->0).
        keep = subject.samples_z < front_z.unsqueeze(1)
        rgba_new = subject.samples_rgba * keep.unsqueeze(-1).float()
        cnt_new = keep.to(torch.int32).sum(dim=1).clamp_max(subject.sample_count)
        return (DeepImage(samples_z=subject.samples_z, samples_rgba=rgba_new,
                          sample_count=cnt_new),)


@resilient
class DeepFlatten:
    DESCRIPTION = "Flatten a DEEP_IMAGE to a regular IMAGE + front-depth MASK using front-to-back over compositing."
    CATEGORY = "NukeMax/Deep"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "depth")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"deep": ("DEEP_IMAGE", {})}}

    def execute(self, deep: DeepImage):
        img, depth = deep.to_image_depth()
        return (img, depth)


@resilient
class DeepRecolor:
    DESCRIPTION = "Tint samples deeper than (or shallower than) a Z threshold. Useful for atmospheric depth/fog passes."
    CATEGORY = "NukeMax/Deep"
    FUNCTION = "execute"
    RETURN_TYPES = ("DEEP_IMAGE",)
    RETURN_NAMES = ("deep",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deep": ("DEEP_IMAGE", {}),
                "z_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1e6, "step": 0.01}),
                "tint_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "tint_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "tint_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "mode": (("deeper", "shallower"), {}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def execute(self, deep: DeepImage, z_threshold, tint_r, tint_g, tint_b, mode, amount):
        tint = torch.tensor([tint_r, tint_g, tint_b, 1.0],
                            device=deep.samples_z.device, dtype=deep.samples_rgba.dtype)
        if mode == "deeper":
            sel = deep.samples_z > z_threshold
        else:
            sel = deep.samples_z < z_threshold
        sel_f = sel.unsqueeze(-1).float() * amount
        tinted = deep.samples_rgba * (1 - sel_f) + (deep.samples_rgba * tint) * sel_f
        return (DeepImage(samples_z=deep.samples_z, samples_rgba=tinted,
                          sample_count=deep.sample_count),)


NODE_CLASS_MAPPINGS = {
    "NukeMax_DeepFromImage": DeepFromImage,
    "NukeMax_DeepMerge": DeepMerge,
    "NukeMax_DeepHoldout": DeepHoldout,
    "NukeMax_DeepFlatten": DeepFlatten,
    "NukeMax_DeepRecolor": DeepRecolor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_DeepFromImage": "Deep From Image (NukeMax)",
    "NukeMax_DeepMerge": "Deep Merge (NukeMax)",
    "NukeMax_DeepHoldout": "Deep Holdout (NukeMax)",
    "NukeMax_DeepFlatten": "Deep Flatten (NukeMax)",
    "NukeMax_DeepRecolor": "Deep Recolor (NukeMax)",
}
