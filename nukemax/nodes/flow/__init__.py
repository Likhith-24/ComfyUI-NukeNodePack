"""Sub-Pixel Optical Flow Compositing — Concept 5."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ...core import flow as nflow
from ...core.color import to_bchw, to_bhwc, luminance
from ...core.composite import merge_over_straight
from ...types import FlowField
from ...utils.resilience import resilient


def _farneback_like_flow(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cheap multi-scale Lucas-Kanade with a few iterations. Math-only,
    no external deps. (B,1,H,W) inputs -> (B,2,H,W) flow.
    Adequate for slow motion and as a fallback when no model is loaded.
    """
    B, _, H, W = a.shape
    flow = torch.zeros(B, 2, H, W, device=a.device, dtype=a.dtype)
    levels = 3
    pyramid_a = [a]
    pyramid_b = [b]
    for _ in range(levels - 1):
        pyramid_a.append(F.avg_pool2d(pyramid_a[-1], 2))
        pyramid_b.append(F.avg_pool2d(pyramid_b[-1], 2))
    for lvl in range(levels - 1, -1, -1):
        ai = pyramid_a[lvl]
        bi = pyramid_b[lvl]
        h, w = ai.shape[-2:]
        flow_l = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False) * (h / max(H, 1))
        for _ in range(3):
            warped = nflow.backward_warp(bi, flow_l)
            kx = torch.tensor([[-0.5, 0, 0.5]], dtype=ai.dtype, device=ai.device).view(1, 1, 1, 3)
            ky = kx.transpose(-1, -2)
            Ix = F.conv2d(F.pad(ai, (1, 1, 0, 0), mode="reflect"), kx)
            Iy = F.conv2d(F.pad(ai, (0, 0, 1, 1), mode="reflect"), ky)
            It = warped - ai
            ksz = 5
            kn = torch.ones(1, 1, ksz, ksz, device=ai.device, dtype=ai.dtype)
            pad = ksz // 2
            def cv(x):
                return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kn)
            A11 = cv(Ix * Ix); A12 = cv(Ix * Iy); A22 = cv(Iy * Iy)
            b1 = -cv(Ix * It); b2 = -cv(Iy * It)
            det = (A11 * A22 - A12 * A12).clamp_min(1e-6)
            du = (A22 * b1 - A12 * b2) / det
            dv = (-A12 * b1 + A11 * b2) / det
            flow_l = flow_l + torch.cat([du, dv], dim=1)
        flow = F.interpolate(flow_l, size=(H, W), mode="bilinear", align_corners=False) * (H / max(h, 1))
    return flow


@resilient
class ComputeOpticalFlow:
    """Compute forward + backward optical flow for an image batch.

    Tries to use OpenCV's Farneback when available (CPU, robust); falls
    back to the pure-PyTorch multi-scale LK above.
    """
    DESCRIPTION = "Compute forward and backward optical flow plus a forward-backward consistency occlusion mask for an image batch."
    CATEGORY = "NukeMax/Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOW_FIELD",)
    RETURN_NAMES = ("flow",)
    OUTPUT_TOOLTIPS = ("Flow field bundle containing forward flow, backward flow, and forward occlusion mask.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Image batch (sequential frames) to compute optical flow on."}),
                "method": (("auto", "torch_lk", "opencv_farneback"), {"tooltip": "Flow algorithm: auto picks OpenCV Farneback if available, else torch LK."}),
                "consistency_threshold": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 32.0, "step": 0.1, "tooltip": "Pixel error threshold for forward-backward consistency check used to flag occlusions."}),
            },
        }

    def execute(self, frames, method, consistency_threshold):
        x = to_bchw(frames)
        y = luminance(x)  # (B,1,H,W)
        B, _, H, W = y.shape
        if B < 2:
            zero = torch.zeros(0, 2, H, W)
            return (FlowField(flow_fwd=zero, flow_bwd=zero, occlusion_fwd=torch.zeros(0, 1, H, W)),)
        cv2 = None
        if method != "torch_lk":
            try:
                import cv2 as _cv2  # type: ignore
                cv2 = _cv2
            except ImportError:
                if method == "opencv_farneback":
                    raise
        if cv2 is not None and method != "torch_lk":
            import numpy as np
            fwd = []
            bwd = []
            ynp = (y.squeeze(1) * 255).clamp(0, 255).byte().cpu().numpy()
            for t in range(B - 1):
                f = cv2.calcOpticalFlowFarneback(ynp[t], ynp[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                b = cv2.calcOpticalFlowFarneback(ynp[t + 1], ynp[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                fwd.append(torch.from_numpy(f).permute(2, 0, 1))
                bwd.append(torch.from_numpy(b).permute(2, 0, 1))
            flow_fwd = torch.stack(fwd, dim=0).to(y.device, y.dtype)
            flow_bwd = torch.stack(bwd, dim=0).to(y.device, y.dtype)
        else:
            flow_fwd = _farneback_like_flow(y[:-1], y[1:])
            flow_bwd = _farneback_like_flow(y[1:], y[:-1])
        occ = nflow.occlusion_from_consistency(flow_fwd, flow_bwd, threshold=consistency_threshold)
        return (FlowField(flow_fwd=flow_fwd, flow_bwd=flow_bwd, occlusion_fwd=occ),)


@resilient
class FlowBackwardWarp:
    DESCRIPTION = "Backward-warp an image batch using a precomputed flow field in either the forward or backward direction."
    CATEGORY = "NukeMax/Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped",)
    OUTPUT_TOOLTIPS = ("Image batch resampled by the chosen flow direction.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image batch to warp."}),
                "flow": ("FLOW_FIELD", {"tooltip": "Flow field bundle from Compute Optical Flow."}),
                "direction": (("forward", "backward"), {"tooltip": "Which flow vector to use for the backward warp."}),
            },
        }

    def execute(self, image, flow, direction):
        x = to_bchw(image)
        f = flow.flow_fwd if direction == "forward" else flow.flow_bwd
        # Pad to len(x) by repeating last flow
        T = x.shape[0]
        if f.shape[0] < T:
            pad = T - f.shape[0]
            f = torch.cat([f, f[-1:].expand(pad, -1, -1, -1)], dim=0)
        else:
            f = f[:T]
        out = nflow.backward_warp(x, f)
        return (to_bhwc(out.clamp(0, 1)),)


@resilient
class FlowForwardWarp:
    DESCRIPTION = "Splat-style forward warp of an image batch using the forward flow; also returns the splat weight as a mask."
    CATEGORY = "NukeMax/Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("warped", "weight")
    OUTPUT_TOOLTIPS = ("Forward-warped image, normalized by accumulated splat weight.", "Per-pixel splat coverage (0=no contribution, 1=full).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"tooltip": "Source image batch to splat-forward."}), "flow": ("FLOW_FIELD", {"tooltip": "Flow field bundle from Compute Optical Flow."})}}

    def execute(self, image, flow):
        x = to_bchw(image)
        f = flow.flow_fwd
        T = x.shape[0]
        if f.shape[0] < T:
            f = torch.cat([f, f[-1:].expand(T - f.shape[0], -1, -1, -1)], dim=0)
        else:
            f = f[:T]
        out, w = nflow.forward_warp(x, f)
        out_norm = out / w.clamp_min(1e-6)
        return (to_bhwc(out_norm.clamp(0, 1)), w.squeeze(1).clamp(0, 1))


@resilient
class FlowOcclusionMask:
    DESCRIPTION = "Extract the occlusion mask from a flow bundle, computing it from forward-backward consistency if missing."
    CATEGORY = "NukeMax/Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("occlusion",)
    OUTPUT_TOOLTIPS = ("Per-pixel forward-flow occlusion mask (1=occluded/inconsistent).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"flow": ("FLOW_FIELD", {"tooltip": "Flow field bundle from Compute Optical Flow."})}}

    def execute(self, flow):
        if flow.occlusion_fwd is None:
            occ = nflow.occlusion_from_consistency(flow.flow_fwd, flow.flow_bwd)
        else:
            occ = flow.occlusion_fwd
        return (occ.squeeze(1),)


@resilient
class CleanPlateMerge:
    """Warp a clean plate along the flow under a moving object mask, then
    merge to remove the object. Output: clean composite IMAGE.
    """
    DESCRIPTION = "Warp a clean plate along the flow chain and merge it under the object mask to remove a moving object from footage."
    CATEGORY = "NukeMax/Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Clean composite with the masked object replaced by the warped plate.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "footage": ("IMAGE", {"tooltip": "Original footage batch containing the unwanted object."}),
                "clean_plate": ("IMAGE", {"tooltip": "Clean reference plate (single frame or batch) to warp into the hole."}),
                "object_mask": ("MASK", {"tooltip": "Mask covering the object to remove (1=replace with plate)."}),
                "flow": ("FLOW_FIELD", {"tooltip": "Flow field bundle used to align the plate to each frame."}),
                "feather_px": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "tooltip": "Gaussian feather radius in pixels applied to the object mask edge."}),
            },
        }

    def execute(self, footage, clean_plate, object_mask, flow, feather_px):
        from ...core import blur as nblur
        foot = to_bchw(footage)
        plate = to_bchw(clean_plate)
        T = foot.shape[0]
        if plate.shape[0] == 1:
            plate = plate.expand(T, -1, -1, -1)
        # Warp plate forward along the flow chain to align with each frame.
        plate_aligned = plate.clone()
        for t in range(1, min(T, flow.flow_fwd.shape[0] + 1)):
            plate_aligned[t:t + 1] = nflow.backward_warp(plate_aligned[t - 1:t], flow.flow_bwd[t - 1:t])
        m = object_mask if object_mask.ndim == 4 else object_mask.unsqueeze(1)
        if feather_px > 0:
            m = nblur.gaussian_blur(m, feather_px / 3.0)
        m = m.expand(T, 1, -1, -1) if m.shape[0] == 1 else m
        rgb_out, _ = merge_over_straight(plate_aligned, m, foot, torch.ones_like(m))
        return (to_bhwc(rgb_out.clamp(0, 1)),)


@resilient
class FlowVisualize:
    """Render a Middlebury-color visualization of forward flow."""
    DESCRIPTION = "Render a Middlebury-style HSV color visualization of the forward flow vectors."
    CATEGORY = "NukeMax/Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("RGB image visualizing flow direction (hue) and magnitude (saturation).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"flow": ("FLOW_FIELD", {"tooltip": "Flow field bundle from Compute Optical Flow."}), "max_magnitude": ("FLOAT", {"default": 32.0, "min": 0.5, "max": 1024.0, "tooltip": "Pixel magnitude that maps to full saturation in the visualization."})}}

    def execute(self, flow, max_magnitude):
        f = flow.flow_fwd
        ang = torch.atan2(f[:, 1], f[:, 0])
        mag = (f.norm(dim=1) / max_magnitude).clamp(0, 1)
        h = (ang / (2 * math.pi) + 0.5).clamp(0, 1)
        s = mag
        v = torch.ones_like(mag)
        # HSV->RGB
        i = (h * 6).floor()
        f6 = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f6 * s)
        t = v * (1 - (1 - f6) * s)
        i_mod = i.long() % 6
        r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
        g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
        b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))
        rgb = torch.stack([r, g, b], dim=1)
        return (to_bhwc(rgb.clamp(0, 1)),)


NODE_CLASS_MAPPINGS = {
    "NukeMax_ComputeOpticalFlow": ComputeOpticalFlow,
    "NukeMax_FlowBackwardWarp": FlowBackwardWarp,
    "NukeMax_FlowForwardWarp": FlowForwardWarp,
    "NukeMax_FlowOcclusionMask": FlowOcclusionMask,
    "NukeMax_CleanPlateMerge": CleanPlateMerge,
    "NukeMax_FlowVisualize": FlowVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_ComputeOpticalFlow": "Compute Optical Flow",
    "NukeMax_FlowBackwardWarp": "Flow Backward Warp",
    "NukeMax_FlowForwardWarp": "Flow Forward Warp",
    "NukeMax_FlowOcclusionMask": "Flow Occlusion Mask",
    "NukeMax_CleanPlateMerge": "Clean Plate Merge",
    "NukeMax_FlowVisualize": "Flow Visualize",
}
