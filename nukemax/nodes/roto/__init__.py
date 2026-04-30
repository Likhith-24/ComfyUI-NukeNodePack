"""Smart Roto System — Concept 1.

Nodes:
  - Roto Spline Editor (Interactive)  -> ROTO_SHAPE
  - Roto Shape From JSON              -> ROTO_SHAPE   (file-based fallback)
  - Roto Shape To AI Tracker          -> ROTO_SHAPE, TRACKING_DATA
  - Roto Shape Renderer               -> MASK
  - Roto Shape To Diffusion Guidance  -> MASK, MASK (soft), MASK (latent), STRING (sam_prompts)
"""
from __future__ import annotations

import json
import math

import torch

from ...core import blur, splines
from ...core.flow import backward_warp
from ...types import RotoShape, TrackingData
from ...utils.resilience import resilient


# ---------------- Roto Spline Editor (interactive) ----------------

@resilient
class RotoSplineEditor:
    """Interactive bezier editor. The JS widget posts a JSON spline state
    into the hidden `spline_state` STRING widget. This node parses it.

    JSON schema:
      {
        "frames": [
          { "points": [[x,y],...], "in": [[x,y],...], "out": [[x,y],...],
            "feather": [f,...] }
        ],
        "closed": true,
        "canvas": {"h": H, "w": W}
      }
    """
    DESCRIPTION = "Parse a JSON spline state from the interactive bezier editor into a ROTO_SHAPE on the given canvas."
    CATEGORY = "NukeMax/Roto"
    FUNCTION = "execute"
    RETURN_TYPES = ("ROTO_SHAPE",)
    RETURN_NAMES = ("roto",)
    OUTPUT_TOOLTIPS = ("Per-frame roto shape (points, handles, feather, canvas).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spline_state": ("STRING", {"multiline": True, "default": "{}", "tooltip": "JSON string from the JS bezier editor describing per-frame points, handles, and feather."}),
                "canvas_height": ("INT", {"default": 512, "min": 8, "max": 8192, "tooltip": "Canvas height in pixels when the JSON does not specify one."}),
                "canvas_width": ("INT", {"default": 512, "min": 8, "max": 8192, "tooltip": "Canvas width in pixels when the JSON does not specify one."}),
            },
        }

    def execute(self, spline_state: str, canvas_height: int, canvas_width: int):
        try:
            data = json.loads(spline_state) if spline_state.strip() else {}
        except json.JSONDecodeError:
            data = {}
        frames = data.get("frames")
        canvas = data.get("canvas", {})
        H = int(canvas.get("h", canvas_height))
        W = int(canvas.get("w", canvas_width))
        closed = bool(data.get("closed", True))
        if not frames:
            # Default 4-vertex rectangle in the middle
            cx, cy = W * 0.5, H * 0.5
            r = min(H, W) * 0.25
            pts = torch.tensor([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]])
            return (RotoShape.from_polygon(pts, (H, W), feather=0.0, closed=closed),)
        T = len(frames)
        N = len(frames[0]["points"])
        pts = torch.zeros(T, N, 2)
        hin = torch.zeros(T, N, 2)
        hout = torch.zeros(T, N, 2)
        feather = torch.zeros(T, N)
        for i, f in enumerate(frames):
            pts[i] = torch.tensor(f["points"], dtype=torch.float32)
            hin[i] = torch.tensor(f.get("in", f["points"]), dtype=torch.float32)
            hout[i] = torch.tensor(f.get("out", f["points"]), dtype=torch.float32)
            fe = f.get("feather", [0.0] * N)
            feather[i] = torch.tensor(fe, dtype=torch.float32)
        return (RotoShape(points=pts, handles_in=hin, handles_out=hout, feather=feather,
                          canvas_h=H, canvas_w=W, closed=closed),)


# ---------------- Roto Shape From JSON (file path) ----------------

@resilient
class RotoShapeFromFile:
    DESCRIPTION = "Load a ROTO_SHAPE from a JSON file on disk using the same schema as the spline editor."
    CATEGORY = "NukeMax/Roto"
    FUNCTION = "execute"
    RETURN_TYPES = ("ROTO_SHAPE",)
    RETURN_NAMES = ("roto",)
    OUTPUT_TOOLTIPS = ("Roto shape parsed from the JSON file.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "", "tooltip": "Filesystem path to a JSON file containing a roto spline state."})}}

    def execute(self, path: str):
        from pathlib import Path
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            txt = f.read()
        editor = RotoSplineEditor()
        return editor.execute(txt, 512, 512)


# ---------------- Roto Shape To AI Tracker ----------------

@resilient
class RotoShapeToAITracker:
    """Propagate a frame-0 ROTO_SHAPE across an IMAGE batch using optical
    flow. If a FLOW_FIELD is provided, we use it directly. Otherwise we
    compute a fast Lucas-Kanade-like estimate per-vertex via local
    cross-correlation (math, not a model).
    """
    DESCRIPTION = "Propagate a frame-0 roto shape across a frame batch using a flow field or per-vertex NCC tracking."
    CATEGORY = "NukeMax/Roto"
    FUNCTION = "execute"
    RETURN_TYPES = ("ROTO_SHAPE", "TRACKING_DATA")
    RETURN_NAMES = ("roto_animated", "tracks")
    OUTPUT_TOOLTIPS = ("Roto shape with per-frame propagated points and handles.", "Per-frame tracking data (coords, velocity, confidence) for the vertices.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "roto": ("ROTO_SHAPE", {"tooltip": "Source roto shape; only frame 0 is used as the seed."}),
                "frames": ("IMAGE", {"tooltip": "Image batch to propagate the shape across."}),
            },
            "optional": {
                "flow": ("FLOW_FIELD", {"tooltip": "Optional precomputed flow field; bypasses NCC search."}),
                "search_radius": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "NCC search window radius in pixels when no flow is provided."}),
            },
        }

    def execute(self, roto: RotoShape, frames: torch.Tensor, flow=None, search_radius: int = 8):
        # frames: (B,H,W,C) ComfyUI -> (B,C,H,W)
        if frames.shape[-1] in (1, 3, 4) and frames.shape[1] not in (1, 3, 4):
            frames_bchw = frames.permute(0, 3, 1, 2).contiguous()
        else:
            frames_bchw = frames
        B, _, H, W = frames_bchw.shape
        # Use first frame's ROTO; replicate or trim points across B.
        pts0 = roto.points[0].clone()  # (N,2)
        hin0 = roto.handles_in[0].clone()
        hout0 = roto.handles_out[0].clone()
        feather0 = roto.feather[0].clone()
        N = pts0.shape[0]
        out_pts = torch.zeros(B, N, 2)
        out_hin = torch.zeros(B, N, 2)
        out_hout = torch.zeros(B, N, 2)
        out_pts[0] = pts0
        out_hin[0] = hin0
        out_hout[0] = hout0
        velocity = torch.zeros(B, N, 2)
        confidence = torch.ones(B, N)

        if flow is not None and B - 1 <= flow.flow_fwd.shape[0]:
            # Use flow field for propagation.
            cur = pts0.clone()
            cur_in = hin0.clone()
            cur_out = hout0.clone()
            for t in range(1, B):
                f_t = flow.flow_fwd[t - 1]  # (2,H,W)
                disp = self._sample_flow(f_t, cur)
                cur = cur + disp
                cur_in = cur_in + self._sample_flow(f_t, cur_in)
                cur_out = cur_out + self._sample_flow(f_t, cur_out)
                out_pts[t] = cur
                out_hin[t] = cur_in
                out_hout[t] = cur_out
                velocity[t] = disp
        else:
            # Fallback: local NCC search per vertex.
            cur = pts0.clone()
            cur_in = hin0.clone()
            cur_out = hout0.clone()
            for t in range(1, B):
                disp = self._ncc_track(frames_bchw[t - 1], frames_bchw[t], cur, search_radius)
                cur = cur + disp
                cur_in = cur_in + disp
                cur_out = cur_out + disp
                out_pts[t] = cur
                out_hin[t] = cur_in
                out_hout[t] = cur_out
                velocity[t] = disp

        feather_b = feather0.unsqueeze(0).expand(B, -1).clone()
        animated = RotoShape(
            points=out_pts, handles_in=out_hin, handles_out=out_hout,
            feather=feather_b, canvas_h=H, canvas_w=W, closed=roto.closed, name=roto.name,
        )
        td = TrackingData(coords=out_pts, velocity=velocity, confidence=confidence,
                          canvas_h=H, canvas_w=W)
        return (animated, td)

    @staticmethod
    def _sample_flow(flow_2hw: torch.Tensor, pts_n2: torch.Tensor) -> torch.Tensor:
        """Sample a (2,H,W) flow at (N,2) sub-pixel points -> (N,2)."""
        H, W = flow_2hw.shape[-2:]
        x = pts_n2[:, 0].clamp(0, W - 1)
        y = pts_n2[:, 1].clamp(0, H - 1)
        x0 = x.floor().long(); x1 = (x0 + 1).clamp(max=W - 1)
        y0 = y.floor().long(); y1 = (y0 + 1).clamp(max=H - 1)
        wx = x - x0.float(); wy = y - y0.float()
        f00 = flow_2hw[:, y0, x0]; f10 = flow_2hw[:, y0, x1]
        f01 = flow_2hw[:, y1, x0]; f11 = flow_2hw[:, y1, x1]
        sampled = (f00 * (1 - wx) * (1 - wy) + f10 * wx * (1 - wy)
                   + f01 * (1 - wx) * wy + f11 * wx * wy)
        return sampled.T  # (N,2)

    @staticmethod
    def _ncc_track(a_chw: torch.Tensor, b_chw: torch.Tensor, pts: torch.Tensor, radius: int) -> torch.Tensor:
        """Cheap normalized cross-correlation per vertex inside a small
        search window. Patch size = 7x7. Returns (N,2) displacement.
        """
        C, H, W = a_chw.shape
        patch = 3
        N = pts.shape[0]
        disp = torch.zeros(N, 2)
        for i in range(N):
            cx = int(round(float(pts[i, 0].item())))
            cy = int(round(float(pts[i, 1].item())))
            x0 = max(cx - patch, 0); x1 = min(cx + patch + 1, W)
            y0 = max(cy - patch, 0); y1 = min(cy + patch + 1, H)
            tmpl = a_chw[:, y0:y1, x0:x1]
            if tmpl.numel() == 0:
                continue
            ph, pw = tmpl.shape[-2:]
            best = (-1e9, 0, 0)
            tmpl_n = tmpl - tmpl.mean()
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    sy0 = y0 + dy; sx0 = x0 + dx
                    sy1 = sy0 + ph; sx1 = sx0 + pw
                    if sy0 < 0 or sx0 < 0 or sy1 > H or sx1 > W:
                        continue
                    cand = b_chw[:, sy0:sy1, sx0:sx1]
                    cand_n = cand - cand.mean()
                    num = (tmpl_n * cand_n).sum().item()
                    den = (tmpl_n.norm() * cand_n.norm()).clamp_min(1e-8).item()
                    score = num / den
                    if score > best[0]:
                        best = (score, dx, dy)
            disp[i, 0] = best[1]; disp[i, 1] = best[2]
        return disp


# ---------------- Roto Shape Renderer ----------------

@resilient
class RotoShapeRenderer:
    DESCRIPTION = "Rasterize a ROTO_SHAPE to a per-frame mask with optional flow-driven directional motion blur."
    CATEGORY = "NukeMax/Roto"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Rasterized per-frame roto mask.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "roto": ("ROTO_SHAPE", {"tooltip": "Roto shape to rasterize."}),
                "samples_per_segment": ("INT", {"default": 16, "min": 2, "max": 128, "tooltip": "Bezier sampling density per segment when building the polyline."}),
                "feather_override": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 256.0, "step": 0.1, "tooltip": "Feather radius in pixels; <0 uses the per-vertex feather mean."}),
            },
            "optional": {
                "flow": ("FLOW_FIELD", {"tooltip": "Optional flow used to drive directional motion blur."}),
                "motion_blur_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Multiplier on flow vectors for the directional blur amount."}),
            },
        }

    def execute(self, roto: RotoShape, samples_per_segment: int, feather_override: float,
                flow=None, motion_blur_strength: float = 0.0):
        polyline = splines.shape_to_polyline(
            roto.points, roto.handles_in, roto.handles_out,
            closed=roto.closed, samples_per_segment=samples_per_segment,
        )
        feather = feather_override if feather_override >= 0 else float(roto.feather.mean().item())
        mask = splines.rasterize_polygon_sdf(
            polyline, H=roto.canvas_h, W=roto.canvas_w, feather=feather, closed=roto.closed,
        )  # (T,H,W)
        # Optional flow-driven motion blur (per-pair).
        if flow is not None and motion_blur_strength > 0 and mask.shape[0] > 1:
            T = mask.shape[0]
            blurred = mask.clone().unsqueeze(1)  # (T,1,H,W)
            for t in range(min(T, flow.flow_fwd.shape[0] + 1) - 1):
                f = flow.flow_fwd[t] * motion_blur_strength  # (2,H,W)
                dx = f[0:1].unsqueeze(0); dy = f[1:2].unsqueeze(0)
                blurred[t:t + 1] = blur.directional_blur(blurred[t:t + 1], dx, dy, samples=8)
            mask = blurred.squeeze(1)
        return (mask.clamp(0, 1),)


# ---------------- Roto Shape To Diffusion Guidance ----------------

@resilient
class RotoShapeToDiffusionGuidance:
    """Dual-output bridge: hard mask, soft inpaint mask, latent-space mask,
    and SAM-style point/box prompts as JSON.
    """
    DESCRIPTION = "Convert a roto shape into hard, soft, and latent-resolution masks plus SAM-style box+point prompts as JSON."
    CATEGORY = "NukeMax/Roto"
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("mask_hard", "mask_soft", "mask_latent", "sam_prompts_json")
    OUTPUT_TOOLTIPS = ("Hard binary roto mask at canvas resolution.", "Soft Gaussian-feathered roto mask for inpainting.", "Roto mask downsampled to latent resolution.", "JSON list of per-frame SAM prompts (boxes, points, labels).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "roto": ("ROTO_SHAPE", {"tooltip": "Roto shape to convert into diffusion guidance."}),
                "soft_radius_px": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 256.0, "tooltip": "Gaussian feather radius in pixels for the soft mask."}),
                "latent_downscale": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "Spatial downscale factor for the latent-resolution mask (e.g. 8 for SD VAE)."}),
                "samples_per_segment": ("INT", {"default": 16, "min": 2, "max": 128, "tooltip": "Bezier sampling density per segment."}),
            },
        }

    def execute(self, roto: RotoShape, soft_radius_px: float, latent_downscale: int,
                samples_per_segment: int):
        polyline = splines.shape_to_polyline(
            roto.points, roto.handles_in, roto.handles_out,
            closed=roto.closed, samples_per_segment=samples_per_segment,
        )
        hard = splines.rasterize_polygon_sdf(polyline, roto.canvas_h, roto.canvas_w, feather=0.0, closed=roto.closed)
        # Soft mask: gaussian-blurred hard mask
        h_bchw = hard.unsqueeze(1)
        soft = blur.gaussian_blur(h_bchw, soft_radius_px / 3.0).squeeze(1).clamp(0, 1)
        # Latent mask: downscale hard
        H_l = max(1, roto.canvas_h // latent_downscale)
        W_l = max(1, roto.canvas_w // latent_downscale)
        latent = torch.nn.functional.interpolate(h_bchw, size=(H_l, W_l), mode="area").squeeze(1)
        # SAM prompts: bounding boxes + centroid points per frame
        prompts = []
        for t in range(roto.T):
            pts = polyline[t]
            x0 = float(pts[:, 0].min()); x1 = float(pts[:, 0].max())
            y0 = float(pts[:, 1].min()); y1 = float(pts[:, 1].max())
            cx = float(pts[:, 0].mean()); cy = float(pts[:, 1].mean())
            prompts.append({
                "frame": t,
                "boxes": [[x0, y0, x1, y1]],
                "points": [[cx, cy]],
                "labels": [1],
            })
        return (hard.clamp(0, 1), soft, latent.clamp(0, 1), json.dumps(prompts))


NODE_CLASS_MAPPINGS = {
    "NukeMax_RotoSplineEditor": RotoSplineEditor,
    "NukeMax_RotoShapeFromFile": RotoShapeFromFile,
    "NukeMax_RotoShapeToAITracker": RotoShapeToAITracker,
    "NukeMax_RotoShapeRenderer": RotoShapeRenderer,
    "NukeMax_RotoShapeToDiffusionGuidance": RotoShapeToDiffusionGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_RotoSplineEditor": "Roto Spline Editor",
    "NukeMax_RotoShapeFromFile": "Roto Shape From File",
    "NukeMax_RotoShapeToAITracker": "Roto Shape → AI Tracker",
    "NukeMax_RotoShapeRenderer": "Roto Shape Renderer",
    "NukeMax_RotoShapeToDiffusionGuidance": "Roto Shape → Diffusion Guidance",
}
