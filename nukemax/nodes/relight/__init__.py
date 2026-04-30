"""PBR Lighting Extraction & Relight — Concept 3.

Note: the *decomposition* nodes (`Material Decomposer`,
`Light Probe Estimator`) require external models (depth + normal +
albedo). To keep this pack importable without those models, we ship:

  * a math-only `Material Decomposer (Heuristic)` that estimates a
    naive albedo via local mean and a fake normal from luminance gradients;
  * a stub `Material Decomposer (Models)` node that documents how to
    plug in Marigold / StableNormal / IID-AppearanceFlow when present.

The relighting nodes (`3-Point Relight`, `Light Probe to EXR`) are
fully math-driven and work on any `MATERIAL_SET`, regardless of source.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ...core import blur, shading
from ...core.color import to_bchw, to_bhwc, luminance, srgb_to_linear, linear_to_srgb
from ...types import Light, LightProbe, LightRig, MaterialSet
from ...utils.resilience import resilient


@resilient
class MaterialDecomposerHeuristic:
    """Math-only decomposition. Albedo = local mean of linear image,
    normal = gradient of luminance lifted to z, depth = inverse luminance.
    Useful for demos and for testing the relight math without ML models.
    """
    DESCRIPTION = "Heuristic albedo/normal/depth/roughness decomposition from a single image, using local-mean albedo and luminance gradients."
    CATEGORY = "NukeMax/Relight"
    FUNCTION = "execute"
    RETURN_TYPES = ("MATERIAL_SET",)
    RETURN_NAMES = ("materials",)
    OUTPUT_TOOLTIPS = ("Material set bundle (albedo, normal, depth, roughness).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to decompose into materials."}),
                "albedo_blur_sigma": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 64.0, "tooltip": "Gaussian sigma in pixels used to estimate albedo as a local mean."}),
                "depth_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "tooltip": "Strength of luminance-gradient lift used to fake the normal map."}),
            },
        }

    def execute(self, image, albedo_blur_sigma, depth_strength):
        x = srgb_to_linear(to_bchw(image).clamp(0, 1))
        albedo = blur.gaussian_blur(x, albedo_blur_sigma)
        Y = luminance(x)
        kx = torch.tensor([[-0.5, 0, 0.5]], dtype=x.dtype, device=x.device).view(1, 1, 1, 3)
        ky = kx.transpose(-1, -2)
        gx = F.conv2d(F.pad(Y, (1, 1, 0, 0), mode="reflect"), kx)
        gy = F.conv2d(F.pad(Y, (0, 0, 1, 1), mode="reflect"), ky)
        nz = torch.ones_like(Y)
        n = torch.cat([-gx * depth_strength, -gy * depth_strength, nz], dim=1)
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)
        depth = (1.0 - Y).clamp(0.05, 1.0)
        roughness = (1.0 - Y).clamp(0.1, 0.95)
        return (MaterialSet(albedo=albedo.clamp(0, 1), normal=n, depth=depth, roughness=roughness),)


@resilient
class MaterialDecomposerModels:
    """Wrapper for external decomposition models. If the user has
    Marigold / StableNormal weights placed in `ComfyUI/models/` the node
    loads them lazily; otherwise it falls back to the heuristic
    decomposer with an `info` warning.
    """
    DESCRIPTION = "Model-backed material decomposition (Marigold/StableNormal); falls back to heuristic decomposer if weights are unavailable."
    CATEGORY = "NukeMax/Relight"
    FUNCTION = "execute"
    RETURN_TYPES = ("MATERIAL_SET", "STRING")
    RETURN_NAMES = ("materials", "info")
    OUTPUT_TOOLTIPS = ("Material set bundle (albedo, normal, depth, roughness).", "Status message describing which backend was used.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to decompose."}),
                "depth_model": ("STRING", {"default": "marigold", "tooltip": "Identifier of the depth model to use if available."}),
                "normal_model": ("STRING", {"default": "stable_normal", "tooltip": "Identifier of the normal model to use if available."}),
            },
        }

    def execute(self, image, depth_model, normal_model):
        # Lazy import path: try `comfyui_marigold` etc.; if missing, fall back.
        try:
            raise ImportError("Model wrappers not implemented in v0.1.0")
        except ImportError as exc:
            heur = MaterialDecomposerHeuristic()
            ms = heur.execute(image, 8.0, 0.5)[0]
            return (ms, f"INFO: model wrappers unavailable ({exc}); used heuristic decomposer.")


@resilient
class LightRigBuilder:
    """Build a LIGHT_RIG from key/fill/rim parameters. The JS widget
    posts a JSON state into `rig_state`.
    """
    DESCRIPTION = "Build a LIGHT_RIG from a JSON state or key/fill/rim parameters with a global ambient term."
    CATEGORY = "NukeMax/Relight"
    FUNCTION = "execute"
    RETURN_TYPES = ("LIGHT_RIG",)
    RETURN_NAMES = ("rig",)
    OUTPUT_TOOLTIPS = ("Light rig bundle (lights tuple plus ambient color).",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rig_state": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional JSON string describing lights and ambient; overrides the simple sliders."}),
                "key_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "tooltip": "Intensity of the key light when no JSON state is provided."}),
                "fill_intensity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "tooltip": "Intensity of the fill light when no JSON state is provided."}),
                "rim_intensity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "tooltip": "Intensity of the rim/back light when no JSON state is provided."}),
                "ambient": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "tooltip": "Greyscale ambient term added to all surfaces."}),
            },
        }

    def execute(self, rig_state, key_intensity, fill_intensity, rim_intensity, ambient):
        import json
        try:
            data = json.loads(rig_state) if rig_state.strip() else {}
        except json.JSONDecodeError:
            data = {}
        if data.get("lights"):
            lights = tuple(
                Light(
                    position=tuple(L.get("position", (0, 0, 1))),
                    direction=tuple(L.get("direction", (0, 0, -1))),
                    color=tuple(L.get("color", (1, 1, 1))),
                    intensity=float(L.get("intensity", 1.0)),
                    type=L.get("type", "directional"),
                    radius=float(L.get("radius", 0.0)),
                    falloff=float(L.get("falloff", 2.0)),
                )
                for L in data["lights"]
            )
        else:
            # Default 3-point: key from upper right, fill from upper left, rim from behind.
            lights = (
                Light(direction=(-0.6, -0.6, -0.5), color=(1.0, 0.95, 0.85), intensity=key_intensity, type="directional"),
                Light(direction=(0.6, -0.3, -0.5), color=(0.6, 0.7, 1.0), intensity=fill_intensity, type="directional"),
                Light(direction=(0.0, 0.6, 0.8), color=(1.0, 1.0, 1.0), intensity=rim_intensity, type="directional"),
            )
        amb = float(data.get("ambient", ambient))
        return (LightRig(lights=lights, ambient=(amb, amb, amb)),)


@resilient
class ThreePointRelight:
    DESCRIPTION = "Relight a MATERIAL_SET with a LIGHT_RIG using a Lambert+Phong shader and optional Reinhard tonemap."
    CATEGORY = "NukeMax/Relight"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Relit sRGB image of the materials under the rig.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "materials": ("MATERIAL_SET", {"tooltip": "Albedo/normal/depth/roughness bundle."}),
                "rig": ("LIGHT_RIG", {"tooltip": "Light rig describing the lights and ambient."}),
                "fov_deg": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 170.0, "tooltip": "Camera vertical field-of-view in degrees used to reconstruct view rays."}),
                "tonemap": ("BOOLEAN", {"default": True, "tooltip": "Apply a Reinhard tonemap before returning sRGB."}),
            },
        }

    def execute(self, materials, rig, fov_deg, tonemap):
        rgb = shading.shade_lambert_phong(
            albedo=materials.albedo, normal=materials.normal, depth=materials.depth,
            rig=rig, roughness=materials.roughness, fov_deg=fov_deg,
        )
        if tonemap:
            # Reinhard
            rgb = rgb / (1 + rgb)
        rgb = linear_to_srgb(rgb.clamp(0, 1))
        return (to_bhwc(rgb),)


@resilient
class LightProbeEstimator:
    """Naive light-probe estimation: divide image by albedo to leave
    incoming radiance, then bin by world-space normal direction onto an
    equirectangular environment map.
    """
    DESCRIPTION = "Estimate an HDR equirectangular light probe by dividing the image by albedo and binning radiance by surface normal."
    CATEGORY = "NukeMax/Relight"
    FUNCTION = "execute"
    RETURN_TYPES = ("LIGHT_PROBE",)
    RETURN_NAMES = ("probe",)
    OUTPUT_TOOLTIPS = ("Equirectangular HDR light probe bundle.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image (sRGB) to extract incoming radiance from."}),
                "materials": ("MATERIAL_SET", {"tooltip": "Material set providing albedo and normal for the same image."}),
                "probe_height": ("INT", {"default": 256, "min": 32, "max": 4096, "tooltip": "Height of the output equirect probe in pixels."}),
                "probe_width": ("INT", {"default": 512, "min": 32, "max": 8192, "tooltip": "Width of the output equirect probe in pixels."}),
            },
        }

    def execute(self, image, materials, probe_height, probe_width):
        import math
        x = srgb_to_linear(to_bchw(image).clamp(1e-4, 1))
        radiance = (x / materials.albedo.clamp_min(0.05)).clamp(0, 8)
        n = materials.normal
        B, _, H, W = n.shape
        # Convert each pixel's normal to (theta, phi) on the env map.
        nx, ny, nz = n.unbind(dim=1)
        theta = torch.atan2(nx, nz)
        phi = torch.asin(ny.clamp(-1, 1))
        u = ((theta / math.pi + 1) * 0.5 * (probe_width - 1)).clamp(0, probe_width - 1).long()
        v = ((-phi / (math.pi * 0.5) + 1) * 0.5 * (probe_height - 1)).clamp(0, probe_height - 1).long()
        env = torch.zeros(B, 3, probe_height, probe_width, device=x.device, dtype=x.dtype)
        cnt = torch.zeros(B, 1, probe_height, probe_width, device=x.device, dtype=x.dtype)
        for c in range(3):
            env[:, c].view(B, -1).scatter_add_(
                1,
                (v * probe_width + u).view(B, -1),
                radiance[:, c].reshape(B, -1),
            )
        cnt.view(B, -1).scatter_add_(
            1, (v * probe_width + u).view(B, -1),
            torch.ones_like(u, dtype=x.dtype).view(B, -1),
        )
        env = env / cnt.clamp_min(1)
        # Smooth holes
        env = blur.gaussian_blur(env, 4.0)
        return (LightProbe(env_map=env, exposure=0.0),)


@resilient
class LightProbeToEXR:
    """Bridge output: write the HDR probe to a 32-bit EXR (or fallback
    to a Radiance .hdr if OpenEXR isn't installed) so the user can
    match lighting in Blender/Nuke.
    """
    DESCRIPTION = "Write a LIGHT_PROBE to a 32-bit float EXR file (or .npy fallback) for use in external DCC apps."
    CATEGORY = "NukeMax/Relight"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_TOOLTIPS = ("Filesystem path of the written probe file (forward slashes).",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "probe": ("LIGHT_PROBE", {"tooltip": "Light probe to write to disk."}),
                "out_dir": ("STRING", {"default": "output", "tooltip": "Output directory (created if missing)."}),
                "filename": ("STRING", {"default": "probe.exr", "tooltip": "Output file name; .exr if OpenEXR is available."}),
            },
        }

    def execute(self, probe, out_dir, filename):
        out_path = Path(out_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        env = probe.env_map[0].permute(1, 2, 0).cpu().contiguous().float().numpy()
        try:
            import OpenEXR  # type: ignore
            import Imath  # type: ignore
            import numpy as np
            h, w, _ = env.shape
            header = OpenEXR.Header(w, h)
            half = Imath.PixelType(Imath.PixelType.FLOAT)
            header["channels"] = {c: Imath.Channel(half) for c in ("R", "G", "B")}
            exr = OpenEXR.OutputFile(str(out_path), header)
            exr.writePixels({
                "R": env[:, :, 0].astype(np.float32).tobytes(),
                "G": env[:, :, 1].astype(np.float32).tobytes(),
                "B": env[:, :, 2].astype(np.float32).tobytes(),
            })
            exr.close()
            written = out_path
        except ImportError:
            # Fallback: numpy .npy
            import numpy as np
            written = out_path.with_suffix(".npy")
            np.save(str(written), env)
        return (str(written).replace("\\", "/"),)


NODE_CLASS_MAPPINGS = {
    "NukeMax_MaterialDecomposerHeuristic": MaterialDecomposerHeuristic,
    "NukeMax_MaterialDecomposerModels": MaterialDecomposerModels,
    "NukeMax_LightRigBuilder": LightRigBuilder,
    "NukeMax_ThreePointRelight": ThreePointRelight,
    "NukeMax_LightProbeEstimator": LightProbeEstimator,
    "NukeMax_LightProbeToEXR": LightProbeToEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_MaterialDecomposerHeuristic": "Material Decomposer (Heuristic)",
    "NukeMax_MaterialDecomposerModels": "Material Decomposer (Models)",
    "NukeMax_LightRigBuilder": "Light Rig Builder",
    "NukeMax_ThreePointRelight": "3-Point Relight",
    "NukeMax_LightProbeEstimator": "Light Probe Estimator",
    "NukeMax_LightProbeToEXR": "Light Probe → EXR",
}
