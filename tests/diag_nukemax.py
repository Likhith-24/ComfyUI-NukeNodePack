"""
NukeMax Node Diagnostics
Runs inside the comfy Python environment with ComfyUI on sys.path.
Reports import health, registration, and per-node attribute completeness.
"""
import sys, os, traceback, json, importlib, inspect, time

COMFYUI_DIR = os.path.dirname(os.path.abspath(__file__)) + r"\ComfyUI"
NUKEMAX_DIR = r"D:\PROJECT\Custom_Nodes\ComfyUI-NukeMaxNodes"

# ── sys.path setup ──────────────────────────────────────────────────────────
if COMFYUI_DIR not in sys.path:
    sys.path.insert(0, COMFYUI_DIR)
if NUKEMAX_DIR not in sys.path:
    sys.path.insert(0, NUKEMAX_DIR)

SEP = "─" * 72
PASS = "✓"
WARN = "⚠"
FAIL = "✗"

results = {"passed": 0, "warned": 0, "failed": 0, "errors": []}

def ok(msg):
    results["passed"] += 1
    print(f"  {PASS}  {msg}")

def warn(msg):
    results["warned"] += 1
    print(f"  {WARN}  {msg}")

def fail(msg, detail=""):
    results["failed"] += 1
    entry = msg + (f": {detail}" if detail else "")
    results["errors"].append(entry)
    print(f"  {FAIL}  {entry}")

# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("NUKEMAX DIAGNOSTICS")
print(f"Python       : {sys.version}")
print(f"Executable   : {sys.executable}")
print(f"ComfyUI dir  : {COMFYUI_DIR}")
print(f"NukeMax dir  : {NUKEMAX_DIR}")
print(SEP)

# ── 1. Environment checks ───────────────────────────────────────────────────
print("\n[1] Environment")
try:
    import torch
    ok(f"torch {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    fail("torch import", str(e))

try:
    import numpy as np
    ok(f"numpy {np.__version__}")
except Exception as e:
    fail("numpy import", str(e))

try:
    import scipy
    ok(f"scipy {scipy.__version__}")
except Exception as e:
    warn(f"scipy not available: {e}")

try:
    import cv2
    ok(f"opencv-python {cv2.__version__}")
except Exception as e:
    warn(f"opencv-python not available (optional): {e}")

try:
    import soundfile
    ok(f"soundfile {soundfile.__version__}")
except Exception as e:
    warn(f"soundfile not available (optional): {e}")

# ── 2. NukeMax module imports ────────────────────────────────────────────────
print(f"\n[2] Module imports")
submodules = ["nukemax.types", "nukemax.core", "nukemax.utils",
              "nukemax.nodes.roto", "nukemax.nodes.fft",
              "nukemax.nodes.relight", "nukemax.nodes.audio",
              "nukemax.nodes.flow", "nukemax.nodes.edges",
              "nukemax.nodes.types_io"]

import_errors = []
for mod in submodules:
    try:
        importlib.import_module(mod)
        ok(mod)
    except Exception as e:
        fail(f"import {mod}", str(e))
        import_errors.append((mod, traceback.format_exc()))

if import_errors:
    print("\n  ── Full tracebacks ──")
    for mod, tb in import_errors:
        print(f"\n  {mod}:\n{tb}")

# ── 3. Node registration ─────────────────────────────────────────────────────
print(f"\n[3] Node class mappings")
from nukemax.nodes import audio, edges, fft, flow, relight, roto, types_io

all_mappings = {}
all_display = {}
for e in (audio, edges, fft, flow, relight, roto, types_io):
    all_mappings.update(e.NODE_CLASS_MAPPINGS)
    all_display.update(e.NODE_DISPLAY_NAME_MAPPINGS)

ok(f"Total nodes registered: {len(all_mappings)}")
if len(all_mappings) != 47:
    fail(f"Expected 47, got {len(all_mappings)}")

# ── 4. Per-node attribute checks ─────────────────────────────────────────────
print(f"\n[4] Per-node attribute audit")
REQUIRED_ATTRS = ["CATEGORY", "FUNCTION", "RETURN_TYPES", "INPUT_TYPES"]

node_issues = []
for key, cls in sorted(all_mappings.items()):
    issues = []
    # Required attributes
    for attr in REQUIRED_ATTRS:
        if not hasattr(cls, attr):
            issues.append(f"missing {attr}")
    # CATEGORY starts with NukeMax/
    cat = getattr(cls, "CATEGORY", "")
    if not cat.startswith("NukeMax/"):
        issues.append(f"CATEGORY '{cat}' doesn't start with 'NukeMax/'")
    # FUNCTION method exists and is callable
    fn = getattr(cls, "FUNCTION", None)
    if fn and not callable(getattr(cls, fn, None)):
        issues.append(f"FUNCTION '{fn}' not callable")
    # RETURN_TYPES is a tuple
    rt = getattr(cls, "RETURN_TYPES", None)
    if rt is not None and not isinstance(rt, tuple):
        issues.append(f"RETURN_TYPES is {type(rt).__name__}, expected tuple")
    elif rt is not None and len(rt) == 0:
        issues.append("RETURN_TYPES is empty")
    # INPUT_TYPES is a classmethod returning dict
    try:
        it = cls.INPUT_TYPES()
        if not isinstance(it, dict):
            issues.append("INPUT_TYPES() didn't return dict")
        else:
            if "required" not in it and "optional" not in it:
                issues.append("INPUT_TYPES() has neither 'required' nor 'optional' key")
    except Exception as e:
        issues.append(f"INPUT_TYPES() raised: {e}")
    # Display name registered
    if key not in all_display:
        issues.append("no display name registered")

    if issues:
        fail(f"{key}", "; ".join(issues))
        node_issues.append(key)
    else:
        ok(key)

# ── 5. Functional smoke tests ─────────────────────────────────────────────────
print(f"\n[5] Functional smoke tests (zero-tensor inputs)")
import torch

def _img(B=1, H=16, W=16, C=3):
    return torch.rand(B, H, W, C)

def _mask(B=1, H=16, W=16):
    return torch.rand(B, H, W)

from nukemax.types import FlowField, RotoShape, AudioFeatures

def _flow(B=1, H=16, W=16):
    return FlowField(flow_fwd=torch.zeros(B,2,H,W), flow_bwd=torch.zeros(B,2,H,W))

def _circle(H=32, W=32):
    import math
    N = 32; r = 8.0; cx, cy = W/2, H/2
    theta = torch.linspace(0, 2*math.pi, N+1)[:-1]
    pts = torch.stack([cx + r*torch.cos(theta), cy + r*torch.sin(theta)], -1)
    return RotoShape.from_polygon(pts, (H,W), feather=0.0, closed=True)

def _audio(T=8):
    return AudioFeatures(
        waveform=torch.zeros(T*256), sr=16000,
        stft_mag=torch.rand(257, T)+0.01,
        onsets=torch.rand(T), bpm=120.0,
        centroid=torch.rand(T), rms=torch.rand(T), hop_length=256)

smoke_tests = {
    # FFT
    "FFTAnalyze + FFTSynthesize round-trip": lambda: (
        lambda ft: fft.FFTSynthesize().execute(ft)[0].shape
    )(fft.FFTAnalyze().execute(_img(H=16,W=16))[0]),
    "FrequencyMask full pass": lambda: (
        lambda ft: fft.FrequencyMask().execute(ft, 0.0, 1.0, 0.0)[0]
    )(fft.FFTAnalyze().execute(_img(H=16,W=16))[0]),
    "FFTTextureSynthesis": lambda: fft.FFTTextureSynthesis().execute(_img(H=16,W=16), 24, 32, 42)[0].shape,
    "LatentFrequencyMatch": lambda: fft.LatentFrequencyMatch().execute(
        {"samples": torch.randn(1,4,8,8)}, _img(H=16,W=16), 8)[0]["samples"].shape,
    # Relight
    "MaterialDecomposerHeuristic": lambda: relight.MaterialDecomposerHeuristic().execute(_img(), 4.0, 0.5)[0].albedo.shape,
    "LightRigBuilder default": lambda: len(relight.LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0].lights),
    "ThreePointRelight": lambda: relight.ThreePointRelight().execute(
        relight.MaterialDecomposerHeuristic().execute(_img(), 4.0, 0.5)[0],
        relight.LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0],
        50.0, True)[0].shape,
    "LightProbeEstimator": lambda: relight.LightProbeEstimator().execute(_img(), relight.MaterialDecomposerHeuristic().execute(_img(),4.0,0.5)[0], 32, 64)[0].env_map.shape,
    # Audio
    "AudioToFloatCurve": lambda: len(audio.AudioToFloatCurve().execute(_audio(), 8, 24.0, "full", 0.5, 1.0)[0]),
    "AudioDriveMask intensity": lambda: audio.AudioDriveMask().execute(_mask(B=4), [0.5]*4, "intensity", 1.0)[0].shape,
    "AudioDriveSchedule": lambda: json.loads(audio.AudioDriveSchedule().execute([0.0,1.0], 4.0, 12.0)[0]),
    "AudioSpectrogram log": lambda: audio.AudioSpectrogram().execute(_audio(), True)[0].shape,
    "AudioSpectrogram linear": lambda: audio.AudioSpectrogram().execute(_audio(), False)[0].shape,
    # Flow
    "ComputeOpticalFlow auto": lambda: flow.ComputeOpticalFlow().execute(_img(B=2,H=16,W=16), "auto", 1.5)[0].flow_fwd.shape,
    "FlowBackwardWarp zero": lambda: flow.FlowBackwardWarp().execute(_img(B=2,H=16,W=16), _flow(B=2,H=16,W=16), "forward")[0].shape,
    "FlowForwardWarp zero": lambda: flow.FlowForwardWarp().execute(_img(B=2,H=16,W=16), _flow(B=2,H=16,W=16))[0].shape,
    "FlowOcclusionMask": lambda: flow.FlowOcclusionMask().execute(_flow(B=2,H=16,W=16))[0].shape,
    "FlowVisualize": lambda: flow.FlowVisualize().execute(_flow(B=2,H=16,W=16), 16.0)[0].shape,
    "CleanPlateMerge zero mask": lambda: flow.CleanPlateMerge().execute(_img(B=2,H=16,W=16), _img(H=16,W=16), torch.zeros(2,16,16), _flow(B=2,H=16,W=16), 0.0)[0].shape,
    # Edges
    "NormalAwareEdgeBlur": lambda: edges.NormalAwareEdgeBlur().execute(_mask(), _img(), 2.0, 0.85)[0].shape,
    "MatteDensityAdjust neutral": lambda: edges.MatteDensityAdjust().execute(_mask(), 1.0, 1.0, 0.0, 1.0)[0].shape,
    "SubPixelEdgeDetect": lambda: edges.SubPixelEdgeDetect().execute(_img(), 8)[0].shape,
    "HairAwareChoke zero": lambda: edges.HairAwareChoke().execute(_mask(), _img(), 0.0, 5)[0].shape,
    "HairAwareChoke positive": lambda: edges.HairAwareChoke().execute(_mask(), _img(), 1.5, 5)[0].shape,
    # Roto
    "RotoSplineEditor default": lambda: roto.RotoSplineEditor().execute("", 32, 32)[0],
    "RotoShapeRenderer circle": lambda: roto.RotoShapeRenderer().execute(_circle(), 16, 0.0)[0].shape,
    "RotoShapeToDiffusionGuidance": lambda: roto.RotoShapeToDiffusionGuidance().execute(_circle(H=64,W=64), 8.0, 8, 8)[0].shape,
    # IO
    "Serialize/Deserialize round-trip (ROTO_SHAPE)": lambda: (
        lambda s: types_io.NODE_CLASS_MAPPINGS["NukeMax_Deserialize_ROTO_SHAPE"]().execute(
            types_io.NODE_CLASS_MAPPINGS["NukeMax_Serialize_ROTO_SHAPE"]().execute(_circle())[0]
        )[0].canvas_h
    )(None),
}

smoke_failures = []
for name, fn in smoke_tests.items():
    try:
        result = fn()
        ok(f"{name}  → {result}")
    except Exception as e:
        fail(f"{name}", str(e))
        smoke_failures.append((name, traceback.format_exc()))

if smoke_failures:
    print("\n  ── Smoke test tracebacks ──")
    for name, tb in smoke_failures:
        print(f"\n  {name}:\n{tb}")

# ── 6. Nuke convention checks ────────────────────────────────────────────────
print(f"\n[6] Nuke output-contract checks")

def check_image(tensor, label):
    if tensor.ndim != 4:
        fail(f"{label}: shape {tuple(tensor.shape)} not BHWC (ndim={tensor.ndim})")
        return
    if tensor.dtype != torch.float32:
        fail(f"{label}: dtype {tensor.dtype} not float32")
        return
    lo, hi = float(tensor.min()), float(tensor.max())
    if lo < -1e-4 or hi > 1.0001:
        fail(f"{label}: values [{lo:.4f}, {hi:.4f}] outside [0,1]")
    else:
        ok(f"{label}: BHWC float32 [{lo:.4f}, {hi:.4f}]")

def check_mask(tensor, label):
    if tensor.ndim != 3:
        fail(f"{label}: shape {tuple(tensor.shape)} not BHW (ndim={tensor.ndim})")
        return
    if tensor.dtype != torch.float32:
        fail(f"{label}: dtype {tensor.dtype} not float32")
        return
    lo, hi = float(tensor.min()), float(tensor.max())
    if lo < -1e-4 or hi > 1.0001:
        fail(f"{label}: values [{lo:.4f}, {hi:.4f}] outside [0,1]")
    else:
        ok(f"{label}: BHW float32 [{lo:.4f}, {hi:.4f}]")

# IMAGE contract
check_image(fft.FFTSynthesize().execute(fft.FFTAnalyze().execute(_img(H=16,W=16))[0])[0], "FFTSynthesize IMAGE")
check_image(fft.FFTTextureSynthesis().execute(_img(H=16,W=16), 24, 32, 0)[0], "FFTTextureSynthesis IMAGE")
check_image(flow.FlowVisualize().execute(_flow(), 16.0)[0], "FlowVisualize IMAGE")
check_image(flow.FlowBackwardWarp().execute(_img(B=2,H=16,W=16), _flow(B=2,H=16,W=16), "forward")[0], "FlowBackwardWarp IMAGE")
check_image(flow.FlowForwardWarp().execute(_img(B=2,H=16,W=16), _flow(B=2,H=16,W=16))[0], "FlowForwardWarp IMAGE")
check_image(audio.AudioSpectrogram().execute(_audio(), True)[0], "AudioSpectrogram IMAGE")
_ms = relight.MaterialDecomposerHeuristic().execute(_img(), 4.0, 0.5)[0]
_rig = relight.LightRigBuilder().execute("", 1.0, 0.4, 0.6, 0.05)[0]
check_image(relight.ThreePointRelight().execute(_ms, _rig, 50.0, True)[0], "ThreePointRelight IMAGE")

# MASK contract
check_mask(roto.RotoShapeRenderer().execute(_circle(), 16, 0.0)[0], "RotoShapeRenderer MASK")
check_mask(flow.FlowOcclusionMask().execute(_flow(B=2))[0], "FlowOcclusionMask MASK")
check_mask(edges.NormalAwareEdgeBlur().execute(_mask(), _img(), 2.0, 0.85)[0], "NormalAwareEdgeBlur MASK")
check_mask(edges.MatteDensityAdjust().execute(_mask(), 1.0, 1.0, 0.0, 1.0)[0], "MatteDensityAdjust MASK")
check_mask(edges.SubPixelEdgeDetect().execute(_img(), 8)[0], "SubPixelEdgeDetect MASK")
check_mask(edges.HairAwareChoke().execute(_mask(), _img(), 0.0, 5)[0], "HairAwareChoke MASK")
check_mask(audio.AudioDriveMask().execute(_mask(B=4), [0.5]*4, "intensity", 1.0)[0], "AudioDriveMask MASK")

# Nuke gamma convention: out = in^(1/gamma).
# gamma=2  → 0.5^(1/2) = 0.707  (BRIGHTER than input) ✓
# gamma=0.5→ 0.5^2     = 0.25   (darker  than input)
base = torch.full((1,16,16), 0.5)
out_gamma2 = edges.MatteDensityAdjust().execute(base, 2.0, 1.0, 0.0, 1.0)[0]
out_gamma05 = edges.MatteDensityAdjust().execute(base, 0.5, 1.0, 0.0, 1.0)[0]
mean2 = float(out_gamma2.mean())
mean05 = float(out_gamma05.mean())
if mean2 > 0.5:
    ok(f"MatteDensityAdjust gamma=2 brightens (Nuke: out=in^(1/2)={mean2:.4f} > 0.5)")
else:
    fail("MatteDensityAdjust gamma=2 should brighten (Nuke out=in^(1/gamma))", f"mean={mean2:.4f}")
if mean05 < 0.5:
    ok(f"MatteDensityAdjust gamma=0.5 darkens (Nuke: out=in^2={mean05:.4f} < 0.5)")
else:
    fail("MatteDensityAdjust gamma=0.5 should darken (Nuke out=in^(1/gamma))", f"mean={mean05:.4f}")

# ── 7. Summary ───────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"SUMMARY  passed={results['passed']}  warned={results['warned']}  failed={results['failed']}")
if results["errors"]:
    print("\nFailed items:")
    for e in results["errors"]:
        print(f"  {FAIL}  {e}")
print(SEP)
sys.exit(1 if results["failed"] > 0 else 0)
