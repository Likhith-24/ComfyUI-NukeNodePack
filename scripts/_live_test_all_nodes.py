"""Submit a per-node /prompt to the live ComfyUI server and verify each
NukeMax_* node executes successfully end-to-end.

For every NukeMax_* node:
  1. Inspect its INPUT_TYPES from /object_info.
  2. Synthesize a minimal valid prompt graph by wiring upstream producers
     (EmptyImage / SolidMask / EmptyLatentImage / NukeMax_* helpers) for
     each required socket type, and using literal values for primitives.
  3. POST to /prompt, poll /history/{id} until complete, log pass/fail.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

BASE = "http://127.0.0.1:8189"
CLIENT_ID = str(uuid.uuid4())


def _http_json(method: str, path: str, body: dict | None = None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _make_wav(path: Path, sr: int = 16000, dur: float = 0.4) -> None:
    n = int(sr * dur)
    samples = bytearray()
    for i in range(n):
        v = int(8000 * (((i // 64) % 2) * 2 - 1))
        samples += struct.pack("<h", v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(bytes(samples))


def _shape_state_json() -> str:
    return json.dumps({
        "frames": [{
            "points": [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0]],
            "in":  [[20, 20], [40, 20], [40, 40], [20, 40]],
            "out": [[20, 20], [40, 20], [40, 40], [20, 40]],
            "feather": [0.0, 0.0, 0.0, 0.0],
        }],
        "closed": True,
        "canvas": {"h": 64, "w": 64},
    })


def _new_id(g: dict) -> str:
    n = len(g) + 1
    return str(n)


def add_node(g: dict, class_type: str, inputs: dict) -> str:
    nid = _new_id(g)
    g[nid] = {"class_type": class_type, "inputs": inputs}
    return nid


def producer(g: dict, type_name: str, ctx: dict) -> tuple:
    """Produce an output socket of `type_name`. Returns (node_id, output_index)."""
    cache = ctx.setdefault("cache", {})
    if type_name in cache:
        return cache[type_name]

    if type_name == "IMAGE":
        nid = add_node(g, "EmptyImage", {"width": 64, "height": 64, "batch_size": 1, "color": 0})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "MASK":
        nid = add_node(g, "SolidMask", {"value": 0.5, "width": 64, "height": 64})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "LATENT":
        nid = add_node(g, "EmptyLatentImage", {"width": 64, "height": 64, "batch_size": 1})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "ROTO_SHAPE":
        nid = add_node(g, "NukeMax_RotoSplineEditor",
                       {"shape_state": _shape_state_json(), "canvas_h": 64, "canvas_w": 64})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "FFT_TENSOR":
        img = producer(g, "IMAGE", ctx)
        nid = add_node(g, "NukeMax_FFTAnalyze", {"image": [img[0], img[1]]})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "FLOW_FIELD":
        img = producer(g, "IMAGE", ctx)
        nid = add_node(g, "NukeMax_ComputeOpticalFlow",
                       {"frames": [img[0], img[1]], "method": "torch_lk", "smoothing_sigma": 1.5})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "MATERIAL_SET":
        img = producer(g, "IMAGE", ctx)
        nid = add_node(g, "NukeMax_MaterialDecomposerHeuristic",
                       {"image": [img[0], img[1]], "albedo_blur_sigma": 4.0, "spec_strength": 0.5})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "LIGHT_RIG":
        nid = add_node(g, "NukeMax_LightRigBuilder",
                       {"rig_state": "", "key_intensity": 1.0, "fill_intensity": 0.5,
                        "back_intensity": 0.7, "ambient": 0.05})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "LIGHT_PROBE":
        img = producer(g, "IMAGE", ctx)
        nid = add_node(g, "NukeMax_LightProbeEstimator",
                       {"image": [img[0], img[1]], "env_h": 64, "env_w": 128})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "AUDIO_FEATURES":
        wav = ctx["wav_path"]
        nid = add_node(g, "NukeMax_AudioLoadAnalyze",
                       {"audio_path": wav, "sample_rate": 16000, "n_fft": 1024, "hop_length": 256})
        cache[type_name] = (nid, 0); return cache[type_name]
    if type_name == "TRACKING_DATA":
        rs = producer(g, "ROTO_SHAPE", ctx)
        img = producer(g, "IMAGE", ctx)
        nid = add_node(g, "NukeMax_RotoShapeToAITracker",
                       {"shape": [rs[0], rs[1]], "frames": [img[0], img[1]]})
        cache[type_name] = (nid, 1); return cache[type_name]  # second output is TRACKING_DATA
    raise ValueError(f"no producer for type {type_name!r}")


def _literal_for(arg_type: str, spec: dict, arg_name: str, ctx: dict):
    if arg_type == "STRING":
        # Path-like inputs need real content
        n = arg_name.lower()
        if "path" in n and "audio" in n:
            return ctx["wav_path"]
        if "path" in n and ("output" in n or "out" in n or "dir" in n):
            return str(ctx["out_dir"])
        if n in ("audio_path",):
            return ctx["wav_path"]
        if "filename" in n:
            return f"probe_{ctx['stamp']}.exr"
        if n == "shape_state":
            return _shape_state_json()
        return spec.get("default", "")
    if arg_type == "INT":
        return int(spec.get("default", 1))
    if arg_type == "FLOAT":
        return float(spec.get("default", 0.0))
    if arg_type == "BOOLEAN":
        return bool(spec.get("default", False))
    return spec.get("default", "")


def build_prompt_for_node(node_name: str, info: dict, ctx: dict) -> dict:
    g: dict = {}
    inputs_spec = info.get("input", {})
    required = inputs_spec.get("required", {})
    inputs: dict = {}
    for arg_name, arg_spec in required.items():
        # arg_spec: [type] or [type, opts] or [list_of_options] or [list_of_options, opts]
        arg_type = arg_spec[0]
        opts = arg_spec[1] if len(arg_spec) > 1 and isinstance(arg_spec[1], dict) else {}
        if isinstance(arg_type, list):
            # Combo: pick first option (or a sane one)
            inputs[arg_name] = arg_type[0]
            continue
        upstream_types = {
            "IMAGE", "MASK", "LATENT", "ROTO_SHAPE", "FFT_TENSOR", "FLOW_FIELD",
            "MATERIAL_SET", "LIGHT_RIG", "LIGHT_PROBE", "AUDIO_FEATURES", "TRACKING_DATA",
        }
        if arg_type in upstream_types:
            src = producer(g, arg_type, ctx)
            inputs[arg_name] = [src[0], src[1]]
        elif arg_type == "FLOAT" and opts.get("forceInput"):
            # AudioToFloatCurve etc. — skip; drive_mask needs FLOAT array
            # No native producer for raw FLOAT; chain through AudioToFloatCurve.
            af = producer(g, "AUDIO_FEATURES", ctx)
            cnode = add_node(g, "NukeMax_AudioToFloatCurve", {
                "audio": [af[0], af[1]],
                "frame_count": 4, "fps": 24.0, "feature": "rms",
                "smoothing": 0.0, "scale": 1.0,
            })
            inputs[arg_name] = [cnode, 0]
        else:
            inputs[arg_name] = _literal_for(arg_type, opts, arg_name, ctx)
    add_node(g, node_name, inputs)
    return g


def submit_and_wait(prompt: dict, timeout_s: float = 60.0) -> tuple[bool, str]:
    body = {"prompt": prompt, "client_id": CLIENT_ID}
    try:
        resp = _http_json("POST", "/prompt", body, timeout=10.0)
    except urllib.error.HTTPError as e:
        try:
            err = json.load(e)
        except Exception:  # noqa: BLE001
            err = e.read().decode("utf-8", "ignore")
        return False, f"HTTP {e.code}: {err}"
    except Exception as e:  # noqa: BLE001
        return False, f"submit err: {e}"
    pid = resp.get("prompt_id")
    if not pid:
        return False, f"no prompt_id: {resp}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            hist = _http_json("GET", f"/history/{pid}", timeout=5.0)
        except Exception as e:  # noqa: BLE001
            time.sleep(0.5); continue
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "success":
                return True, "ok"
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                return False, f"exec error: {msgs[-2:] if msgs else status}"
            # Some versions report completion without status_str: presence in
            # history with "outputs" populated indicates done
            if entry.get("outputs"):
                return True, "ok (outputs present)"
        time.sleep(0.4)
    return False, f"timeout after {timeout_s}s"


def main() -> int:
    info = _http_json("GET", "/object_info", timeout=5.0)
    nodes = sorted(k for k in info if k.startswith("NukeMax_"))
    print(f"Discovered {len(nodes)} NukeMax_* nodes; running per-node live tests")
    out_dir = Path(tempfile.gettempdir()) / "nukemax_live_probe"
    out_dir.mkdir(exist_ok=True, parents=True)
    wav_path = out_dir / "probe.wav"
    _make_wav(wav_path)
    ctx = {
        "wav_path": str(wav_path),
        "out_dir": out_dir,
        "stamp": str(int(time.time())),
    }
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    for name in nodes:
        try:
            prompt = build_prompt_for_node(name, info[name], ctx)
        except Exception as e:  # noqa: BLE001
            failed.append((name, f"build err: {e}")); print(f"  FAIL {name}: build err: {e}"); continue
        ok, msg = submit_and_wait(prompt, timeout_s=90.0)
        if ok:
            passed.append(name); print(f"  PASS {name}")
        else:
            failed.append((name, msg)); print(f"  FAIL {name}: {msg}")
    print("\n=== summary ===")
    print(f"PASS: {len(passed)}/{len(nodes)}")
    print(f"FAIL: {len(failed)}")
    for name, msg in failed:
        print(f"  - {name}: {msg}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
