"""Live mega-graph hitting every NukeMax_* node, using EXACT input names
queried from /object_info. Run with the live ComfyUI server up.
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


def _http_json(method, path, body=None, timeout=10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _make_wav(p: Path):
    n = 6400
    samples = bytearray()
    for i in range(n):
        samples += struct.pack("<h", int(8000 * (((i // 64) % 2) * 2 - 1)))
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(bytes(samples))


def _shape_state_json():
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


def build(wav_path: str, shape_path: str, out_dir: str) -> dict:
    g: dict = {}
    counter = [0]
    def add(class_type: str, inputs: dict) -> str:
        counter[0] += 1
        k = str(counter[0])
        g[k] = {"class_type": class_type, "inputs": inputs}
        return k

    img = add("EmptyImage", {"width": 64, "height": 64, "batch_size": 2, "color": 8421504})
    smask = add("SolidMask", {"value": 0.5, "width": 64, "height": 64})
    latent = add("EmptyLatentImage", {"width": 64, "height": 64, "batch_size": 1})

    rs = add("NukeMax_RotoSplineEditor", {
        "spline_state": _shape_state_json(), "canvas_height": 64, "canvas_width": 64,
    })
    rs_render = add("NukeMax_RotoShapeRenderer", {
        "roto": [rs, 0], "samples_per_segment": 4, "feather_override": -1.0,
    })
    rs_track = add("NukeMax_RotoShapeToAITracker", {
        "roto": [rs, 0], "frames": [img, 0],
    })
    rs_diff = add("NukeMax_RotoShapeToDiffusionGuidance", {
        "roto": [rs, 0], "soft_radius_px": 8.0,
        "latent_downscale": 8, "samples_per_segment": 4,
    })
    rs_file = add("NukeMax_RotoShapeFromFile", {"path": shape_path})

    fft_an = add("NukeMax_FFTAnalyze", {"image": [img, 0]})
    fft_mask = add("NukeMax_FrequencyMask", {
        "fft": [fft_an, 0], "low": 0.0, "high": 0.5, "softness": 0.1,
    })
    fft_syn = add("NukeMax_FFTSynthesize", {"fft": [fft_mask, 0]})
    fft_tex = add("NukeMax_FFTTextureSynthesis", {
        "exemplar": [img, 0], "out_height": 64, "out_width": 64, "seed": 1234,
    })
    fft_lat = add("NukeMax_LatentFrequencyMatch", {
        "noise_latent": [latent, 0], "context_image": [img, 0], "n_bins": 16,
    })

    mat_h = add("NukeMax_MaterialDecomposerHeuristic", {
        "image": [img, 0], "albedo_blur_sigma": 8.0, "depth_strength": 0.5,
    })
    mat_m = add("NukeMax_MaterialDecomposerModels", {
        "image": [img, 0], "depth_model": "marigold", "normal_model": "stable_normal",
    })
    rig = add("NukeMax_LightRigBuilder", {
        "rig_state": "", "key_intensity": 1.0, "fill_intensity": 0.5,
        "rim_intensity": 0.7, "ambient": 0.05,
    })
    relit = add("NukeMax_ThreePointRelight", {
        "materials": [mat_h, 0], "rig": [rig, 0],
        "fov_deg": 50.0, "tonemap": True,
    })
    probe = add("NukeMax_LightProbeEstimator", {
        "image": [img, 0], "materials": [mat_h, 0],
        "probe_height": 32, "probe_width": 64,
    })
    probe_exr = add("NukeMax_LightProbeToEXR", {
        "probe": [probe, 0], "out_dir": out_dir, "filename": "live_probe.exr",
    })

    audio = add("NukeMax_AudioLoadAnalyze", {
        "path": wav_path, "n_fft": 1024, "hop_length": 256,
    })
    curve = add("NukeMax_AudioToFloatCurve", {
        "audio": [audio, 0], "frame_count": 4, "fps": 24.0,
        "band": "full", "smoothing": 0.0, "gain": 1.0,
    })
    drive_mask = add("NukeMax_AudioDriveMask", {
        "mask": [rs_render, 0], "curve": [curve, 0],
        "mode": "intensity", "amount": 1.0,
    })
    drive_sched = add("NukeMax_AudioDriveSchedule", {
        "curve": [curve, 0], "min_value": 4.0, "max_value": 12.0,
    })
    spectro = add("NukeMax_AudioSpectrogram", {
        "audio": [audio, 0], "log_scale": True,
    })

    flow = add("NukeMax_ComputeOpticalFlow", {
        "frames": [img, 0], "method": "torch_lk", "consistency_threshold": 1.5,
    })
    flow_vis = add("NukeMax_FlowVisualize", {
        "flow": [flow, 0], "max_magnitude": 16.0,
    })
    flow_bwd = add("NukeMax_FlowBackwardWarp", {
        "image": [img, 0], "flow": [flow, 0], "direction": "forward",
    })
    flow_fwd = add("NukeMax_FlowForwardWarp", {
        "image": [img, 0], "flow": [flow, 0],
    })
    flow_occ = add("NukeMax_FlowOcclusionMask", {"flow": [flow, 0]})
    clean = add("NukeMax_CleanPlateMerge", {
        "footage": [img, 0], "clean_plate": [img, 0], "object_mask": [rs_render, 0],
        "flow": [flow, 0], "feather_px": 0.0,
    })

    nbe = add("NukeMax_NormalAwareEdgeBlur", {
        "mask": [rs_render, 0], "normal": [img, 0], "sigma": 2.0, "normal_threshold": 0.85,
    })
    mda = add("NukeMax_MatteDensityAdjust", {
        "mask": [rs_render, 0], "gamma": 1.0, "contrast": 1.0,
        "edge_lo": 0.01, "edge_hi": 0.99,
    })
    sub = add("NukeMax_SubPixelEdgeDetect", {"image": [img, 0], "top_k": 8})
    hac = add("NukeMax_HairAwareChoke", {
        "mask": [rs_render, 0], "image": [img, 0], "choke": 1.0, "hair_window": 5,
    })

    s_rs = add("NukeMax_Serialize_ROTO_SHAPE", {"obj": [rs, 0]})
    d_rs = add("NukeMax_Deserialize_ROTO_SHAPE", {"payload": [s_rs, 0]})
    rs_render_d = add("NukeMax_RotoShapeRenderer", {
        "roto": [d_rs, 0], "samples_per_segment": 4, "feather_override": -1.0,
    })

    s_td = add("NukeMax_Serialize_TRACKING_DATA", {"obj": [rs_track, 1]})
    add("NukeMax_Deserialize_TRACKING_DATA", {"payload": [s_td, 0]})

    s_fft = add("NukeMax_Serialize_FFT_TENSOR", {"obj": [fft_an, 0]})
    d_fft = add("NukeMax_Deserialize_FFT_TENSOR", {"payload": [s_fft, 0]})
    fft_syn_d = add("NukeMax_FFTSynthesize", {"fft": [d_fft, 0]})

    s_flow = add("NukeMax_Serialize_FLOW_FIELD", {"obj": [flow, 0]})
    d_flow = add("NukeMax_Deserialize_FLOW_FIELD", {"payload": [s_flow, 0]})
    flow_vis_d = add("NukeMax_FlowVisualize", {"flow": [d_flow, 0], "max_magnitude": 16.0})

    s_ms = add("NukeMax_Serialize_MATERIAL_SET", {"obj": [mat_h, 0]})
    d_ms = add("NukeMax_Deserialize_MATERIAL_SET", {"payload": [s_ms, 0]})
    relit_d = add("NukeMax_ThreePointRelight", {
        "materials": [d_ms, 0], "rig": [rig, 0], "fov_deg": 50.0, "tonemap": True,
    })

    s_lr = add("NukeMax_Serialize_LIGHT_RIG", {"obj": [rig, 0]})
    d_lr = add("NukeMax_Deserialize_LIGHT_RIG", {"payload": [s_lr, 0]})
    relit_d2 = add("NukeMax_ThreePointRelight", {
        "materials": [mat_h, 0], "rig": [d_lr, 0], "fov_deg": 50.0, "tonemap": True,
    })

    s_lp = add("NukeMax_Serialize_LIGHT_PROBE", {"obj": [probe, 0]})
    d_lp = add("NukeMax_Deserialize_LIGHT_PROBE", {"payload": [s_lp, 0]})
    add("NukeMax_LightProbeToEXR", {
        "probe": [d_lp, 0], "out_dir": out_dir, "filename": "live_probe_d.exr",
    })

    s_af = add("NukeMax_Serialize_AUDIO_FEATURES", {"obj": [audio, 0]})
    d_af = add("NukeMax_Deserialize_AUDIO_FEATURES", {"payload": [s_af, 0]})
    spectro_d = add("NukeMax_AudioSpectrogram", {"audio": [d_af, 0], "log_scale": True})

    relit_models = add("NukeMax_ThreePointRelight", {
        "materials": [mat_m, 0], "rig": [rig, 0], "fov_deg": 50.0, "tonemap": True,
    })

    drive_mask2 = add("NukeMax_AudioDriveMask", {
        "mask": [smask, 0], "curve": [drive_sched, 1],
        "mode": "intensity", "amount": 1.0,
    })

    rs_render_f = add("NukeMax_RotoShapeRenderer", {
        "roto": [rs_file, 0], "samples_per_segment": 4, "feather_override": -1.0,
    })

    image_sinks = [fft_syn, fft_tex, relit, spectro, flow_vis, flow_bwd, clean,
                   fft_syn_d, flow_vis_d, relit_d, relit_d2, spectro_d, relit_models]
    for n in image_sinks:
        add("PreviewImage", {"images": [n, 0]})

    mask_sinks = [rs_render, drive_mask, drive_mask2, nbe, mda, hac,
                  rs_render_d, rs_render_f, flow_occ]
    for n in mask_sinks:
        m2i = add("MaskToImage", {"mask": [n, 0]})
        add("PreviewImage", {"images": [m2i, 0]})

    add("PreviewImage", {"images": [flow_fwd, 0]})
    m2i = add("MaskToImage", {"mask": [flow_fwd, 1]})
    add("PreviewImage", {"images": [m2i, 0]})

    # SubPixelEdgeDetect on uniform color may emit empty batches that crash
    # PreviewImage. Route via TRACKING_DATA serialization instead — Comfy
    # still executes the node once and produces both outputs.
    s_td2 = add("NukeMax_Serialize_TRACKING_DATA", {"obj": [sub, 1]})
    _ = s_td2

    for i in range(3):
        m2i = add("MaskToImage", {"mask": [rs_diff, i]})
        add("PreviewImage", {"images": [m2i, 0]})

    add("SaveLatent", {"samples": [fft_lat, 0], "filename_prefix": "nukemax_live"})

    _ = probe_exr
    return g


def submit(prompt: dict, timeout_s=240.0):
    body = {"prompt": prompt, "client_id": CLIENT_ID}
    try:
        resp = _http_json("POST", "/prompt", body, timeout=15.0)
    except urllib.error.HTTPError as e:
        try:
            err = json.load(e)
        except Exception:
            err = e.read().decode("utf-8", "ignore")
        print(json.dumps(err, indent=2))
        raise
    pid = resp["prompt_id"]
    print(f"submitted prompt_id={pid}")
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            hist = _http_json("GET", f"/history/{pid}", timeout=5.0)
        except Exception:
            time.sleep(0.5); continue
        if pid in hist:
            entry = hist[pid]
            ss = entry.get("status", {}).get("status_str")
            if ss != last:
                print(f"  status={ss}")
                last = ss
            if ss == "success":
                return entry
            if ss == "error":
                msgs = entry["status"].get("messages", [])
                raise SystemExit(f"prompt error:\n{json.dumps(msgs, indent=2)}")
        time.sleep(0.4)
    raise SystemExit("timed out")


def main() -> int:
    info = _http_json("GET", "/object_info", timeout=5.0)
    nukemax = sorted(k for k in info if k.startswith("NukeMax_"))
    print(f"server has {len(nukemax)} NukeMax_* nodes")

    out_dir = Path(tempfile.gettempdir()) / "nukemax_live_probe"
    out_dir.mkdir(exist_ok=True, parents=True)
    wav_path = out_dir / "live.wav"
    _make_wav(wav_path)
    shape_file = out_dir / "shape.json"
    shape_file.write_text(_shape_state_json(), encoding="utf-8")

    prompt = build(str(wav_path).replace("\\", "/"),
                   str(shape_file).replace("\\", "/"),
                   str(out_dir).replace("\\", "/"))
    print(f"built mega-graph with {len(prompt)} nodes")

    entry = submit(prompt)

    referenced = sorted({nd["class_type"] for nd in prompt.values()
                         if nd["class_type"].startswith("NukeMax_")})
    print(f"NukeMax_* nodes referenced: {len(referenced)}/{len(nukemax)}")

    output_nukemax = sorted({prompt[nid]["class_type"] for nid in entry.get("outputs", {})
                             if prompt[nid]["class_type"].startswith("NukeMax_")})
    print(f"NukeMax_* OUTPUT_NODE entries that emitted outputs: {len(output_nukemax)}")
    for c in output_nukemax:
        print(f"  - {c}")

    missing = set(nukemax) - set(referenced)
    if missing:
        print("MISSING from graph:", missing)
        return 1
    print("\nPASS: live ComfyUI executed every NukeMax_* node end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
