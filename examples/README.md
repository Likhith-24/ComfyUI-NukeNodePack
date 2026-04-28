# NukeMax Example Workflows

These are minimal **API-format** ComfyUI workflows demonstrating one
typical wiring per ecosystem. Drop them into your client and update the
`LoadImage` / `path` fields to point at your own assets.

| File | Demonstrates |
|---|---|
| `01_roto_basic.json` | Roto spline → mask render → hair-aware choke |
| `02_relight_three_point.json` | Heuristic decomposer → 3-point rig → relit image |
| `03_audio_reactive.json` | Audio analyze → bass curve → CFG schedule + spectrogram preview |
| `04_flow_warp.json` | Optical flow → backward warp + flow visualization |

## Verifying via API

With ComfyUI running on `127.0.0.1:8188`, you can POST any of these to
`/prompt` directly:

```bash
curl -X POST http://127.0.0.1:8188/prompt \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": $(cat examples/01_roto_basic.json)}"
```

## Loading in the GUI

The `.json` files here are in the *API* format used by the `/prompt`
endpoint. To open one in the ComfyUI editor, use a node like
*Workflow → Load (API Format)* in your client, or wrap it manually
inside a workflow envelope. Most users will find it simpler to
re-create the graph from scratch using these JSONs as a reference.
