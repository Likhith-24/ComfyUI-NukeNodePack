# ComfyUI-NukeMaxNodes

Hybrid VFX/AI bridge nodes for ComfyUI. Six ecosystems on a shared
foundation of strict custom data types, vectorized PyTorch math, a
temporal-batch-first execution model, and a JS web extension layer.

Every traditional VFX node ships a **Dual-Output Bridge**: classical
outputs (`MASK`/`IMAGE`) plus AI-consumable outputs (SAM prompts, latent
guidance, conditioning curves).

## Ecosystems

1. **Smart Roto** — `ROTO_SHAPE` type, interactive bezier editor, AI
   tracker, diffusion guidance bridge.
2. **Frequency-Guided Generation** — FFT analyze/synthesize, latent
   frequency matching for seamless inpaint.
3. **PBR Relight** — material decomposition, light probe estimation,
   3-point relight, EXR bridge to Blender/Nuke.
4. **Audio-Reactive** — native audio tensor processing, per-frame
   curves, mask/CFG modulation.
5. **Sub-Pixel Flow Compositing** — forward/backward warp, occlusion,
   clean-plate merge.
6. **Smart Edge Tools** — normal-aware blur, matte density, sub-pixel
   edge detect, hair-aware choke.

## Install

```sh
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-NukeMaxNodes
pip install -r ComfyUI-NukeMaxNodes/requirements.txt  # optional extras
```

Restart ComfyUI. Nodes appear under `NukeMax/*`.

## Status

Phase 0 foundation: shipping. Phases 1–6 in progress. See
`nukemax/nodes/<ecosystem>/README.md` for per-ecosystem docs.
