# ComfyUI-NukeMaxNodes

Hybrid VFX × AI bridge nodes for ComfyUI. ~50 nodes across 13 file-grouped categories — every traditional VFX operation also exposes an AI-consumable side output (SAM prompts, latent guidance, conditioning curves, EXR metadata) so you can bridge a Nuke/Blender-style compositing flow into a Flux / Qwen-Image / Wan / Z-Image graph without round-tripping to disk.

Foundation: strict custom datatypes (`ROTO_SHAPE`, `FFT_DATA`, `MATERIAL_PBR`, `LIGHT_PROBE`, `AUDIO_TENSOR`, `FLOW_VECTOR`, `EDGE_MAP`, `MEC_RENDER_PASSES`, …), vectorized PyTorch math (BCHW with batch dim = time), per-batch temporal execution, and an optional JS web-extension layer (`web/`) for interactive editors.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> ComfyUI-NukeMaxNodes
pip install -r ComfyUI-NukeMaxNodes/requirements.txt
```

Embedded-Python (ComfyUI portable):

```bash
..\..\python_embeded\python.exe -m pip install -r ComfyUI-NukeMaxNodes\requirements.txt
```

`requirements.txt`: `torch>=2.0`, `numpy>=1.24`, `librosa>=0.10`, `soundfile>=0.12`, `imageio>=2.31` (and optionally `OpenEXR` for native EXR I/O — `imageio` is the cross-platform fallback).

`WEB_DIRECTORY` is set to `./web`, so JS extensions for the interactive bezier editor and curve viewer load automatically. Restart ComfyUI; nodes appear under `NukeMax/<category>`.

## Node groups

Organised by source file under `nukemax/nodes/`. Counts are exact; class names listed for traceability.

### Smart Roto (`roto/`) — 5 nodes

`NukeMax_RotoShapeCreate`, `NukeMax_RotoShapeEdit`, `NukeMax_RotoShapeRender`, `NukeMax_RotoTrack`, `NukeMax_RotoToDiffusionGuidance`.

- **Custom type:** `ROTO_SHAPE` (per-frame bezier control points + tangents).
- **Inputs (typical):** initial keyframe shape (JSON), source IMAGE batch, optional tracker model.
- **Outputs:** `ROTO_SHAPE`, `MASK` (rasterised), and on `RotoToDiffusionGuidance` a CONDITIONING-style guidance bundle.
- **Use case:** classical bezier roto with sub-pixel rasterisation; the `RotoToDiffusionGuidance` bridge converts shapes into spatial conditioning consumable by ControlNet / inpaint nodes downstream.

### Frequency-Guided Generation (`fft/`) — 5 nodes

`NukeMax_FFTAnalyze`, `NukeMax_FFTSynthesize`, `NukeMax_FFTSpectrumViewer`, `NukeMax_FFTLatentMatch`, `NukeMax_FFTBandFilter`.

- **Custom type:** `FFT_DATA` (complex spectrum, log magnitude, phase).
- **Inputs:** IMAGE / LATENT, frequency band specs, optional reference latent.
- **Outputs:** `FFT_DATA`, IMAGE preview, LATENT (frequency-matched).
- **Use case:** seamless inpaint by matching the frequency profile of a generation to surrounding plate; band-isolated noise injection; debugging tile/seam artefacts on Flux/Qwen-Image upscales.

### PBR Relight (`relight/`) — 6 nodes

`NukeMax_MaterialDecompose`, `NukeMax_LightProbeEstimate`, `NukeMax_ThreePointLight`, `NukeMax_PBRRelight`, `NukeMax_MaterialEdit`, `NukeMax_EXRMaterialBridge`.

- **Custom types:** `MATERIAL_PBR` (albedo / normal / roughness / metalness), `LIGHT_PROBE` (HDR environment).
- **Inputs:** IMAGE (or EXR via the bridge), light positions/colors/intensities.
- **Outputs:** decomposed PBR channels, relit IMAGE, exportable EXR.
- **Use case:** estimate a light probe from a single still, relight under three-point lighting, or hand a `MATERIAL_PBR` bundle to Blender/Nuke through `EXRMaterialBridge`. Useful as an upstream stage before Flux/Qwen-Image refines.

### Audio-Reactive (`audio/`) — 5 nodes

`NukeMax_AudioLoad`, `NukeMax_AudioFeatureExtract`, `NukeMax_AudioToCurve`, `NukeMax_AudioMaskModulate`, `NukeMax_AudioCFGModulate`.

- **Custom types:** `AUDIO_TENSOR`, `CURVE` (per-frame floats aligned to a target FPS).
- **Inputs:** audio file path or AUDIO tensor, FPS, frame count, feature ('rms' / 'onset' / 'spectral_centroid' / band energies).
- **Outputs:** `AUDIO_TENSOR`, per-frame `CURVE`, modulated MASK, modulated CFG schedule.
- **Use case:** drive mask intensity or sampler CFG from beat/onset for music-reactive Wan or AnimateDiff renders.

### Sub-Pixel Flow Compositing (`flow/`) — 6 nodes

`NukeMax_FlowEstimate`, `NukeMax_FlowForwardWarp`, `NukeMax_FlowBackwardWarp`, `NukeMax_FlowOcclusion`, `NukeMax_FlowVisualize`, `NukeMax_CleanPlateMerge`.

- **Custom type:** `FLOW_VECTOR` (B×2×H×W).
- **Inputs:** IMAGE batch (T frames), optional reference plate.
- **Outputs:** `FLOW_VECTOR`, warped IMAGE, occlusion MASK, RGB flow viz, clean-plate composite.
- **Use case:** temporal coherence for Wan/AnimateDiff; clean-plate generation for inpaint-anything plates that downstream Flux fills.

### Smart Edge Tools (`edges/`) — 4 nodes

`NukeMax_NormalAwareBlur`, `NukeMax_MatteDensityAdjust`, `NukeMax_SubPixelEdgeDetect`, `NukeMax_HairAwareChoke`.

- **Custom type:** `EDGE_MAP`.
- **Inputs:** IMAGE, optional NORMAL or MATTE.
- **Outputs:** IMAGE / MASK with edge-respecting filtering.
- **Use case:** clean compositing-grade edges before/after a diffusion pass — particularly hair where standard chokes destroy detail.

### Utilities (`utils/`) — 1 node

`UniversalRerouteMEC` — typed reroute node for the MEC type system. Keeps long graphs readable without forcing scalar collapse.

### EXR I/O (`io/`) — 3 nodes

`LoadEXRMEC`, `SaveEXRMEC`, `EXRMetadataReaderMEC`.

- **Custom type:** `MEC_EXR_METADATA`.
- **Inputs:** filepath, channel selection, optional metadata to embed on save.
- **Outputs:** IMAGE (with optional alpha / Z / multi-channel split), full EXR metadata dict.
- **Use case:** read multi-channel EXRs from Nuke / Blender renders, ingest as IMAGE batches for diffusion preprocessing, write back round-trip with metadata preserved.

### Render Passes (`passes/`) — 2 nodes

`MergeRenderPassesMEC`, `DepthOfFieldMaskMEC`.

- **Custom type:** `MEC_RENDER_PASSES`.
- **Inputs:** beauty + diffuse + specular + normal + depth IMAGEs, focus distance / aperture.
- **Outputs:** merged IMAGE, DoF MASK.
- **Use case:** bring 3D render passes into the diffusion graph; build per-region masks (in-focus subject vs out-of-focus background) for targeted inpainting.

### Plate Tools (`plate/`) — 4 nodes

`GrainMatchMEC`, `PlateStabilizerMEC`, `CleanPlateExtractorMEC`, `DifferenceMatteMEC`.

- **Inputs:** plate IMAGE, generated IMAGE, motion estimate.
- **Outputs:** grain-matched IMAGE, stabilised IMAGE batch, clean plate, difference MASK.
- **Use case:** make Flux/Qwen/Wan output composite cleanly into live-action plates — match grain, stabilise jitter, extract clean plates for set extension.

### Geometry Extensions (`geometry_ext/`) — 3 nodes

`DepthWarpMEC`, `NormalToCurvatureMEC`, `PositionPassSplitterMEC`.

- **Inputs:** DEPTH / NORMAL / POSITION IMAGEs.
- **Outputs:** warped IMAGE, curvature MASK, X/Y/Z split IMAGEs.
- **Use case:** turn a depth pass into camera-relative warps; convert normals into curvature masks for ControlNet conditioning; split P-pass for region masking.

### Metadata (`metadata/`) — 3 nodes

`MetadataWriterMEC`, `FrameRangeRouterMEC`, `ShotMetadataNodeMEC`.

- **Inputs:** IMAGE batch, key/value pairs, frame range spec.
- **Outputs:** IMAGE (with metadata attached), routed IMAGE batches per range, shot metadata bundle.
- **Use case:** keep shot codes, frame numbers, and scene/take info traveling with the tensors — pair with `ComfyUI-FolderIncrementer` for fully-versioned saves.

### Color (`color/`) — 3 nodes

`ColorSpaceConvertMEC`, `LUTApplyMEC`, `ExposureGradeMEC`.

- **Inputs:** IMAGE, source/target color space, `.cube` LUT path, exposure / lift / gamma / gain.
- **Outputs:** color-managed IMAGE.
- **Use case:** convert sRGB ↔ ACEScg for diffusion (always do diffusion in linear/sRGB, then grade via LUT/exposure on the way out).

## Use in image/video generation pipelines (Flux / Qwen-Image / Wan / Z-Image / ERNIE-VL)

| Model family | Key bridges from this pack |
|---|---|
| **Flux** | `RotoToDiffusionGuidance` → ControlNet/inpaint conditioning. `FFTLatentMatch` for tile-seamless upscale. `GrainMatchMEC` for live-action composites. `ColorSpaceConvertMEC` + `LUTApplyMEC` to keep the look consistent. |
| **Qwen-Image** | Same pattern as Flux. Use `NormalToCurvatureMEC` outputs as ControlNet hints if a Qwen-compatible curvature ControlNet is loaded. |
| **Wan 2.x** | `FlowEstimate` + `FlowForwardWarp` give per-frame flow conditioning that can be wired into Wan-Animate or AnimateDiff temporal nodes. `AudioToCurve` + `AudioCFGModulate` for music-reactive video. `LoadEXRMEC` for video frame ingest from a renderer. `PlateStabilizerMEC` to denoise camera jitter before sampling. |
| **Z-Image** | Use the FFT and edge tools for refinement passes; same Flux-style ControlNet/inpaint bridges apply where Z-Image exposes them. |
| **GLM-Image** | The IMAGE outputs are model-agnostic and feed cleanly back into the GLM-Image sampler I2I path (via `denoise_strength`). |
| **ERNIE-VL** | Not directly applicable (LLM, not a sampler). The metadata nodes can still be useful for cataloguing ERNIE-VL described frames. |

The strict typing means a `MASK` produced by, for example, `RotoShapeRender` is a normal ComfyUI MASK and connects to any node that accepts MASK — only the *bundle* types (`ROTO_SHAPE`, `FFT_DATA`, `MATERIAL_PBR`, …) are pack-internal.

## Notes

- `types_io.py` intentionally exposes empty mappings; types are registered by their owning category modules.
- All temporal nodes operate batch-first (B = T), no per-frame Python loops in the hot path.
- The dual-output discipline is enforced: classical VFX socket on the left, AI-consumable socket on the right, on every node where both are meaningful.

## License

Apache-2.0 (see `LICENSE`).
