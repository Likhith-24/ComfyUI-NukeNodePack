# PBR Lighting Extraction & Relight

Decompose 2D images into albedo / normal / depth / roughness, estimate
an HDR light probe, then re-render with a procedural 3-point rig.
Bridge output: write the probe to EXR for Blender/Nuke import.

## Nodes

- **Material Decomposer (Heuristic)** — math-only baseline. No models
  required; useful for testing relight math.
- **Material Decomposer (Models)** — lazy wrapper for Marigold /
  StableNormal etc. Falls back to the heuristic if weights are missing.
- **Light Rig Builder** — JS widget for spherical placement of key/
  fill/rim lights.
- **3-Point Relight** — pure-PyTorch Lambert + Blinn-Phong shading.
- **Light Probe Estimator** — divide image by albedo, bin radiance by
  normal direction onto an equirectangular env map.
- **Light Probe → EXR** — bridge output; writes 32-bit EXR (or `.npy`
  if OpenEXR is missing).
