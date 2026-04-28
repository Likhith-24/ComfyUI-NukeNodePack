# Sub-Pixel Optical Flow Compositing

Optical flow used the way VFX uses it: not just for interpolation, but
for *compositing*. Warp clean plates along motion to remove objects,
mask occlusions, and visualize flow with the Middlebury color wheel.

## Nodes

- **Compute Optical Flow** — outputs `FLOW_FIELD` (forward + backward
  + occlusion). Uses OpenCV Farneback if available, else a pure-PyTorch
  multi-scale Lucas-Kanade fallback.
- **Flow Forward Warp** — splatting with bilinear weights and a
  weight-buffer output for hole detection.
- **Flow Backward Warp** — `grid_sample` bilinear.
- **Flow Occlusion Mask** — forward-backward consistency check.
- **Clean Plate Merge** — warps a clean plate under a moving object
  mask along the flow chain, Porter-Duff Over composite.
- **Flow Visualize** — Middlebury HSV color wheel.
