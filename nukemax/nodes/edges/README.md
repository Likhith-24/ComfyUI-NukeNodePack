# Smart Edge Tools

Geometry-aware edge manipulation. Goes beyond stock choke/blur by using
normal maps, local frequency, and matte semi-transparency masks.

## Nodes

- **Normal-Aware Edge Blur** — separable Gaussian gated by normal-map
  similarity. Blurs within a surface, stops at edges.
- **Matte Density Adjust** — gamma/contrast applied only to the
  semi-transparent band (`edge_lo < α < edge_hi`); fully-opaque or
  fully-transparent regions are bit-exact passthrough.
- **Sub-Pixel Edge Detect** — Sobel + parabolic interpolation.
  Bridge: emits `TRACKING_DATA` of the strongest edge points.
- **Hair-Aware Choke** — uses local stddev as a hair proxy; regions of
  high frequency get less aggressive choke.
