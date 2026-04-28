# Smart Roto System

`ROTO_SHAPE` is a frame-batched cubic-bezier spline. Edit it in the
graph, propagate it across a video using optical flow, render it to
masks (with optional flow-driven motion blur), and emit AI guidance
(soft inpaint masks, latent-space masks, SAM points/boxes).

## Nodes
- **Roto Spline Editor** — JS canvas widget (see `web/widgets/roto/`)
  that posts a JSON state into a hidden string widget. Output:
  `ROTO_SHAPE`.
- **Roto Shape From File** — loads the same JSON from disk for
  workflows generated outside ComfyUI.
- **Roto Shape → AI Tracker** — propagates frame-0 vertices using
  either a `FLOW_FIELD` (preferred) or a fallback NCC tracker.
- **Roto Shape Renderer** — rasterizes per-frame masks (signed-distance
  + raycast) with optional flow-stretch motion blur.
- **Roto Shape → Diffusion Guidance** — dual-output bridge. Emits hard
  mask, soft inpaint mask, latent-space mask (÷8 by default), and a
  JSON of SAM point/box prompts per frame.
