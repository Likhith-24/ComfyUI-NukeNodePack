"""Core vectorized math primitives.

ComfyUI image convention: `(B, H, W, C)` float32 in [0,1].
PyTorch conv convention: `(B, C, H, W)`.

All public helpers in `nukemax.core` operate on `(B, C, H, W)` internally.
Use `to_bchw` / `to_bhwc` at the node boundary.
"""

from . import color, blur, composite, fft, flow, shading, splines  # noqa: F401
