"""Runtime capability probe.

Detects hardware/software features once at import time and exposes a
single `CAPS` object that the rest of the pack consults to choose code
paths. Every capability has a CPU/fp32 fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _torch_version_tuple() -> tuple[int, int]:
    parts = torch.__version__.split("+", 1)[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


@dataclass(frozen=True)
class Capabilities:
    has_cuda: bool
    has_mps: bool
    torch_major: int
    torch_minor: int
    supports_compile: bool
    supports_bf16: bool
    supports_fp16: bool

    @property
    def device(self) -> torch.device:
        if self.has_cuda:
            return torch.device("cuda")
        if self.has_mps:
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def preferred_dtype(self) -> torch.dtype:
        if self.has_cuda and self.supports_bf16:
            return torch.bfloat16
        if self.has_cuda and self.supports_fp16:
            return torch.float16
        return torch.float32

    @property
    def use_compile(self) -> bool:
        # `torch.compile` is unstable on Windows + older torch; gate strictly.
        return self.supports_compile and self.has_cuda


def _probe() -> Capabilities:
    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    major, minor = _torch_version_tuple()
    supports_compile = (major, minor) >= (2, 1) and hasattr(torch, "compile")
    supports_bf16 = has_cuda and torch.cuda.is_bf16_supported() if has_cuda else False
    supports_fp16 = has_cuda
    return Capabilities(
        has_cuda=has_cuda,
        has_mps=has_mps,
        torch_major=major,
        torch_minor=minor,
        supports_compile=supports_compile,
        supports_bf16=supports_bf16,
        supports_fp16=supports_fp16,
    )


CAPS = _probe()


def maybe_compile(fn):
    """Decorate a hot-path function with torch.compile when safe."""
    if CAPS.use_compile:
        try:
            return torch.compile(fn, dynamic=True, fullgraph=False)
        except Exception:
            return fn
    return fn
