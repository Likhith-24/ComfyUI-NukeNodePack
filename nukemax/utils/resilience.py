"""Graph-never-crashes contract for nodes.

Wrap a node class with `@resilient` to catch any exception in `execute`,
log the traceback, and return a safe default tuple matching the node's
declared `RETURN_TYPES` plus an `info` string prefixed `"ERROR: "`.

The decorator inspects `RETURN_TYPES` and synthesizes a passthrough
default. If a node declares an `ERROR_DEFAULTS` classvar (a tuple of
callables `() -> value`), those are used in preference. Nodes whose
last `RETURN_TYPE` is `"STRING"` named `info`/`status`/`error` get an
informative error message in that slot.
"""
from __future__ import annotations

import functools
import logging
import traceback
from typing import Any, Callable

import torch

log = logging.getLogger("nukemax")


def _zero_for(rt: str) -> Any:
    rt_u = rt.upper()
    if rt_u == "IMAGE":
        return torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    if rt_u == "MASK":
        return torch.zeros(1, 64, 64, dtype=torch.float32)
    if rt_u == "LATENT":
        return {"samples": torch.zeros(1, 4, 8, 8, dtype=torch.float32)}
    if rt_u in ("FLOAT",):
        return 0.0
    if rt_u in ("INT",):
        return 0
    if rt_u in ("BOOLEAN", "BOOL"):
        return False
    if rt_u == "STRING":
        return ""
    return None


def resilient(cls):
    """Class decorator. Wraps the node's FUNCTION method."""
    fn_name = getattr(cls, "FUNCTION", None)
    if not fn_name or not hasattr(cls, fn_name):
        return cls
    original = getattr(cls, fn_name)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("[%s] %s\n%s", cls.__name__, exc, tb)
            rt = getattr(cls, "RETURN_TYPES", ())
            names = getattr(cls, "RETURN_NAMES", ()) or ()
            defaults = list(getattr(cls, "ERROR_DEFAULTS", ()))
            out: list[Any] = []
            for i, t in enumerate(rt):
                if i < len(defaults):
                    try:
                        out.append(defaults[i]())
                        continue
                    except Exception:
                        pass
                v = _zero_for(t)
                if t.upper() == "STRING":
                    name = names[i] if i < len(names) else ""
                    if name.lower() in ("info", "status", "error", "message"):
                        v = f"ERROR: {exc}"
                out.append(v)
            return tuple(out)

    setattr(cls, fn_name, wrapped)
    return cls


def resilient_fn(returns: tuple[str, ...]) -> Callable:
    """Function-level variant for ad-hoc node functions."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                log.error("[%s] %s\n%s", fn.__name__, exc, traceback.format_exc())
                out = [_zero_for(t) for t in returns]
                if returns and returns[-1].upper() == "STRING":
                    out[-1] = f"ERROR: {exc}"
                return tuple(out)
        return wrapped
    return deco
