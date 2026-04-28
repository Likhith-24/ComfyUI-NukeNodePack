"""Serialize/deserialize custom types for workflow-JSON portability.

Tensor blobs go through `safetensors`-style raw bytes encoded with
base64. We avoid an external dep by using torch's own pickle-free
mechanism: `torch.save` to a BytesIO. This is acceptable because the
serialized blobs are only ever consumed by our own nodes.
"""
from __future__ import annotations

import base64
import io
from dataclasses import asdict, fields, is_dataclass
from typing import Any

import torch

from . import (
    RotoShape,
    TrackingData,
    FFTTensor,
    MaterialSet,
    LightProbe,
    LightRig,
    Light,
    AudioFeatures,
    FlowField,
)


_TYPE_REGISTRY = {
    "RotoShape": RotoShape,
    "TrackingData": TrackingData,
    "FFTTensor": FFTTensor,
    "MaterialSet": MaterialSet,
    "LightProbe": LightProbe,
    "LightRig": LightRig,
    "Light": Light,
    "AudioFeatures": AudioFeatures,
    "FlowField": FlowField,
}


def _enc_tensor(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t.detach().cpu().contiguous(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _dec_tensor(s: str) -> torch.Tensor:
    raw = base64.b64decode(s.encode("ascii"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _enc_value(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return {"__tensor__": _enc_tensor(v)}
    if is_dataclass(v):
        return _to_dict(v)
    if isinstance(v, (list, tuple)):
        return [_enc_value(x) for x in v]
    return v


def _dec_value(v: Any) -> Any:
    if isinstance(v, dict) and "__tensor__" in v:
        return _dec_tensor(v["__tensor__"])
    if isinstance(v, dict) and "__type__" in v:
        return _from_dict(v)
    if isinstance(v, list):
        return [_dec_value(x) for x in v]
    return v


def _to_dict(obj: Any) -> dict:
    assert is_dataclass(obj)
    d: dict[str, Any] = {"__type__": type(obj).__name__}
    for f in fields(obj):
        d[f.name] = _enc_value(getattr(obj, f.name))
    return d


def _from_dict(d: dict) -> Any:
    tname = d.pop("__type__")
    cls = _TYPE_REGISTRY[tname]
    kwargs = {k: _dec_value(v) for k, v in d.items()}
    # Tuples that became lists need to round-trip as tuples for frozen dataclasses
    # that declare Tuple types.
    if cls is LightRig:
        kwargs["lights"] = tuple(_from_dict(x) if isinstance(x, dict) and "__type__" in x else x for x in kwargs.get("lights", []))
        kwargs["ambient"] = tuple(kwargs.get("ambient", (0.0, 0.0, 0.0)))
    if cls is Light:
        for tk in ("position", "direction", "color"):
            if tk in kwargs:
                kwargs[tk] = tuple(kwargs[tk])
    return cls(**kwargs)


def serialize(obj: Any) -> dict:
    return _to_dict(obj)


def deserialize(d: dict) -> Any:
    # Don't mutate caller's dict.
    return _from_dict(dict(d))
