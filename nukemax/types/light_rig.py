"""LIGHT_RIG: list of analytic lights for procedural relight."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


LightType = Literal["point", "directional", "area"]


@dataclass(frozen=True)
class Light:
    position: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    type: LightType = "directional"
    radius: float = 0.0       # area light radius (world units)
    falloff: float = 2.0      # inverse-square = 2.0


@dataclass(frozen=True)
class LightRig:
    lights: Tuple[Light, ...] = field(default_factory=tuple)
    ambient: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __iter__(self):
        return iter(self.lights)

    def __len__(self) -> int:
        return len(self.lights)
