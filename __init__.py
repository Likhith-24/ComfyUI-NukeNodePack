"""Node registration aggregator. Imports all ecosystem subpackages and
merges their `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`.
"""
from __future__ import annotations

import importlib
import logging

log = logging.getLogger("nukemax")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

# Each ecosystem subpackage exposes the same two dicts.
# Use relative imports so this works no matter what folder name
# ComfyUI loads us under (e.g. "ComfyUI-NukeMaxNodes" with a hyphen).
_ECOSYSTEMS = (
    ".nukemax.nodes.types_io",  # serialize/deserialize for custom types
    ".nukemax.nodes.roto",
    ".nukemax.nodes.fft",
    ".nukemax.nodes.relight",
    ".nukemax.nodes.audio",
    ".nukemax.nodes.flow",
    ".nukemax.nodes.edges",
    # Migrated from ComfyUI-CustomNodePacks (Apr 2026)
    ".nukemax.nodes.utils",
    ".nukemax.nodes.io",
    ".nukemax.nodes.passes",
    ".nukemax.nodes.plate",
    ".nukemax.nodes.geometry_ext",
    ".nukemax.nodes.metadata",
    ".nukemax.nodes.color",
    # New (May 2026): deep compositing, shuffle, Nuke-style copy/paste.
    ".nukemax.nodes.deep",
    ".nukemax.nodes.shuffle",
    ".nukemax.nodes.nkscript",
)

for mod_name in _ECOSYSTEMS:
    try:
        mod = importlib.import_module(mod_name, package=__name__)
        NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as exc:  # noqa: BLE001
        log.warning("[NukeMax] failed to import %s: %s", mod_name, exc)

# Tell ComfyUI where to find our JS.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
