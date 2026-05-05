"""nkscript I/O nodes + server routes.

User-facing nodes:
  - NkScriptExport : take a STRING (e.g. produced from outside) and
    pass it through; primarily used to embed canned setups in workflows.
  - NkScriptImport : parse a Nuke-style text block at runtime and
    return diagnostic info. Actual graph mutation happens client-side
    via the JS extension; this node exists for inspection and CI.

Registers an aiohttp route under /nukemax/nkscript/{serialize,parse}
so the front-end JS can request server-side serialise/parse without
duplicating the TCL grammar in JavaScript.
"""
from __future__ import annotations

import json
import logging

from ...core import nkscript
from ...utils.resilience import resilient

log = logging.getLogger("nukemax.nkscript")

# ---------- Server routes (registered once on import) ----------

def _register_routes() -> None:
    try:
        from server import PromptServer  # type: ignore
    except Exception:
        return
    try:
        from aiohttp import web  # type: ignore
    except Exception:
        return
    routes = PromptServer.instance.routes

    @routes.post("/nukemax/nkscript/serialize")
    async def _serialize(request):  # noqa: ANN001
        try:
            body = await request.json()
            text = nkscript.serialize(body.get("nodes", []), body.get("links", []))
            return web.json_response({"ok": True, "text": text})
        except Exception as e:  # noqa: BLE001
            log.exception("nkscript serialize failed")
            return web.json_response({"ok": False, "error": str(e)}, status=400)

    @routes.post("/nukemax/nkscript/parse")
    async def _parse(request):  # noqa: ANN001
        try:
            body = await request.json()
            data = nkscript.parse_to_json(body.get("text", ""))
            return web.json_response({"ok": True, "data": json.loads(data)})
        except Exception as e:  # noqa: BLE001
            log.exception("nkscript parse failed")
            return web.json_response({"ok": False, "error": str(e)}, status=400)


_register_routes()


# ---------- Nodes ----------

@resilient
class NkScriptParse:
    DESCRIPTION = "Parse a Nuke-style text block (.nk syntax) and emit a JSON description of the encoded subgraph."
    CATEGORY = "NukeMax/NkScript"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("json", "node_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "Nuke-style .nk text block to parse."}),
            },
        }

    def execute(self, text: str):
        js = nkscript.parse_to_json(text)
        try:
            n = len(json.loads(js).get("nodes", []))
        except Exception:
            n = 0
        return (js, n)


@resilient
class NkScriptSerialize:
    DESCRIPTION = "Convert a JSON subgraph description (same shape produced by NkScriptParse) into Nuke-style text."
    CATEGORY = "NukeMax/NkScript"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subgraph_json": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "JSON describing nodes + links to serialise."}),
            },
        }

    def execute(self, subgraph_json: str):
        return (nkscript.serialize_from_json(subgraph_json),)


NODE_CLASS_MAPPINGS = {
    "NukeMax_NkScriptParse": NkScriptParse,
    "NukeMax_NkScriptSerialize": NkScriptSerialize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeMax_NkScriptParse": "NkScript Parse (NukeMax)",
    "NukeMax_NkScriptSerialize": "NkScript Serialize (NukeMax)",
}
