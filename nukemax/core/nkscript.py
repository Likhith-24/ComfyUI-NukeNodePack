"""nkscript — Nuke-style text-based copy/paste for ComfyUI nodes.

PROTOCOL (matches Nuke .nk syntax exactly):

    # nukemax_nk 1.0 v1
    set cut_paste_input [stack 0]
    version 1.0 v1
    push $cut_paste_input
    Blur {
     inputs 1
     size 15
     name Blur1
     selected true
     xpos 100
     ypos -50
    }
    set N1_Blur1 [stack 0]

PURE STACK LOGIC (no `input_N name` knobs, no node-id strings in knobs):
  - Every node pops `inputs N` items off the top of the stack and pushes
    itself. Top of stack = input slot 0, next = slot 1, etc.
  - `set NAME [stack 0]` aliases the current top so it can be recalled.
  - `push $NAME` duplicates that aliased node onto the top of the stack.
  - Wiring is determined purely by push order + `inputs` count. This is
    what makes the format git-diffable: a slider change is one line.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


HEADER_VERSION = "nukemax_nk 1.0 v1"


# --- Tokenizer --------------------------------------------------------------

_TOK = re.compile(
    r"""
    \s+ |
    \{ | \} |
    " ( (?: \\. | [^"\\] )* ) " |
    \[ ( [^\]]* ) \] |
    \$ ( [A-Za-z_][A-Za-z0-9_]* ) |
    ( [^\s{}\[\]"]+ )
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in _TOK.finditer(text):
        if m.group(0).isspace():
            continue
        if m.group(1) is not None:
            out.append(("str", m.group(1).encode("utf-8").decode("unicode_escape")))
        elif m.group(2) is not None:
            out.append(("bracket", m.group(2).strip()))
        elif m.group(3) is not None:
            out.append(("var", m.group(3)))
        elif m.group(4) is not None:
            out.append(("tok", m.group(4)))
        elif m.group(0) == "{":
            out.append(("lb", "{"))
        elif m.group(0) == "}":
            out.append(("rb", "}"))
    return out


# --- Data model -------------------------------------------------------------

@dataclass
class NodeDef:
    class_type: str
    name: str
    knobs: dict[str, Any] = field(default_factory=dict)
    inputs: list[tuple[int, str]] = field(default_factory=list)
    xpos: float = 0.0
    ypos: float = 0.0
    selected: bool = False


# --- Helpers ----------------------------------------------------------------

_RESERVED_KNOBS = {"name", "xpos", "ypos", "selected", "inputs", "class"}


def _esc(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if s == "":
        return '""'
    if any(ch in s for ch in " \t\n{}[]\"$"):
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s


def _alias_for(node_id: Any, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return f"N{node_id}_{safe}"


# --- Serializer -------------------------------------------------------------

def serialize(nodes: Iterable[dict], links: Iterable[tuple]) -> str:
    """Subgraph -> Nuke .nk text using PURE stack logic."""
    nodes = list(nodes)
    links = list(links)
    in_map: dict[Any, list[tuple[int, Any]]] = {}
    for src_id, _src_slot, dst_id, dst_slot in links:
        in_map.setdefault(dst_id, []).append((dst_slot, src_id))
    for v in in_map.values():
        v.sort()

    aliases: dict[Any, str] = {n["id"]: _alias_for(n["id"], n["name"]) for n in nodes}
    in_subgraph = {n["id"] for n in nodes}

    buf = io.StringIO()
    buf.write(f"# {HEADER_VERSION}\n")
    buf.write("set cut_paste_input [stack 0]\n")
    buf.write("version 1.0 v1\n")
    buf.write("push $cut_paste_input\n")

    stack_top: Any | None = None
    for n in nodes:
        nid = n["id"]
        ins = [(slot, src) for slot, src in in_map.get(nid, []) if src in in_subgraph]
        N = len(ins)
        if N == 0:
            pass
        elif N == 1 and stack_top == ins[0][1]:
            pass  # natural flow
        else:
            # reverse so input slot 0 ends up on top of stack
            for _slot, src in reversed(ins):
                buf.write(f"push ${aliases[src]}\n")
        buf.write(f"{n['class_type']} {{\n")
        if N > 0:
            buf.write(f" inputs {N}\n")
        for k, v in (n.get("widgets") or {}).items():
            if k in _RESERVED_KNOBS:
                continue
            buf.write(f" {k} {_esc(v)}\n")
        buf.write(f" name {_esc(n['name'])}\n")
        buf.write(f" xpos {int(n.get('xpos', 0))}\n")
        buf.write(f" ypos {int(n.get('ypos', 0))}\n")
        if n.get("selected"):
            buf.write(" selected true\n")
        buf.write("}\n")
        buf.write(f"set {aliases[nid]} [stack 0]\n")
        stack_top = nid
    buf.write("end_group\n")
    return buf.getvalue()


# --- Parser -----------------------------------------------------------------

class _Cursor:
    def __init__(self, toks): self.toks = toks; self.i = 0
    def peek(self): return self.toks[self.i] if self.i < len(self.toks) else None
    def take(self): t = self.toks[self.i]; self.i += 1; return t
    def eof(self): return self.i >= len(self.toks)


def _coerce(s: str) -> Any:
    if s in ("true", "True"): return True
    if s in ("false", "False"): return False
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_value(cur: _Cursor) -> Any:
    kind, val = cur.take()
    if kind in ("str", "bracket", "var"):
        return val
    if kind == "tok":
        return _coerce(val)
    if kind == "lb":
        items: list[Any] = []
        depth = 1
        while not cur.eof() and depth > 0:
            k, _v = cur.peek()
            if k == "rb":
                cur.take(); depth -= 1
                if depth == 0: break
                items.append("}"); continue
            if k == "lb":
                cur.take(); depth += 1
                items.append("{"); continue
            items.append(_parse_value(cur))
        return items
    return val


def parse(text: str) -> list[NodeDef]:
    """Parse Nuke-style text. Inputs resolved purely from stack."""
    text = re.sub(r"(?m)^\s*#.*$", "", text)
    toks = _tokenize(text)
    cur = _Cursor(toks)
    stack: list[str] = []
    aliases: dict[str, str] = {}
    nodes: list[NodeDef] = []

    while not cur.eof():
        kind, val = cur.peek()
        if kind == "tok" and val == "set":
            cur.take()
            _, var = cur.take()
            ref_kind, ref_val = cur.take()
            if ref_kind == "bracket" and ref_val.startswith("stack") and stack:
                aliases[var] = stack[-1]
            continue
        if kind == "tok" and val == "push":
            cur.take()
            v_kind, v_val = cur.take()
            if v_kind == "var" and v_val in aliases:
                stack.append(aliases[v_val])
            continue
        if kind == "tok" and val in ("version", "end_group", "Root"):
            cur.take()
            # consume optional trailing tokens / block
            for _ in range(2):
                nxt = cur.peek()
                if nxt and nxt[0] == "tok" and nxt[1] not in ("set", "push", "version", "end_group", "Root"):
                    # could be version arg; only swallow if class-like token isn't expected
                    if val == "version":
                        cur.take(); continue
                break
            nxt = cur.peek()
            if nxt and nxt[0] == "lb":
                _skip_block(cur)
            continue
        if kind == "tok":
            class_type = val
            cur.take()
            if cur.peek() and cur.peek()[0] == "lb":
                cur.take()
                node = _parse_node_block(cur, class_type)
                inputs_n = int(node.knobs.pop("__inputs", 0))
                if inputs_n > 0:
                    take = min(inputs_n, len(stack))
                    popped = []
                    for _ in range(take):
                        popped.append(stack.pop())
                    for slot, src in enumerate(popped):
                        node.inputs.append((slot, src))
                stack.append(node.name)
                nodes.append(node)
                continue
        cur.take()
    return nodes


def _skip_block(cur: _Cursor) -> None:
    if cur.eof() or cur.peek()[0] != "lb":
        return
    cur.take()
    depth = 1
    while not cur.eof() and depth > 0:
        k, _ = cur.take()
        if k == "lb": depth += 1
        elif k == "rb": depth -= 1


def _parse_node_block(cur: _Cursor, class_type: str) -> NodeDef:
    knobs: dict[str, Any] = {}
    name = f"{class_type}_anon"
    xpos = ypos = 0.0
    selected = False
    while not cur.eof():
        k, v = cur.take()
        if k == "rb":
            break
        if k != "tok":
            continue
        knob = v
        if cur.eof():
            break
        if cur.peek()[0] == "lb":
            cur.take()
            val: Any = []
            depth = 1
            while not cur.eof() and depth > 0:
                kk, _vv = cur.peek()
                if kk == "rb":
                    cur.take(); depth -= 1
                    if depth == 0: break
                    val.append("}"); continue
                if kk == "lb":
                    cur.take(); depth += 1
                    val.append("{"); continue
                val.append(_parse_value(cur))
            value = val
        else:
            value = _parse_value(cur)
        if knob == "name":
            name = str(value)
        elif knob == "xpos":
            xpos = float(value)
        elif knob == "ypos":
            ypos = float(value)
        elif knob == "selected":
            selected = bool(value) if isinstance(value, bool) else (str(value).lower() == "true")
        elif knob == "inputs":
            knobs["__inputs"] = int(value)
        else:
            knobs[knob] = value
    return NodeDef(class_type=class_type, name=name, knobs=knobs,
                   inputs=[], xpos=xpos, ypos=ypos, selected=selected)


# --- High-level helpers -----------------------------------------------------

def parse_to_json(text: str) -> str:
    nodes = parse(text)
    return json.dumps({
        "version": HEADER_VERSION,
        "nodes": [{
            "class_type": n.class_type, "name": n.name, "knobs": n.knobs,
            "inputs": [{"slot": s, "src": src} for s, src in n.inputs],
            "xpos": n.xpos, "ypos": n.ypos, "selected": n.selected,
        } for n in nodes],
    }, indent=2)


def serialize_from_json(data: str) -> str:
    obj = json.loads(data)
    return serialize(obj.get("nodes", []), [tuple(x) for x in obj.get("links", [])])
