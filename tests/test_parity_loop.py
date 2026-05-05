"""Phase-loop verification suite for the new NukeMax features.

Run: python -m pytest tests/test_parity_loop.py -q
or:  python tests/test_parity_loop.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nks = _load("nks", "nukemax/core/nkscript.py")


# --------- Phase A/B: text protocol ---------------------------------

def test_serialize_chain_uses_natural_stack_no_pushes():
    nodes = [
        {"id": 1, "class_type": "Shuffle",  "name": "Shuffle1",
         "widgets": {"out_R": "B"}, "xpos": 0, "ypos": 0, "selected": True},
        {"id": 2, "class_type": "DeepFrom", "name": "Deep1",
         "widgets": {}, "xpos": 0, "ypos": 50, "selected": True},
        {"id": 3, "class_type": "Blur",     "name": "Blur1",
         "widgets": {"size": 15}, "xpos": 0, "ypos": 100, "selected": True},
    ]
    links = [(1, 0, 2, 0), (2, 0, 3, 0)]
    text = nks.serialize(nodes, links)
    # No 'input_' lines in stack-only protocol.
    assert "input_" not in text
    # Linear chain: only 1 push (the prologue cut_paste_input). No extra pushes.
    extra_pushes = [l for l in text.splitlines() if l.startswith("push $") and "$cut_paste_input" not in l]
    assert extra_pushes == []
    # All knobs present.
    assert " size 15" in text
    assert "name Shuffle1" in text


def test_serialize_branch_uses_push():
    # Merge1 takes input 0 from Deep1 (natural) and input 1 from Shuffle1 (branch).
    nodes = [
        {"id": 1, "class_type": "Shuffle",  "name": "Shuffle1", "widgets": {}, "xpos": 0, "ypos": 0},
        {"id": 2, "class_type": "DeepFrom", "name": "Deep1",    "widgets": {}, "xpos": 0, "ypos": 50},
        {"id": 3, "class_type": "Merge2",   "name": "Merge1",   "widgets": {}, "xpos": 0, "ypos": 100},
    ]
    links = [(1, 0, 2, 0), (2, 0, 3, 0), (1, 0, 3, 1)]
    text = nks.serialize(nodes, links)
    # Should NOT use input_N hardcoded refs:
    assert "input_1" not in text
    # Should branch via push:
    assert "push $N1_Shuffle1" in text
    assert "inputs 2" in text


def test_round_trip_preserves_topology():
    nodes = [
        {"id": 1, "class_type": "Shuffle",  "name": "Shuffle1",
         "widgets": {"out_R": "B"}, "xpos": 0, "ypos": 0},
        {"id": 2, "class_type": "DeepFrom", "name": "Deep1",
         "widgets": {}, "xpos": 0, "ypos": 50},
        {"id": 3, "class_type": "Merge2",   "name": "Merge1",
         "widgets": {"max_samples": 8}, "xpos": 0, "ypos": 100},
    ]
    links = [(1, 0, 2, 0), (2, 0, 3, 0), (1, 0, 3, 1)]
    text = nks.serialize(nodes, links)
    parsed = nks.parse(text)
    assert [p.class_type for p in parsed] == ["Shuffle", "DeepFrom", "Merge2"]
    by_name = {p.name: p for p in parsed}
    # Links: Deep1 -> Merge slot 0, Shuffle1 -> Merge slot 1.
    merge_in = dict(by_name["Merge1"].inputs)
    assert merge_in[0] == "Deep1"
    assert merge_in[1] == "Shuffle1"
    # Linear chain: Deep1 input 0 = Shuffle1.
    deep_in = dict(by_name["Deep1"].inputs)
    assert deep_in[0] == "Shuffle1"
    # Knobs preserved with exact types.
    assert by_name["Shuffle1"].knobs["out_R"] == "B"
    assert by_name["Merge1"].knobs["max_samples"] == 8


def test_diff_friendly_single_slider_change_is_one_line():
    n0 = [{"id": 1, "class_type": "Blur", "name": "Blur1",
           "widgets": {"size": 15}, "xpos": 0, "ypos": 0}]
    n1 = [{"id": 1, "class_type": "Blur", "name": "Blur1",
           "widgets": {"size": 22}, "xpos": 0, "ypos": 0}]
    a = nks.serialize(n0, [])
    b = nks.serialize(n1, [])
    diff = [(x, y) for x, y in zip(a.splitlines(), b.splitlines()) if x != y]
    assert len(diff) == 1
    assert "size" in diff[0][0] and "size" in diff[0][1]


def test_python_api_modify_and_repaste():
    """A Python script outside ComfyUI can read text, mutate a knob, repaste."""
    nodes = [{"id": 1, "class_type": "Blur", "name": "Blur1",
              "widgets": {"size": 15}, "xpos": 100, "ypos": -50}]
    text = nks.serialize(nodes, [])
    # Round-trip via JSON and mutate.
    j = json.loads(nks.parse_to_json(text))
    j["nodes"][0]["knobs"]["size"] = 33
    # Reconstruct serialize input shape and re-emit.
    new_nodes = [{
        "id": 1,
        "class_type": j["nodes"][0]["class_type"],
        "name": j["nodes"][0]["name"],
        "widgets": j["nodes"][0]["knobs"],
        "xpos": j["nodes"][0]["xpos"],
        "ypos": j["nodes"][0]["ypos"],
    }]
    text2 = nks.serialize(new_nodes, [])
    parsed = nks.parse(text2)
    assert parsed[0].knobs["size"] == 33


def test_no_binary_or_uuid_links():
    """Hard rule: copied text contains zero ComfyUI link UUIDs / numeric IDs."""
    nodes = [
        {"id": 7, "class_type": "A", "name": "A1", "widgets": {}, "xpos": 0, "ypos": 0},
        {"id": 12, "class_type": "B", "name": "B1", "widgets": {}, "xpos": 0, "ypos": 50},
    ]
    text = nks.serialize(nodes, [(7, 0, 12, 0)])
    # Must not contain raw ComfyUI workflow JSON markers.
    forbidden = ['"link_id"', '"links":', '"order":', '"type":"primitive"', "\\u0000"]
    for f in forbidden:
        assert f not in text, f"forbidden token leaked: {f}"


# --------- Phase A: branching by editing text -----------------------

def test_user_can_branch_by_editing_text():
    # Original: linear A -> B
    text = nks.serialize([
        {"id": 1, "class_type": "A", "name": "A1", "widgets": {}, "xpos": 0, "ypos": 0},
        {"id": 2, "class_type": "B", "name": "B1", "widgets": {}, "xpos": 0, "ypos": 50},
    ], [(1, 0, 2, 0)])
    # User manually adds a Merge2 that takes A1 twice (mirror branch).
    edited = text.replace(
        "end_group",
        "push $N1_A1\nMerge2 {\n inputs 2\n name Merge1\n xpos 0\n ypos 100\n}\nset N3_Merge1 [stack 0]\nend_group",
    )
    parsed = nks.parse(edited)
    assert any(p.name == "Merge1" for p in parsed)
    merge = next(p for p in parsed if p.name == "Merge1")
    srcs = [src for _slot, src in merge.inputs]
    assert srcs.count("A1") == 1   # one from explicit push
    assert srcs.count("B1") == 1   # one from natural top after B1


if __name__ == "__main__":
    failures = 0
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {t.__name__}: {e!r}")
    print(f"\n{'=' * 50}\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
