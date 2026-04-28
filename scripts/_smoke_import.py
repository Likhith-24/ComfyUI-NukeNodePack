"""Smoke-test: import all NukeMax ecosystems and report node count."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Put the parent of the pack on sys.path and import as a normal package.
sys.path.insert(0, str(ROOT.parent))
PKG = ROOT.name  # "ComfyUI-NukeMaxNodes" — hyphenated but importable via importlib.
import importlib
try:
    mod = importlib.import_module(PKG)
except Exception:
    traceback.print_exc()
    sys.exit(1)

print(f"NODE_CLASS_MAPPINGS: {len(mod.NODE_CLASS_MAPPINGS)} nodes")
for name in sorted(mod.NODE_CLASS_MAPPINGS):
    cls = mod.NODE_CLASS_MAPPINGS[name]
    print(f"  - {name:40s}  cat={getattr(cls,'CATEGORY','?')}")
print(f"WEB_DIRECTORY: {mod.WEB_DIRECTORY}")
