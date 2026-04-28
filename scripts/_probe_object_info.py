"""Wait for ComfyUI to become live, then dump NukeMax_* node count."""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8189/object_info"
deadline = time.time() + 90
last = ""
data = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(URL, timeout=3) as r:
            data = json.load(r)
        break
    except Exception as e:  # noqa: BLE001
        last = str(e)
        time.sleep(2)
if data is None:
    print(f"FAIL waiting for server: {last}")
    sys.exit(1)
keys = sorted(k for k in data if k.startswith("NukeMax_"))
print(f"OK live; total nodes={len(data)}; NukeMax_* nodes={len(keys)}")
for k in keys:
    cat = data[k].get("category", "?")
    print(f"  {k:42s} cat={cat}")
