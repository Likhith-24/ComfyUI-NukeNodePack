// NukeMax — base ComfyUI web extension.
// Phase 0 ships only registration + a "hello" proof widget.
// Per-ecosystem widgets live under web/widgets/<eco>/ and are imported here.

import { app } from "../../scripts/app.js";

// Per-ecosystem widget modules. Each one self-registers via
// app.registerExtension when imported.
import "./widgets/roto/roto_editor.js";
import "./widgets/relight/light_placer.js";
import "./widgets/audio/waveform_preview.js";

app.registerExtension({
    name: "NukeMax.Base",
    async setup() {
        console.log("[NukeMax] web extension loaded");
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Hook point: per-ecosystem widget files attach themselves to specific
        // node types here by checking nodeData.name.
    },
});
