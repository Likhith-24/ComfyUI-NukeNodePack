// Audio waveform / spectrogram preview widget.
// Attaches a read-only waveform display to nodes named
// "NukeMax_AudioLoadAnalyze". Reads the file path from the node's
// `path` widget and renders a stylized stripe plus BPM/duration
// readout. The actual audio analysis happens server-side; this is a
// preview helper so the user can confirm the file is loaded.

import { app } from "../../../../scripts/app.js";

const NODE_NAME = "NukeMax_AudioLoadAnalyze";

function createWaveformWidget(node) {
    const state = {
        path: "",
        loaded: false,
        bpm: 0,
        duration: 0,
        bars: new Array(96).fill(0),
    };
    const pathWidget = node.widgets.find(w => w.name === "path");

    function loadPreview() {
        if (!pathWidget?.value || pathWidget.value === state.path) return;
        state.path = pathWidget.value;
        // Best-effort fetch of the file via ComfyUI's /view endpoint.
        // This is purely for the UI preview; the executor still does
        // the real loading from disk. If decoding fails we just keep
        // a neutral display.
        try {
            const url = `/view?filename=${encodeURIComponent(state.path)}`;
            fetch(url).then(r => r.arrayBuffer()).then(buf => {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                ctx.decodeAudioData(buf, (decoded) => {
                    const ch0 = decoded.getChannelData(0);
                    const N = state.bars.length;
                    const step = Math.max(1, (ch0.length / N) | 0);
                    for (let i = 0; i < N; i++) {
                        let peak = 0;
                        const start = i * step;
                        const end = Math.min(ch0.length, start + step);
                        for (let j = start; j < end; j++) {
                            const v = Math.abs(ch0[j]);
                            if (v > peak) peak = v;
                        }
                        state.bars[i] = peak;
                    }
                    state.duration = decoded.duration;
                    state.loaded = true;
                    node.setDirtyCanvas(true, true);
                }, () => { state.loaded = false; });
            }).catch(() => { state.loaded = false; });
        } catch (e) {
            // Silent — preview is best-effort.
        }
    }

    const widget = {
        type: "custom",
        name: "audio_waveform",
        size: [320, 96],
        draw(ctx, node, w, y, h) {
            const height = 96;
            ctx.save();
            ctx.fillStyle = "#1a1a22";
            ctx.fillRect(0, y, w, height);
            // Center line
            ctx.strokeStyle = "#2a2a32";
            ctx.beginPath();
            ctx.moveTo(0, y + height / 2);
            ctx.lineTo(w, y + height / 2);
            ctx.stroke();
            // Bars
            const N = state.bars.length;
            const bw = w / N;
            ctx.fillStyle = state.loaded ? "#5cf" : "#444";
            for (let i = 0; i < N; i++) {
                const v = state.bars[i];
                const bh = Math.max(1, v * (height - 8));
                ctx.fillRect(i * bw + 1, y + (height - bh) / 2, bw - 2, bh);
            }
            // Label
            ctx.fillStyle = "#aaa";
            ctx.font = "10px monospace";
            const label = state.loaded
                ? `${state.path.split(/[\\/]/).pop() || ""}  ${state.duration.toFixed(2)}s`
                : (state.path ? "preview unavailable (server-side load only)" : "set 'path' to preview");
            ctx.fillText(label, 6, y + 12);
            ctx.restore();
            loadPreview();
        },
    };

    node.addCustomWidget(widget);
    // Trigger initial load if path is already set
    setTimeout(loadPreview, 100);
    return widget;
}

app.registerExtension({
    name: "NukeMax.Audio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            try { createWaveformWidget(this); } catch (e) {
                console.error("[NukeMax] waveform widget failed", e);
            }
        };
    },
});
