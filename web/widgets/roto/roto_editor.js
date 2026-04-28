// Roto Spline Editor canvas widget.
// Attaches a click-to-add / drag bezier editor to nodes named
// "NukeMax_RotoSplineEditor". The widget writes its JSON state into
// the hidden `spline_state` STRING widget so the value persists in
// the workflow.

import { app } from "../../../../scripts/app.js";

const NODE_NAME = "NukeMax_RotoSplineEditor";

function createRotoWidget(node) {
    const state = {
        frames: [
            { points: [], in: [], out: [], feather: [] },
        ],
        closed: true,
        canvas: { h: 512, w: 512 },
        currentFrame: 0,
    };

    // Find the hidden state widget (created from the Python INPUT_TYPES).
    const stateWidget = node.widgets.find(w => w.name === "spline_state");

    const widget = {
        type: "custom",
        name: "roto_canvas",
        size: [320, 320],
        draw(ctx, node, widget_width, y, widget_height) {
            const x = 0;
            const w = widget_width;
            const h = 320;
            ctx.save();
            ctx.fillStyle = "#222";
            ctx.fillRect(x, y, w, h);
            const frame = state.frames[state.currentFrame] || { points: [] };
            const sx = w / state.canvas.w;
            const sy = h / state.canvas.h;
            // Polyline
            ctx.strokeStyle = "#5cf";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            frame.points.forEach((p, i) => {
                const px = x + p[0] * sx;
                const py = y + p[1] * sy;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            if (state.closed && frame.points.length > 2) ctx.closePath();
            ctx.stroke();
            // Vertices
            ctx.fillStyle = "#fc6";
            frame.points.forEach(p => {
                ctx.beginPath();
                ctx.arc(x + p[0] * sx, y + p[1] * sy, 4, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.fillStyle = "#aaa";
            ctx.font = "10px monospace";
            ctx.fillText(`frame ${state.currentFrame + 1}/${state.frames.length}  pts:${frame.points.length}`, x + 6, y + 14);
            ctx.restore();
        },
        mouse(event, pos, node) {
            if (event.type !== "pointerdown") return false;
            const w = widget.size?.[0] || 320;
            const h = 320;
            const sx = state.canvas.w / w;
            const sy = state.canvas.h / h;
            const px = pos[0] * sx;
            const py = pos[1] * sy;
            const frame = state.frames[state.currentFrame];
            // Right-click removes nearest; left-click adds.
            if (event.button === 2) {
                let bestI = -1, bestD = 1e9;
                frame.points.forEach((p, i) => {
                    const d = (p[0] - px) ** 2 + (p[1] - py) ** 2;
                    if (d < bestD) { bestD = d; bestI = i; }
                });
                if (bestI >= 0 && bestD < 100) {
                    frame.points.splice(bestI, 1);
                    frame.in.splice(bestI, 1);
                    frame.out.splice(bestI, 1);
                    frame.feather.splice(bestI, 1);
                }
            } else {
                frame.points.push([px, py]);
                frame.in.push([px, py]);
                frame.out.push([px, py]);
                frame.feather.push(0);
            }
            sync();
            node.setDirtyCanvas(true, true);
            return true;
        },
    };

    function sync() {
        if (stateWidget) stateWidget.value = JSON.stringify({
            frames: state.frames,
            closed: state.closed,
            canvas: state.canvas,
        });
    }

    sync();
    node.addCustomWidget(widget);
    return widget;
}

app.registerExtension({
    name: "NukeMax.Roto",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            try { createRotoWidget(this); } catch (e) { console.error("[NukeMax] roto widget failed", e); }
        };
    },
});
