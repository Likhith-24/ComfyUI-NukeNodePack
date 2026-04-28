// Light Rig Builder spherical light placer.
// Attaches an interactive 2D-projected sphere widget to nodes named
// "NukeMax_LightRigBuilder". Click and drag to position lights on a
// hemisphere; the widget writes JSON state into the hidden
// `rig_state` STRING widget.
//
// Coordinate convention matches `nukemax/core/shading.py`:
//   +X right, +Y up, -Z toward camera. Light direction points FROM
//   light TOWARD scene, so a light placed in front-right is
//   (-x, -y, -z) when projected.

import { app } from "../../../../scripts/app.js";

const NODE_NAME = "NukeMax_LightRigBuilder";

const DEFAULT_LIGHTS = [
    { name: "key",  azimuth: -45, elevation: 30, color: [1.0, 0.95, 0.85], intensity: 1.0, type: "directional" },
    { name: "fill", azimuth:  60, elevation: 15, color: [0.6, 0.7, 1.0],   intensity: 0.4, type: "directional" },
    { name: "rim",  azimuth: 170, elevation: 40, color: [1.0, 1.0, 1.0],   intensity: 0.6, type: "directional" },
];

function azElToDirection(azDeg, elDeg) {
    // Direction the light points toward the scene origin.
    const az = (azDeg * Math.PI) / 180;
    const el = (elDeg * Math.PI) / 180;
    const x = -Math.cos(el) * Math.sin(az);
    const y = -Math.sin(el);
    const z = -Math.cos(el) * Math.cos(az);
    return [x, y, z];
}

function lightToPayload(L) {
    return {
        direction: azElToDirection(L.azimuth, L.elevation),
        color: L.color,
        intensity: L.intensity,
        type: L.type,
        radius: 0.0,
        falloff: 2.0,
    };
}

function createLightRigWidget(node) {
    const state = {
        lights: DEFAULT_LIGHTS.map(L => ({ ...L })),
        ambient: 0.05,
        selected: 0,
        dragging: false,
    };
    const stateWidget = node.widgets.find(w => w.name === "rig_state");

    function sync() {
        if (!stateWidget) return;
        stateWidget.value = JSON.stringify({
            lights: state.lights.map(lightToPayload),
            ambient: state.ambient,
        });
    }

    const widget = {
        type: "custom",
        name: "light_sphere",
        size: [320, 320],
        draw(ctx, node, w, y, h) {
            const size = 320;
            const cx = w / 2;
            const cy = y + size / 2;
            const r = size * 0.45;
            ctx.save();
            ctx.fillStyle = "#1a1a22";
            ctx.fillRect(0, y, w, size);
            // Sphere outline
            ctx.strokeStyle = "#444";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();
            // Crosshair
            ctx.strokeStyle = "#2a2a32";
            ctx.beginPath();
            ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy);
            ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r);
            ctx.stroke();
            // Lights as colored dots
            state.lights.forEach((L, i) => {
                const az = (L.azimuth * Math.PI) / 180;
                const el = (L.elevation * Math.PI) / 180;
                const px = cx + Math.cos(el) * Math.sin(az) * r;
                const py = cy - Math.sin(el) * r;
                const col = `rgb(${(L.color[0] * 255) | 0},${(L.color[1] * 255) | 0},${(L.color[2] * 255) | 0})`;
                ctx.fillStyle = col;
                ctx.beginPath();
                ctx.arc(px, py, i === state.selected ? 9 : 6, 0, Math.PI * 2);
                ctx.fill();
                if (i === state.selected) {
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
                ctx.fillStyle = "#ddd";
                ctx.font = "10px monospace";
                ctx.fillText(L.name, px + 10, py + 4);
            });
            // HUD
            const sel = state.lights[state.selected];
            ctx.fillStyle = "#aaa";
            ctx.font = "11px monospace";
            ctx.fillText(`[${sel.name}] az=${sel.azimuth.toFixed(0)}° el=${sel.elevation.toFixed(0)}° I=${sel.intensity.toFixed(2)}`, 8, y + size - 10);
            ctx.fillText("L-click sphere: move selected. R-click dot: cycle. Wheel: intensity.", 8, y + 14);
            ctx.restore();
        },
        mouse(event, pos, node) {
            const size = 320;
            const w = widget.size?.[0] || 320;
            const cx = w / 2;
            const cy = size / 2;
            const r = size * 0.45;
            const dx = pos[0] - cx;
            const dy = pos[1] - cy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (event.type === "pointerdown") {
                if (event.button === 2) {
                    // Cycle selected light
                    state.selected = (state.selected + 1) % state.lights.length;
                    node.setDirtyCanvas(true, true);
                    return true;
                }
                // Pick nearest dot if close, else move selected.
                let best = -1, bestD = 1e9;
                state.lights.forEach((L, i) => {
                    const az = (L.azimuth * Math.PI) / 180;
                    const el = (L.elevation * Math.PI) / 180;
                    const px = Math.cos(el) * Math.sin(az) * r;
                    const py = -Math.sin(el) * r;
                    const d = (px - dx) ** 2 + (py - dy) ** 2;
                    if (d < bestD) { bestD = d; best = i; }
                });
                if (bestD < 144) state.selected = best;
                state.dragging = true;
            }
            if ((event.type === "pointerdown" || event.type === "pointermove") && state.dragging) {
                if (dist > r * 1.2) return false;
                const nx = Math.max(-1, Math.min(1, dx / r));
                const ny = Math.max(-1, Math.min(1, -dy / r));
                const az = (Math.atan2(nx, Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny))) * 180) / Math.PI;
                const el = (Math.asin(ny) * 180) / Math.PI;
                const L = state.lights[state.selected];
                L.azimuth = az;
                L.elevation = el;
                sync();
                node.setDirtyCanvas(true, true);
                return true;
            }
            if (event.type === "pointerup") state.dragging = false;
            if (event.type === "wheel") {
                const L = state.lights[state.selected];
                L.intensity = Math.max(0, Math.min(10, L.intensity + (event.deltaY < 0 ? 0.1 : -0.1)));
                sync();
                node.setDirtyCanvas(true, true);
                return true;
            }
            return false;
        },
    };

    sync();
    node.addCustomWidget(widget);
    return widget;
}

app.registerExtension({
    name: "NukeMax.Relight",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            try { createLightRigWidget(this); } catch (e) {
                console.error("[NukeMax] light rig widget failed", e);
            }
        };
    },
});
