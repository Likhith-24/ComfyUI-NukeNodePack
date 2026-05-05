// nkscript copy/paste — Nuke-style text clipboard for ComfyUI nodes.
//
// Hotkeys:
//   Ctrl+Shift+C  → serialise the selected nodes to Nuke .nk text and
//                    write to the OS clipboard. Also drops a toast.
//   Ctrl+Shift+V  → read text from the OS clipboard, parse it, and
//                    instantiate the nodes (with widget values + links)
//                    pasted at the cursor position.
//
// The heavy lifting (TCL grammar, stack semantics) happens server-side
// at /nukemax/nkscript/{serialize,parse}; the JS just packages the live
// graph state and applies the result.

import { app } from "../../../../scripts/app.js";

function _toast(msg, type = "info") {
    try {
        app.extensionManager?.toast?.add({ severity: type, summary: "NkScript", detail: msg, life: 3500 });
    } catch (_) { console.log("[NkScript]", msg); }
}

function _selectedNodes() {
    const sel = app.canvas?.selected_nodes;
    if (!sel) return [];
    return Object.values(sel);
}

function _gatherSubgraph(nodes) {
    const ids = new Set(nodes.map((n) => n.id));
    const out_nodes = nodes.map((n) => {
        // Capture widget values keyed by widget name.
        const widgets = {};
        for (const w of n.widgets || []) {
            if (w.name && w.value !== undefined) widgets[w.name] = w.value;
        }
        return {
            id: n.id,
            class_type: n.comfyClass || n.type,
            name: n.title || n.type + n.id,
            widgets,
            xpos: Math.round(n.pos?.[0] ?? 0),
            ypos: Math.round(n.pos?.[1] ?? 0),
            selected: true,
        };
    });
    // Walk graph.links, keep only links where both endpoints are selected.
    const out_links = [];
    const links = app.graph?.links || {};
    for (const k in links) {
        const l = links[k];
        if (!l) continue;
        if (ids.has(l.origin_id) && ids.has(l.target_id)) {
            out_links.push([l.origin_id, l.origin_slot, l.target_id, l.target_slot]);
        }
    }
    return { nodes: out_nodes, links: out_links };
}

async function _copy() {
    const nodes = _selectedNodes();
    if (!nodes.length) { _toast("Nothing selected", "warn"); return; }
    const payload = _gatherSubgraph(nodes);
    try {
        const r = await fetch("/nukemax/nkscript/serialize", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const j = await r.json();
        if (!j.ok) throw new Error(j.error || "serialize failed");
        await navigator.clipboard.writeText(j.text);
        _toast(`Copied ${nodes.length} node(s) as NkScript`, "success");
    } catch (e) {
        // Fallback: client-side minimal serialiser.
        const text = _clientSideSerialize(payload);
        await navigator.clipboard.writeText(text);
        _toast(`Copied ${nodes.length} node(s) (offline mode)`, "success");
    }
}

function _clientSideSerialize(p) {
    const lines = [];
    lines.push("# nukemax_nk 1.0 v1");
    lines.push("set cut_paste_input [stack 0]");
    lines.push("version 1.0 v1");
    lines.push("push $cut_paste_input");
    const nameOf = {};
    const aliasOf = {};
    p.nodes.forEach((n) => {
        nameOf[n.id] = n.name;
        aliasOf[n.id] = `N${n.id}_${String(n.name).replace(/[^A-Za-z0-9_]/g, "_")}`;
    });
    const inMap = {};
    const inSub = new Set(p.nodes.map((n) => n.id));
    p.links.forEach(([sid, sslot, did, dslot]) => {
        if (!inSub.has(sid) || !inSub.has(did)) return;
        (inMap[did] ||= []).push([dslot, sid]);
    });
    Object.values(inMap).forEach((arr) => arr.sort((a, b) => a[0] - b[0]));
    let stackTop = null;
    for (const n of p.nodes) {
        const ins = inMap[n.id] || [];
        const N = ins.length;
        if (N === 0) {
            // no pushes
        } else if (N === 1 && stackTop === ins[0][1]) {
            // natural flow
        } else {
            for (let i = ins.length - 1; i >= 0; i--) {
                lines.push(`push $${aliasOf[ins[i][1]]}`);
            }
        }
        lines.push(`${n.class_type} {`);
        if (N > 0) lines.push(` inputs ${N}`);
        for (const [k, v] of Object.entries(n.widgets || {})) {
            const sv = (typeof v === "string" && /[\s{}\[\]"$]/.test(v)) ? JSON.stringify(v) : v;
            lines.push(` ${k} ${sv}`);
        }
        lines.push(` name ${n.name}`);
        lines.push(` xpos ${n.xpos}`);
        lines.push(` ypos ${n.ypos}`);
        if (n.selected) lines.push(" selected true");
        lines.push("}");
        lines.push(`set ${aliasOf[n.id]} [stack 0]`);
        stackTop = n.id;
    }
    lines.push("end_group");
    return lines.join("\n");
}

async function _paste() {
    let text = "";
    try { text = await navigator.clipboard.readText(); }
    catch (_) { _toast("Clipboard read denied — paste into a NkScript Parse node instead", "warn"); return; }
    if (!text || !/{[\s\S]*}/.test(text)) { _toast("Clipboard does not look like NkScript", "warn"); return; }
    let parsed;
    try {
        const r = await fetch("/nukemax/nkscript/parse", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        const j = await r.json();
        if (!j.ok) throw new Error(j.error || "parse failed");
        parsed = j.data;
    } catch (e) {
        _toast("Parse failed: " + e.message, "error"); return;
    }
    const cursor = app.canvas?.graph_mouse || [0, 0];
    const created = {};
    let baseX = cursor[0], baseY = cursor[1];
    let originX = null, originY = null;
    for (const nd of parsed.nodes) {
        const node = LiteGraph.createNode(nd.class_type);
        if (!node) { console.warn("[NkScript] unknown class", nd.class_type); continue; }
        if (originX === null) { originX = nd.xpos; originY = nd.ypos; }
        node.pos = [baseX + (nd.xpos - originX), baseY + (nd.ypos - originY)];
        if (nd.name) node.title = nd.name;
        app.graph.add(node);
        // Apply widget values.
        for (const [k, v] of Object.entries(nd.knobs || {})) {
            const w = (node.widgets || []).find((x) => x.name === k);
            if (w) { w.value = v; w.callback?.(v); }
        }
        created[nd.name] = node;
    }
    // Wire links by NAME (NodeDef.inputs[i] = {slot, src}).
    for (const nd of parsed.nodes) {
        const dst = created[nd.name];
        if (!dst) continue;
        for (const e of nd.inputs || []) {
            const src = created[e.src];
            if (!src) continue;
            try {
                src.connect(0, dst, e.slot);
            } catch (err) { console.warn("[NkScript] link failed", err); }
        }
    }
    app.graph.setDirtyCanvas(true, true);
    _toast(`Pasted ${parsed.nodes.length} node(s)`, "success");
}

window.addEventListener("keydown", (ev) => {
    if (!ev.ctrlKey || !ev.shiftKey) return;
    const k = ev.key.toLowerCase();
    if (k === "c") { ev.preventDefault(); _copy(); }
    else if (k === "v") { ev.preventDefault(); _paste(); }
});

app.registerExtension({
    name: "NukeMax.NkScript",
    async setup() {
        console.log("[NukeMax] NkScript clipboard extension loaded (Ctrl+Shift+C / Ctrl+Shift+V)");
        // Add right-click canvas menu items.
        const origMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const opts = origMenu.apply(this, arguments) || [];
            opts.push(null);
            opts.push({ content: "Copy as NkScript (Ctrl+Shift+C)", callback: _copy });
            opts.push({ content: "Paste NkScript (Ctrl+Shift+V)", callback: _paste });
            return opts;
        };
    },
});

export { _copy as nkCopy, _paste as nkPaste };
