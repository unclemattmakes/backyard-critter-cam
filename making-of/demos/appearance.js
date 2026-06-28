/* Demo 3 — the appearance map. Crops overlap; visit prototypes pull the regulars apart. */
(function () {
  const M = window.MakingOf, h = M.h, clamp = M.clamp, lerp = M.lerp;

  M.register("appearance", function (mount, data) {
    const pts = data.points || [];
    const protos = data.prototypes || [];
    const tri = data.cos_tri || [];
    const pcos = data.proto_cos || [];
    const feat = data.featured || {};
    const n = pts.length;
    if (!n) { mount.appendChild(h("p.muted", null, "No appearance data exported.")); return; }

    // map each point to its visit prototype index (for the collapse animation)
    const protoIndex = {};
    protos.forEach((p, i) => { protoIndex[p.individual + "|" + p.visit] = i; });
    pts.forEach((p) => { p._proto = protoIndex[p.individual + "|" + p.visit]; });

    // visit colours (by-visit mode): golden-angle hue per distinct visit
    const visits = [...new Set(pts.map((p) => p.visit))];
    const visitColor = {};
    visits.forEach((v, i) => { visitColor[v] = `hsl(${(i * 137.5) % 360} 45% 58%)`; });

    let t = 0;          // 0 = crops, 1 = prototypes
    let target = 0, anim = null;
    let colorBy = "individual";
    let hover = -1, sel = [], pair = null;

    // ---- layout ----
    const canvas = h("canvas", { style: { width: "100%", display: "block", borderRadius: "4px", background: "rgba(0,0,0,.22)", border: "1px solid var(--rule)", cursor: "crosshair", touchAction: "none" } });
    const side = h("div", { style: { width: "190px", flex: "0 0 auto" } });
    const stage = h("div", { style: { display: "flex", gap: "16px", alignItems: "stretch" } }, [h("div", { style: { flex: "1", minWidth: "0" } }, canvas), side]);

    const seg = h("div.seg", null, [
      segBtn("by individual", true, () => setColor("individual")),
      segBtn("by visit", false, () => setColor("visit")),
    ]);
    const collapseBtn = h("button.btn", { onclick: () => setView(target ? 0 : 1) }, "Collapse to prototypes ▸");
    const controls = h("div", { style: { display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center", margin: "14px 0 4px" } }, [seg, collapseBtn]);

    const featRow = h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap", marginTop: "10px" } });
    if (feat.same_visit) featRow.appendChild(featChip("Same visit", feat.same_visit.cos, "crop", feat.same_visit));
    if (feat.cross_session) featRow.appendChild(featChip("Same raccoon, another night", feat.cross_session.cos, "proto", feat.cross_session));
    if (feat.different) featRow.appendChild(featChip("Two different raccoons", feat.different.cos, "proto", feat.different));

    mount.append(controls, stage, featRow);

    // side panel contents
    const legend = h("div", { style: { display: "flex", flexDirection: "column", gap: "6px", marginBottom: "14px" } });
    const preview = h("div", { style: { textAlign: "center" } });
    const readout = h("div", { style: { marginTop: "12px", minHeight: "44px" } });
    side.append(legend, preview, readout);

    function segBtn(label, on, fn) { const b = h("button", { onclick: fn }, label); if (on) b.classList.add("on"); return b; }
    function setColor(c) { colorBy = c; [...seg.children].forEach((b, i) => b.classList.toggle("on", (i === 0) === (c === "individual"))); renderLegend(); draw(); }

    function featChip(label, cos, space, info) {
      return h("button.btn", { onclick: () => {
        setView(space === "proto" ? 1 : 0);
        pair = { space, a: info.a, b: info.b, cos };
        sel = []; draw(); showReadout();
      }, style: { fontSize: "10px" } }, [label + " ", h("span.val", { style: { marginLeft: "4px" } }, cos.toFixed(2))]);
    }

    function setView(to) {
      target = to; pair = null; sel = [];
      if (anim) cancelAnimationFrame(anim);
      const from = t, t0 = performance.now(), dur = 700;
      collapseBtn.textContent = to ? "◂ Back to all crops" : "Collapse to prototypes ▸";
      collapseBtn.classList.toggle("on", !!to);
      const step = (now) => {
        const k = clamp((now - t0) / dur, 0, 1);
        const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; // ease in-out
        t = lerp(from, to, e);
        draw();
        if (k < 1) anim = requestAnimationFrame(step); else { showReadout(); }
      };
      anim = requestAnimationFrame(step);
    }

    // ---- canvas plumbing ----
    let W = 0, Hh = 0, dpr = Math.min(window.devicePixelRatio || 1, 2), pad = 26;
    const ctx = canvas.getContext("2d");
    function resize() {
      W = canvas.clientWidth || 520; Hh = Math.max(300, Math.round(W * 0.62));
      canvas.width = W * dpr; canvas.height = Hh * dpr; canvas.style.height = Hh + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
    }
    const px = (x) => pad + x * (W - 2 * pad);
    const py = (y) => pad + y * (Hh - 2 * pad);

    function pointPos(p) {
      if (p._proto == null) return [px(p.x), py(p.y)];
      const pr = protos[p._proto];
      return [lerp(px(p.x), px(pr.x), t), lerp(py(p.y), py(pr.y), t)];
    }
    function colorOf(p) { return colorBy === "individual" ? M.castColor(p.individual) : visitColor[p.visit]; }

    function draw() {
      ctx.clearRect(0, 0, W, Hh);
      // pair line
      if (pair) {
        const A = pair.space === "proto" ? protoPos(pair.a) : pointPos(pts[pair.a]);
        const B = pair.space === "proto" ? protoPos(pair.b) : pointPos(pts[pair.b]);
        ctx.strokeStyle = "rgba(201,162,75,.8)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); ctx.setLineDash([]);
      }
      // crops (fade out as t->1)
      const cropA = 1 - t;
      if (cropA > 0.02) {
        pts.forEach((p, i) => {
          const [x, y] = pointPos(p);
          const hl = pair && pair.space === "crop" && (i === pair.a || i === pair.b);
          const r = hl ? 6 : (i === hover ? 5 : 3);
          ctx.globalAlpha = hl ? 1 : cropA * 0.85;
          ctx.fillStyle = colorOf(p);
          ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
          if (hl || (sel.includes(i) && t < 0.5)) { ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
        });
      }
      // prototypes (fade in)
      if (t > 0.02) {
        protos.forEach((p, i) => {
          const [x, y] = protoPos(i);
          const hl = pair && pair.space === "proto" && (i === pair.a || i === pair.b);
          ctx.globalAlpha = t;
          ctx.fillStyle = M.castColor(p.individual);
          const r = hl ? 11 : (i === hover && t >= 0.5 ? 10 : 8);
          ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
          ctx.globalAlpha = t; ctx.strokeStyle = "rgba(16,13,7,.7)"; ctx.lineWidth = 2; ctx.stroke();
          if (hl || (sel.includes(i) && t >= 0.5)) { ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
        });
      }
      ctx.globalAlpha = 1;
    }
    function protoPos(i) { const pr = protos[i]; return [px(pr.x), py(pr.y)]; }

    // ---- hit testing ----
    function evtXY(e) { const r = canvas.getBoundingClientRect(); const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left; const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top; return [cx, cy]; }
    function nearest(cx, cy) {
      const useProto = t >= 0.5;
      let best = -1, bd = useProto ? 16 : 9;
      if (useProto) protos.forEach((p, i) => { const [x, y] = protoPos(i); const d = Math.hypot(x - cx, y - cy); if (d < bd) { bd = d; best = i; } });
      else pts.forEach((p, i) => { const [x, y] = pointPos(p); const d = Math.hypot(x - cx, y - cy); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    canvas.addEventListener("mousemove", (e) => { const [x, y] = evtXY(e); const idx = nearest(x, y); if (idx !== hover) { hover = idx; draw(); updatePreview(); } });
    canvas.addEventListener("mouseleave", () => { hover = -1; draw(); updatePreview(); });
    canvas.addEventListener("click", (e) => {
      const [x, y] = evtXY(e); const idx = nearest(x, y); if (idx < 0) return;
      pair = null;
      if (sel.includes(idx)) sel = sel.filter((s) => s !== idx);
      else { sel.push(idx); if (sel.length > 2) sel.shift(); }
      draw(); showReadout();
    });

    function cropCos(i, j) {
      if (i === j) return 1;
      if (i > j) { const k = i; i = j; j = k; }
      const idx = i * n - (i * (i + 1)) / 2 + (j - i - 1);
      return (tri[idx] || 0) / 100;
    }
    function protoCos(i, j) { return (pcos[i] && pcos[i][j] != null ? pcos[i][j] : 0) / 100; }

    function renderLegend() {
      legend.innerHTML = "";
      legend.appendChild(h("div.lbl", null, colorBy === "individual" ? "individuals" : "each visit a colour"));
      if (colorBy === "individual") {
        [...new Set(pts.map((p) => p.individual))].forEach((nm) =>
          legend.appendChild(h("div", { style: { fontSize: "13px", color: "var(--dim)" } }, [h("span.swatch", { style: { background: M.castColor(nm) } }), nm])));
      } else {
        legend.appendChild(h("div.muted", { style: { fontSize: "12px" } }, visits.length + " visits — same-visit crops share a hue and clump together."));
      }
    }

    function updatePreview() {
      preview.innerHTML = "";
      const useProto = t >= 0.5;
      if (!useProto && hover >= 0) {
        const p = pts[hover];
        preview.append(
          h("img", { src: p.thumb, alt: "", style: { width: "96px", height: "96px", objectFit: "cover", borderRadius: "4px", border: "2px solid " + M.castColor(p.individual) } }),
          h("div", { style: { fontVariant: "small-caps", fontSize: "15px", marginTop: "5px" } }, p.individual),
          h("div.muted", { style: { fontSize: "11px", fontFamily: "var(--mono)" } }, "visit " + p.visit));
      } else if (useProto && hover >= 0) {
        const p = protos[hover];
        preview.append(
          h("div", { style: { width: "60px", height: "60px", borderRadius: "50%", margin: "0 auto", background: M.castColor(p.individual) } }),
          h("div", { style: { fontVariant: "small-caps", fontSize: "15px", marginTop: "6px" } }, p.individual),
          h("div.muted", { style: { fontSize: "11px", fontFamily: "var(--mono)" } }, "prototype · " + p.n + " crops"));
      } else {
        preview.append(h("div.muted", { style: { fontSize: "12px", padding: "20px 0", textAlign: "center" } }, t >= 0.5 ? "hover a prototype" : "hover a crop"));
      }
    }

    function showReadout() {
      readout.innerHTML = "";
      const useProto = t >= 0.5;
      let a, b, cos, na, nb;
      if (pair) {
        a = pair.a; b = pair.b; cos = pair.cos;
        const arr = pair.space === "proto" ? protos : pts;
        na = arr[a].individual; nb = arr[b].individual;
      } else if (sel.length === 2) {
        a = sel[0]; b = sel[1];
        cos = useProto ? protoCos(a, b) : cropCos(a, b);
        const arr = useProto ? protos : pts;
        na = arr[a].individual; nb = arr[b].individual;
      } else {
        readout.appendChild(h("p.muted", { style: { fontSize: "12px" } }, "Click two dots for their cosine similarity.")); return;
      }
      const same = na === nb;
      readout.append(
        h("div.lbl", { style: { marginBottom: "4px" } }, "cosine similarity"),
        h("div", { style: { fontFamily: "var(--mono)", fontSize: "30px", color: cos >= 0.6 ? "var(--ok)" : (cos >= 0.4 ? "var(--gilt)" : "var(--rust)"), lineHeight: "1" } }, cos.toFixed(2)),
        h("div", { style: { fontSize: "12px", color: "var(--dim)", marginTop: "5px" } }, [h("span.swatch", { style: { background: M.castColor(na) } }), na, " ↔ ", h("span.swatch", { style: { background: M.castColor(nb), marginLeft: "5px" } }), nb]),
        h("div.muted", { style: { fontSize: "11.5px", marginTop: "4px" } }, same ? "same individual" : "different individuals"));
    }

    renderLegend(); updatePreview(); showReadout();
    const mqA = window.matchMedia("(max-width:560px)");
    const applyA = () => {
      stage.style.flexDirection = mqA.matches ? "column" : "row";
      side.style.width = mqA.matches ? "100%" : "190px";
    };
    mqA.addEventListener("change", () => { applyA(); resize(); });
    applyA();
    new ResizeObserver(resize).observe(stage);
    resize();
  });
})();
