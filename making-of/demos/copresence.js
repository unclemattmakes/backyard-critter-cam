/* Bonus demo — two at once. Drag two detection boxes; IoU < 0.45 reads as two animals. */
(function () {
  const M = window.MakingOf, h = M.h, clamp = M.clamp;

  M.register("copresence", function (mount, data) {
    const THR = data.iou_threshold || 0.45;
    const ar = (data.w && data.h) ? data.h / data.w : 0.5625;
    // two boxes in stage-fraction coords {x,y,w,h}; start clearly apart (two animals)
    const boxes = [
      { x: 0.10, y: 0.34, w: 0.30, h: 0.42, c: "var(--notch)" },
      { x: 0.60, y: 0.30, w: 0.30, h: 0.44, c: "var(--elliot)" },
    ];

    const stage = h("div", { style: { position: "relative", width: "100%", paddingTop: (ar * 100) + "%", borderRadius: "4px", overflow: "hidden", border: "1px solid var(--rule)", background: data.frame ? "#000" : "rgba(0,0,0,.3)", touchAction: "none", userSelect: "none", cursor: "default" } });
    if (data.frame) stage.appendChild(h("img", { src: data.frame, alt: "night yard", style: { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "cover", opacity: ".62" } }));
    const inter = h("div", { style: { position: "absolute", background: "rgba(201,162,75,.32)", border: "1px solid var(--gilt)", pointerEvents: "none" } });
    stage.appendChild(inter);

    const readout = h("div", { style: { display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap", marginTop: "14px" } });
    mount.append(
      h("p.muted", { style: { marginTop: "0", fontSize: "13.5px" } }, "Drag either box. The detector double-boxes one animal at high overlap; two separate bodies sit low."),
      stage, readout);

    function iou(a, b) {
      const ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
      const ix2 = Math.min(a.x + a.w, b.x + b.w), iy2 = Math.min(a.y + a.h, b.y + b.h);
      const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
      const I = iw * ih, U = a.w * a.h + b.w * b.h - I;
      return { iou: U > 0 ? I / U : 0, rect: iw > 0 && ih > 0 ? { x: ix1, y: iy1, w: iw, h: ih } : null };
    }

    const els = boxes.map((b) => {
      const el = h("div", { style: { position: "absolute", border: "2px solid " + b.c, borderRadius: "2px", cursor: "grab", boxShadow: "0 0 0 1px rgba(0,0,0,.5)" } });
      el.appendChild(h("div", { style: { position: "absolute", inset: "auto auto 100% 0", fontFamily: "var(--mono)", fontSize: "9px", color: "#16120a", background: b.c, padding: "0 4px", borderRadius: "2px", whiteSpace: "nowrap" } }, "animal"));
      stage.appendChild(el);
      return el;
    });

    let drag = null;
    function place() {
      const W = stage.clientWidth, H = stage.clientHeight;
      boxes.forEach((b, k) => { const e = els[k]; e.style.left = (b.x * W) + "px"; e.style.top = (b.y * H) + "px"; e.style.width = (b.w * W) + "px"; e.style.height = (b.h * H) + "px"; });
      const r = iou(boxes[0], boxes[1]);
      if (r.rect) { inter.style.display = "block"; inter.style.left = (r.rect.x * W) + "px"; inter.style.top = (r.rect.y * H) + "px"; inter.style.width = (r.rect.w * W) + "px"; inter.style.height = (r.rect.h * H) + "px"; }
      else inter.style.display = "none";
      renderReadout(r.iou);
    }
    function renderReadout(v) {
      const two = v < THR;
      readout.innerHTML = "";
      readout.append(
        h("div", null, [h("div.lbl", null, "intersection-over-union"), h("div", { style: { fontFamily: "var(--mono)", fontSize: "30px", color: two ? "var(--ok)" : "var(--rust)", lineHeight: "1" } }, v.toFixed(2))]),
        h("div", { style: { flex: "1", minWidth: "180px" } }, [
          h("div", { style: { fontVariant: "small-caps", fontSize: "19px", color: two ? "var(--ok)" : "var(--rust)" } }, two ? "✓ two raccoons — co-presence" : "one animal, double-boxed"),
          h("div.muted", { style: { fontSize: "12.5px", marginTop: "3px" } }, two
            ? "Below the 0.45 cut: two separate bodies. This visit gets the “2+ raccoons” badge — its blended template is set aside, and the clip un-blend (chapter 06) splits them."
            : "At or above 0.45 the boxes overlap too much to be two animals — it's the detector boxing one twice, so non-max suppression merges them."),
        ]));
    }

    function evt(e) { const r = stage.getBoundingClientRect(); const cx = (e.touches ? e.touches[0].clientX : e.clientX); const cy = (e.touches ? e.touches[0].clientY : e.clientY); return { x: (cx - r.left) / r.width, y: (cy - r.top) / r.height }; }
    function down(k) { return (e) => { drag = { k, off: null }; const p = evt(e); drag.off = { x: p.x - boxes[k].x, y: p.y - boxes[k].y }; els[k].style.cursor = "grabbing"; e.preventDefault(); }; }
    function move(e) { if (!drag) return; const p = evt(e); const b = boxes[drag.k]; b.x = clamp(p.x - drag.off.x, 0, 1 - b.w); b.y = clamp(p.y - drag.off.y, 0, 1 - b.h); place(); e.preventDefault(); }
    function up() { if (drag) els[drag.k].style.cursor = "grab"; drag = null; }
    els.forEach((e, k) => { e.addEventListener("mousedown", down(k)); e.addEventListener("touchstart", down(k), { passive: false }); });
    window.addEventListener("mousemove", move); window.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("mouseup", up); window.addEventListener("touchend", up);

    new ResizeObserver(place).observe(stage);
    requestAnimationFrame(place);
  });
})();
