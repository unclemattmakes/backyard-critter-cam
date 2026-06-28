/* Demo 1 — the pipeline scrubber. Step one real capture through every stage of the rig. */
(function () {
  const M = window.MakingOf, h = M.h;

  M.register("pipeline", function (mount, data) {
    const samples = (data.samples || []).filter((s) => s.frames && s.frames.frame);
    if (!samples.length) { mount.appendChild(h("p.muted", null, "No pipeline samples exported.")); return; }
    let si = 0, stage = 0;

    const STAGES = [
      { tag: "watch", label: "Frame", title: "A frame arrives",
        note: "The camera streams ~30 fps. A cheap MOG2 background-subtraction gate watches every frame for movement — and the expensive GPU detector stays <b>asleep</b>." },
      { tag: "motion", label: "Motion", title: "Motion trips the gate",
        note: "Movement lights up the motion mask. Once the largest blob clears the area threshold, and only then, the rig <b>wakes the detector</b>." },
      { tag: "detect", label: "Detect", title: "MegaDetector fires",
        note: (s) => `MegaDetector v6 (via Ultralytics) returns one box: <b>animal</b>, confidence <span class="val">${s.confidence.toFixed(2)}</span>. Street traffic and people are filtered out before this.` },
      { tag: "crop", label: "Crop", title: "Crop & score",
        note: (s) => `The box is cropped out and scored for shot quality — sharpness, with a bump for night eyeshine — so the gallery can lead with the cutest frame. Quality <span class="val">${Math.round(s.crop_quality)}</span>.` },
      { tag: "id", label: "Identify", title: "Gate, then name",
        note: (s) => `The general-CLIP gate confirms it's an animal, then BioCLIP 2 names the species: <b>${s.species}</b> <span class="val">${(s.species_confidence||0).toFixed(2)}</span>. (Individual ID comes later — chapters 03–04.)` },
      { tag: "log", label: "Log", title: "One row in SQLite",
        note: "Everything lands as a single database row — the spine the whole system reads back. A behaviour clip is recorded around the visit too." },
    ];

    const stepper = h("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "14px" } });
    const stageWrap = h("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1.5fr) minmax(0,1fr)", gap: "18px", alignItems: "start" } });
    const visual = h("div");
    const info = h("div");
    const nav = h("div", { style: { display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "16px", gap: "10px" } });

    const prev = h("button.btn", { onclick: () => { stage = Math.max(0, stage - 1); render(); } }, "‹ Back");
    const next = h("button.btn", { onclick: () => { stage = Math.min(STAGES.length - 1, stage + 1); render(); } }, "Next ›");
    const cyc = h("button.btn", { onclick: () => { si = (si + 1) % samples.length; stage = 0; render(); } });
    nav.append(prev, h("div", { style: { display: "flex", gap: "8px" } }, [cyc]), next);

    mount.append(stepper, stageWrap, nav);
    stageWrap.append(visual, info);

    function frameBox(s, opts) {
      opts = opts || {};
      const f = s.frames;
      const box = h("div", { style: { position: "relative", borderRadius: "3px", overflow: "hidden", border: "1px solid var(--rule2)", background: "#000", aspectRatio: (f.w && f.h) ? `${f.w}/${f.h}` : "16/9" } });
      const src = opts.empty && f.empty ? f.empty : f.frame;
      box.appendChild(h("img", { src, alt: "frame", style: { display: "block", width: "100%", height: "100%", objectFit: "cover", opacity: opts.dim ? "0.78" : "1" } }));
      if (opts.mask && f.mask) {
        box.appendChild(h("img", { src: f.mask, alt: "motion", style: { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "cover", mixBlendMode: "screen", opacity: ".75" } }));
      }
      if (opts.bbox && s.bbox) {
        const [x1, y1, x2, y2] = s.bbox;
        box.appendChild(h("div", { style: { position: "absolute", left: (x1 * 100) + "%", top: (y1 * 100) + "%", width: ((x2 - x1) * 100) + "%", height: ((y2 - y1) * 100) + "%", border: "2px solid var(--gilt)", borderRadius: "2px", boxShadow: "0 0 0 1px rgba(0,0,0,.5)" } }));
        box.appendChild(h("div", { style: { position: "absolute", left: (x1 * 100) + "%", top: "calc(" + (y1 * 100) + "% - 18px)", background: "var(--gilt)", color: "#16120a", fontFamily: "var(--mono)", fontSize: "10px", padding: "1px 5px", borderRadius: "2px" } }, `animal ${s.confidence.toFixed(2)}`));
      }
      return box;
    }

    function detectorBadge(awake) {
      return h("div", { style: { display: "inline-flex", alignItems: "center", gap: "7px", fontFamily: "var(--mono)", fontSize: "10px", letterSpacing: ".14em", textTransform: "uppercase", color: awake ? "var(--rust)" : "var(--faint)", marginBottom: "10px" } },
        [h("span", { style: { width: "8px", height: "8px", borderRadius: "50%", background: awake ? "var(--rust)" : "var(--rule2)", boxShadow: awake ? "0 0 8px var(--rust)" : "none" } }), awake ? "GPU detector — awake" : "GPU detector — asleep"]);
    }

    function dbRow(s) {
      const t = (s.timestamp || "").slice(11, 19);
      const rows = [
        ["id", s.id], ["time", t], ["class", "animal"], ["confidence", s.confidence.toFixed(3)],
        ["species", s.species], ["species_conf", (s.species_confidence || 0).toFixed(3)],
        ["crop_quality", Math.round(s.crop_quality)], ["source", "glass_door_cam"],
      ];
      const tbl = h("table", { style: { width: "100%", borderCollapse: "collapse", fontFamily: "var(--mono)", fontSize: "12px" } });
      rows.forEach(([k, v]) => tbl.appendChild(h("tr", null, [
        h("td", { style: { color: "var(--faint)", padding: "4px 10px 4px 0", whiteSpace: "nowrap" } }, k),
        h("td", { style: { color: "var(--ink)", padding: "4px 0", borderBottom: "1px dotted var(--rule)" } }, String(v)),
      ])));
      return h("div", { style: { border: "1px solid var(--rule2)", borderRadius: "4px", padding: "12px 14px", background: "rgba(0,0,0,.18)" } },
        [h("div.lbl", { style: { marginBottom: "8px" } }, "INSERT INTO detections"), tbl]);
    }

    function render() {
      const s = samples[si];
      cyc.textContent = samples.length > 1 ? `capture ${si + 1} / ${samples.length} ↻` : "Stan";
      // stepper
      stepper.innerHTML = "";
      STAGES.forEach((st, i) => {
        const b = h("button", {
          onclick: () => { stage = i; render(); },
          style: {
            fontFamily: "var(--mono)", fontSize: "10px", letterSpacing: ".1em", textTransform: "uppercase",
            cursor: "pointer", borderRadius: "3px", padding: "6px 9px", transition: "all .15s",
            border: "1px solid " + (i === stage ? "var(--gilt)" : "var(--rule2)"),
            background: i === stage ? "var(--gilt)" : (i < stage ? "rgba(201,162,75,.12)" : "none"),
            color: i === stage ? "#16120a" : (i < stage ? "var(--gilt)" : "var(--dim)"),
          },
        }, `${i}·${st.label}`);
        stepper.appendChild(b);
      });
      // visual
      visual.innerHTML = "";
      const st = STAGES[stage];
      if (st.tag === "watch") visual.appendChild(frameBox(s, { empty: true }));
      else if (st.tag === "motion") visual.appendChild(frameBox(s, { mask: true, dim: true }));
      else if (st.tag === "detect") visual.appendChild(frameBox(s, { bbox: true }));
      else if (st.tag === "crop" || st.tag === "id") {
        const row = h("div", { style: { display: "flex", gap: "14px", alignItems: "flex-start" } });
        row.appendChild(h("img", { src: s.crop, alt: "crop", style: { width: "150px", borderRadius: "3px", border: "1px solid var(--rule2)" } }));
        if (st.tag === "id") {
          const bars = h("div", { style: { flex: "1", minWidth: "0" } });
          bars.appendChild(gateRow());
          bars.appendChild(bar(s.species, s.species_confidence || 0, "var(--gilt)"));
          bars.appendChild(h("div.muted", { style: { fontSize: "12px", marginTop: "8px" } }, "BioCLIP is forced-choice — it would never volunteer “not an animal,” which is why the gate runs first."));
          row.appendChild(bars);
        }
        visual.appendChild(row);
      } else if (st.tag === "log") visual.appendChild(dbRow(s));

      // info
      info.innerHTML = "";
      info.appendChild(detectorBadge(stage >= 1 && stage <= 4));
      info.appendChild(h("div", { style: { fontFamily: "var(--serif)", fontVariant: "small-caps", fontSize: "21px", letterSpacing: ".02em", marginBottom: "8px" } }, st.title));
      info.appendChild(h("p", { style: { color: "var(--dim)", margin: "0", fontSize: "15px" }, html: typeof st.note === "function" ? st.note(s) : st.note }));

      prev.disabled = stage === 0; next.disabled = stage === STAGES.length - 1;
      prev.style.opacity = stage === 0 ? ".4" : "1"; next.style.opacity = stage === STAGES.length - 1 ? ".4" : "1";
    }

    function gateRow() {
      return h("div", { style: { display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px", fontFamily: "var(--mono)", fontSize: "12px", color: "var(--sage)" } },
        [h("span", { style: { color: "var(--ok)" } }, "✓"), "CLIP gate: is this an animal? — yes"]);
    }

    function bar(label, v, color) {
      const wrap = h("div", { style: { margin: "4px 0" } });
      wrap.appendChild(h("div", { style: { display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: "11px", marginBottom: "3px" } },
        [h("span", { style: { color: "var(--ink)" } }, label), h("span", { style: { color: "var(--gilt)" } }, v.toFixed(2))]));
      wrap.appendChild(h("div", { style: { height: "8px", background: "var(--rule)", borderRadius: "3px", overflow: "hidden" } },
        [h("div", { style: { width: (v * 100) + "%", height: "100%", background: color } })]));
      return wrap;
    }

    render();
    // responsive: single column on narrow
    const mq = window.matchMedia("(max-width:560px)");
    const apply = () => { stageWrap.style.gridTemplateColumns = mq.matches ? "1fr" : "minmax(0,1.5fr) minmax(0,1fr)"; };
    mq.addEventListener("change", apply); apply();
  });
})();
