/* Demo 2 — the decoy bench. BioCLIP can't say "no"; the general-CLIP gate can. */
(function () {
  const M = window.MakingOf, h = M.h;

  M.register("decoy", function (mount, data) {
    const tray = data.tray || [];
    const thr0 = data.threshold || 0.6;
    const dist = data.dist;
    let sel = 0, thr = thr0;

    // Category derived from the real BioCLIP label, so the tray never contradicts the gate:
    //   animal   — a confident real species (gate passes)
    //   uncertain — BioCLIP's weak catch-all 'brown rat' calls (gate passes; it still commits)
    //   nonanimal — genuine 'not an animal' crops the gate rejected
    const CATCOL = { animal: "var(--ok)", uncertain: "var(--gilt)", nonanimal: "var(--rust)" };
    const CATTIP = { animal: "a real animal", uncertain: "BioCLIP's unsure call", nonanimal: "not an animal" };
    const cat = (t) => t.bioclip_species === "not an animal" ? "nonanimal"
      : (t.bioclip_species === "brown rat" ? "uncertain" : "animal");

    // ---- the tray of clickable crops ----
    const trayEl = h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "16px" } });
    tray.forEach((t, i) => {
      const c = cat(t);
      const cell = h("button", {
        onclick: () => { sel = i; render(); },
        title: CATTIP[c],
        style: { padding: "0", border: "1px solid var(--rule2)", borderRadius: "4px", overflow: "hidden", cursor: "pointer", background: "none", position: "relative", lineHeight: "0" },
      }, [
        h("img", { src: t.crop, alt: "", style: { width: "62px", height: "62px", objectFit: "cover", display: "block" } }),
        h("span", { style: { position: "absolute", top: "3px", right: "3px", width: "9px", height: "9px", borderRadius: "50%", background: CATCOL[c], boxShadow: "0 0 0 1px #000" } }),
      ]);
      trayEl.appendChild(cell);
    });

    const detail = h("div", { style: { display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: "16px", alignItems: "start", padding: "14px", border: "1px solid var(--rule)", borderRadius: "4px", background: "rgba(0,0,0,.16)" } });
    const stripWrap = h("div", { style: { marginTop: "20px" } });

    mount.append(
      h("p.muted", { style: { marginTop: "0", fontSize: "13.5px" } },
        "Pick a crop. Green = a real animal · amber = a call BioCLIP wasn't sure of · red = not an animal at all."),
      trayEl, detail, stripWrap);

    function gateMeter(p, rejected) {
      const wrap = h("div");
      wrap.appendChild(h("div.lbl", { style: { marginBottom: "8px" } }, "① is it an animal?"));
      const track = h("div", { style: { position: "relative", height: "10px", background: "linear-gradient(90deg, rgba(134,171,93,.25), rgba(192,95,62,.3))", borderRadius: "5px" } });
      track.appendChild(h("div", { style: { position: "absolute", left: (thr * 100) + "%", top: "-4px", bottom: "-4px", width: "2px", background: "var(--gilt)" } }));
      if (p != null) track.appendChild(h("div", { style: { position: "absolute", left: "calc(" + (p * 100) + "% - 6px)", top: "-3px", width: "12px", height: "12px", borderRadius: "50%", background: "var(--ink)", border: "2px solid #100d07" } }));
      wrap.appendChild(track);
      wrap.appendChild(h("div", { style: { display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: "10px", color: "var(--faint)", marginTop: "4px" } }, [h("span", null, "animal"), h("span", null, "not an animal")]));
      wrap.appendChild(h("div", { style: { marginTop: "10px", fontFamily: "var(--mono)", fontSize: "12px" } },
        p == null ? [h("span.muted", null, "gate score not available")]
          : [h("span", { style: { color: "var(--dim)" } }, "p(not animal) = "), h("span", { style: { color: "var(--gilt)" } }, p.toFixed(3))]));
      wrap.appendChild(h("div", { style: { marginTop: "6px", fontVariant: "small-caps", fontSize: "16px", letterSpacing: ".02em", color: rejected ? "var(--rust)" : "var(--ok)" } },
        rejected ? "✗ rejected — not an animal" : "✓ passes — send to BioCLIP"));
      return wrap;
    }

    function speciesPanel(t, rejected) {
      const wrap = h("div");
      wrap.appendChild(h("div.lbl", { style: { marginBottom: "8px" } }, "② what species? (BioCLIP)"));
      if (rejected) {
        wrap.appendChild(h("div", { style: { fontFamily: "var(--mono)", fontSize: "12px", color: "var(--faint)", fontStyle: "italic", padding: "8px 0" } }, "skipped — BioCLIP never sees it."));
        wrap.appendChild(h("p.muted", { style: { fontSize: "12px", marginTop: "4px" } }, "The gate caught it first, so it's never force-labelled a species."));
        return wrap;
      }
      const v = t.bioclip_confidence || 0;
      wrap.appendChild(h("div", { style: { display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: "12px", marginBottom: "4px" } },
        [h("span", { style: { color: "var(--ink)" } }, t.bioclip_species), h("span", { style: { color: "var(--gilt)" } }, v.toFixed(2))]));
      wrap.appendChild(h("div", { style: { height: "9px", background: "var(--rule)", borderRadius: "3px", overflow: "hidden" } },
        [h("div", { style: { width: (v * 100) + "%", height: "100%", background: "var(--gilt)" } })]));
      const note = cat(t) === "uncertain"
        ? "Even BioCLIP's weakest call still commits to a species — it has no “none of the above”. That's the blind spot the gate covers."
        : "Confident — because there really is one.";
      wrap.appendChild(h("p.muted", { style: { fontSize: "12px", marginTop: "10px" } }, note));
      return wrap;
    }

    function render() {
      [...trayEl.children].forEach((c, i) => { c.style.borderColor = i === sel ? "var(--gilt)" : "var(--rule2)"; c.style.boxShadow = i === sel ? "0 0 0 2px rgba(201,162,75,.4)" : "none"; });
      const t = tray[sel];
      const p = t.gate_p_nonanimal;
      const rejected = p != null ? p >= thr : t.truth === "nonanimal";
      detail.innerHTML = "";
      detail.appendChild(h("img", { src: t.crop, alt: "selected crop", style: { width: "120px", borderRadius: "3px", border: "1px solid var(--rule2)" } }));
      detail.appendChild(gateMeter(p, rejected));
      detail.appendChild(speciesPanel(t, rejected));
      renderStrip();
    }

    function renderStrip() {
      stripWrap.innerHTML = "";
      if (!dist || !dist.animal_pnon) {
        stripWrap.appendChild(h("p.muted", { style: { fontSize: "12.5px" } }, "Run the exporter with --score to see the full separation."));
        return;
      }
      const A = dist.animal_pnon, N = dist.nonanimal_pnon;
      stripWrap.appendChild(h("div.lbl", { style: { marginBottom: "10px" } }, "the gate across " + (A.length + N.length) + " more real crops"));

      const lane = (vals, color, label) => {
        const row = h("div", { style: { display: "flex", alignItems: "center", gap: "10px", margin: "6px 0" } });
        row.appendChild(h("span", { style: { width: "84px", textAlign: "right", fontFamily: "var(--mono)", fontSize: "10px", color: "var(--faint)" } }, label));
        const strip = h("div", { style: { position: "relative", flex: "1", height: "26px", borderRadius: "4px", background: "rgba(0,0,0,.22)", border: "1px solid var(--rule)" } });
        vals.forEach((v, i) => {
          const jitter = 4 + ((i * 37) % 16);
          strip.appendChild(h("div", { style: { position: "absolute", left: (v * 100) + "%", top: jitter + "%", width: "5px", height: "5px", marginLeft: "-2px", borderRadius: "50%", background: color, opacity: ".75" } }));
        });
        strip.appendChild(h("div", { style: { position: "absolute", left: (thr * 100) + "%", top: "-3px", bottom: "-3px", width: "2px", background: "var(--gilt)", zIndex: "2" } }));
        row.appendChild(strip);
        return row;
      };
      stripWrap.appendChild(lane(A, "var(--ok)", "real animals"));
      stripWrap.appendChild(lane(N, "var(--rust)", "not animals"));

      const sliderRow = h("div", { style: { display: "flex", alignItems: "center", gap: "12px", margin: "14px 0 8px", paddingLeft: "94px" } });
      const slider = h("input", { type: "range", min: "0", max: "1", step: "0.01", value: String(thr), style: { flex: "1" },
        oninput: (e) => { thr = parseFloat(e.target.value); render(); } });
      sliderRow.append(h("span", { style: { fontFamily: "var(--mono)", fontSize: "11px", color: "var(--dim)" } }, "threshold"), slider, h("span.val", null, thr.toFixed(2)));
      stripWrap.appendChild(sliderRow);

      const wrongAnimals = A.filter((v) => v >= thr).length;
      const caughtNon = N.filter((v) => v >= thr).length;
      const tally = h("div", { style: { display: "flex", gap: "24px", flexWrap: "wrap", paddingLeft: "94px", fontFamily: "var(--mono)", fontSize: "12px" } }, [
        h("span", null, [h("span", { style: { color: "var(--rust)" } }, wrongAnimals + "/" + A.length), h("span", { style: { color: "var(--faint)" } }, " animals wrongly rejected")]),
        h("span", null, [h("span", { style: { color: "var(--ok)" } }, caughtNon + "/" + N.length), h("span", { style: { color: "var(--faint)" } }, " non-animals caught")]),
      ]);
      stripWrap.appendChild(tally);
      stripWrap.appendChild(h("p.muted", { style: { fontSize: "12.5px", marginTop: "10px" } }, "The two clouds barely touch — that wide gap is why one fixed cut at 0.60 holds."));
    }

    const mq = window.matchMedia("(max-width:560px)");
    const apply = () => { detail.style.gridTemplateColumns = mq.matches ? "1fr" : "auto 1fr 1fr"; };
    mq.addEventListener("change", apply); apply();
    render();
  });
})();
