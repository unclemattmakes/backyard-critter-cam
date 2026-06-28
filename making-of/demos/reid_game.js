/* Demo 4 — you be the re-ID. Guess against the model's hunch; finale un-blends a pair visit. */
(function () {
  const M = window.MakingOf, h = M.h;

  M.register("reid_game", function (mount, data) {
    const rounds = (data.rounds || []).filter((r) => r.crops && r.crops.length);
    const finale = data.finale;
    const CHOICES = ["Stan", "Notch", "Elliot", "Someone new"];
    let ri = -1, you = 0, model = 0, locked = false, picked = null;

    const view = h("div");
    mount.appendChild(view);

    function strip(crops, size) {
      size = size || 90;
      return h("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap", margin: "4px 0 0" } },
        crops.map((c) => h("img", { src: c, alt: "", style: { width: size + "px", height: size + "px", objectFit: "cover", borderRadius: "3px", border: "1px solid var(--rule2)" } })));
    }
    function fmtDate(ts) { return (ts || "").slice(5, 16).replace("T", " "); }

    function renderIntro() {
      view.innerHTML = "";
      view.append(
        h("div", { style: { fontVariant: "small-caps", fontSize: "22px", letterSpacing: ".02em", marginBottom: "8px" } }, "Same, or someone new?"),
        h("p", { style: { color: "var(--dim)", margin: "0 0 16px", fontSize: "15px" } },
          ["The system never names a raccoon on its own — it shows its best guess and waits for you. Here are ",
            h("b", null, String(rounds.length)), " real visits. Beat the machine."]),
        h("button.btn.on", { onclick: () => { ri = 0; renderRound(); } }, "Start ▸"));
    }

    function renderRound() {
      locked = false; picked = null;
      const r = rounds[ri];
      const top = r.candidates && r.candidates[0];
      view.innerHTML = "";

      const head = h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "10px" } }, [
        h("span.lbl", null, "Round " + (ri + 1) + " / " + rounds.length),
        h("span", { style: { fontFamily: "var(--mono)", fontSize: "12px" } }, [h("span.gilt", null, "you " + you), h("span.muted", { style: { color: "var(--faint)" } }, "  ·  model " + model)]),
      ]);
      view.appendChild(head);
      view.appendChild(h("div.muted", { style: { fontFamily: "var(--mono)", fontSize: "11px", marginBottom: "2px" } }, "a raccoon visit · " + fmtDate(r.started) + " · " + r.n_crops + " crops"));
      view.appendChild(strip(r.crops));

      const hunch = h("div", { style: { margin: "16px 0 8px", padding: "10px 14px", borderLeft: "2px solid var(--gilt)", background: "rgba(201,162,75,.06)" } });
      if (r.novel) hunch.append(h("div", { style: { fontStyle: "italic", color: "var(--ink)" } }, "“Nearest is " + (top ? top.name + " " + top.sim.toFixed(2) : "—") + ", but that's below the 0.55 bar — possibly someone new.”"));
      else hunch.append(h("div", { style: { fontStyle: "italic", color: "var(--ink)" } }, "“Looks like " + (top ? top.name : "—") + " (" + (top ? top.sim.toFixed(2) : "—") + ").”"));
      hunch.append(h("div.lbl", { style: { marginTop: "6px" } }, "the model's hunch"));
      view.appendChild(hunch);

      view.appendChild(h("div.lbl", { style: { margin: "14px 0 8px" } }, "your call"));
      const btns = h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap" } });
      CHOICES.forEach((c) => btns.appendChild(h("button.btn", { onclick: () => pick(c) },
        c === "Someone new" ? c : [h("span.swatch", { style: { background: M.castColor(c) } }), c])));
      view.appendChild(btns);
      view.appendChild(h("div", { id: "reveal" }));
    }

    function pick(choice) {
      if (locked) return;
      locked = true; picked = choice;
      const r = rounds[ri];
      const top = r.candidates && r.candidates[0];
      const modelGuess = r.novel ? "Someone new" : (top ? top.name : "Someone new");
      const youRight = choice === r.truth;
      const modelRight = modelGuess === r.truth;
      if (youRight) you++; if (modelRight) model++;

      const rev = view.querySelector("#reveal");
      rev.innerHTML = "";
      rev.style.marginTop = "16px";
      rev.appendChild(h("div", { style: { display: "flex", gap: "18px", flexWrap: "wrap", alignItems: "baseline" } }, [
        h("div", null, [h("span.lbl", null, "truth "), h("span", { style: { fontVariant: "small-caps", fontSize: "20px", color: M.castColor(r.truth) } }, r.truth)]),
        verdict("you said " + choice, youRight),
        verdict("model said " + modelGuess, modelRight),
      ]));
      let note = "";
      if (r.truth === "Elliot") note = "Elliot almost never visits alone, so he has barely any solo template — the model leans toward the regulars. This is the case the un-blend trick below was built for.";
      else if (!modelRight) note = "A real miss: on a busy night the appearance signal blends two animals, and the nearest template wins. Behaviour (chapter 05) is the tie-breaker.";
      else note = "Confirmed. Each confirmation becomes a template, so the next suggestion is a little sharper.";
      rev.appendChild(h("p.muted", { style: { fontSize: "13px", marginTop: "10px" } }, note));

      const moreRounds = ri < rounds.length - 1;
      rev.appendChild(h("button.btn.on", { style: { marginTop: "6px" }, onclick: () => { if (moreRounds) { ri++; renderRound(); } else renderFinale(); } },
        moreRounds ? "Next visit ▸" : "The bonus round ▸"));
    }

    function verdict(label, ok) {
      return h("div", { style: { fontFamily: "var(--mono)", fontSize: "12px", color: ok ? "var(--ok)" : "var(--rust)" } }, (ok ? "✓ " : "✗ ") + label);
    }

    function renderFinale() {
      view.innerHTML = "";
      view.append(
        h("div", { style: { fontVariant: "small-caps", fontSize: "22px", letterSpacing: ".02em", marginBottom: "4px" } }, "The never-solo raccoon"),
        h("div", { style: { fontFamily: "var(--mono)", fontSize: "12px", marginBottom: "12px" } }, [h("span.gilt", null, "Final score — you " + you), h("span.muted", { style: { color: "var(--faint)" } }, " · model " + model)]));
      if (!finale || !finale.groups || finale.groups.length < 2) {
        view.appendChild(h("p.muted", null, "No un-blend example was exported.")); return;
      }
      view.appendChild(h("p", { style: { color: "var(--dim)", margin: "0 0 14px", fontSize: "15px" } },
        ["Some raccoons only ever turn up ", h("span", { style: { fontStyle: "italic", color: "var(--ink)" } }, "with"), " another — so you never get a clean solo photo to learn their face from. But a behaviour clip tracks each animal ",
          h("span", { style: { fontStyle: "italic", color: "var(--ink)" } }, "separately"), ". This one real visit holds ", h("b", null, String(finale.n_tracklets)), " motion tracklets, tangled together."]));

      const splitBtn = h("button.btn.on", { onclick: doSplit }, "Un-blend from the clips ▸");
      const out = h("div", { style: { marginTop: "16px" } });
      view.append(splitBtn, out);

      function doSplit() {
        splitBtn.style.display = "none";
        out.innerHTML = "";
        const grid = h("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" } });
        finale.groups.slice(0, 2).forEach((g, i) => {
          const name = g.label || (g.suggestion && g.suggestion.name) || ("Animal " + String.fromCharCode(65 + i));
          const card = h("div", { style: { border: "1px solid var(--rule2)", borderRadius: "4px", padding: "12px", background: "rgba(0,0,0,.16)" } });
          card.append(
            h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "8px" } }, [
              h("span", { style: { fontVariant: "small-caps", fontSize: "18px" } }, name),
              h("span.val", null, g.n + " frames")]),
            strip(g.rep_crops || [], 64),
            h("div.muted", { style: { fontFamily: "var(--mono)", fontSize: "10px", marginTop: "8px" } }, "cohesion " + (g.cohesion != null ? g.cohesion.toFixed(2) : "—") + (g.suggestion ? " · looks like " + g.suggestion.name : "")));
          grid.appendChild(card);
        });
        out.appendChild(grid);
        const sizes = finale.groups.slice(0, 2).map((g) => g.n).join(" and ");
        out.appendChild(h("p.muted", { style: { fontSize: "13.5px", marginTop: "14px" } },
          "Two distinct animals — " + sizes + " frames — pulled out of one blurry crowd. Match either cluster to a known raccoon and the other is named by elimination. That's how a raccoon who's never photographed alone finally earns an appearance template of his own."));
        out.appendChild(h("button.btn", { style: { marginTop: "10px" }, onclick: () => { ri = -1; you = 0; model = 0; renderIntro(); } }, "↻ Play again"));
      }
      const mq = window.matchMedia("(max-width:520px)");
      const apply = () => { const g = out.querySelector("div"); if (g) g.style.gridTemplateColumns = mq.matches ? "1fr" : "1fr 1fr"; };
      mq.addEventListener("change", apply);
    }

    if (!rounds.length) { mount.appendChild(h("p.muted", null, "No re-ID rounds exported.")); return; }
    renderIntro();
  });
})();
