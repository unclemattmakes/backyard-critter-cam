/* Demo 5 — the two-axis dial. Looks like Notch × acts like Notch; the disagreement is the alert. */
(function () {
  const M = window.MakingOf, h = M.h, clamp = M.clamp;

  M.register("twoaxis", function (mount, data) {
    const prof = data.notch || data.raccoon;
    const win = prof && prof.typical_window;
    const NOVEL = data.novel_threshold || 0.55;
    const examples = (data.examples || []).filter((e) => e.hour != null);
    const subject = data.notch ? "Notch" : "a raccoon";
    if (!win) { mount.appendChild(h("p.muted", null, "No behaviour profile exported.")); return; }

    const liveApp = 0.88;       // "this visit looks like Notch" — Notch's real cross-night self-match
    let hour = 14;              // start mid-afternoon: deliberately off-pattern

    function inWindow(hr) { const s = win.start_hour, e = win.end_hour; return s <= e ? (hr >= s && hr <= e) : (hr >= s || hr <= e); }
    function circDist(a, b) { let d = Math.abs(a - b) % 24; return Math.min(d, 24 - d); }
    function behaviourFit(hr) {
      if (inWindow(hr)) return 1;
      const out = Math.min(circDist(hr, win.start_hour), circDist(hr, win.end_hour));
      return clamp(1 - out / 6, 0, 1);
    }
    function winText() { return String(win.start_hour).padStart(2, "0") + ":00–" + String(win.end_hour).padStart(2, "0") + ":00"; }

    // ---- layout ----
    const banner = h("div", { style: { padding: "12px 16px", borderRadius: "4px", marginBottom: "16px", border: "1px solid var(--rule2)" } });
    const clockC = h("canvas", { style: { width: "190px", height: "190px", touchAction: "none", cursor: "pointer" } });
    const planeC = h("canvas", { style: { width: "100%", height: "320px", borderRadius: "4px", background: "rgba(0,0,0,.22)", border: "1px solid var(--rule)" } });
    const stage = h("div", { style: { display: "flex", gap: "18px", alignItems: "center", flexWrap: "wrap" } }, [
      h("div", { style: { flex: "0 0 auto", textAlign: "center" } }, [clockC, h("div.lbl", { style: { marginTop: "4px" } }, "drag the arrival time")]),
      h("div", { style: { flex: "1", minWidth: "240px" } }, planeC),
    ]);

    const sliderRow = h("div", { style: { display: "flex", alignItems: "center", gap: "12px", marginTop: "14px" } }, [
      h("span", { style: { fontFamily: "var(--mono)", fontSize: "11px", color: "var(--dim)" } }, "arrival"),
    ]);
    const slider = h("input", { type: "range", min: "0", max: "23", step: "1", value: String(hour), style: { flex: "1" }, oninput: (e) => { hour = +e.target.value; render(); } });
    const hourOut = h("span.val", { style: { minWidth: "52px" } });
    sliderRow.append(slider, hourOut);

    mount.append(banner, stage, sliderRow);

    // ---- clock ----
    const cctx = clockC.getContext("2d"), cdpr = Math.min(window.devicePixelRatio || 1, 2);
    function sizeClock() { clockC.width = 190 * cdpr; clockC.height = 190 * cdpr; cctx.setTransform(cdpr, 0, 0, cdpr, 0, 0); }
    function ang(hr) { return (hr / 24) * Math.PI * 2 - Math.PI / 2; }
    function drawClock() {
      const cx = 95, cy = 95, R = 74;
      cctx.clearRect(0, 0, 190, 190);
      // window arc
      cctx.strokeStyle = "rgba(201,162,75,.85)"; cctx.lineWidth = 8; cctx.lineCap = "round";
      cctx.beginPath(); cctx.arc(cx, cy, R, ang(win.start_hour), ang(win.end_hour + (win.end_hour < win.start_hour ? 24 : 0))); cctx.stroke();
      // ticks
      cctx.strokeStyle = "var(--rule2)"; cctx.lineWidth = 1;
      for (let hr = 0; hr < 24; hr += 1) { const a = ang(hr), big = hr % 6 === 0; cctx.globalAlpha = big ? 1 : .5; cctx.beginPath(); cctx.moveTo(cx + Math.cos(a) * (R - 4), cy + Math.sin(a) * (R - 4)); cctx.lineTo(cx + Math.cos(a) * (R + (big ? 6 : 3)), cy + Math.sin(a) * (R + (big ? 6 : 3))); cctx.stroke(); }
      cctx.globalAlpha = 1;
      // labels 0/6/12/18
      cctx.fillStyle = "var(--faint)"; cctx.font = "10px " + getMono(); cctx.textAlign = "center"; cctx.textBaseline = "middle";
      [["0", 0], ["6", 6], ["12", 12], ["18", 18]].forEach(([t, hr]) => { const a = ang(hr); cctx.fillText(t, cx + Math.cos(a) * (R - 18), cy + Math.sin(a) * (R - 18)); });
      // hand
      const fit = behaviourFit(hour), inw = inWindow(hour);
      const a = ang(hour), col = inw ? "var(--ok)" : (fit < 0.4 ? "var(--rust)" : "var(--gilt)");
      cctx.strokeStyle = col; cctx.lineWidth = 2.5; cctx.beginPath(); cctx.moveTo(cx, cy); cctx.lineTo(cx + Math.cos(a) * (R - 8), cy + Math.sin(a) * (R - 8)); cctx.stroke();
      cctx.fillStyle = col; cctx.beginPath(); cctx.arc(cx + Math.cos(a) * (R - 8), cy + Math.sin(a) * (R - 8), 5, 0, 7); cctx.fill();
      cctx.fillStyle = "var(--ink)"; cctx.font = "600 18px " + getMono();
      cctx.fillText(String(hour).padStart(2, "0") + ":00", cx, cy);
    }
    function getMono() { return getComputedStyle(document.body).getPropertyValue("--mono") || "monospace"; }

    function clockHour(e) {
      const r = clockC.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left - 95;
      const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top - 95;
      let a = Math.atan2(y, x) + Math.PI / 2; if (a < 0) a += Math.PI * 2;
      return Math.round((a / (Math.PI * 2)) * 24) % 24;
    }
    let dragging = false;
    const onMove = (e) => { if (!dragging) return; hour = clockHour(e); render(); e.preventDefault(); };
    clockC.addEventListener("mousedown", (e) => { dragging = true; hour = clockHour(e); render(); });
    clockC.addEventListener("touchstart", (e) => { dragging = true; hour = clockHour(e); render(); });
    window.addEventListener("mousemove", onMove); window.addEventListener("touchmove", onMove, { passive: false });
    window.addEventListener("mouseup", () => { dragging = false; }); window.addEventListener("touchend", () => { dragging = false; });

    // ---- plane ----
    const pctx = planeC.getContext("2d"), pdpr = Math.min(window.devicePixelRatio || 1, 2);
    let PW = 0, PH = 320, pad = 34;
    function sizePlane() { PW = planeC.clientWidth || 360; planeC.width = PW * pdpr; planeC.height = PH * pdpr; pctx.setTransform(pdpr, 0, 0, pdpr, 0, 0); }
    const X = (app) => pad + app * (PW - 2 * pad);
    const Y = (fit) => (PH - pad) - fit * (PH - 2 * pad);

    function drawPlane() {
      pctx.clearRect(0, 0, PW, PH);
      // zones
      const xT = X(NOVEL), yMid = Y(0.5);
      pctx.fillStyle = "rgba(192,95,62,.10)"; pctx.fillRect(xT, pad, PW - pad - xT, yMid - pad);   // high-app low-fit = alert
      pctx.fillStyle = "rgba(148,163,107,.08)"; pctx.fillRect(xT, yMid, PW - pad - xT, (PH - pad) - yMid); // agree
      pctx.fillStyle = "var(--faint)"; pctx.font = "10px " + getMono(); pctx.textAlign = "left";
      pctx.fillText("⚠ looks like, doesn't act like", xT + 6, pad + 14);
      pctx.fillStyle = "var(--sage)"; pctx.fillText("✓ agrees — quiet", xT + 6, PH - pad - 8);
      // axes
      pctx.strokeStyle = "var(--rule2)"; pctx.lineWidth = 1;
      pctx.beginPath(); pctx.moveTo(pad, pad); pctx.lineTo(pad, PH - pad); pctx.lineTo(PW - pad, PH - pad); pctx.stroke();
      pctx.setLineDash([3, 3]); pctx.strokeStyle = "rgba(201,162,75,.4)";
      pctx.beginPath(); pctx.moveTo(xT, pad); pctx.lineTo(xT, PH - pad); pctx.stroke(); pctx.setLineDash([]);
      pctx.fillStyle = "var(--dim)"; pctx.textAlign = "center";
      pctx.fillText("looks like " + subject + " →", PW / 2, PH - 10);
      pctx.save(); pctx.translate(12, PH / 2); pctx.rotate(-Math.PI / 2); pctx.fillText("↑ acts like " + subject, 0, 0); pctx.restore();
      // example dots
      examples.forEach((e) => {
        const f = behaviourFit(e.hour);
        pctx.globalAlpha = .8; pctx.fillStyle = e.truth ? M.castColor(e.truth) : "var(--unknown)";
        pctx.beginPath(); pctx.arc(X(e.appearance), Y(f), 4, 0, 7); pctx.fill();
      });
      pctx.globalAlpha = 1;
      // live dot
      const f = behaviourFit(hour);
      const lx = X(liveApp), ly = Y(f), alert = liveApp >= NOVEL && f < 0.5;
      pctx.strokeStyle = "var(--ink)"; pctx.lineWidth = 1; pctx.setLineDash([2, 2]);
      pctx.beginPath(); pctx.moveTo(lx, PH - pad); pctx.lineTo(lx, ly); pctx.moveTo(pad, ly); pctx.lineTo(lx, ly); pctx.stroke(); pctx.setLineDash([]);
      pctx.fillStyle = alert ? "var(--rust)" : "var(--gilt)";
      pctx.beginPath(); pctx.arc(lx, ly, 8, 0, 7); pctx.fill();
      pctx.strokeStyle = "#fff"; pctx.lineWidth = 2; pctx.stroke();
      pctx.fillStyle = "var(--ink)"; pctx.font = "11px " + getMono(); pctx.textAlign = "left";
      pctx.fillText("this visit", lx + 12, ly - 8);
    }

    function render() {
      hour = ((hour % 24) + 24) % 24;
      slider.value = String(hour);
      hourOut.textContent = String(hour).padStart(2, "0") + ":00";
      drawClock(); drawPlane();
      const fit = behaviourFit(hour), inw = inWindow(hour), alert = liveApp >= NOVEL && fit < 0.5;
      banner.style.borderColor = alert ? "var(--rust)" : "var(--rule2)";
      banner.style.background = alert ? "rgba(192,95,62,.10)" : "rgba(148,163,107,.07)";
      banner.innerHTML = "";
      banner.append(
        h("div", { style: { fontVariant: "small-caps", fontSize: "17px", letterSpacing: ".02em", color: alert ? "var(--rust)" : "var(--ok)" } },
          alert ? "⚠ the two axes disagree" : "✓ the two axes agree"),
        h("div", { style: { fontSize: "14px", color: "var(--dim)", marginTop: "4px" } },
          alert
            ? ["Looks like ", h("b", { style: { color: "var(--ink)" } }, subject + " (" + liveApp.toFixed(2) + ")"), ", but arrived at ", h("b", { style: { color: "var(--ink)" } }, String(hour).padStart(2, "0") + ":00"), " — and " + subject + " runs " + winText() + ". A look-alike, or " + subject + " off-pattern?"]
            : ["Looks like ", h("b", { style: { color: "var(--ink)" } }, subject + " (" + liveApp.toFixed(2) + ")"), " and arrived ", h("b", { style: { color: "var(--ink)" } }, String(hour).padStart(2, "0") + ":00"), " — squarely in the usual " + winText() + " window. Nothing to flag."]));
    }

    function sizeAll() { sizeClock(); sizePlane(); render(); }
    new ResizeObserver(sizePlane).observe(planeC);
    new ResizeObserver(render).observe(stage);
    sizeAll();
  });
})();
