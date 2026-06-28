/* Bonus demo — how it moves. A metronome ticking at each tracklet's real stride cadence. */
(function () {
  const M = window.MakingOf, h = M.h;

  M.register("gait", function (mount, data) {
    const tracks = data.tracks || [];
    if (!tracks.length) { mount.appendChild(h("p.muted", null, "No gait estimates exported.")); return; }
    let sel = Math.min(2, tracks.length - 1), playing = true, raf = null, beats = 0, lastPhase = 0;

    // tracklet picker
    const picker = h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "16px" } });
    tracks.forEach((t, i) => {
      const c = h("button", { onclick: () => { sel = i; beats = 0; render(); }, style: { padding: "0", border: "1px solid var(--rule2)", borderRadius: "4px", overflow: "hidden", cursor: "pointer", background: "none", position: "relative", lineHeight: "0" } }, [
        t.crop ? h("img", { src: t.crop, alt: "", style: { width: "58px", height: "58px", objectFit: "cover", display: "block" } }) : h("div", { style: { width: "58px", height: "58px", background: "var(--panel)" } }),
        h("span", { style: { position: "absolute", bottom: "0", left: "0", right: "0", fontFamily: "var(--mono)", fontSize: "9px", color: "var(--ink)", background: "rgba(16,13,7,.72)", textAlign: "center" } }, t.hz + "Hz"),
      ]);
      picker.appendChild(c);
    });

    const stageBob = h("div", { style: { position: "relative", width: "120px", height: "150px", flex: "0 0 auto" } });
    const info = h("div", { style: { flex: "1", minWidth: "0" } });
    const stage = h("div", { style: { display: "flex", gap: "22px", alignItems: "center", flexWrap: "wrap" } }, [stageBob, info]);

    mount.append(picker, stage);

    function bar(label, v, max, color, unit) {
      return h("div", { style: { margin: "8px 0" } }, [
        h("div", { style: { display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: "11px", marginBottom: "3px" } }, [h("span", { style: { color: "var(--dim)" } }, label), h("span", { style: { color: "var(--gilt)" } }, v + (unit || ""))]),
        h("div", { style: { height: "7px", background: "var(--rule)", borderRadius: "3px", overflow: "hidden" } }, [h("div", { style: { width: M.clamp(v / max, 0, 1) * 100 + "%", height: "100%", background: color } })]),
      ]);
    }

    function render() {
      [...picker.children].forEach((c, i) => { c.style.borderColor = i === sel ? "var(--gilt)" : "var(--rule2)"; c.style.boxShadow = i === sel ? "0 0 0 2px rgba(201,162,75,.4)" : "none"; });
      const t = tracks[sel];
      // bob stage: a crop that bobs + a beat ring
      stageBob.innerHTML = "";
      const dot = h("div", { id: "bobdot", style: { position: "absolute", left: "50%", top: "50%", width: "74px", height: "74px", marginLeft: "-37px", marginTop: "-37px", borderRadius: "8px", overflow: "hidden", border: "2px solid var(--gilt)", background: "var(--panel)", willChange: "transform" } },
        t.crop ? [h("img", { src: t.crop, alt: "", style: { width: "100%", height: "100%", objectFit: "cover" } })] : []);
      const ring = h("div", { id: "bobring", style: { position: "absolute", left: "50%", top: "50%", width: "74px", height: "74px", marginLeft: "-37px", marginTop: "-37px", borderRadius: "10px", border: "2px solid var(--gilt)", opacity: "0" } });
      stageBob.append(ring, dot);

      info.innerHTML = "";
      info.append(
        h("div.lbl", null, "stride cadence (clipmotion.py)"),
        h("div", { style: { display: "flex", alignItems: "baseline", gap: "8px", margin: "4px 0 10px" } }, [
          h("span", { style: { fontFamily: "var(--mono)", fontSize: "34px", color: "var(--gilt)", lineHeight: "1" } }, t.hz.toFixed(2)),
          h("span", { style: { fontVariant: "small-caps", fontSize: "16px", color: "var(--dim)" } }, "body-bobs per second"),
          h("span", { id: "beatcount", style: { fontFamily: "var(--mono)", fontSize: "12px", color: "var(--faint)", marginLeft: "auto" } }, "")]),
        bar("trust (autocorrelation)", t.strength, 1, "var(--sage)", ""),
        bar("walking observed", t.walk_s, 6, "var(--notch)", "s"),
        t.speed != null ? bar("avg speed", t.speed, 0.25, "var(--elliot)", "") : null,
        h("p.muted", { style: { fontSize: "12.5px", marginTop: "10px" } },
          "From the up-and-down bob of the body while walking — measured off the clip, no extra sensor. It's a confound-robust hint for re-ID: a limp reads the same from any angle, where a single still (pose + soft glass) does not."),
        h("button.btn", { id: "playbtn", style: { marginTop: "8px" }, onclick: toggle }, playing ? "❚❚ pause" : "▶ play"));
      lastPhase = 0;
    }

    function toggle() { playing = !playing; const b = info.querySelector("#playbtn"); if (b) b.textContent = playing ? "❚❚ pause" : "▶ play"; }

    let t0 = null;
    function loop(now) {
      if (t0 == null) t0 = now;
      const t = tracks[sel];
      const dot = stageBob.querySelector("#bobdot"), ring = stageBob.querySelector("#bobring"), bc = info.querySelector("#beatcount");
      if (dot && playing) {
        const secs = (now - t0) / 1000;
        const phase = secs * t.hz;             // cycles elapsed
        const yp = Math.sin(phase * Math.PI * 2);
        dot.style.transform = "translateY(" + (-yp * 22).toFixed(1) + "px)";
        if (Math.floor(phase) > Math.floor(lastPhase)) { beats++; if (bc) bc.textContent = beats + " bobs"; if (ring) { ring.style.transition = "none"; ring.style.opacity = ".9"; ring.style.transform = "scale(1)"; requestAnimationFrame(() => { ring.style.transition = "opacity .4s, transform .4s"; ring.style.opacity = "0"; ring.style.transform = "scale(1.5)"; }); } }
        lastPhase = phase;
      } else if (dot) { t0 = now - (lastPhase / t.hz) * 1000; }
      raf = requestAnimationFrame(loop);
    }

    render();
    raf = requestAnimationFrame(loop);
  });
})();
