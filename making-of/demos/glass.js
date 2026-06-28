/* Bonus demo — through the glass. Scrub real crops worst→best; quality.py scores each shot. */
(function () {
  const M = window.MakingOf, h = M.h;

  M.register("glass", function (mount, data) {
    const crops = data.crops || [];
    if (!crops.length) { mount.appendChild(h("p.muted", null, "No crops exported.")); return; }
    let i = Math.floor(crops.length * 0.3);

    function verdict(q) {
      if (q < 45) return ["flared & soft", "var(--rust)"];
      if (q < 120) return ["soft — the glass haze", "var(--gilt)"];
      if (q < 220) return ["readable", "var(--sage)"];
      return ["sharp", "var(--ok)"];
    }

    const photo = h("div", { style: { position: "relative", width: "260px", flex: "0 0 auto" } });
    const info = h("div", { style: { flex: "1", minWidth: "0" } });
    const top = h("div", { style: { display: "flex", gap: "18px", alignItems: "flex-start", flexWrap: "wrap" } }, [photo, info]);

    const sliderRow = h("div", { style: { display: "flex", alignItems: "center", gap: "12px", marginTop: "18px" } });
    const slider = h("input", { type: "range", min: "0", max: String(crops.length - 1), step: "1", value: String(i), style: { flex: "1" }, oninput: (e) => { i = +e.target.value; render(); } });
    sliderRow.append(h("span", { style: { fontFamily: "var(--mono)", fontSize: "11px", color: "var(--faint)" } }, "softest"), slider, h("span", { style: { fontFamily: "var(--mono)", fontSize: "11px", color: "var(--faint)" } }, "sharpest"));

    mount.append(top, sliderRow);

    function render() {
      const c = crops[i];
      const [vtxt, vcol] = verdict(c.quality);
      photo.innerHTML = "";
      photo.append(
        h("img", { src: c.crop, alt: "raccoon crop", style: { width: "260px", height: "200px", objectFit: "cover", borderRadius: "4px", border: "1px solid var(--rule2)", display: "block" } }),
        h("div", { style: { position: "absolute", top: "8px", left: "8px", fontFamily: "var(--mono)", fontSize: "10px", letterSpacing: ".1em", textTransform: "uppercase", color: "var(--ink)", background: "rgba(16,13,7,.7)", padding: "2px 7px", borderRadius: "2px" } }, c.night ? "night · through glass" : "daylight"));
      info.innerHTML = "";
      info.append(
        h("div.lbl", null, "shot quality (quality.py)"),
        h("div", { style: { display: "flex", alignItems: "baseline", gap: "10px", margin: "4px 0 2px" } }, [
          h("span", { style: { fontFamily: "var(--mono)", fontSize: "34px", color: vcol, lineHeight: "1" } }, String(c.quality)),
          h("span", { style: { fontVariant: "small-caps", fontSize: "18px", color: vcol } }, vtxt)]),
        scale(c.quality, data),
        h("p.muted", { style: { fontSize: "13.5px", marginTop: "14px" } },
          "Almost everything arrives at night, through a sliding glass door — soft, a little flared. So every crop is scored for sharpness (with a bump for night eyeshine), and the journal leads each visit with its best frame, not just its most confident one."),
        h("div.muted", { style: { fontFamily: "var(--mono)", fontSize: "11px", marginTop: "6px" } }, "crop " + (i + 1) + " of " + crops.length + " · this yard spans quality " + data.min + "–" + data.max + ", median " + data.median));
    }

    function scale(q, d) {
      const lo = Math.log(d.min + 1), hi = Math.log(d.max + 1), at = (Math.log(q + 1) - lo) / (hi - lo);
      const bar = h("div", { style: { position: "relative", height: "8px", borderRadius: "4px", background: "linear-gradient(90deg, var(--rust), var(--gilt) 40%, var(--ok))", marginTop: "10px" } });
      bar.appendChild(h("div", { style: { position: "absolute", left: "calc(" + (at * 100).toFixed(1) + "% - 7px)", top: "-4px", width: "14px", height: "14px", borderRadius: "50%", background: "var(--ink)", border: "2px solid #100d07" } }));
      return bar;
    }

    const mq = window.matchMedia("(max-width:520px)");
    const apply = () => { photo.style.width = mq.matches ? "100%" : "260px"; const im = photo.querySelector("img"); if (im) im.style.width = mq.matches ? "100%" : "260px"; };
    mq.addEventListener("change", apply); apply();
    render();
  });
})();
