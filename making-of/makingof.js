/* Behind the Glass — boot + shared helpers. No framework; demos register against MakingOf. */
(function () {
  "use strict";

  const DEMOS = [
    ["demo-pipeline", "pipeline", "pipeline.json"],
    ["demo-glass", "glass", "glass.json"],
    ["demo-decoy", "decoy", "decoy.json"],
    ["demo-appearance", "appearance", "appearance.json"],
    ["demo-copresence", "copresence", "copresence.json"],
    ["demo-reidgame", "reid_game", "reid_game.json"],
    ["demo-gait", "gait", "gait.json"],
    ["demo-twoaxis", "twoaxis", "twoaxis.json"],
  ];

  const CAST = {
    Stan: "#c9a24b", Notch: "#6f9bd1", Elliot: "#c07f9e",
    "Miss B.": "#8fb08a", Pepsi: "#c8856a",
  };
  const UNKNOWN = "#7d7458";

  const MakingOf = {
    demos: {},
    data: {},
    register(name, fn) { this.demos[name] = fn; },
    castColor(name) { return CAST[name] || UNKNOWN; },
    boot,
  };
  window.MakingOf = MakingOf;

  /* ---- tiny DOM helper: h('div.cls', {attr}, [children|string]) ---- */
  function h(sel, attrs, kids) {
    const parts = sel.split(/(?=[.#])/);
    const tag = parts[0].match(/^[a-z0-9]+/i) ? parts[0].replace(/[.#].*/, "") : "div";
    const node = document.createElement(tag || "div");
    parts.forEach((p) => {
      if (p[0] === ".") node.classList.add(p.slice(1));
      else if (p[0] === "#") node.id = p.slice(1);
    });
    if (attrs) for (const k in attrs) {
      if (k === "style" && typeof attrs[k] === "object") Object.assign(node.style, attrs[k]);
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k.startsWith("on") && typeof attrs[k] === "function") node.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] != null) node.setAttribute(k, attrs[k]);
    }
    appendKids(node, kids);
    return node;
  }
  function appendKids(node, kids) {
    if (kids == null) return;
    (Array.isArray(kids) ? kids : [kids]).forEach((c) => {
      if (c == null || c === false) return;
      node.appendChild(typeof c === "string" || typeof c === "number"
        ? document.createTextNode(String(c)) : c);
    });
  }
  MakingOf.h = h;
  MakingOf.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  MakingOf.lerp = (a, b, t) => a + (b - a) * t;

  async function boot() {
    let metaErr = false;
    try {
      const files = ["meta.json", ...DEMOS.map((d) => d[2])];
      const loaded = await Promise.all(files.map((f) =>
        fetch("data/" + f).then((r) => { if (!r.ok) throw new Error(f); return r.json(); })));
      MakingOf.data.meta = loaded[0];
      DEMOS.forEach((d, i) => { MakingOf.data[d[1]] = loaded[i + 1]; });
    } catch (e) {
      metaErr = true;
      console.error("data load failed", e);
    }

    if (!metaErr) fillMeta(MakingOf.data.meta);

    DEMOS.forEach(([id, name]) => {
      const mount = document.querySelector("#" + id + " .mount");
      if (!mount) return;
      const fn = MakingOf.demos[name];
      const data = MakingOf.data[name];
      mount.innerHTML = "";
      try {
        if (fn && data) fn(mount, data);
        else mount.appendChild(h("p.muted", null, "Demo data not available."));
      } catch (e) {
        console.error("demo " + name + " failed", e);
        mount.appendChild(h("p.muted", null, "This demo hit a snag — see the console."));
      }
    });

    setupNav();
    setupReveal();
  }

  function fillMeta(meta) {
    if (!meta) return;
    const days = document.getElementById("dek-days");
    if (days) days.textContent = meta.days;
    const sl = document.getElementById("statline");
    if (sl) {
      const stats = [
        [fmt(meta.crops), "crops"],
        [fmt(meta.clips), "clips"],
        [fmt(meta.visits), "visits"],
        [meta.days, "nights"],
        [meta.species_named, "species named"],
        [(meta.cast || []).length, "named individuals"],
      ];
      stats.forEach(([n, k]) => sl.appendChild(h("div.stat", null, [h("b", null, String(n)), h("span", null, k)])));
    }
  }
  function fmt(n) { return (n || 0).toLocaleString("en-US"); }
  MakingOf.fmt = fmt;

  function setupNav() {
    const links = [...document.querySelectorAll("nav.tabs a")];
    const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          links.forEach((l) => l.classList.remove("on"));
          const a = map.get(en.target.id);
          if (a) a.classList.add("on");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("section.chapter").forEach((s) => obs.observe(s));
  }

  function setupReveal() {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); obs.unobserve(en.target); } });
    }, { rootMargin: "0px 0px -8% 0px" });
    document.querySelectorAll(".demo").forEach((d) => { d.classList.add("reveal"); obs.observe(d); });
  }
})();
