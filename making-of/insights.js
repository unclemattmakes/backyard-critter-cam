/* Twenty Nights — chart engine.
   Vanilla JS over the same baked JSON the making-of demos read; no libraries.
   Every figure recomputes from data/*.json, so a regenerated export updates the page.
   Linear charts render in real pixels (and redraw on resize) so marks and labels
   never shrink with the viewport; only the radial dial scales as a composition. */
(function () {
  "use strict";

  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var AMBER = "#c38616", BLUE = "#5887c3", GLOW = "#e8ae4a";
  var INK2 = "#a7aeba", INK3 = "#6f7683", LINE = "#232a36", GRID = "#1a2029", CARD = "#151a23";
  var NS = "http://www.w3.org/2000/svg";

  /* ---------- tiny DOM helpers (labels are data — text nodes only) ---------- */
  function h(tag, cls, kids, attrs) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (attrs) for (var k in attrs) n.setAttribute(k, attrs[k]);
    append(n, kids);
    return n;
  }
  function s(tag, attrs, kids) {
    var n = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) n.setAttribute(k, attrs[k]);
    append(n, kids);
    return n;
  }
  function append(n, kids) {
    if (kids == null) return;
    (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" || typeof c === "number"
        ? document.createTextNode(String(c)) : c);
    });
  }
  function fmt(n) { return Math.round(n).toLocaleString("en-US"); }
  function mulberry(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }
  function quant(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]; }

  /* ---------- responsive redraw registry (pixel-space charts) ---------- */
  var redrawers = [];
  var resizeT = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeT);
    resizeT = setTimeout(function () { redrawers.forEach(function (f) { f(); }); }, 160);
  });

  /* ---------- tooltip ---------- */
  var tip = document.getElementById("tip");
  function tipShow(x, y, lines) {
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    append(tip, [h("b", null, lines[0])].concat(lines.slice(1).map(function (l) {
      return h("span", null, [l, document.createElement("br")]);
    })));
    tip.hidden = false;
    var r = tip.getBoundingClientRect();
    var px = Math.min(Math.max(10, x + 14), window.innerWidth - r.width - 10);
    var py = y - r.height - 14; if (py < 8) py = y + 18;
    tip.style.left = px + "px"; tip.style.top = py + "px";
  }
  function tipHide() { tip.hidden = true; }
  function hoverable(el, get) {
    el.addEventListener("pointermove", function (e) { tipShow(e.clientX, e.clientY, get()); });
    el.addEventListener("pointerleave", tipHide);
    el.addEventListener("focus", function () {
      var r = el.getBoundingClientRect();
      tipShow(r.left + r.width / 2, r.top, get());
    });
    el.addEventListener("blur", tipHide);
  }

  /* ---------- count-up ---------- */
  function countUp(el) {
    var target = +el.getAttribute("data-count");
    if (REDUCED || !isFinite(target)) { el.textContent = fmt(target); return; }
    var t0 = null, dur = 1400;
    function step(t) {
      if (!t0) t0 = t;
      var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * e);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* ---------- reveal ---------- */
  var pendingAnims = new Map(); /* section el -> [fn] */
  function onReveal(section, fn) {
    if (!pendingAnims.has(section)) pendingAnims.set(section, []);
    pendingAnims.get(section).push(fn);
  }
  function setupReveal() {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        en.target.classList.add("in");
        (pendingAnims.get(en.target) || []).forEach(function (fn) { fn(); });
        pendingAnims.delete(en.target);
        en.target.querySelectorAll("[data-count]").forEach(countUp);
        obs.unobserve(en.target);
      });
    }, { rootMargin: "0px 0px -12% 0px" });
    document.querySelectorAll(".chapter").forEach(function (c) {
      c.classList.add("reveal"); obs.observe(c);
    });
  }

  /* ---------- top bar after hero ---------- */
  function setupBar() {
    var bar = document.getElementById("bar"), hero = document.querySelector(".hero");
    new IntersectionObserver(function (en) {
      var off = !en[0].isIntersecting;
      bar.classList.toggle("on", off);
      bar.setAttribute("aria-hidden", off ? "false" : "true");
    }, { threshold: 0.04 }).observe(hero);
  }

  /* ---------- table twin ---------- */
  function table(mount, head, rows) {
    var el = document.getElementById(mount);
    if (!el) return;
    el.appendChild(h("div", "tbl-scroll", h("table", null, [
      h("thead", null, h("tr", null, head.map(function (t) { return h("th", null, t); }))),
      h("tbody", null, rows.map(function (r) {
        return h("tr", null, r.map(function (c) { return h("td", null, c); }));
      }))
    ])));
  }

  /* =====================================================================
     02 · the arrival dial (viewBox-scaled: a composition, marks stay big)
  ===================================================================== */
  function drawDial(data) {
    var mount = document.getElementById("dial");
    var hours = data.raccoon.arrival_hours;
    var counts = [], maxC = 0, total = 0;
    for (var i = 0; i < 24; i++) {
      var c = +hours[String(i)] || 0;
      counts.push(c); if (c > maxC) maxC = c; total += c;
    }
    var peakH = counts.indexOf(maxC);
    var W = 640, H = 568, cx = 320, cy = 322, r0 = 118, maxLen = 142, barW = 13;
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, "aria-hidden": "true" });

    function pol(hr, r) { /* hour (fractional) -> x,y; midnight at top, clockwise */
      var a = (hr / 24) * Math.PI * 2 - Math.PI / 2;
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    }
    function arcPath(h0, h1, rIn, rOut) { /* annulus sector, h0 -> h1 clockwise */
      var span = ((h1 - h0 + 24) % 24) || 24, large = span > 12 ? 1 : 0;
      var a = pol(h0, rOut), b = pol(h1, rOut), c2 = pol(h1, rIn), d = pol(h0, rIn);
      return "M" + a + " A" + rOut + " " + rOut + " 0 " + large + " 1 " + b +
             " L" + c2 + " A" + rIn + " " + rIn + " 0 " + large + " 0 " + d + " Z";
    }

    /* night band 19:00 -> 07:00 — a thin bezel just inside the clock face */
    svg.appendChild(s("path", { d: arcPath(19, 7, r0 - 22, r0 - 13), fill: BLUE, "fill-opacity": 0.32 }));
    /* baseline ring + hour dots */
    svg.appendChild(s("circle", { cx: cx, cy: cy, r: r0, fill: "none", stroke: LINE, "stroke-width": 1 }));
    for (var d = 0; d < 24; d++) {
      var p = pol(d, r0 - 9);
      svg.appendChild(s("circle", { cx: p[0], cy: p[1], r: 1.3, fill: INK3, "fill-opacity": 0.55 }));
    }
    /* cardinal labels */
    [["12 am", 0], ["6 am", 6], ["12 pm", 12], ["6 pm", 18]].forEach(function (lab) {
      var p = pol(lab[1], r0 - 38);
      svg.appendChild(s("text", {
        x: p[0], y: p[1], "text-anchor": "middle", "dominant-baseline": "middle",
        "font-size": 14.5, fill: INK3, "letter-spacing": "0.08em"
      }, lab[0]));
    });

    /* bars: butt-cap radial line + round tip = square base, rounded data end */
    var anims = [];
    counts.forEach(function (c, hr) {
      if (!c) return;
      var len = (c / maxC) * maxLen, capR = barW / 2;
      var g = s("g", { transform: "rotate(" + (hr / 24) * 360 + " " + cx + " " + cy + ")" });
      var line = s("line", { x1: cx, y1: cy - r0, x2: cx, y2: cy - r0, stroke: AMBER, "stroke-width": barW });
      var cap = s("circle", { cx: cx, cy: cy - r0, r: capR, fill: AMBER });
      g.appendChild(line); g.appendChild(cap); svg.appendChild(g);
      anims.push(function (t) {
        var l = len * t;
        line.setAttribute("y2", cy - r0 - Math.max(0, l - capR));
        cap.setAttribute("cy", cy - r0 - Math.max(0, l - capR));
      });
      if (c === 1 && hr >= 7 && hr < 19) { /* the daylight oddballs */
        var tipP = pol(hr, r0 + len + 14);
        svg.appendChild(s("circle", { cx: tipP[0], cy: tipP[1], r: 8, fill: "none", stroke: GLOW, "stroke-width": 1.4 }));
        var lp = pol(hr, r0 + len + 37);
        svg.appendChild(s("text", {
          x: lp[0], y: lp[1], "text-anchor": "middle", "dominant-baseline": "middle",
          "font-size": 13, fill: INK3
        }, (hr === 12 ? "12 pm" : hr % 12 + (hr < 12 ? " am" : " pm"))));
      }
    });

    /* peak label with a hairline leader */
    (function () {
      var p1 = pol(peakH + 0.5, r0 + maxLen + 10), p2 = pol(peakH + 0.5, r0 + maxLen + 28);
      svg.appendChild(s("line", { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], stroke: INK3, "stroke-width": 1 }));
      var anchor = p2[0] > cx + 20 ? "start" : p2[0] < cx - 20 ? "end" : "middle";
      svg.appendChild(s("text", { x: p2[0], y: p2[1] - 5, "text-anchor": anchor, "font-size": 15, fill: "#eef1f5", "font-weight": 500, "font-family": "'Hanken Grotesk', sans-serif" }, maxC + " arrivals"));
      svg.appendChild(s("text", { x: p2[0], y: p2[1] + 13, "text-anchor": anchor, "font-size": 13, fill: INK3 },
        (peakH % 12 || 12) + "–" + ((peakH + 1) % 12 || 12) + (peakH >= 12 ? " pm" : " am")));
    })();

    /* center readout */
    svg.appendChild(s("text", { x: cx, y: cy - 8, "text-anchor": "middle", "font-size": 44, fill: "#eef1f5", "font-family": "'Hanken Grotesk', sans-serif", "font-weight": 300 }, String(total)));
    svg.appendChild(s("text", { x: cx, y: cy + 20, "text-anchor": "middle", "font-size": 12.5, fill: INK3, "letter-spacing": "0.12em" }, "RACCOON VISITS"));

    /* invisible per-hour hit wedges (tooltip) */
    counts.forEach(function (c, hr) {
      var outer = Math.max(r0 + 44, r0 + (c / maxC) * maxLen + 22);
      var wedge = s("path", { d: arcPath(hr, hr + 1, r0 - 34, outer), fill: "transparent" });
      hoverable(wedge, function () {
        var lab = (hr % 12 || 12) + (hr < 12 ? " am" : " pm") + " – " + ((hr + 1) % 12 || 12) + ((hr + 1) % 24 < 12 ? " am" : " pm");
        return [c + (c === 1 ? " arrival" : " arrivals"), lab];
      });
      svg.appendChild(wedge);
    });

    mount.appendChild(svg);

    /* sweep in */
    if (REDUCED) { anims.forEach(function (f) { f(1); }); }
    else onReveal(document.getElementById("ch2"), function () {
      var t0 = null, dur = 1100;
      function stepA(t) {
        if (!t0) t0 = t;
        var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
        anims.forEach(function (f) { f(e); });
        if (p < 1) requestAnimationFrame(stepA);
      }
      requestAnimationFrame(stepA);
    });

    /* table twin */
    var rows = [];
    counts.forEach(function (c, hr) {
      if (c) rows.push([(hr % 12 || 12) + (hr < 12 ? " am" : " pm"), String(c)]);
    });
    rows.push(["hours not listed", "0"]);
    table("dial-table", ["hour of arrival", "visits"], rows);
    var peakStat = document.getElementById("peak-stat");
    if (peakStat) peakStat.textContent = String(maxC);
  }

  /* =====================================================================
     03 · the cast
  ===================================================================== */
  var PORTRAITS = {
    "Stan":   { src: "media/crops/game_6209.jpg",  alt: "Stan, sitting upright at night, mouth open mid-chatter, one paw raised." },
    "Notch":  { src: "media/crops/game_12414.jpg", alt: "Notch, standing on the pavers at night, looking straight into the camera." },
    "Elliot": { src: "media/crops/game_18680.jpg", alt: "Elliot at dusk, soft-focus and mid-stride — one of only a few dozen frames of him." }
  };
  function drawCast(meta) {
    var mount = document.getElementById("cast");
    var cast = meta.cast || [];
    var totalNamed = cast.reduce(function (a, c) { return a + c.crops; }, 0);
    var maxN = cast.reduce(function (a, c) { return Math.max(a, c.crops); }, 0);
    cast.forEach(function (c) {
      var pct = Math.round((c.crops / totalNamed) * 100);
      var w = Math.max(0.9, (c.crops / maxN) * 74); /* % of track; leave room for the value */
      var p = PORTRAITS[c.name];
      var face = p
        ? h("div", "cast-face", h("img", null, null, { src: p.src, alt: p.alt, loading: "lazy" }))
        : h("div", "cast-face cast-face--none", h("span", null, "☾"), { role: "img", "aria-label": c.name + " — no photograph in the public export" });
      var row = h("div", "cast-row", [
        face,
        h("div", "cast-meta", [
          h("div", "cast-name-row", [
            h("span", "cast-name", c.name),
            h("span", "cast-sub", pct + "% of named photographs")
          ]),
          h("div", "cast-track", [
            h("span", "cast-bar", null, { style: "--w:" + w }),
            h("span", "cast-n", fmt(c.crops), { style: "--w:" + w })
          ])
        ])
      ]);
      hoverable(row, function () {
        return [fmt(c.crops) + " photographs", c.name + " · " + pct + "% of all named crops"];
      });
      mount.appendChild(row);
    });
  }

  /* =====================================================================
     the strip renderer (04 similarity · 05 gate) — pixel-space, HTML labels
     rowsSpec: [{label, sub, note?, values, color, mode:'bins'|'dots', dotInfo?}]
  ===================================================================== */
  function stripFigure(mountId, rowsSpec, opts) {
    var mount = document.getElementById(mountId);
    mount.classList.add("strips");
    if (opts.threshold != null) mount.classList.add("has-cut");

    /* HTML skeleton once: label rows + plot slots + shared axis */
    var slots = rowsSpec.map(function (row, ri) {
      var lab = h("div", "strip-lab", [
        h("span", "strip-name", row.label),
        h("span", "strip-sub", row.sub + (row.note ? " · " : ""),),
        row.note ? h("span", "strip-note", row.note) : null,
        h("span", "strip-mean", "mean " + mean(row.values).toFixed(2))
      ]);
      var plot = h("div", "strip-plot");
      if (ri === 0 && opts.threshold != null) {
        plot.appendChild(h("span", "cut-chip", "the cut · " + opts.threshold.toFixed(2),
          { style: "--x:" + (opts.threshold * 100) + "%" }));
      }
      var rowEl = h("div", "strip-row", [lab, plot]);
      mount.appendChild(rowEl);
      return plot;
    });
    var axis = h("div", "strip-axis", [0, 0.25, 0.5, 0.75, 1].map(function (v) {
      return h("span", null, String(v));
    }));
    mount.appendChild(axis);

    function render() {
      var plotW = slots[0].clientWidth;
      if (!plotW) return;
      rowsSpec.forEach(function (row, ri) {
        var plot = slots[ri];
        var old = plot.querySelector("svg");
        if (old) plot.removeChild(old);
        var rowH = opts.rowH || 56;
        var svg = s("svg", { width: plotW, height: rowH, viewBox: "0 0 " + plotW + " " + rowH, role: "img", "aria-label": row.aria || row.label });
        var x = function (v) { return v * plotW; };

        /* faint quartile grid */
        [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
          var gx = Math.min(plotW - 0.5, Math.max(0.5, x(v)));
          svg.appendChild(s("line", { x1: gx, y1: 0, x2: gx, y2: rowH, stroke: GRID, "stroke-width": 1 }));
        });

        if (row.mode === "bins") {
          var BINS = 72, bins = new Array(BINS).fill(0), bMax = 0;
          row.values.forEach(function (v) {
            var b = Math.min(BINS - 1, Math.floor(v * BINS)); bins[b]++;
            if (bins[b] > bMax) bMax = bins[b];
          });
          var bw = plotW / BINS;
          bins.forEach(function (n, bi) {
            if (!n) return;
            var op = 0.08 + 0.92 * Math.pow(n / bMax, 0.55);
            var r = s("rect", { x: x(bi / BINS) + 0.5, y: 6, width: Math.max(1, bw - 1.5), height: rowH - 12, fill: row.color, "fill-opacity": op, rx: 2 });
            hoverable(r, function () {
              return [n + (n === 1 ? " pair" : " pairs"), "similarity " + (bi / BINS).toFixed(2) + "–" + ((bi + 1) / BINS).toFixed(2)];
            });
            svg.appendChild(r);
          });
        } else {
          var rnd = mulberry(ri * 7919 + 17);
          row.values.forEach(function (v, vi) {
            var cy2 = 9 + rnd() * (rowH - 18);
            var cxp = Math.min(plotW - 6, Math.max(6, x(v)));
            svg.appendChild(s("circle", { cx: cxp, cy: cy2, r: 4.5, fill: row.color, "fill-opacity": 0.8, stroke: CARD, "stroke-width": 1.5 }));
            var hit = s("circle", { cx: cxp, cy: cy2, r: 12, fill: "transparent" });
            hoverable(hit, function () { return row.dotInfo ? row.dotInfo(vi, v) : [v.toFixed(2)]; });
            svg.appendChild(hit);
          });
        }

        /* threshold + mean tick */
        if (opts.threshold != null) {
          svg.appendChild(s("line", { x1: x(opts.threshold), y1: 0, x2: x(opts.threshold), y2: rowH, stroke: GLOW, "stroke-width": 1.4 }));
        }
        var m = mean(row.values);
        svg.appendChild(s("line", { x1: x(m), y1: 1, x2: x(m), y2: rowH - 1, stroke: "#eef1f5", "stroke-width": 2 }));

        plot.appendChild(svg);
      });
    }
    render();
    redrawers.push(render);
  }

  /* ---------- 04 · similarity ---------- */
  function drawSimilarity(app) {
    var pts = app.points, tri = app.cos_tri, n = app.n_points;
    function cos(i, j) { /* row-major upper triangle */
      if (i > j) { var k = i; i = j; j = k; }
      return tri[i * n - (i * (i + 1)) / 2 + (j - i - 1)] / 100;
    }
    var sameVisit = [], sameInd = [], diffInd = [];
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
      var c = cos(i, j), a = pts[i], b = pts[j];
      if (a.visit === b.visit) sameVisit.push(c);
      else if (a.individual === b.individual) sameInd.push(c);
      else diffInd.push(c);
    }
    stripFigure("simstrips", [
      { label: "Same visit", sub: sameVisit.length + " pairs", values: sameVisit, color: AMBER, mode: "bins",
        note: "bursts touch " + Math.max.apply(null, sameVisit).toFixed(2),
        aria: "Same-visit pairs: mean 0.51, reaching 0.99." },
      { label: "Same raccoon, different night", sub: sameInd.length + " pairs", values: sameInd, color: AMBER, mode: "bins",
        aria: "Same raccoon on different nights: mean 0.23." },
      { label: "Different raccoons", sub: diffInd.length + " pairs", values: diffInd, color: AMBER, mode: "bins",
        aria: "Different raccoons: mean 0.15." }
    ], { rowH: 54 });

    function statRow(name, arr) {
      var so = arr.slice().sort(function (a, b) { return a - b; });
      return [name, String(arr.length), mean(arr).toFixed(2), quant(so, 0.5).toFixed(2), quant(so, 0.9).toFixed(2), so[so.length - 1].toFixed(2)];
    }
    table("sim-table", ["pair kind", "pairs", "mean", "median", "p90", "max"], [
      statRow("same visit", sameVisit),
      statRow("same raccoon, different night", sameInd),
      statRow("different raccoons", diffInd)
    ]);

    /* prototypes */
    var protos = app.prototypes, pcos = app.proto_cos, sameP = [], diffP = [], samePairs = [], diffPairs = [];
    for (var pi = 0; pi < protos.length; pi++) for (var pj = pi + 1; pj < protos.length; pj++) {
      var pv = pcos[pi][pj] / 100;
      if (protos[pi].individual === protos[pj].individual) { sameP.push(pv); samePairs.push([pi, pj]); }
      else { diffP.push(pv); diffPairs.push([pi, pj]); }
    }
    function protoInfo(pairs) {
      return function (vi, v) {
        var pr = pairs[vi];
        return [v.toFixed(2), protos[pr[0]].individual + " × " + protos[pr[1]].individual + " · two visits"];
      };
    }
    stripFigure("protostrips", [
      { label: "Same raccoon", sub: sameP.length + " prototype pairs", values: sameP, color: AMBER, mode: "dots",
        dotInfo: protoInfo(samePairs), note: "best match " + Math.max.apply(null, sameP).toFixed(2),
        aria: "Same-raccoon prototype pairs: mean 0.51, best 0.84." },
      { label: "Different raccoons", sub: diffP.length + " prototype pairs", values: diffP, color: AMBER, mode: "dots",
        dotInfo: protoInfo(diffPairs), aria: "Different-raccoon prototype pairs: mean 0.37." }
    ], { rowH: 64 });
    function statRowP(name, arr) {
      var so = arr.slice().sort(function (a, b) { return a - b; });
      return [name, String(arr.length), mean(arr).toFixed(2), quant(so, 0.5).toFixed(2), so[0].toFixed(2), so[so.length - 1].toFixed(2)];
    }
    table("proto-table", ["pair kind", "pairs", "mean", "median", "min", "max"], [
      statRowP("same raccoon", sameP), statRowP("different raccoons", diffP)
    ]);
  }

  /* ---------- 05 · the gate ---------- */
  function drawGate(decoy) {
    var A = decoy.dist.animal_pnon, N = decoy.dist.nonanimal_pnon, cut = decoy.threshold;
    var maxA = Math.max.apply(null, A), minN = Math.min.apply(null, N);
    stripFigure("gate", [
      { label: "Real animals", sub: A.length + " crops", values: A, color: AMBER, mode: "dots",
        dotInfo: function (i, v) { return [v.toFixed(3), "p(not an animal) — a real animal"]; },
        note: "worst " + maxA.toFixed(2),
        aria: "All 160 real animals score at or below 0.43." },
      { label: "Not animals", sub: N.length + " crops — plates, empty deck", values: N, color: BLUE, mode: "dots",
        dotInfo: function (i, v) { return [v.toFixed(3), "p(not an animal) — a genuine non-animal"]; },
        note: "closest " + minN.toFixed(2),
        aria: "All 160 non-animals score at or above 0.60." }
    ], { threshold: cut, rowH: 68 });

    var aOK = A.filter(function (v) { return v < cut; }).length;
    var nOK = N.filter(function (v) { return v >= cut; }).length;
    table("gate-table", ["population", "crops", "min", "max", "on the right side of 0.60"], [
      ["real animals", String(A.length), Math.min.apply(null, A).toFixed(3), maxA.toFixed(3), aOK + " of " + A.length],
      ["non-animals", String(N.length), minN.toFixed(3), Math.max.apply(null, N).toFixed(3), nOK + " of " + N.length]
    ]);
  }

  /* ---------- 06 · gait (pixel-space, HTML annotations) ---------- */
  function drawGait(gait) {
    var mount = document.getElementById("gait");
    mount.classList.add("gaitfig");
    var tracks = gait.tracks.slice().sort(function (a, b) { return a.hz - b.hz; });
    var lo = 0.5, hi = 3.6;
    var plot = h("div", "strip-plot gait-plot");
    mount.appendChild(plot);
    var first = tracks[0], last = tracks[tracks.length - 1];
    /* HTML annotations pinned by percent */
    plot.appendChild(h("span", "gait-note", "the amble · " + first.hz + " Hz",
      { style: "--x:" + ((first.hz - lo) / (hi - lo) * 100) + "%; --align:0" }));
    plot.appendChild(h("span", "gait-note gait-note--end", "the trot · " + last.hz + " Hz",
      { style: "--x:" + ((last.hz - lo) / (hi - lo) * 100) + "%; --align:1" }));
    var axis = h("div", "strip-axis gait-axis", [1, 2, 3].map(function (v) {
      return h("span", null, v + " Hz", { style: "--x:" + ((v - lo) / (hi - lo) * 100) + "%" });
    }));
    mount.appendChild(axis);

    function render() {
      var W = plot.clientWidth;
      if (!W) return;
      var old = plot.querySelector("svg");
      if (old) plot.removeChild(old);
      var H = 150, axisY = 128;
      var x = function (v) { return ((v - lo) / (hi - lo)) * W; };
      var svg = s("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Dot plot of stride cadence for seven tracked walks, from 0.83 to 3.33 hertz. Hovering a dot pulses a ring at that walk's real cadence." });
      svg.appendChild(s("line", { x1: 0, y1: axisY, x2: W, y2: axisY, stroke: LINE, "stroke-width": 1 }));
      [1, 2, 3].forEach(function (v) {
        svg.appendChild(s("line", { x1: x(v), y1: axisY - 5, x2: x(v), y2: axisY + 5, stroke: INK3, "stroke-width": 1 }));
      });
      var lastX = -99, lane = 0;
      tracks.forEach(function (t) {
        var cx2 = x(t.hz);
        lane = (cx2 - lastX < 26) ? lane + 1 : 0;
        lastX = cx2;
        var cy2 = axisY - 36 - lane * 30;
        var g = s("g", { tabindex: "0", role: "img", "aria-label": "Walk " + t.id + ": " + t.hz + " hertz, straightness " + t.straightness + ", " + t.walk_s + " seconds walking." });
        var ring = s("circle", { cx: cx2, cy: cy2, r: 9, fill: "none", stroke: GLOW, "stroke-width": 1.6, opacity: 0, "class": "pulse-ring" });
        var dot = s("circle", { cx: cx2, cy: cy2, r: 7, fill: AMBER, stroke: CARD, "stroke-width": 2 });
        var stem = s("line", { x1: cx2, y1: cy2 + 9, x2: cx2, y2: axisY, stroke: LINE, "stroke-width": 1 });
        var hit = s("circle", { cx: cx2, cy: cy2, r: 17, fill: "transparent" });
        g.appendChild(stem); g.appendChild(ring); g.appendChild(dot); g.appendChild(hit);
        g.style.setProperty("--period", (1 / t.hz).toFixed(3) + "s");
        function on() { g.classList.add("pulsing"); ring.setAttribute("opacity", 1); }
        function off() { g.classList.remove("pulsing"); ring.setAttribute("opacity", 0); }
        g.addEventListener("pointerenter", on); g.addEventListener("pointerleave", off);
        g.addEventListener("focus", on); g.addEventListener("blur", off);
        hoverable(hit, function () {
          return [t.hz + " Hz", "straightness " + t.straightness.toFixed(2) + " · " + t.walk_s + "s walking", "avg speed " + t.speed.toFixed(2) + " frame-widths/s"];
        });
        svg.appendChild(g);
      });
      plot.appendChild(svg);
    }
    render();
    redrawers.push(render);

    table("gait-table", ["walk", "cadence (Hz)", "trust", "straightness", "walking (s)", "avg speed"], tracks.map(function (t) {
      return ["#" + t.id, t.hz.toFixed(2), t.strength.toFixed(2), t.straightness.toFixed(2), t.walk_s.toFixed(1), t.speed.toFixed(3)];
    }));
  }

  /* =====================================================================
     boot
  ===================================================================== */
  function fillStat(name, v) {
    document.querySelectorAll("[data-stat='" + name + "']").forEach(function (el) {
      el.textContent = fmt(v);
    });
  }

  function boot() {
    setupBar();
    var files = ["meta", "twoaxis", "appearance", "decoy", "gait", "reid_game"];
    Promise.all(files.map(function (f) {
      return fetch("data/" + f + ".json").then(function (r) {
        if (!r.ok) throw new Error(f); return r.json();
      });
    })).then(function (loaded) {
      var meta = loaded[0], two = loaded[1], app = loaded[2], decoy = loaded[3], gait = loaded[4], game = loaded[5];

      /* keep the static numbers honest against a regenerated export */
      var heroTargets = [meta.crops, meta.visits, meta.species_named, (meta.cast || []).length];
      document.querySelectorAll("#hero-stats dd").forEach(function (dd, i) {
        dd.setAttribute("data-count", heroTargets[i]); dd.textContent = fmt(heroTargets[i]);
      });
      fillStat("clips", meta.clips); fillStat("species", meta.species_named);
      fillStat("embeddings", meta.embeddings);
      if (game && game.finale) fillStat("tracklets", game.finale.n_tracklets);

      drawDial(two);
      drawCast(meta);
      drawSimilarity(app);
      drawGate(decoy);
      drawGait(gait);
    }).catch(function (e) {
      console.error("data load failed", e);
      ["dial", "cast", "simstrips", "protostrips", "gate", "gait"].forEach(function (id) {
        var el = document.getElementById(id);
        if (el) el.appendChild(h("p", null,
          "The charts read data/*.json, so this page needs to be served: python -m http.server 8011 --directory making-of",
          { style: "font:13px/1.6 var(--mono);color:#6f7683;padding:20px 0" }));
      });
    }).finally(function () {
      setupReveal();
      /* the hero counts up immediately */
      document.querySelectorAll(".hero [data-count]").forEach(countUp);
    });
  }

  boot();
})();
