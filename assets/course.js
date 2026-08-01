(function () {
  "use strict";

  var chapters = window.ATTENTION_CHAPTERS || [];
  var STORAGE_KEY = "attention_atlas_completed";

  function esc(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function readProgress() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch (_) {
      return [];
    }
  }

  function writeProgress(ids) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
    } catch (_) {
      /* Local files or privacy mode may disable storage. */
    }
  }

  function renderMath(scope) {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(scope || document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false }
      ],
      throwOnError: false,
      strict: "ignore"
    });
  }

  function marker(id, color) {
    return '<marker id="' + id + '" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="' + color + '"/></marker>';
  }

  function headsDiagram(mode) {
    var q = ["Q₁", "Q₂", "Q₃", "Q₄", "Q₅", "Q₆", "Q₇", "Q₈"];
    var kvCount = mode === "mha" ? 8 : mode === "mqa" ? 1 : 4;
    var color = "#285f8f";
    var nodes = "";
    var lines = "";
    q.forEach(function (label, i) {
      var y = 44 + i * 37;
      var k = mode === "mha" ? i : mode === "mqa" ? 0 : Math.floor(i / 2);
      var ky = kvCount === 1 ? 174 : 44 + k * (kvCount === 8 ? 37 : 74);
      nodes += '<rect x="70" y="' + (y - 15) + '" width="72" height="29" rx="4" fill="#eaf2f8" stroke="' + color + '"/><text x="106" y="' + (y + 5) + '" text-anchor="middle" font-size="12" fill="' + color + '">' + label + "</text>";
      lines += '<path d="M142 ' + y + ' C250 ' + y + ',265 ' + ky + ',360 ' + ky + '" fill="none" stroke="#aab5c1" stroke-width="1.2"/>';
    });
    for (var j = 0; j < kvCount; j++) {
      var y2 = kvCount === 1 ? 174 : 44 + j * (kvCount === 8 ? 37 : 74);
      nodes += '<rect x="360" y="' + (y2 - 18) + '" width="122" height="36" rx="4" fill="#fff" stroke="' + color + '"/><text x="421" y="' + (y2 + 4) + '" text-anchor="middle" font-size="11" fill="' + color + '">K/V ' + (j + 1) + "</text>";
      lines += '<path d="M482 ' + y2 + ' L625 ' + y2 + '" stroke="' + color + '" marker-end="url(#arrowHead)" />';
    }
    return '<svg viewBox="0 0 760 350" role="img" aria-label="Query 与 KV 头连接图"><defs>' + marker("arrowHead", color) + '</defs><text x="70" y="22" font-size="10" fill="#718096">QUERY HEADS</text><text x="360" y="22" font-size="10" fill="#718096">KEY / VALUE HEADS</text><text x="625" y="22" font-size="10" fill="#718096">OUTPUT CHANNELS</text>' + lines + nodes + '<rect x="625" y="80" width="78" height="188" rx="5" fill="#f3f1e9" stroke="#10243b"/><text x="664" y="174" text-anchor="middle" font-size="11" fill="#10243b" transform="rotate(-90 664 174)">Concat → Wᴼ</text></svg>';
  }

  function latentDiagram() {
    return '<svg viewBox="0 0 820 360" role="img" aria-label="MLA 潜变量缓存图"><defs>' + marker("aLat", "#285f8f") + '</defs>' +
      '<text x="40" y="28" font-size="10" fill="#718096">TOKEN STATE</text><rect x="40" y="130" width="115" height="70" rx="5" fill="#fff" stroke="#10243b"/><text x="97" y="170" text-anchor="middle" font-size="15" fill="#10243b">hₜ</text>' +
      '<path d="M155 165 L250 165" stroke="#285f8f" marker-end="url(#aLat)"/><text x="200" y="150" text-anchor="middle" font-size="10" fill="#718096">Wᴰᴷⱽ</text>' +
      '<rect x="255" y="120" width="150" height="90" rx="7" fill="#eaf2f8" stroke="#285f8f" stroke-width="2"/><text x="330" y="158" text-anchor="middle" font-size="14" fill="#285f8f">cₜᴷⱽ</text><text x="330" y="183" text-anchor="middle" font-size="10" fill="#718096">唯一主要缓存</text>' +
      '<path d="M405 145 C470 145 465 72 530 72" fill="none" stroke="#285f8f" marker-end="url(#aLat)"/><path d="M405 165 L530 165" stroke="#285f8f" marker-end="url(#aLat)"/><path d="M405 185 C470 185 465 258 530 258" fill="none" stroke="#285f8f" marker-end="url(#aLat)"/>' +
      '<g fill="#fff" stroke="#285f8f"><rect x="535" y="45" width="110" height="54" rx="4"/><rect x="535" y="138" width="110" height="54" rx="4"/><rect x="535" y="231" width="110" height="54" rx="4"/></g>' +
      '<g font-size="11" text-anchor="middle" fill="#285f8f"><text x="590" y="77">K¹ᶜ / V¹</text><text x="590" y="170">K²ᶜ / V²</text><text x="590" y="263">Kᴴᶜ / Vᴴ</text></g>' +
      '<path d="M155 192 C270 300 500 325 678 300" fill="none" stroke="#b6531b" stroke-dasharray="5 4" marker-end="url(#aLat)"/><rect x="680" y="270" width="105" height="60" rx="4" fill="#fbecdc" stroke="#b6531b"/><text x="732" y="296" text-anchor="middle" font-size="11" fill="#b6531b">RoPE key</text><text x="732" y="315" text-anchor="middle" font-size="10" fill="#718096">解耦缓存</text></svg>';
  }

  function sparseDiagram() {
    var tokens = "";
    for (var i = 0; i < 14; i++) {
      var chosen = [2, 6, 9, 12].indexOf(i) >= 0;
      tokens += '<rect x="' + (48 + i * 49) + '" y="63" width="34" height="34" rx="3" fill="' + (chosen ? "#f8f0df" : "#fff") + '" stroke="' + (chosen ? "#8b5d12" : "#cfd5dc") + '"/>';
      if (chosen) tokens += '<path d="M' + (65 + i * 49) + ' 97 L' + (330 + ([2, 6, 9, 12].indexOf(i) * 55)) + ' 220" stroke="#8b5d12" stroke-width="1.4"/>';
    }
    return '<svg viewBox="0 0 820 340" role="img" aria-label="DSA 两阶段稀疏选择"><defs>' + marker("aSpa", "#8b5d12") + '</defs><text x="48" y="42" font-size="10" fill="#718096">FULL HISTORY · LOW-DIMENSION INDEX SCAN</text>' + tokens +
      '<rect x="305" y="127" width="210" height="55" rx="5" fill="#f8f0df" stroke="#8b5d12"/><text x="410" y="151" text-anchor="middle" font-size="12" fill="#8b5d12">Lightning Indexer</text><text x="410" y="169" text-anchor="middle" font-size="10" fill="#718096">score all → Top-k</text>' +
      '<path d="M410 182 L410 216" stroke="#8b5d12" marker-end="url(#aSpa)"/><g fill="#fff" stroke="#8b5d12"><rect x="305" y="220" width="44" height="44" rx="3"/><rect x="360" y="220" width="44" height="44" rx="3"/><rect x="415" y="220" width="44" height="44" rx="3"/><rect x="470" y="220" width="44" height="44" rx="3"/></g><path d="M515 242 L650 242" stroke="#8b5d12" marker-end="url(#aSpa)"/><rect x="655" y="207" width="120" height="70" rx="5" fill="#fff" stroke="#10243b"/><text x="715" y="237" text-anchor="middle" font-size="12" fill="#10243b">Core Attention</text><text x="715" y="257" text-anchor="middle" font-size="10" fill="#718096">high-dim MLA</text><text x="305" y="294" font-size="10" fill="#718096">SELECTED TOKENS ONLY</text></svg>';
  }

  function compressedDiagram(mode) {
    var rawTokens = Array.from({ length: 16 }, function (_, i) {
      return '<rect x="' + (38 + i * 31) + '" y="50" width="23" height="26" rx="2"/>';
    }).join("");
    if (mode === "hca") {
      return '<svg viewBox="0 0 820 340" role="img" aria-label="HCA 重压缩后稠密读取"><defs>' + marker("aHca", "#684b91") + '</defs>' +
        '<text x="38" y="34" font-size="10" fill="#718096">RAW TOKEN HISTORY · REPRESENTATIVE SAMPLE</text><g fill="#fff" stroke="#cfd5dc">' + rawTokens + '</g>' +
        '<path d="M286 82 L286 126" stroke="#684b91" marker-end="url(#aHca)"/><text x="312" y="107" font-size="10" fill="#684b91">m′ = 128 : 1</text>' +
        '<g fill="#f0ebf7" stroke="#684b91"><rect x="115" y="140" width="150" height="56" rx="4"/><rect x="285" y="140" width="150" height="56" rx="4"/></g><text x="275" y="172" text-anchor="middle" font-size="11" fill="#684b91">very short compressed history</text>' +
        '<path d="M445 168 L560 168" stroke="#684b91" marker-end="url(#aHca)"/><rect x="565" y="132" width="150" height="72" rx="5" fill="#fff" stroke="#684b91" stroke-width="2"/><text x="640" y="162" text-anchor="middle" font-size="12" fill="#684b91">Dense core</text><text x="640" y="184" text-anchor="middle" font-size="10" fill="#718096">read every summary</text>' +
        '<rect x="565" y="238" width="150" height="54" rx="4" fill="#fff" stroke="#b6531b"/><text x="640" y="261" text-anchor="middle" font-size="11" fill="#b6531b">Local SWA</text><text x="640" y="280" text-anchor="middle" font-size="10" fill="#718096">recent 128 raw tokens</text><path d="M640 238 L640 207" stroke="#b6531b" marker-end="url(#aHca)"/></svg>';
    }
    return '<svg viewBox="0 0 820 360" role="img" aria-label="CSA 压缩后稀疏选择"><defs>' + marker("aCmp", "#8b5d12") + '</defs>' +
      '<text x="38" y="34" font-size="10" fill="#718096">RAW TOKEN HISTORY</text><g fill="#fff" stroke="#cfd5dc">' + rawTokens + '</g>' +
      '<path d="M286 82 L286 116" stroke="#8b5d12" marker-end="url(#aCmp)"/><text x="310" y="105" font-size="10" fill="#8b5d12">overlap · stride m=4</text>' +
      '<g fill="#f8f0df" stroke="#8b5d12"><rect x="75" y="130" width="80" height="42" rx="3"/><rect x="168" y="130" width="80" height="42" rx="3"/><rect x="261" y="130" width="80" height="42" rx="3"/><rect x="354" y="130" width="80" height="42" rx="3"/></g><text x="254" y="157" text-anchor="middle" font-size="10" fill="#8b5d12">L/m compressed entries</text>' +
      '<path d="M445 151 L535 151" stroke="#8b5d12" marker-end="url(#aCmp)"/><rect x="540" y="120" width="145" height="62" rx="4" fill="#f8f0df" stroke="#8b5d12"/><text x="612" y="146" text-anchor="middle" font-size="11" fill="#8b5d12">Lightning Indexer</text><text x="612" y="166" text-anchor="middle" font-size="10" fill="#718096">scan L/m → Top-k</text>' +
      '<path d="M612 182 L612 222" stroke="#8b5d12" marker-end="url(#aCmp)"/><rect x="535" y="228" width="155" height="60" rx="4" fill="#fff" stroke="#8b5d12" stroke-width="2"/><text x="612" y="254" text-anchor="middle" font-size="11" fill="#8b5d12">Sparse core</text><text x="612" y="274" text-anchor="middle" font-size="10" fill="#718096">selected summaries only</text>' +
      '<rect x="710" y="228" width="85" height="60" rx="4" fill="#fff" stroke="#b6531b"/><text x="752" y="253" text-anchor="middle" font-size="10" fill="#b6531b">Local</text><text x="752" y="271" text-anchor="middle" font-size="10" fill="#b6531b">SWA 128</text><path d="M710 258 L692 258" stroke="#b6531b" marker-end="url(#aCmp)"/></svg>';
  }

  function linearDiagram() {
    return '<svg viewBox="0 0 820 340" role="img" aria-label="线性注意力状态递推"><defs>' + marker("aLin", "#39704f") + '</defs><g fill="#fff" stroke="#39704f"><rect x="40" y="55" width="90" height="50" rx="4"/><rect x="40" y="145" width="90" height="50" rx="4"/><rect x="40" y="235" width="90" height="50" rx="4"/></g><g text-anchor="middle" font-size="11" fill="#39704f"><text x="85" y="85">(k₁,v₁)</text><text x="85" y="175">(k₂,v₂)</text><text x="85" y="265">(kₜ,vₜ)</text></g><path d="M130 80 C220 80 205 145 290 145" fill="none" stroke="#39704f" marker-end="url(#aLin)"/><path d="M130 170 L290 170" stroke="#39704f" marker-end="url(#aLin)"/><path d="M130 260 C220 260 205 195 290 195" fill="none" stroke="#39704f" marker-end="url(#aLin)"/><rect x="295" y="110" width="205" height="120" rx="7" fill="#eaf3ed" stroke="#39704f" stroke-width="2"/><text x="397" y="153" text-anchor="middle" font-size="14" fill="#39704f">Sₜ = Σ φ(kⱼ)vⱼᵀ</text><text x="397" y="180" text-anchor="middle" font-size="12" fill="#39704f">zₜ = Σ φ(kⱼ)</text><text x="397" y="207" text-anchor="middle" font-size="10" fill="#718096">fixed-size recurrent state</text><rect x="560" y="65" width="90" height="50" rx="4" fill="#fff" stroke="#39704f"/><text x="605" y="95" text-anchor="middle" font-size="11" fill="#39704f">qₜ</text><path d="M650 90 C710 90 700 150 740 150" fill="none" stroke="#39704f" marker-end="url(#aLin)"/><path d="M500 170 L735 170" stroke="#39704f" marker-end="url(#aLin)"/><rect x="738" y="140" width="55" height="60" rx="4" fill="#fff" stroke="#10243b"/><text x="765" y="175" text-anchor="middle" font-size="12" fill="#10243b">yₜ</text></svg>';
  }

  function deltaDiagram() {
    return '<svg viewBox="0 0 820 340" role="img" aria-label="Delta rule 状态更新"><defs>' + marker("aDel", "#39704f") + '</defs><rect x="50" y="120" width="150" height="90" rx="5" fill="#eaf3ed" stroke="#39704f"/><text x="125" y="156" text-anchor="middle" font-size="13" fill="#39704f">旧状态 Sₜ₋₁</text><text x="125" y="184" text-anchor="middle" font-size="11" fill="#718096">先乘 gate αₜ</text><path d="M200 165 L300 165" stroke="#39704f" marker-end="url(#aDel)"/><rect x="305" y="115" width="170" height="100" rx="5" fill="#fff" stroke="#39704f"/><text x="390" y="148" text-anchor="middle" font-size="12" fill="#39704f">读当前预测</text><text x="390" y="176" text-anchor="middle" font-size="13" fill="#10243b">v̂ₜ = Sᵀkₜ</text><text x="390" y="198" text-anchor="middle" font-size="10" fill="#718096">误差 eₜ = vₜ − v̂ₜ</text><path d="M475 165 L575 165" stroke="#39704f" marker-end="url(#aDel)"/><rect x="580" y="105" width="185" height="120" rx="5" fill="#eaf3ed" stroke="#39704f" stroke-width="2"/><text x="672" y="143" text-anchor="middle" font-size="12" fill="#39704f">定点修正</text><text x="672" y="173" text-anchor="middle" font-size="12" fill="#10243b">Sₜ = αSₜ₋₁ + βkₜeₜᵀ</text><text x="672" y="202" text-anchor="middle" font-size="10" fill="#718096">rank-1 update</text><path d="M390 65 L390 110" stroke="#b6531b" marker-end="url(#aDel)"/><text x="390" y="48" text-anchor="middle" font-size="11" fill="#b6531b">目标 (kₜ,vₜ)</text></svg>';
  }

  function kdaDiagram() {
    return '<svg viewBox="0 0 820 370" role="img" aria-label="KDA 与 MLA 混合架构"><defs>' + marker("aKda", "#684b91") + '</defs><text x="38" y="34" font-size="10" fill="#718096">KDA STATE TRANSITION</text><rect x="38" y="78" width="135" height="70" rx="4" fill="#f0ebf7" stroke="#684b91"/><text x="105" y="107" text-anchor="middle" font-size="11" fill="#684b91">Diag(αₜ)</text><text x="105" y="128" text-anchor="middle" font-size="10" fill="#718096">channel-wise decay</text><path d="M173 113 L255 113" stroke="#684b91" marker-end="url(#aKda)"/><rect x="260" y="78" width="155" height="70" rx="4" fill="#f0ebf7" stroke="#684b91"/><text x="337" y="107" text-anchor="middle" font-size="11" fill="#684b91">I − βₜkₜkₜᵀ</text><text x="337" y="128" text-anchor="middle" font-size="10" fill="#718096">rank-1 correction</text><path d="M415 113 L500 113" stroke="#684b91" marker-end="url(#aKda)"/><rect x="505" y="78" width="135" height="70" rx="4" fill="#fff" stroke="#10243b"/><text x="572" y="107" text-anchor="middle" font-size="11" fill="#10243b">Sₜ</text><text x="572" y="128" text-anchor="middle" font-size="10" fill="#718096">fixed state</text><path d="M640 113 L755 113" stroke="#684b91" marker-end="url(#aKda)"/><text x="710" y="96" text-anchor="middle" font-size="10" fill="#718096">qₜ reads</text>' +
      '<text x="38" y="224" font-size="10" fill="#718096">KIMI LINEAR · LAYERWISE 3 : 1 HYBRID</text><g><rect x="38" y="252" width="155" height="64" rx="4" fill="#f0ebf7" stroke="#684b91"/><rect x="210" y="252" width="155" height="64" rx="4" fill="#f0ebf7" stroke="#684b91"/><rect x="382" y="252" width="155" height="64" rx="4" fill="#f0ebf7" stroke="#684b91"/><rect x="554" y="252" width="215" height="64" rx="4" fill="#eaf2f8" stroke="#285f8f"/></g><g text-anchor="middle" font-size="11"><text x="115" y="289" fill="#684b91">KDA</text><text x="287" y="289" fill="#684b91">KDA</text><text x="459" y="289" fill="#684b91">KDA</text><text x="661" y="280" fill="#285f8f">Global MLA</text><text x="661" y="299" fill="#718096" font-size="10">full-context correction</text></g></svg>';
  }

  function renderDiagram(config) {
    var svg = "";
    if (config.type === "heads") svg = headsDiagram(config.mode);
    if (config.type === "latent") svg = latentDiagram();
    if (config.type === "sparse") svg = sparseDiagram();
    if (config.type === "compressed") svg = compressedDiagram(config.mode);
    if (config.type === "linear") svg = linearDiagram();
    if (config.type === "delta") svg = deltaDiagram();
    if (config.type === "kda") svg = kdaDiagram();
    return '<figure><div class="diagram">' + svg + '</div><figcaption class="figcaption">' + esc(config.caption) + "</figcaption></figure>";
  }

  function renderCards(items, className) {
    return '<div class="' + className + '-grid">' + items.map(function (item) {
      return '<article class="' + className + '"><span class="label">' + esc(item.label) + '</span><strong>' + esc(item.title) + '</strong><p>' + esc(item.body) + "</p></article>";
    }).join("") + "</div>";
  }

  function renderChapter() {
    var root = document.getElementById("chapter-root");
    if (!root) return;
    var id = new URLSearchParams(window.location.search).get("id") || "mha";
    var index = chapters.findIndex(function (c) { return c.id === id; });
    if (index < 0) index = 0;
    var c = chapters[index];
    document.title = c.title + " · Attention Atlas";
    document.body.classList.add(c.category);

    var motivation = c.motivation.map(function (p) { return "<p>" + p + "</p>"; }).join("");
    var derivations = c.derivations.map(function (d, i) {
      return '<article class="formula"><span class="formula-label">Derivation ' + (i + 1) + '</span><h3>' + d.title + "</h3><div>" + d.body + "</div></article>";
    }).join("");
    var exercises = c.exercises.map(function (e, i) {
      return '<article class="exercise"><h3>练习 ' + (i + 1) + '</h3><div class="exercise-body"><p>' + e.q + '</p><details><summary>提示</summary><p>' + e.hint + '</p></details><details><summary>答案</summary><p>' + e.answer + "</p></details></div></article>";
    }).join("");
    var sources = c.sources.map(function (s) {
      return '<li><span><a href="' + esc(s.url) + '" target="_blank" rel="noreferrer">' + esc(s.label) + "</a></span></li>";
    }).join("");
    var prev = chapters[index - 1];
    var next = chapters[index + 1];
    var nav = '<nav class="chapter-nav">' +
      (prev ? '<a href="?id=' + prev.id + '"><small>上一章</small>' + esc(prev.title + " · " + prev.zhTitle) + "</a>" : "<span></span>") +
      (next ? '<a class="next" href="?id=' + next.id + '"><small>下一章</small>' + esc(next.title + " · " + next.zhTitle) + "</a>" : '<a class="next" href="index.html"><small>课程完成</small>回到架构地图</a>') +
      "</nav>";

    root.innerHTML =
      '<nav class="breadcrumbs"><a href="index.html">Attention Atlas</a> &nbsp;/&nbsp; Chapter ' + String(c.order).padStart(2, "0") + " &nbsp;/&nbsp; " + esc(c.title) + "</nav>" +
      '<header class="chapter-hero"><p class="eyebrow">Chapter ' + String(c.order).padStart(2, "0") + " · " + esc(c.fullTitle) + '</p><h1>' + esc(c.title) + '</h1><p class="chapter-deck">' + esc(c.zhTitle) + "。 " + esc(c.deck) + '</p><div class="chapter-meta"><b>' + esc(c.category) + "</b><span>" + esc(c.year) + "</span><span>难度 · " + esc(c.difficulty) + "</span><span>" + esc(c.report) + '</span></div></header>' +
      '<aside class="takeaway"><span>Mathematical Takeaway</span><p>' + c.takeaway + "</p></aside>" +
      '<main class="chapter-main"><h2 data-no="01">问题从哪里来</h2>' + motivation +
      '<h2 data-no="02">设计限定条件</h2>' + renderCards(c.constraints, "constraint") +
      '<h2 data-no="03">先抓住数学直觉</h2>' + renderCards(c.intuitions, "intuition") +
      '<h2 data-no="04">架构图</h2>' + renderDiagram(c.diagram) +
      '<h2 data-no="05">数学推导</h2>' + derivations +
      '<div class="warning"><strong>边界与误区：</strong> ' + esc(c.warning) + "</div>" +
      '<h2 data-no="06">练习与答案</h2>' + exercises +
      '<h2 data-no="07">权威来源</h2><ol class="source-list">' + sources + "</ol>" +
      '<button class="button" id="complete-chapter" type="button">标记本章完成</button>' +
      nav + "</main>";

    var done = readProgress();
    var button = document.getElementById("complete-chapter");
    function syncButton() {
      var completed = done.indexOf(c.id) >= 0;
      button.textContent = completed ? "✓ 已完成 · 点击撤销" : "标记本章完成";
      button.classList.toggle("primary", completed);
    }
    button.addEventListener("click", function () {
      var at = done.indexOf(c.id);
      if (at >= 0) done.splice(at, 1);
      else done.push(c.id);
      writeProgress(done);
      syncButton();
    });
    syncButton();
    renderMath(root);
  }

  function populateHome() {
    var grid = document.getElementById("chapter-grid");
    if (grid) {
      grid.innerHTML = chapters.map(function (c) {
        return '<a class="chapter-card ' + c.category + '" href="chapter.html?id=' + c.id + '"><span class="card-kicker">Chapter ' + String(c.order).padStart(2, "0") + '</span><span class="year">' + esc(c.year) + '</span><h3>' + esc(c.title) + '</h3><p>' + esc(c.zhTitle) + '</p><span class="arrow">↗</span></a>';
      }).join("");
    }
    var done = readProgress();
    document.querySelectorAll("[data-progress-count]").forEach(function (el) {
      el.textContent = done.length + " / " + chapters.length;
    });
    setupCalculator();
    renderMath(document.body);
  }

  function setupCalculator() {
    var form = document.getElementById("kv-calculator");
    if (!form) return;
    var output = document.getElementById("kv-result");
    function calculate() {
      var layers = Number(form.elements.layers.value);
      var length = Number(form.elements.length.value);
      var heads = Number(form.elements.heads.value);
      var dim = Number(form.elements.dim.value);
      var bytes = Number(form.elements.bytes.value);
      var total = 2 * layers * length * heads * dim * bytes;
      var gib = total / Math.pow(1024, 3);
      output.textContent = isFinite(gib) ? gib.toFixed(gib < 1 ? 3 : 2) + " GiB" : "—";
    }
    form.addEventListener("input", calculate);
    calculate();
  }

  window.addEventListener("DOMContentLoaded", function () {
    renderChapter();
    populateHome();
  });
})();
