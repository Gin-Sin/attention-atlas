(function () {
  "use strict";

  var chapters = window.ATTENTION_CHAPTERS || [];
  var implementations = window.ATTENTION_IMPLEMENTATIONS || {};
  var STORAGE_KEY = "attention_atlas_completed";

  function esc(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function technicalText(value) {
    var replacements = {
      cKV: "\\(c^{KV}\\)",
      cQ: "\\(c^{Q}\\)",
      kR: "\\(k^{R}\\)",
      qI: "\\(q^{I}\\)",
      kI: "\\(k^{I}\\)",
      Hkv: "\\(H_{kv}\\)",
      Hq: "\\(H_q\\)",
      Lq: "\\(L_q\\)",
      Lk: "\\(L_k\\)",
      dk: "\\(d_k\\)",
      dv: "\\(d_v\\)",
      dh: "\\(d_h\\)",
      dc: "\\(d_c\\)",
      dR: "\\(d_R\\)",
      WUK: "\\(W^{UK}\\)",
      WUV: "\\(W^{UV}\\)",
      WOA: "\\(W^{OA}\\)",
      WOB: "\\(W^{OB}\\)",
      WKR: "\\(W^{KR}\\)",
      WQR: "\\(W^{QR}\\)",
      WIUQ: "\\(W^{IUQ}\\)",
      WIK: "\\(W^{IK}\\)"
    };
    return esc(value).replace(/\b(cKV|cQ|kR|qI|kI|Hkv|Hq|Lq|Lk|dk|dv|dh|dc|dR|WUK|WUV|WOA|WOB|WKR|WQR|WIUQ|WIK)\b/g, function (token) {
      return replacements[token];
    });
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

  function renderDiagram(config) {
    if (window.AttentionDiagrams) {
      var report = window.AttentionDiagrams.build(config);
      var badges = report.badges.map(function (badge) {
        return "<span>" + esc(badge) + "</span>";
      }).join("");
      var notes = report.notes.map(function (note, index) {
        return '<li><span class="diagram-guide__num">' + String(index + 1).padStart(2, "0") +
          '</span><div><strong>' + technicalText(note[0]) + '</strong><p>' + technicalText(note[1]) + "</p></div></li>";
      }).join("");
      return '<figure class="report-figure" id="architecture-block"><div class="diagram-header"><div><span class="diagram-header__eyebrow">Architecture Deconstruction</span><strong>完整 Attention Block · 教学重绘</strong></div><div class="diagram-header__tools"><div class="diagram-legend">' +
        badges + '</div><button class="diagram-expand" type="button" data-diagram-expand aria-expanded="false">⤢ 放大查看</button></div></div><div class="diagram diagram--report"><div class="diagram-canvas">' +
        report.svg + '</div></div><figcaption class="figcaption">' + technicalText(config.caption) +
        '</figcaption><aside class="diagram-memory"><span>One-line Memory · 一眼记住</span><p>' +
        technicalText(report.memory) + '</p></aside><ol class="diagram-guide">' + notes + "</ol></figure>";
    }
    return '<section class="warning diagram-load-failure" role="alert" aria-labelledby="diagram-load-failure-title">' +
      '<strong id="diagram-load-failure-title">架构图加载失败</strong>' +
      "<p>完整图表资源未能载入。请刷新页面，或检查 assets/diagrams.js 是否可访问。</p></section>";
  }

  function renderCards(items, className) {
    return '<div class="' + className + '-grid">' + items.map(function (item) {
      return '<article class="' + className + '"><span class="label">' + esc(item.label) + '</span><strong>' + esc(item.title) + '</strong><p>' + esc(item.body) + "</p></article>";
    }).join("") + "</div>";
  }

  function renderPositionEncoding(position) {
    if (!position) return '<div class="warning" role="status">本章暂无位置编码说明。</div>';
    return '<article class="formula position-encoding"><span class="formula-label">Position &amp; Sequence</span>' +
      "<h3>" + esc(position.title) + "</h3><p>" + esc(position.summary) + "</p><div>" +
      position.equation + "</div></article>" +
      renderCards(position.steps || [], "intuition") +
      '<div class="warning"><strong>实现边界：</strong> ' + esc(position.caveat) + "</div>";
  }

  function renderImplementations(chapterId) {
    var implementation = implementations[chapterId];
    if (!implementation) {
      return '<div class="warning" role="status">本章的 PyTorch 教学实现未能载入。</div>';
    }
    var prefix = "implementation-" + chapterId + "-";
    var blocks = (implementation.blocks || []).map(function (block, index) {
      var number = block.id || String(index + 1).padStart(2, "0");
      var codeId = prefix + number;
      return '<article class="formula implementation-block"><span class="formula-label">Block ' +
        esc(number) + " · Lines " + esc(block.start) + "–" + esc(block.end) +
        "</span><h3>" + esc(block.title) + '</h3><button class="button" type="button" data-copy-target="' +
        esc(codeId) + '">复制代码</button><pre><code id="' + esc(codeId) + '">' +
        esc(block.code) + "</code></pre></article>";
    }).join("");
    var fullSourceId = prefix + "full-source";
    return '<p>源文件：<a href="' + esc(implementation.path) + '"><code>' +
      esc(implementation.path) + "</code></a></p>" + blocks +
      '<details class="formula implementation-source"><summary>展开完整 PyTorch 源码</summary>' +
      '<p>Source path · <code>' + esc(implementation.path) + '</code></p>' +
      '<button class="button" type="button" data-copy-target="' + esc(fullSourceId) +
      '">复制完整源码</button><pre><code id="' + esc(fullSourceId) + '">' +
      esc(implementation.source) + "</code></pre></details>";
  }

  function initDiagramExpand(scope) {
    var figure = scope.querySelector(".report-figure");
    var button = scope.querySelector("[data-diagram-expand]");
    if (!figure || !button) return;
    function setExpanded(expanded) {
      figure.classList.toggle("is-expanded", expanded);
      document.body.classList.toggle("diagram-is-open", expanded);
      button.setAttribute("aria-expanded", expanded ? "true" : "false");
      button.textContent = expanded ? "× 关闭大图" : "⤢ 放大查看";
    }
    button.addEventListener("click", function () {
      setExpanded(!figure.classList.contains("is-expanded"));
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && figure.classList.contains("is-expanded")) {
        setExpanded(false);
      }
    });
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        if (!document.execCommand("copy")) throw new Error("copy command failed");
        resolve();
      } catch (error) {
        reject(error);
      } finally {
        document.body.removeChild(textarea);
      }
    });
  }

  function initCopyButtons(scope) {
    scope.querySelectorAll("[data-copy-target]").forEach(function (button) {
      button.addEventListener("click", function () {
        var target = document.getElementById(button.getAttribute("data-copy-target"));
        if (!target) return;
        var original = button.textContent;
        copyText(target.textContent).then(function () {
          button.textContent = "✓ 已复制";
        }).catch(function () {
          button.textContent = "复制失败";
        }).finally(function () {
          window.setTimeout(function () { button.textContent = original; }, 1500);
        });
      });
    });
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
      var source = d.source
        ? '<p><a class="formula-label" href="#authoritative-sources">Source anchor · ' + esc(d.source) + "</a></p>"
        : "";
      return '<article class="formula"><span class="formula-label">Derivation ' + (i + 1) +
        "</span><h3>" + d.title + "</h3><div>" + d.body + "</div>" + source + "</article>";
    }).join("");
    var exercises = c.exercises.map(function (e, i) {
      return '<article class="exercise"><div class="diagram-legend" aria-label="练习分类与难度"><span>Kind · ' + esc(e.kind) +
        '</span><span>Level · ' + esc(e.level) + "</span></div><h3>练习 " +
        (i + 1) + '</h3><div class="exercise-body"><p>' + e.q +
        '</p><details><summary>提示</summary><p>' + e.hint +
        '</p></details><details><summary>答案</summary><p>' + e.answer +
        "</p></details></div></article>";
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
      '<h2 data-no="05">位置编码与时序注入</h2>' + renderPositionEncoding(c.positionEncoding) +
      '<h2 data-no="06">数学推导</h2>' + derivations +
      '<div class="warning"><strong>边界与误区：</strong> ' + esc(c.warning) + "</div>" +
      '<h2 data-no="07">PyTorch 逐块实现</h2>' + renderImplementations(c.id) +
      '<h2 data-no="08">练习与答案</h2>' + exercises +
      '<h2 data-no="09" id="authoritative-sources">权威来源</h2><ol class="source-list">' + sources + "</ol>" +
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
    initDiagramExpand(root);
    initCopyButtons(root);
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
