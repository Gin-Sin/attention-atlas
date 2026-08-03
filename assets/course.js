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

  function richText(value) {
    return esc(value).replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
  }

  /* Diagram notes/captions/memory carry real KaTeX \(...\) markup, so they
     only need HTML escaping before the auto-renderer runs. */
  function technicalText(value) {
    return esc(value);
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

  function renderImplementationEditor(implementation) {
    if (!implementation || !(implementation.blocks || []).length) {
      return '<aside class="code-ide code-ide--unavailable" role="status">' +
        "<strong>PyTorch 教学实现未能载入</strong>" +
        "<p>架构图仍可阅读；请检查 assets/implementations.js 是否可访问。</p></aside>";
    }

    var blocks = implementation.blocks;
    var initial = blocks.find(function (block) {
      return String(block.id) === "01";
    }) || blocks[0];
    var initialId = String(initial.id || "01").padStart(2, "0");
    var options = blocks.map(function (block, index) {
      var blockId = String(block.id || index + 1).padStart(2, "0");
      return '<option value="' + esc(blockId) + '"' +
        (blockId === initialId ? " selected" : "") + ">Block " + esc(blockId) +
        " · " + esc(block.title) + "</option>";
    }).join("");

    return '<aside class="code-ide" data-architecture-ide tabindex="-1" aria-label="PyTorch architecture code workbench">' +
      '<div class="code-ide__titlebar"><span class="code-ide__traffic" aria-hidden="true">' +
      '<i></i><i></i><i></i></span><a class="code-ide__meta" href="' +
      esc(implementation.path) + '" title="打开完整源码 ' + esc(implementation.path) +
      '">' + esc(implementation.path) + '</a><span class="code-ide__icons">' +
      '<button class="code-ide__icon" type="button" data-workbench-mode aria-pressed="false" title="查看完整源码" aria-label="查看完整源码">⛶</button>' +
      '<button class="code-ide__icon" type="button" data-workbench-copy title="复制当前代码" aria-label="复制当前代码">⧉</button>' +
      '<a class="code-ide__icon" href="' + esc(implementation.path) +
      '" download title="下载完整源码" aria-label="下载完整源码">↓</a></span></div>' +
      '<div class="code-ide__toolbar">' +
      '<select class="code-ide__jump" data-workbench-select aria-label="选择实现代码块" title="跳转到指定代码块">' +
      options + '</select><span class="code-ide__lines" data-workbench-lines>Lines ' +
      esc(initial.start) + "–" + esc(initial.end) + "</span></div>" +
      '<p class="code-ide__status" data-workbench-status aria-live="polite">' +
      "选择架构图节点，或从列表跳转到对应实现。</p>" +
      '<pre class="code-ide__editor language-python line-numbers" data-workbench-pre data-start="' +
      esc(initial.start) + '"><code class="language-python" data-workbench-editor>' +
      esc(initial.code) + "</code></pre></aside>";
  }

  function renderDiagram(config, implementation) {
    if (window.AttentionDiagrams) {
      var report = window.AttentionDiagrams.build(config);
      var badges = report.badges.map(function (badge) {
        return "<span>" + esc(badge) + "</span>";
      }).join("");
      var notes = report.notes.map(function (note, index) {
        return '<li><span class="diagram-guide__num">' + String(index + 1).padStart(2, "0") +
          '</span><div><strong>' + technicalText(note[0]) + '</strong><p>' + technicalText(note[1]) + "</p></div></li>";
      }).join("");
      return '<figure class="report-figure report-figure--workbench" id="architecture-block"><div class="diagram-header"><div><span class="diagram-header__eyebrow">Architecture Deconstruction</span><strong>完整 Attention Block · 教学重绘</strong></div><div class="diagram-header__tools"><div class="diagram-legend">' +
        badges + '</div><button class="diagram-expand" type="button" data-diagram-expand aria-expanded="false">⤢ 放大查看</button></div></div><div class="architecture-workbench"><div class="architecture-pane"><div class="diagram diagram--report"><div class="diagram-canvas">' +
        report.svg + "</div></div></div>" + renderImplementationEditor(implementation) +
        '</div><figcaption class="figcaption">' + technicalText(config.caption) +
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
      window.dispatchEvent(new Event("resize"));
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && figure.classList.contains("is-expanded")) {
        setExpanded(false);
        window.dispatchEvent(new Event("resize"));
      }
    });
  }

  function initWorkbenchHeightSync(scope) {
    var workbench = scope.querySelector(".architecture-workbench");
    if (!workbench) return;
    var pane = workbench.querySelector(".architecture-pane");
    var ide = workbench.querySelector(".code-ide");
    if (!pane || !ide) return;

    function apply() {
      workbench.classList.remove("is-height-synced");
      workbench.style.removeProperty("--workbench-height");
      var sideBySide = window.matchMedia("(min-width: 1500px)").matches;
      if (!sideBySide) return;
      /* Both panes follow the diagram's natural height; the editor flexes. */
      var unified = pane.offsetHeight;
      if (!isFinite(unified) || unified < 200) return;
      workbench.style.setProperty("--workbench-height", unified + "px");
      workbench.classList.add("is-height-synced");
    }

    var pending = 0;
    function schedule() {
      window.cancelAnimationFrame(pending);
      pending = window.requestAnimationFrame(apply);
    }
    window.addEventListener("resize", schedule);
    /* KaTeX and web fonts settle after first paint and shift heights. */
    window.addEventListener("load", schedule);
    window.setTimeout(schedule, 400);
    apply();
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

  function initArchitectureWorkbench(scope, implementation) {
    var workbench = scope.querySelector(".architecture-workbench");
    var ide = scope.querySelector("[data-architecture-ide]");
    if (!workbench || !ide || !implementation || !(implementation.blocks || []).length) return;

    var blocks = implementation.blocks;
    var byId = {};
    blocks.forEach(function (block, index) {
      var id = String(block.id || index + 1).padStart(2, "0");
      byId[id] = block;
    });

    var select = ide.querySelector("[data-workbench-select]");
    var modeButton = ide.querySelector("[data-workbench-mode]");
    var copyButton = ide.querySelector("[data-workbench-copy]");
    var lines = ide.querySelector("[data-workbench-lines]");
    var status = ide.querySelector("[data-workbench-status]");
    var pre = ide.querySelector("[data-workbench-pre]");
    var code = ide.querySelector("[data-workbench-editor]");
    var diagramNodes = workbench.querySelectorAll("svg [data-code-block]");
    var firstDiagramId = diagramNodes.length
      ? String(diagramNodes[0].getAttribute("data-code-block") || "").padStart(2, "0")
      : "";
    var initialId = byId[firstDiagramId]
      ? firstDiagramId
      : byId["01"] ? "01" : Object.keys(byId)[0];
    var currentId = initialId;
    var fullSource = false;
    var displayedCode = "";
    var scrollRequest = 0;

    function normalizeId(value) {
      var id = String(value || "").trim();
      return /^\d+$/.test(id) ? id.padStart(2, "0") : id;
    }

    function setStatus(message) {
      status.textContent = message;
    }

    function clearPrismDecorations() {
      pre.querySelectorAll(".line-highlight").forEach(function (element) {
        element.remove();
      });
    }

    function highlightCode() {
      clearPrismDecorations();
      code.textContent = displayedCode;
      if (!window.Prism || typeof window.Prism.highlightElement !== "function") return;
      try {
        window.Prism.highlightElement(code);
      } catch (_) {
        code.textContent = displayedCode;
      }
    }

    function scrollToSourceLine(block) {
      var request = ++scrollRequest;
      window.requestAnimationFrame(function () {
        if (request !== scrollRequest || !fullSource) return;
        var lineHeight = parseFloat(window.getComputedStyle(code).lineHeight) || 20;
        pre.scrollTop = Math.max(0, (Number(block.start) - 2) * lineHeight);
      });
    }

    function refreshEditor(scrollToLine) {
      var block = byId[currentId];
      if (!block) return;
      displayedCode = fullSource ? implementation.source : block.code;
      if (fullSource) {
        pre.setAttribute("data-line", block.start + "-" + block.end);
        pre.removeAttribute("data-start");
      } else {
        pre.removeAttribute("data-line");
        pre.setAttribute("data-start", block.start);
        pre.scrollTop = 0;
      }
      highlightCode();
      if (fullSource && scrollToLine) scrollToSourceLine(block);
    }

    function syncDiagramNodes(sourceNode) {
      var primary = sourceNode || null;
      diagramNodes.forEach(function (node) {
        var matches = normalizeId(node.getAttribute("data-code-block")) === currentId;
        node.classList.remove("is-active", "is-code-active", "is-code-related");
        node.setAttribute("aria-pressed", "false");
        if (!matches) return;
        if (!primary) primary = node;
        if (node !== primary) node.classList.add("is-code-related");
      });
      if (primary) {
        primary.classList.add("is-code-active");
        primary.setAttribute("aria-pressed", "true");
      }
    }

    function revealIdeIfNeeded() {
      var rect = ide.getBoundingClientRect();
      var outsideViewport = rect.top < 0 || rect.bottom > window.innerHeight;
      if (!outsideViewport) return;
      try {
        ide.focus({ preventScroll: true });
      } catch (_) {
        ide.focus();
      }
      var reduceMotion = window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      ide.scrollIntoView({
        behavior: reduceMotion ? "auto" : "smooth",
        block: "nearest",
        inline: "nearest"
      });
    }

    function selectBlock(id, options) {
      var normalized = normalizeId(id);
      var block = byId[normalized];
      if (!block) return;
      currentId = normalized;
      select.value = currentId;
      lines.textContent = "Lines " + block.start + "–" + block.end;
      syncDiagramNodes(options && options.sourceNode);
      refreshEditor(true);
      setStatus("Block " + currentId + " 已选中 · " +
        (fullSource ? "完整源码已定位到高亮行；" : "编辑器显示当前代码块；") +
        "点击图中节点或使用列表继续切换。");
      if (options && options.reveal) revealIdeIfNeeded();
    }

    diagramNodes.forEach(function (node) {
      var id = normalizeId(node.getAttribute("data-code-block"));
      var block = byId[id];
      if (!block) return;
      node.setAttribute("tabindex", "0");
      node.setAttribute("role", "button");
      node.setAttribute("aria-label", "查看代码块 " + id + "：" + block.title);
      node.addEventListener("click", function () {
        selectBlock(id, { reveal: true, sourceNode: node });
      });
      node.addEventListener("keydown", function (event) {
        if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
        event.preventDefault();
        selectBlock(id, { reveal: true, sourceNode: node });
      });
    });

    select.addEventListener("change", function () {
      selectBlock(select.value);
    });
    modeButton.addEventListener("click", function () {
      fullSource = !fullSource;
      modeButton.setAttribute("aria-pressed", fullSource ? "true" : "false");
      var modeHint = fullSource ? "仅看当前块" : "查看完整源码";
      modeButton.setAttribute("title", modeHint);
      modeButton.setAttribute("aria-label", modeHint);
      refreshEditor(true);
      setStatus(fullSource
        ? "完整源码模式 · 当前块行号已高亮。"
        : "当前代码块模式 · 仅显示选中实现。");
    });
    copyButton.addEventListener("click", function () {
      copyText(displayedCode).then(function () {
        copyButton.textContent = "✓";
        setStatus(fullSource ? "完整源码已复制。" : "当前代码块已复制。");
      }).catch(function () {
        copyButton.textContent = "✕";
        setStatus("复制失败，请从编辑器中手动选择代码。");
      }).finally(function () {
        window.setTimeout(function () {
          copyButton.textContent = "⧉";
        }, 1500);
      });
    });

    selectBlock(initialId);
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
        "</span><h3>" + esc(d.title) + "</h3><div>" + richText(d.body) + "</div>" + source + "</article>";
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
    var tocEntries = [
      ["01", "问题从哪里来", "sec-01"],
      ["02", "设计限定条件", "sec-02"],
      ["03", "先抓住数学直觉", "sec-03"],
      ["04", "架构图与交互实现", "sec-04"],
      ["05", "位置编码与时序注入", "sec-05"],
      ["06", "数学推导", "sec-06"],
      ["07", "练习与答案", "sec-07"],
      ["08", "权威来源", "authoritative-sources"]
    ];
    var toc = '<nav class="chapter-toc" aria-label="本章目录"><span class="chapter-toc__label">Contents · 本章目录</span><ol>' +
      tocEntries.map(function (entry) {
        return '<li><a href="#' + entry[2] + '"><i>' + entry[0] + "</i>" + esc(entry[1]) + "</a></li>";
      }).join("") + "</ol></nav>";
    var prev = chapters[index - 1];
    var next = chapters[index + 1];
    var nav = '<nav class="chapter-nav">' +
      (prev ? '<a href="?id=' + prev.id + '"><small>上一章</small>' + esc(prev.title + " · " + prev.zhTitle) + "</a>" : "<span></span>") +
      (next ? '<a class="next" href="?id=' + next.id + '"><small>下一章</small>' + esc(next.title + " · " + next.zhTitle) + "</a>" : '<a class="next" href="index.html"><small>课程完成</small>回到架构地图</a>') +
      "</nav>";

    root.innerHTML =
      '<nav class="breadcrumbs"><a href="index.html">Attention Atlas</a> &nbsp;/&nbsp; Chapter ' + String(c.order).padStart(2, "0") + " &nbsp;/&nbsp; " + esc(c.title) + "</nav>" +
      '<header class="chapter-hero"><p class="eyebrow">Chapter ' + String(c.order).padStart(2, "0") + " · " + esc(c.fullTitle) + '</p><h1>' + esc(c.title) + '</h1><p class="chapter-deck">' + esc(c.zhTitle) + "。 " + esc(c.deck) + '</p><div class="chapter-meta"><b>' + esc(c.category) + "</b><span>" + esc(c.year) + "</span><span>难度 · " + esc(c.difficulty) + "</span><span>" + esc(c.report) + '</span></div></header>' +
      '<aside class="takeaway"><span>Mathematical Takeaway</span><p>' + c.takeaway + "</p></aside>" + toc +
      '<main class="chapter-main"><h2 data-no="01" id="sec-01">问题从哪里来</h2>' + motivation +
      '<h2 data-no="02" id="sec-02">设计限定条件</h2>' + renderCards(c.constraints, "constraint") +
      '<h2 data-no="03" id="sec-03">先抓住数学直觉</h2>' + renderCards(c.intuitions, "intuition") +
      '<h2 data-no="04" id="sec-04">架构图与交互实现</h2>' + renderDiagram(c.diagram, implementations[c.id]) +
      '<h2 data-no="05" id="sec-05">位置编码与时序注入</h2>' + renderPositionEncoding(c.positionEncoding) +
      '<h2 data-no="06" id="sec-06">数学推导</h2>' + derivations +
      '<div class="warning"><strong>边界与误区：</strong> ' + esc(c.warning) + "</div>" +
      '<h2 data-no="07" id="sec-07">练习与答案</h2>' + exercises +
      '<h2 data-no="08" id="authoritative-sources">权威来源</h2><ol class="source-list">' + sources + "</ol>" +
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
    initArchitectureWorkbench(root, implementations[c.id]);
    initCopyButtons(root);
    renderMath(root);
    initWorkbenchHeightSync(root);
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
