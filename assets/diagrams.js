(function () {
  "use strict";

  var P = {
    canvas: "var(--diagram-canvas, #fbf8f1)",
    ink: "var(--diagram-ink, #27323c)",
    muted: "var(--diagram-muted, #5f6b75)",
    rule: "var(--diagram-rule, #cbd2d5)",
    compute: "var(--diagram-compute, #dceafa)",
    computeStroke: "var(--diagram-compute-stroke, #5c82aa)",
    control: "var(--diagram-control, #e2f2e4)",
    controlStroke: "var(--diagram-control-stroke, #5f8d69)",
    state: "var(--diagram-state, #f6e1e4)",
    stateStroke: "var(--diagram-state-stroke, #a96773)",
    gather: "var(--diagram-gather, #ece4f6)",
    gatherStroke: "var(--diagram-gather-stroke, #826ca2)",
    cyan: "var(--diagram-cyan, #dff3f5)",
    cyanStroke: "var(--diagram-cyan-stroke, #397f8b)",
    orange: "var(--diagram-orange, #faead9)",
    orangeStroke: "var(--diagram-orange-stroke, #b87938)",
    paper: "var(--diagram-paper, #fffdf9)"
  };

  var buildSerial = 0;

  var guides = {
    mha: [
      ["原论文入口", "词嵌入先乘模型维度平方根，再加固定正弦位置编码；Q/K/V 投影发生在其后。"],
      ["完整子层", "缩放点积可选 decoder causal mask，多头拼接与输出投影之后才做 residual Add & Norm。"],
      ["现代缓存只是叠加层", "2017 训练图没有 KV cache；玫瑰色框与带标签虚线只说明现代增量解码。"],
      ["视觉语义", "蓝=计算，绿=控制，玫瑰=缓存/状态，薰衣草=聚合/写回；青色是公式注释，橙色是论文边界。"]
    ],
    mqa: [
      ["论文剖面", "Shazeer 基线使用 learned input positions；MQA 的创新是保留多 Q，只写一套共享 K/V。"],
      ["广播不是复制", "绿色路由表示 Hq 个 query 逻辑读取同一 K/V；高效实现不 materialize repeat。"],
      ["缓存收益", "玫瑰色历史状态形状为两份 [B,1,L,dh]，softmax 本身仍然精确。"],
      ["视觉语义", "蓝=计算，绿=控制，玫瑰=缓存/状态，薰衣草=聚合/写回；青/橙分别标注公式与年代边界。"]
    ],
    gqa: [
      ["T5 论文剖面", "原 GQA uptraining 继承 T5 的按 query head 相对位置偏置，而不是现代 Llama RoPE。"],
      ["显式组映射", "g(h)=floor(h/r)，r=Hq/Hkv；每个 Q 头只读取所属组的 K/V。"],
      ["Uptraining", "虚线训练 inset 表示组内 K/V 权重均值池化初始化，再继续训练；它不是推理路径。"],
      ["视觉语义", "蓝=计算，绿=控制，玫瑰=缓存/状态，薰衣草=聚合/写回；虚线只表示训练或可选路径。"]
    ],
    mla: [
      ["Decode 主路径", "缓存只含 cKV 与 kR；内容分数把 WUK 吸收到 query 侧，value 混合把 WUV 吸收到输出侧。"],
      ["正确缩放", "即使内容 query 被吸收到 dc 维，注意力仍按原始完整 head 宽度 sqrt(dh+dhR) 缩放。"],
      ["重建仅为等价解释", "虚线 inset 中的 kC/v 是训练或概念重建，不是 decode 主路径，也不是缓存内容。"],
      ["归一化边界", "图把原始下投影 latent 与 checkpoint 可选 RMSNorm 分开，避免把实现配方写成 MLA 定义。"],
      ["视觉语义", "玫瑰=持久状态，蓝=投影/attention，绿=可选控制，薰衣草=latent gather/write；青/橙为注释。"]
    ],
    dsa: [
      ["两条清晰车道", "上方 Indexer 生成 qI、kI、wI、全历史 logits 与 TopK；下方从 MLA latent cache gather 后运行候选 MLA。"],
      ["低精度对称路径", "qI 与 kI 都经过 partial RoPE、Hadamard 与 FP8；Hadamard 服务数值范围，不是位置编码。"],
      ["没有固定局部窗", "DSA 原型由内容 TopK 选择候选；图中不添加 local-window 捷径。"],
      ["训练监督", "teacher full logits 与 KL 仅通过带标签虚线连接，detach 后不属于推理图。"],
      ["视觉语义", "绿=TopK/路由，玫瑰=历史 cache，薰衣草=Gather；青/橙为精度和训练注释。"]
    ],
    csa: [
      ["双压缩器、双缓存", "core compressor 产生 CComp；独立 index compressor 产生 KIComp，二者参数和缓存职责不能合并。"],
      ["地址到内容", "KIComp 只负责打分与 TopK；索引垂直下传给 Gather，从 CComp cache 取候选。"],
      ["三路进入同一核心", "选中全局摘要、独立 SWA lane 与 query lane 汇入一次 shared-KV MQA + sink normalization。"],
      ["输出坐标", "partial-RoPE value 混合后先 inverse RoPE，再按组 WOA→concat→WOB。"],
      ["视觉语义", "蓝=计算，绿=选择，玫瑰=两类 cache，薰衣草=Gather/写回；青/橙为位置与因果注释。"]
    ],
    hca: [
      ["只压缩、不索引", "非重叠 compressor 只发布已完成块到 CComp cache；HCA 没有 indexer 或 TopK。"],
      ["Dense 读取短历史", "全部 completed CComp、独立 SWA lane 与 query lane 进入 shared-KV dense MQA + sink。"],
      ["位置与输出", "压缩和局部 entry 使用 partial RoPE；输出 inverse RoPE 后按组写回。"],
      ["视觉语义", "蓝=计算，绿=因果完成控制，玫瑰=cache，薰衣草=汇合/写回；青/橙为位置和边界注释。"]
    ],
    linear: [
      ["原始特征映射", "2020 Linear Transformer 使用 phi(x)=ELU(x)+1，使核可结合且非负。"],
      ["两份固定状态", "S 累积 phi(k)v^T，z 累积 phi(k)；query 分别读取分子和分母。"],
      ["训练表述保持克制", "原论文给出因果递推及自定义 GPU 实现；图不把后来的并行 scan/chunk kernel 冒充官方实现。"],
      ["视觉语义", "蓝=投影/读取，玫瑰=递推状态，薰衣草=归一化写回；橙色注释标明执行边界。"]
    ],
    delta: [
      ["参数路径必须分开", "只有 q/k/v 经过 causal ShortConv；alpha、beta、output gate g 直接由当前 hidden 投影。"],
      ["转置状态约定", "图用 F=S^T，形状 dv×dk；因此预测和读取写成 Fk、Fq。"],
      ["五步更新", "先 decay，再 predict，再形成 error，再 rank-1 write，最后用 q read；顺序决定语义。"],
      ["视觉语义", "蓝=特征计算，绿=gate/control，玫瑰=F 状态，薰衣草=write/read；虚线只标训练执行。"]
    ],
    kda: [
      ["逐通道衰减", "alpha 是 dk 维 direct gate；在转置约定 F=S^T 下，decay 写成 F Diag(alpha)。"],
      ["DPLR 次序不可交换", "列向量原式严格为 (I-beta kk^T)Diag(alpha)S + beta kv^T。"],
      ["参数路径", "q/k/v 各走 ShortConv；alpha、beta、g 直接投影，避免把 gate 错接到卷积支路。"],
      ["Checkpoint 尾部", "图按官方尾部写出六个 3:1 周期，再接 KDA×2→MLA-NoPE。"],
      ["视觉语义", "蓝=计算，绿=逐通道控制，玫瑰=状态，薰衣草=写回/层栈；橙色标注精确层序。"]
    ]
  };

  var memories = {
    mha: "2017 MHA：缩放 embedding 加正弦位置后投影多头，精确注意力写回，再做 post Add & Norm。",
    mqa: "MQA：很多独立 Q 逻辑广播到唯一共享 K/V；减少历史搬运，不近似 softmax。",
    gqa: "GQA：T5 query head 用 g(h) 找组内 K/V；MHA checkpoint 先组内均值再 uptrain。",
    mla: "MLA decode：直接用缓存 cKV+kR 做吸收式打分与读出；kC/v 重建只是等价解释。",
    dsa: "DSA：低维 FP8 Indexer 选地址，Gather 再把原始 MLA latent 交给精确候选 attention。",
    csa: "CSA：KIComp 负责找地址，CComp 提供内容；再与 SWA 一起进入唯一的 MQA+sink 核心。",
    hca: "HCA：只缓存已完成的重压缩块，不做 TopK；全部摘要与 SWA 一起 dense 读取。",
    linear: "Linear Transformer：ELU+1 把历史折进 S/z，query 用分子除以分母读取固定状态。",
    delta: "GDN：q/k/v 走 ShortConv，direct gates 控制 F=S^T 的 decay→predict→error→write→read。",
    kda: "KDA：逐 key-channel decay 后做 delta 纠写，并按精确 checkpoint 尾部与 NoPE MLA 交错。"
  };

  function toneFill(tone) {
    return P[tone] || P.paper;
  }

  function toneStroke(tone) {
    return P[tone + "Stroke"] || P.muted;
  }

  function defs(rootId) {
    var tones = ["compute", "control", "state", "gather", "cyan", "orange", "muted"];
    return "<defs>" + tones.map(function (tone) {
      var color = tone === "muted" ? P.muted : toneStroke(tone);
      return '<marker id="' + rootId + '-arrow-' + tone +
        '" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" ' +
        'refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="' +
        color + '"/></marker>';
    }).join("") + "</defs>";
  }

  function escapeText(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function M(tex, fallback) {
    return {
      tex: tex,
      fallback: fallback || tex
    };
  }

  function fallbackLabel(value) {
    if (value && typeof value === "object" && value.fallback) {
      return value.fallback;
    }
    return value == null ? "" : String(value);
  }

  function textLabel(x, y, value, fontSize, color, weight) {
    return '<text x="' + x + '" y="' + (y + fontSize * 0.34) +
      '" text-anchor="middle" font-family="JetBrains Mono" font-size="' + fontSize +
      '" font-weight="' + (weight || 400) + '" fill="' + color + '">' +
      escapeText(value) + '</text>';
  }

  function mathLabel(x, y, width, height, value, fontSize, color, weight) {
    var fallback = escapeText(value.fallback);
    var aria = escapeText(value.fallback);
    var tex = escapeText(value.tex);
    var left = x - width / 2;
    var top = y - height / 2;
    return (
      '<switch class="svg-math-switch" role="img" aria-label="' + aria + '">' +
      '<foreignObject class="svg-math-label-wrap" x="' + left + '" y="' + top + '" width="' + width + '" height="' + height + '" requiredExtensions="http://www.w3.org/1999/xhtml">' +
      '<div xmlns="http://www.w3.org/1999/xhtml" class="svg-math-label" style="color:' + color + ';font-size:' + fontSize + 'px;font-weight:' + (weight || 400) + '"><span aria-hidden="true">\\(' + tex + '\\)</span></div>' +
      '</foreignObject>' +
      '<text class="svg-math-fallback" x="' + x + '" y="' + (y + fontSize * 0.34) + '" text-anchor="middle" font-family="JetBrains Mono" font-size="' + fontSize + '" font-weight="' + (weight || 400) + '" fill="' + color + '">' + fallback + '</text>' +
      '</switch>'
    );
  }

  function labelMarkup(x, y, width, height, value, fontSize, color, weight) {
    if (value && typeof value === "object" && value.tex) {
      return mathLabel(x, y, width, height, value, fontSize, color, weight);
    }
    return textLabel(x, y, value, fontSize, color, weight);
  }

  function panel(x, y, w, h, title, tone, dashed) {
    var stroke = toneStroke(tone);
    return (
      '<g class="diagram-panel"><rect x="' + x + '" y="' + y + '" width="' + w +
      '" height="' + h + '" rx="14" fill="' + toneFill(tone) +
      '" fill-opacity=".22" stroke="' + stroke + '" stroke-opacity=".62" stroke-width="1.2" ' +
      (dashed ? 'stroke-dasharray="7 6" ' : "") + '/>' +
      '<rect x="' + (x + 12) + '" y="' + (y - 9) + '" width="' +
      Math.max(124, title.length * 7.2 + 20) + '" height="20" fill="' + P.canvas + '"/>' +
      '<text x="' + (x + 20) + '" y="' + (y + 5) +
      '" font-family="JetBrains Mono" font-size="10.5" font-weight="600" fill="' +
      stroke + '">' + escapeText(title) + '</text></g>'
    );
  }

  function box(x, y, w, h, title, sub, tone, n, codeBlockId, options) {
    options = options || {};
    if (x < 24 || y < 24 || x + w > 1076) {
      throw new Error("Diagram node violates horizontal/top margin: " + fallbackLabel(title));
    }
    var fill = toneFill(tone);
    var stroke = toneStroke(tone);
    var number = n != null
      ? '<circle cx="' + (x + 14) + '" cy="' + (y + 14) + '" r="9" fill="' +
        stroke + '"/>' + textLabel(x + 14, y + 14, n, 8.5, P.paper, 700)
      : "";
    var titleY = y + h / 2 - (sub ? 5 : 0);
    var blockId = escapeText(codeBlockId);
    var aria = escapeText("查看代码块 " + codeBlockId + "：" + fallbackLabel(title));
    return (
      '<g class="diagram-code-node" data-code-block="' + blockId +
      '" role="button" tabindex="0" aria-label="' + aria + '">' +
      '<rect class="diagram-node-box" x="' + x + '" y="' + y + '" width="' + w +
      '" height="' + h + '" rx="9" fill="' + fill + '" stroke="' + stroke +
      '" stroke-width="1.35" ' + (options.dashed ? 'stroke-dasharray="6 5" ' : "") + '/>' +
      number +
      labelMarkup(x + w / 2, titleY, w - 22, 34, title, options.titleSize || 11.5, P.ink, 600) +
      (sub ? labelMarkup(x + w / 2, y + h / 2 + 16, w - 20, 25, sub,
        options.subSize || 8.8, P.muted, 500) : "") +
      '</g>'
    );
  }

  function cacheBox(x, y, w, h, title, sub, n, codeBlockId, options) {
    return box(x, y, w, h, title, sub, "state", n, codeBlockId, options);
  }

  function ortho(x1, y1, x2, y2, axis, turn) {
    if (x1 === x2) return "M" + x1 + " " + y1 + "V" + y2;
    if (y1 === y2) return "M" + x1 + " " + y1 + "H" + x2;
    if (axis === "y") {
      var bendY = turn == null ? (y1 + y2) / 2 : turn;
      return "M" + x1 + " " + y1 + "V" + bendY + "H" + x2 + "V" + y2;
    }
    var bendX = turn == null ? (x1 + x2) / 2 : turn;
    return "M" + x1 + " " + y1 + "H" + bendX + "V" + y2 + "H" + x2;
  }

  function edge(rootId, d, label, tone, dashed) {
    tone = tone || "muted";
    if (dashed && !label) {
      throw new Error("Dashed diagram edge requires an explicit label");
    }
    var color = tone === "muted" ? P.muted : toneStroke(tone);
    return (
      '<g><path d="' + d + '" fill="none" stroke="' + (color || P.muted) + '" stroke-width="1.5" ' +
      (dashed ? 'stroke-dasharray="6 5" ' : "") +
      'stroke-linecap="square" stroke-linejoin="round" marker-end="url(#' +
      rootId + '-arrow-' + tone + ')"/>' +
      (label ? labelMarkup(label[0], label[1], label[3] || 190, 24, label[2],
        label[4] || 8.8, color, 600) : "") +
      '</g>'
    );
  }

  function baseSvg(rootId, diagramKey, height, body, label) {
    var style = [
      "--diagram-canvas:#fbf8f1",
      "--diagram-paper:#fffdf9",
      "--diagram-ink:#27323c",
      "--diagram-muted:#5f6b75",
      "--diagram-rule:#cbd2d5",
      "--diagram-compute:#dceafa",
      "--diagram-compute-stroke:#5c82aa",
      "--diagram-control:#e2f2e4",
      "--diagram-control-stroke:#5f8d69",
      "--diagram-state:#f6e1e4",
      "--diagram-state-stroke:#a96773",
      "--diagram-gather:#ece4f6",
      "--diagram-gather-stroke:#826ca2",
      "--diagram-cyan:#dff3f5",
      "--diagram-cyan-stroke:#397f8b",
      "--diagram-orange:#faead9",
      "--diagram-orange-stroke:#b87938"
    ].join(";");
    return (
      '<svg viewBox="0 0 1100 ' + height + '" role="img" aria-label="' + escapeText(label) + '" data-diagram-key="' + escapeText(diagramKey) + '" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono">' +
      '<g style="' + style + '">' + defs(rootId) +
      '<rect width="1100" height="' + height + '" fill="' + P.canvas + '"/>' +
      body +
      '</g></svg>'
    );
  }

  // Static SVG-coordinate guard: every connector is axis-aligned and may touch
  // node boundaries only at its endpoints; no segment may traverse a node.
  function validateStaticGeometry(svg, diagramKey) {
    function attributes(tag) {
      var result = {};
      var match;
      var pattern = /([\w:-]+)="([^"]*)"/g;
      while ((match = pattern.exec(tag))) result[match[1]] = match[2];
      return result;
    }

    function pathPoints(d) {
      var points = [];
      var x = 0;
      var y = 0;
      var match;
      var commands = /([MHV])\s*(-?[\d.]+)(?:\s+(-?[\d.]+))?/g;
      while ((match = commands.exec(d))) {
        if (match[1] === "M") {
          x = Number(match[2]);
          y = Number(match[3]);
        } else if (match[1] === "H") {
          x = Number(match[2]);
        } else {
          y = Number(match[2]);
        }
        points.push({ x: x, y: y });
      }
      return points;
    }

    function crossesInterior(a, z, node) {
      var epsilon = 0.75;
      if (a.x === z.x) {
        return a.x > node.x + epsilon && a.x < node.x + node.w - epsilon &&
          Math.max(Math.min(a.y, z.y), node.y + epsilon) <
          Math.min(Math.max(a.y, z.y), node.y + node.h - epsilon);
      }
      if (a.y === z.y) {
        return a.y > node.y + epsilon && a.y < node.y + node.h - epsilon &&
          Math.max(Math.min(a.x, z.x), node.x + epsilon) <
          Math.min(Math.max(a.x, z.x), node.x + node.w - epsilon);
      }
      throw new Error(diagramKey + ": connector is not orthogonal");
    }

    if (/<pattern\b/i.test(svg) || /[\u2080-\u2089]/.test(svg)) {
      throw new Error(diagramKey + ": forbidden pattern or Unicode subscript");
    }

    var boxes = [];
    var boxMatch;
    var boxPattern = /<rect class="diagram-node-box"[^>]*>/g;
    while ((boxMatch = boxPattern.exec(svg))) {
      var boxAttrs = attributes(boxMatch[0]);
      boxes.push({
        x: Number(boxAttrs.x),
        y: Number(boxAttrs.y),
        w: Number(boxAttrs.width),
        h: Number(boxAttrs.height)
      });
    }

    var edgeMatch;
    var edgePattern = /<path d="([^"]+)"[^>]*marker-end/g;
    while ((edgeMatch = edgePattern.exec(svg))) {
      var d = edgeMatch[1];
      var points = pathPoints(d);
      for (var i = 1; i < points.length; i += 1) {
        for (var j = 0; j < boxes.length; j += 1) {
          if (crossesInterior(points[i - 1], points[i], boxes[j])) {
            throw new Error(diagramKey + ": connector traverses node: " + d);
          }
        }
      }
    }

    var dashedMatch;
    var dashedPattern = /<g><path\b[^>]*stroke-dasharray="[^"]+"[^>]*\/>([\s\S]*?)<\/g>/g;
    while ((dashedMatch = dashedPattern.exec(svg))) {
      if (!/<(?:text|switch)\b/.test(dashedMatch[1])) {
        throw new Error(diagramKey + ": dashed connector lacks a label");
      }
    }

    var implementationKey = diagramKey === "delta" ? "gated-delta" : diagramKey;
    var implementation = window.ATTENTION_IMPLEMENTATIONS &&
      window.ATTENTION_IMPLEMENTATIONS[implementationKey];
    if (implementation) {
      var validIds = {};
      implementation.blocks.forEach(function (block) { validIds[block.id] = true; });
      var idMatch;
      var idPattern = /data-code-block="([^"]+)"/g;
      while ((idMatch = idPattern.exec(svg))) {
        if (!validIds[String(idMatch[1]).padStart(2, "0")]) {
          throw new Error(diagramKey + ": invalid code block " + idMatch[1]);
        }
      }
    }
  }

  function mhaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 526, 548, "2017 TRANSFORMER INPUT & MULTI-HEAD SUBLAYER", "compute");
    b += panel(568, 52, 508, 548, "EXACT ATTENTION, WRITE-BACK & MODERN OVERLAY", "gather");

    b += box(40, 254, 128, 72, M("E_{\\mathrm{token}}", "Token embedding"),
      M("[B,L,d_{\\mathrm{model}}]", "[B,L,dmodel]"), "compute", 1, "03");
    b += box(202, 254, 154, 72, M("\\sqrt{d_{\\mathrm{model}}}\\,E", "Scale embedding"),
      "2017 input scaling", "compute", 2, "03");
    b += box(202, 104, 154, 72, M("\\operatorname{PE}_{\\sin/\\cos}(p)", "Sinusoidal PE"),
      "fixed absolute position", "compute", 3, "01");
    b += box(392, 226, 142, 92, M("X=\\sqrt d\\,E+\\operatorname{PE}", "Add position"),
      "shared Q/K/V source", "gather", 4, "03");

    b += box(584, 92, 174, 66, M("Q_h=XW_h^Q", "Independent Q heads"),
      M("h=1,\\ldots,H", "h = 1…H"), "compute", null, "04");
    b += box(584, 218, 174, 66, M("K_h=XW_h^K", "Independent K heads"),
      M("H_{kv}=H_q=H", "Hkv = Hq = H"), "compute", null, "04");
    b += box(584, 344, 174, 66, M("V_h=XW_h^V", "Independent V heads"),
      "one value space / head", "compute", null, "04");
    b += box(584, 476, 174, 66, M("M_{\\mathrm{causal}}", "Decoder causal mask"),
      "optional · decoder only", "control", null, "06", { dashed: true });

    b += box(794, 92, 266, 92,
      M("S_h=Q_hK_h^{\\mathsf T}/\\sqrt{d_h}+M", "Scaled dot-product scores"),
      "M omitted in encoder self-attention", "compute", 5, "06");
    b += box(794, 234, 266, 82,
      M("O_h=\\operatorname{softmax}(S_h)V_h", "Exact softmax × V"),
      "one output per head", "compute", 6, "06");
    b += box(794, 362, 126, 72, M("\\operatorname{Concat}(O_h)W^O", "Concat → WO"),
      M("Hd_h\\to d", "H·dh → d"), "gather", 7, "07", { titleSize: 10.6 });
    b += box(934, 362, 126, 72, "Add & Norm",
      "post-norm · 2017", "gather", 8, "07");

    b += cacheBox(794, 490, 266, 70, "Modern decode KV cache",
      M("K_{1:t},V_{1:t}\\;\\text{only}", "solid rose · K/V only"), null, "05", { dashed: true });

    b += edge(rootId, ortho(168, 290, 202, 290), null, "compute");
    b += edge(rootId, ortho(356, 290, 392, 272), null, "gather");
    b += edge(rootId, ortho(279, 176, 463, 226, "y", 202), null, "compute");
    b += edge(rootId, ortho(534, 272, 584, 125), null, "compute");
    b += edge(rootId, ortho(534, 272, 584, 251), null, "compute");
    b += edge(rootId, ortho(534, 272, 584, 377), null, "compute");
    b += edge(rootId, ortho(758, 125, 794, 138), null, "compute");
    b += edge(rootId, ortho(758, 251, 794, 138), null, "compute");
    b += edge(rootId, ortho(671, 476, 794, 166, "x", 776),
      [748, 462, "OPTIONAL · decoder mask", 176], "control", true);
    b += edge(rootId, ortho(927, 184, 927, 234), null, "compute");
    b += edge(rootId, ortho(758, 377, 794, 275), null, "compute");
    b += edge(rootId, ortho(927, 316, 857, 362), null, "gather");
    b += edge(rootId, ortho(920, 398, 934, 398), null, "gather");
    b += edge(rootId, "M104 326V582H1070V398H1060",
      [522, 584, "residual stream · right rail", 214], "gather");
    b += edge(rootId, "M758 251H780V466H827V490",
      [714, 448, "OPTIONAL · modern K append", 214], "state", true);
    b += edge(rootId, ortho(671, 410, 1027, 490, "y", 466),
      [886, 460, "OPTIONAL · modern V append", 214], "state", true);
    return baseSvg(rootId, "mha", 630, b,
      "2017 MHA with scaled embedding, sinusoidal positions, post Add and Norm, and optional modern cache overlay");
  }

  function mqaDiagram(rootId) {
    var b = "";
    b += panel(24, 54, 512, 510, "SHAZEER 2019 INPUT & ONE WRITE-HEAD", "compute");
    b += panel(554, 54, 522, 510, "LOGICAL BROADCAST & EXACT MULTI-QUERY READ", "control");

    b += box(40, 252, 126, 70, M("E_t", "Token embedding"),
      M("[B,L,d]", "[B,L,d]"), "compute", 1, "03");
    b += box(196, 100, 184, 70, M("P_t^{\\mathrm{learned}}", "Learned input position"),
      "paper baseline profile", "compute", 2, "04");
    b += box(196, 252, 184, 70, M("X_t=E_t+P_t", "Add learned position"),
      "before projections", "gather", 3, "03");
    b += box(410, 102, 110, 80, M("Q_{1:H_q}", "Many Q heads"),
      M("[B,H_q,L,d_h]", "[B,Hq,L,dh]"), "compute", null, "03");
    b += box(410, 340, 110, 80, M("K,V", "One shared K/V"),
      M("2\\times[B,1,L,d_h]", "2 × [B,1,L,dh]"), "compute", null, "03");

    b += cacheBox(578, 340, 180, 80, "Shared KV cache",
      M("H_{kv}=1", "Hkv = 1 · one write-head"), 4, "05");
    b += box(578, 112, 180, 82, "Logical KV broadcast",
      M("[B,1,L,d_h]\\rightsquigarrow H_q", "stride-0 read · no repeat"), "control", 5, "06");
    b += box(806, 122, 246, 112,
      M("S_h=Q_hK^{\\mathsf T}/\\sqrt{d_h}+M", "Per-Q-head exact scores"),
      M("h=1,\\ldots,H_q", "shared K · independent logits"), "compute", 6, "06");
    b += box(806, 288, 246, 84,
      M("O_h=\\operatorname{softmax}(S_h)V", "Softmax × shared V"),
      "logical broadcast on V", "compute", 7, "06");
    b += box(806, 430, 246, 72,
      M("\\operatorname{Concat}(O_h)W^O", "Concat heads → WO"),
      "residual-stream output", "gather", 8, "07");
    b += box(578, 462, 180, 62, "2019 scope",
      "sharing change · not a new PE", "orange", null, "01");

    b += edge(rootId, ortho(166, 287, 196, 287), null, "compute");
    b += edge(rootId, ortho(288, 170, 288, 252), null, "compute");
    b += edge(rootId, ortho(380, 287, 410, 142), null, "compute");
    b += edge(rootId, ortho(380, 287, 410, 380), null, "compute");
    b += edge(rootId, ortho(520, 142, 578, 153), null, "control");
    b += edge(rootId, ortho(520, 380, 578, 380), null, "state");
    b += edge(rootId, ortho(668, 340, 668, 194), null, "control");
    b += edge(rootId, ortho(758, 153, 806, 178), null, "control");
    b += edge(rootId, ortho(929, 234, 929, 288), null, "compute");
    b += edge(rootId, ortho(758, 380, 806, 330), null, "state");
    b += edge(rootId, ortho(929, 372, 929, 430), null, "gather");
    b += edge(rootId, ortho(668, 420, 668, 462), null, "orange");
    return baseSvg(rootId, "mqa", 592, b,
      "2019 MQA with learned input positions, many query heads, one shared KV head, and logical broadcasting");
  }

  function gqaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 1052, 398, "GQA PAPER PROFILE · T5 RELATIVE BIAS & GROUPED KV", "compute");
    b += panel(74, 492, 952, 164, "TRAINING ONLY · MHA CHECKPOINT TO GQA UPtraining", "orange", true);

    b += box(40, 202, 118, 70, M("X_t", "T5 hidden"),
      M("[B,L,d]", "[B,L,d]"), "compute", 1, "03");
    b += box(190, 88, 170, 72, M("Q_h=XW_h^Q", "Hq query heads"),
      M("h=0,\\ldots,H_q-1", "query head h"), "compute", null, "03");
    b += box(190, 304, 170, 72, M("K_g,V_g", "Hkv grouped K/V"),
      M("g=0,\\ldots,H_{kv}-1", "group index g"), "compute", null, "03");
    b += box(400, 174, 188, 86,
      M("r=H_q/H_{kv},\\quad g(h)=\\lfloor h/r\\rfloor", "Map query head to KV group"),
      "logical lookup · no repeat", "control", 2, "06", { titleSize: 10.4 });
    b += box(400, 70, 188, 72,
      M("b_{h,\\operatorname{bucket}(t-s)}", "T5 relative bias by head"),
      "inherited from checkpoint", "compute", 3, "04");
    b += cacheBox(400, 304, 188, 72, "Grouped KV cache",
      M("2\\times[B,H_{kv},L,d_h]", "K_g and V_g"), 4, "05");
    b += box(632, 94, 244, 106,
      M("S_h=Q_hK_{g(h)}^{\\mathsf T}/\\sqrt{d_h}+B_h^{\\mathrm{T5}}+M",
        "Grouped score + head-indexed T5 bias"),
      "paper-faithful score", "compute", 5, "07", { titleSize: 9.6 });
    b += box(632, 270, 244, 88,
      M("O_h=\\operatorname{softmax}_s(S_h)V_{g(h)}", "Exact groupwise read"),
      "one output / query head", "compute", 6, "07");
    b += box(914, 176, 140, 96,
      M("\\operatorname{Concat}(O_h)W^O", "Concat → WO"),
      "write residual", "gather", 7, "08", { titleSize: 10.4 });

    b += box(100, 536, 170, 72, M("W_{1:H_q}^{K,V}", "MHA K/V weights"),
      "pretrained checkpoint", "state", null, "01");
    b += box(330, 536, 190, 72,
      M("\\bar W_g^{K,V}=r^{-1}\\!\\sum_{h:g(h)=g}W_h^{K,V}", "Groupwise mean pool"),
      "initialization only", "gather", null, "01", { titleSize: 9.4 });
    b += box(580, 536, 170, 72, M("W_{1:H_{kv}}^{K,V}", "GQA K/V init"),
      "fewer write heads", "state", null, "02");
    b += box(810, 536, 190, 72, "Continue pretraining",
      "paper recipe · about 5%", "control", null, "02");

    b += edge(rootId, ortho(158, 237, 190, 124), null, "compute");
    b += edge(rootId, ortho(158, 237, 190, 340), null, "compute");
    b += edge(rootId, ortho(360, 124, 400, 217), null, "control");
    b += edge(rootId, ortho(360, 340, 400, 340), null, "state");
    b += edge(rootId, ortho(494, 304, 494, 260), null, "control");
    b += edge(rootId, ortho(588, 217, 632, 147), null, "control");
    b += edge(rootId, ortho(494, 142, 632, 147), null, "compute");
    b += edge(rootId, ortho(754, 200, 754, 270), null, "compute");
    b += edge(rootId, ortho(588, 340, 632, 314), null, "state");
    b += edge(rootId, ortho(876, 314, 914, 224), null, "gather");
    b += edge(rootId, ortho(270, 572, 330, 572),
      [300, 516, "TRAINING · pool within g", 180], "orange", true);
    b += edge(rootId, ortho(520, 572, 580, 572),
      [550, 630, "TRAINING · initialize", 174], "orange", true);
    b += edge(rootId, ortho(750, 572, 810, 572),
      [780, 516, "TRAINING · uptrain", 160], "orange", true);
    return baseSvg(rootId, "gqa", 680, b,
      "GQA with T5 relative position bias, explicit query-to-KV grouping, and groupwise mean-pool uptraining");
  }

  function mlaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 520, 222, "QUERY LATENT · RAW DEFINITION VS CHECKPOINT NORM", "compute");
    b += panel(24, 300, 520, 220, "JOINT KV LATENT & DECOUPLED ROPE CACHE", "state");
    b += panel(566, 52, 510, 468, "ABSORBED DECODE · NO EXPLICIT kC OR v", "compute");
    b += panel(180, 558, 740, 116, "CONCEPTUAL EQUIVALENCE ONLY · RECONSTRUCTION IS NOT CACHED", "orange", true);

    b += box(40, 222, 116, 62, M("h_t", "Input hidden"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "03");
    b += box(186, 82, 156, 68, M("c_t^Q=W^{DQ}h_t", "Raw query latent"),
      M("c_t^Q\\in\\mathbb R^{d_q}", "definition · raw cq"), "compute", 2, "04");
    b += box(186, 180, 156, 62, M("\\operatorname{RMSNorm}(c_t^Q)", "RMSNorm(cq)"),
      "checkpoint-dependent", "control", null, "04", { dashed: true });
    b += box(362, 92, 166, 82,
      M("q_i^C=W_i^{UQ}\\bar c_t^Q", "Content query qC"),
      M("q_i^R=R_tW_i^{QR}\\bar c_t^Q", "RoPE query qR"),
      "compute", 3, "06", { titleSize: 8.8, subSize: 8.0 });

    b += box(186, 330, 156, 68, M("c_t^{KV}=W^{DKV}h_t", "Raw joint KV latent"),
      M("c_t^{KV}\\in\\mathbb R^{d_c}", "definition · raw cKV"), "compute", 4, "05");
    b += box(186, 430, 156, 62, M("\\operatorname{RMSNorm}(c_t^{KV})", "RMSNorm(cKV)"),
      "checkpoint-dependent", "control", null, "05", { dashed: true });
    b += cacheBox(382, 320, 146, 78, M("c_{1:t}^{KV}", "Cached cKV history"),
      "raw or checkpoint-normalized", 5, "07");
    b += cacheBox(382, 430, 146, 66, M("k_{1:t}^{R}=R_sW^{KR}h_s", "Cached RoPE key kR"),
      M("d_h^R\\;\\text{shared}", "shared positional slice"), 6, "06", { titleSize: 9.8 });

    b += box(566, 92, 210, 84,
      M("\\widetilde q_i^C=(W_i^{UK})^{\\mathsf T}q_i^C", "Define absorbed query q-tilde"),
      M("\\widetilde q_i^C\\in\\mathbb R^{d_c}", "one compact latent-space symbol"),
      "compute", 7, "08", { titleSize: 9.2, subSize: 8.2 });
    b += box(798, 104, 250, 82,
      M("n_{i,s}=\\widetilde q_i^{C\\mathsf T}c_s^{KV}+q_i^{R\\mathsf T}k_s^R",
        "Define compact score numerator n"),
      "content + decoupled RoPE", "compute", 8, "09", { titleSize: 8.8 });
    b += box(798, 212, 250, 62,
      M("s_{i,s}=n_{i,s}/\\sqrt{d_h+d_h^R}+M_{t,s}", "Scale n, then add mask"),
      "correct full-head scaling", "compute", null, "09", { titleSize: 9.1 });
    b += box(798, 308, 250, 82,
      M("a_{i,:}=\\operatorname{softmax}(s_{i,:})", "Candidate weights"),
      M("m_i=\\sum_s a_{i,s}c_s^{KV}", "latent read m_i"), "gather", 9, "10",
      { titleSize: 9.2, subSize: 8.5 });
    b += box(798, 424, 250, 72,
      M("u_t=W^O\\operatorname{Concat}_i(W_i^{UV}m_{t,i})", "Absorbed value/output write"),
      "no historical v reconstruction", "gather", 10, "11", { titleSize: 9.7 });

    b += box(230, 586, 170, 62, M("k_{s,i}^C=W_i^{UK}c_s^{KV}", "Conceptual kC"),
      "equivalent training graph", "orange", null, "08", { dashed: true, titleSize: 9.8 });
    b += box(465, 586, 170, 62, M("v_{s,i}=W_i^{UV}c_s^{KV}", "Conceptual v"),
      "equivalent training graph", "orange", null, "10", { dashed: true, titleSize: 9.8 });
    b += box(700, 586, 170, 62, M("[q_i^C;q_i^R]\\cdot[k_i^C;k^R]", "Reconstructed score"),
      "numerically equivalent", "cyan", null, "09", { dashed: true, titleSize: 9.6 });

    b += edge(rootId, ortho(156, 253, 186, 116), null, "compute");
    b += edge(rootId, ortho(156, 253, 186, 364), null, "compute");
    b += edge(rootId, ortho(156, 253, 382, 463, "x", 366), null, "state");
    b += edge(rootId, ortho(342, 116, 362, 133), null, "compute");
    b += edge(rootId, ortho(264, 150, 264, 180),
      [142, 166, "OPTIONAL · checkpoint RMSNorm", 222], "control", true);
    b += edge(rootId, ortho(342, 211, 362, 150),
      [354, 198, "OPTIONAL · normalized cq", 196], "control", true);
    b += edge(rootId, ortho(342, 364, 382, 359), null, "state");
    b += edge(rootId, ortho(264, 398, 264, 430),
      [142, 414, "OPTIONAL · checkpoint RMSNorm", 222], "control", true);
    b += edge(rootId, ortho(342, 461, 382, 378),
      [362, 446, "OPTIONAL · normalized cKV", 210], "control", true);
    b += edge(rootId, ortho(528, 133, 566, 134), null, "compute");
    b += edge(rootId, ortho(776, 134, 798, 145), null, "compute");
    b += edge(rootId, "M528 359H780V138H798", null, "state");
    b += edge(rootId, "M528 463H786V164H798", null, "state");
    b += edge(rootId, ortho(923, 186, 923, 212), null, "compute");
    b += edge(rootId, ortho(923, 274, 923, 308), null, "compute");
    b += edge(rootId, ortho(923, 390, 923, 424), null, "gather");
    b += edge(rootId, "M455 398H548V548H315V586",
      [314, 540, "OPTIONAL · conceptual reconstruction", 254], "orange", true);
    b += edge(rootId, "M475 398H552V566H550V586",
      [558, 538, "OPTIONAL · conceptual reconstruction", 254], "orange", true);
    b += edge(rootId, "M315 648V660H785V648",
      [550, 660, "OPTIONAL · equivalence check", 220], "orange", true);
    return baseSvg(rootId, "mla", 704, b,
      "MLA absorbed decode over cached cKV and kR, with optional checkpoint normalization and conceptual reconstruction inset");
  }

  function dsaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 1052, 256, "UPPER LANE · LIGHTNING INDEXER SCORES FULL HISTORY", "control");
    b += panel(24, 340, 1052, 260, "LOWER LANE · MLA LATENT CACHE → GATHER → CANDIDATE MLA", "gather");

    b += box(40, 104, 112, 62, M("h_t", "Query hidden"),
      "detached index input", "compute", 1, "03");
    b += box(40, 214, 112, 62, M("h_{1:L}", "History hidden"),
      "index-key source", "compute", null, "03");
    b += box(184, 88, 178, 82, M("q_{t,j}^I", "Indexer qI heads"),
      "pRoPE → Hadamard → FP8", "compute", 2, "03");
    b += cacheBox(184, 202, 178, 82, M("k_{1:L}^I", "Indexer kI cache"),
      "pRoPE → Hadamard → FP8", 3, "03");
    b += box(398, 74, 132, 64, M("w_{t,j}^I", "Head weights wI"),
      "direct query projection", "control", null, "03");
    b += box(398, 170, 214, 94,
      M("I_{t,s}=\\sum_jw_{t,j}^I\\operatorname{ReLU}(q_{t,j}^{I\\mathsf T}k_s^I)",
        "Full-history Indexer logits"),
      M("[B,L_q,L_k]", "score every history position"), "compute", 4, "03", { titleSize: 8.8 });
    b += box(646, 176, 166, 80, M("\\mathcal I_t=\\operatorname{TopK}_s(I_{t,s},k)", "TopK addresses"),
      "indices only · no values", "control", 5, "03", { titleSize: 9.8 });

    b += box(844, 68, 202, 72, "Teacher full MLA logits",
      "detach · dense warm-up target", "orange", null, "02", { dashed: true });
    b += box(844, 194, 202, 72,
      M("D_{\\mathrm{KL}}(p_t\\Vert\\operatorname{softmax}I_t)", "Indexer KL loss"),
      "training only", "orange", null, "02", { dashed: true });

    b += cacheBox(328, 402, 250, 86, "MLA latent cache",
      M("\\{c_s^{KV},k_s^R\\}_{s=1}^{L}", "full history · original latent"), 6, "03");
    b += box(646, 390, 166, 88,
      M("\\operatorname{Gather}(\\{c^{KV},k^R\\},\\mathcal I_t)", "Gather latent candidates"),
      "addresses select original MLA state", "gather", 7, "01", { titleSize: 9.6 });
    b += box(646, 508, 166, 64, M("q_t^C,q_t^R", "MLA query"),
      "high-dimensional query", "compute", null, "01");
    b += box(844, 386, 202, 108,
      M("a_t=\\operatorname{softmax}_{\\mathcal I_t}(QK^{\\mathsf T}/\\sqrt d),\\quad o_t=a_tV",
        "Candidate-only exact MLA"),
      "renormalize inside selected set", "compute", 8, "03", { titleSize: 9.5 });
    b += box(844, 524, 202, 58, M("W^O\\to u_t", "Output projection"),
      "residual write", "gather", 9, "03");

    b += edge(rootId, ortho(152, 135, 184, 129), null, "compute");
    b += edge(rootId, ortho(152, 245, 184, 243), null, "state");
    b += edge(rootId, "M96 104V62H464V74", null, "control");
    b += edge(rootId, ortho(362, 129, 398, 212), null, "compute");
    b += edge(rootId, ortho(362, 243, 398, 228), null, "state");
    b += edge(rootId, ortho(464, 138, 464, 170), null, "control");
    b += edge(rootId, ortho(612, 217, 646, 216), null, "control");
    b += edge(rootId, ortho(729, 256, 729, 390),
      [776, 326, "selected addresses", 174], "control");
    b += edge(rootId, ortho(578, 445, 646, 434), null, "gather");
    b += edge(rootId, ortho(812, 434, 844, 440), null, "gather");
    b += edge(rootId, ortho(812, 540, 844, 470), null, "compute");
    b += edge(rootId, ortho(945, 494, 945, 524), null, "gather");
    b += edge(rootId, ortho(945, 140, 945, 194),
      [1008, 165, "TRAINING · teacher p", 180], "orange", true);
    b += edge(rootId, "M612 180H626V154H826V230H844",
      [736, 150, "TRAINING · full logits I", 190], "orange", true);
    return baseSvg(rootId, "dsa", 628, b,
      "DSA with a full-history low-precision Indexer lane and a separate gathered MLA candidate lane");
  }

  function csaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 348, 548, "TWO INDEPENDENT OVERLAP COMPRESSORS", "compute");
    b += panel(390, 52, 370, 548, "SEPARATE CComp / KIComp CACHES & ROUTING", "control");
    b += panel(778, 52, 298, 598, "SWA + QUERY + ONE MQA/SINK NORMALIZATION", "gather");

    b += box(40, 252, 112, 64, M("H_{1:L}", "Hidden sequence"),
      M("[B,L,d]", "[B,L,d]"), "compute", 1, "02");
    b += box(184, 94, 168, 106,
      M("C^{\\mathrm{Comp}}=\\mathcal C_{\\mathrm{overlap}}(H;\\theta_C)", "Core overlap compressor"),
      "a/b streams · per-channel softmax over 2m", "compute", 2, "02", { titleSize: 9.8, subSize: 8.2 });
    b += box(184, 344, 168, 106,
      M("K^{I\\mathrm{Comp}}=\\mathcal C_{\\mathrm{overlap}}(H;\\theta_I)", "Index overlap compressor"),
      M("\\theta_I\\ne\\theta_C", "independent a/b projections"), "compute", 3, "02", { titleSize: 9.8 });

    b += cacheBox(410, 92, 160, 88, M("C^{\\mathrm{Comp}}_{1:L/m}", "CComp cache"),
      "core content entries", 4, "03");
    b += cacheBox(410, 242, 160, 88, M("K^{I\\mathrm{Comp}}_{1:L/m}", "KIComp cache"),
      "index keys only", 5, "03");
    b += box(410, 382, 160, 70, M("q_t^I,w_t^I", "Indexer query + weights"),
      "independent query path", "compute", null, "03");
    b += box(594, 222, 146, 86,
      M("I_t=\\operatorname{Index}(q_t^I,w_t^I,K^{I\\mathrm{Comp}})", "Score KIComp entries"),
      "weighted ReLU logits", "compute", 6, "03", { titleSize: 8.8 });
    b += box(594, 346, 146, 70, M("\\mathcal J_t=\\operatorname{TopK}(I_t)", "TopK addresses"),
      "V4-Flash / Pro config", "control", 7, "03", { titleSize: 9.8 });
    b += box(594, 468, 146, 78,
      M("\\operatorname{Gather}(C^{\\mathrm{Comp}},\\mathcal J_t)", "Gather CComp"),
      "selected global summaries", "gather", 8, "03", { titleSize: 9.6 });

    b += cacheBox(800, 82, 116, 84, "SWA cache lane",
      "raw recent entries · pRoPE", 9, "03", { subSize: 8.2 });
    b += box(938, 82, 116, 84, "Query lane",
      M("q_t\\;\\cdot\\;\\operatorname{pRoPE}_{64}", "Hq heads · trailing-64"),
      "compute", 10, "03",
      { subSize: 8.1 });
    b += box(800, 222, 254, 112,
      M("\\operatorname{MQA}_{K=V}(q_t;\\mathrm{sink},C_{\\mathcal J_t},C_{\\mathrm{SWA}})",
        "One shared-KV MQA + sink"),
      M("a_t=\\operatorname{softmax}([s_{\\mathrm{sink}},s_{\\mathrm{global}},s_{\\mathrm{SWA}}])",
        "single normalization: sink, global, SWA"), "compute", 11, "03",
      { titleSize: 9.2, subSize: 8.1 });
    b += box(800, 374, 254, 66, M("\\operatorname{RoPE}^{-1}_{64}(-t)", "Inverse partial RoPE"),
      "return value slice to query frame", "compute", 12, "01");
    b += box(800, 478, 254, 70,
      M("W^{OA}_{\\mathrm{group}}\\to\\operatorname{Concat}\\to W^{OB}", "Grouped output projection"),
      "low-rank groups → residual width", "gather", 13, "03", { titleSize: 9.6 });
    b += box(800, 584, 254, 48, M("u_t", "Output hidden"),
      M("[B,L,d]", "[B,L,d]"), "gather", 14, "03");

    b += edge(rootId, ortho(152, 284, 184, 147), null, "compute");
    b += edge(rootId, ortho(152, 284, 184, 397), null, "compute");
    b += edge(rootId, ortho(352, 147, 410, 136), null, "state");
    b += edge(rootId, ortho(352, 397, 410, 286), null, "state");
    b += edge(rootId, ortho(570, 286, 594, 265), null, "compute");
    b += edge(rootId, ortho(570, 417, 594, 280), null, "compute");
    b += edge(rootId, ortho(667, 308, 667, 346), null, "control");
    b += edge(rootId, ortho(667, 416, 667, 468),
      [720, 442, "addresses", 112], "control");
    b += edge(rootId, ortho(570, 136, 594, 507, "x", 580), null, "gather");
    b += edge(rootId, ortho(740, 507, 800, 294), null, "gather");
    b += edge(rootId, ortho(858, 166, 858, 222), null, "state");
    b += edge(rootId, ortho(996, 166, 996, 222), null, "compute");
    b += edge(rootId, ortho(927, 334, 927, 374), null, "compute");
    b += edge(rootId, ortho(927, 440, 927, 478), null, "gather");
    b += edge(rootId, ortho(927, 548, 927, 584), null, "gather");
    return baseSvg(rootId, "csa", 678, b,
      "CSA with independent core and index compressors, separate caches, top-k gather, SWA, and one shared-KV MQA sink normalization");
  }

  function hcaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 356, 548, "NON-OVERLAP HEAVY COMPRESSOR", "compute");
    b += panel(398, 52, 360, 548, "COMPLETED GLOBAL CACHE + INDEPENDENT SWA", "state");
    b += panel(776, 52, 300, 548, "DENSE SHARED-KV MQA + SINK", "gather");

    b += box(40, 268, 112, 64, M("H_{1:L}", "Hidden sequence"),
      M("[B,L,d]", "[B,L,d]"), "compute", 1, "02");
    b += box(184, 200, 176, 126,
      M("C_i^{\\mathrm{Comp}}=\\mathcal C_{m'}(H_{m'i:m'(i+1)})", "Non-overlap compressor"),
      M("m'=128\\;\\cdot\\;\\operatorname{softmax}_{m'}(Z+B)\\odot C",
        "per-channel weighted block summary"),
      "compute", 2, "02", { titleSize: 9.2, subSize: 8.1 });
    b += box(184, 86, 176, 72, "Causal publish gate",
      "only closed blocks become visible", "control", 3, "02");

    b += cacheBox(426, 116, 304, 92, "Completed CComp cache",
      M("\\{C_i^{\\mathrm{Comp}}:m'(i+1)\\le t\\}", "all closed summaries · pRoPE"),
      4, "03");
    b += cacheBox(426, 356, 138, 88, "SWA cache lane",
      "recent raw entries · w=128", 5, "03", { subSize: 8.1 });
    b += box(592, 356, 138, 88, "Query lane",
      M("q_t\\;\\cdot\\;\\operatorname{pRoPE}_{64}", "Hq heads · trailing-64"),
      "compute", 6, "03", { subSize: 8.1 });

    b += box(800, 188, 252, 118,
      M("\\operatorname{MQA}_{K=V}(q_t;\\mathrm{sink},C_{\\mathrm{all}},C_{\\mathrm{SWA}})",
        "Dense shared-KV MQA + sink"),
      M("a_t=\\operatorname{softmax}([s_{\\mathrm{sink}},s_{\\mathrm{all}},s_{\\mathrm{SWA}}])",
        "one normalization: sink, all CComp, SWA"), "compute", 7, "03",
      { titleSize: 9.1, subSize: 8.2 });
    b += box(800, 346, 252, 66, M("\\operatorname{RoPE}^{-1}_{64}(-t)", "Inverse partial RoPE"),
      "return value slice to query frame", "compute", 8, "01");
    b += box(800, 452, 252, 70,
      M("W^{OA}_{\\mathrm{group}}\\to\\operatorname{Concat}\\to W^{OB}", "Grouped output projection"),
      "low-rank groups → residual width", "gather", 9, "03", { titleSize: 9.6 });
    b += box(800, 552, 252, 48, M("u_t", "Output hidden"),
      M("[B,L,d]", "[B,L,d]"), "gather", 10, "03");

    b += edge(rootId, ortho(152, 300, 184, 263), null, "compute");
    b += edge(rootId, ortho(272, 200, 272, 158), null, "control");
    b += edge(rootId, ortho(360, 122, 426, 162), null, "state");
    b += edge(rootId, ortho(730, 162, 800, 232), null, "state");
    b += edge(rootId, "M495 444V468H780V270H800", null, "state");
    b += edge(rootId, ortho(730, 400, 800, 286), null, "compute");
    b += edge(rootId, ortho(926, 306, 926, 346), null, "compute");
    b += edge(rootId, ortho(926, 412, 926, 452), null, "gather");
    b += edge(rootId, ortho(926, 522, 926, 552), null, "gather");
    return baseSvg(rootId, "hca", 628, b,
      "HCA with a causal non-overlap compressor, completed CComp cache, independent SWA, and dense shared-KV MQA sink");
  }

  function linearDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 340, 526, "2020 ELU+1 TOKEN FEATURES", "compute");
    b += panel(382, 52, 380, 526, "CAUSAL S / z RECURRENT STATE", "state");
    b += panel(780, 52, 296, 526, "NUMERATOR, DENOMINATOR & OUTPUT", "gather");

    b += box(40, 250, 106, 64, M("x_t", "Input token"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "04");
    b += box(176, 104, 82, 58, M("W_Q", "WQ"), "query", "compute", null, "04");
    b += box(176, 252, 82, 58, M("W_K", "WK"), "key", "compute", null, "04");
    b += box(176, 420, 82, 58, M("W_V", "WV"), "value", "compute", null, "04");
    b += box(278, 98, 70, 70, M("\\phi(q_t)", "phi(q)"),
      M("\\operatorname{ELU}(q_t)+1", "ELU(q)+1"), "compute", 2, "01", { titleSize: 10.2 });
    b += box(278, 246, 70, 70, M("\\phi(k_t)", "phi(k)"),
      M("\\operatorname{ELU}(k_t)+1", "ELU(k)+1"), "compute", 3, "01", { titleSize: 10.2 });
    b += box(278, 414, 70, 70, M("v_t", "Value v"),
      M("\\mathbb R^{d_v}", "dv channels"), "compute", 4, "04");

    b += cacheBox(402, 112, 160, 82, M("S_{t-1}", "Previous S state"),
      M("\\mathbb R^{r\\times d_v}", "feature × value"), 5, "03");
    b += cacheBox(582, 112, 160, 82, M("z_{t-1}", "Previous z state"),
      M("\\mathbb R^r", "feature normalizer"), 6, "03");
    b += box(402, 250, 160, 92,
      M("S_t=S_{t-1}+\\phi(k_t)v_t^{\\mathsf T}", "Update S with outer product"),
      "fixed-size associative state", "state", 7, "03", { titleSize: 9.6 });
    b += box(582, 250, 160, 92,
      M("z_t=z_{t-1}+\\phi(k_t)", "Update z"),
      "fixed-size denominator state", "state", 8, "03", { titleSize: 9.6 });
    b += box(402, 424, 340, 112, "2020 training execution boundary",
      "paper gives causal recurrence and a custom GPU implementation; this diagram makes no later-kernel claim",
      "orange", null, "03", { subSize: 7.9 });

    b += box(800, 112, 252, 76,
      M("n_t=\\phi(q_t)^{\\mathsf T}S_t", "Numerator"),
      M("n_t\\in\\mathbb R^{d_v}", "query-dependent value read"), "compute", 9, "03");
    b += box(800, 230, 252, 76,
      M("d_t=\\phi(q_t)^{\\mathsf T}z_t+\\varepsilon", "Denominator"),
      "query-dependent normalization", "compute", 10, "03");
    b += box(800, 368, 252, 78,
      M("y_t=n_t/d_t", "Normalize numerator / denominator"),
      "then output projection WO", "gather", 11, "04");
    b += box(800, 502, 252, 54, M("u_t=W^Oy_t", "Output hidden"),
      "constant-state decode", "gather", 12, "04");

    b += edge(rootId, ortho(146, 282, 176, 133), null, "compute");
    b += edge(rootId, ortho(146, 282, 176, 281), null, "compute");
    b += edge(rootId, ortho(146, 282, 176, 449), null, "compute");
    b += edge(rootId, ortho(258, 133, 278, 133), null, "compute");
    b += edge(rootId, ortho(258, 281, 278, 281), null, "compute");
    b += edge(rootId, ortho(258, 449, 278, 449), null, "compute");
    b += edge(rootId, ortho(348, 281, 402, 296), null, "state");
    b += edge(rootId, ortho(348, 449, 402, 320), null, "state");
    b += edge(rootId, ortho(482, 194, 482, 250), null, "state");
    b += edge(rootId, "M348 281H372V226H572V296H582", null, "state");
    b += edge(rootId, ortho(662, 194, 662, 250), null, "state");
    b += edge(rootId, "M482 342V366H776V150H800", null, "compute");
    b += edge(rootId, ortho(742, 296, 800, 268), null, "compute");
    b += edge(rootId, "M313 98V78H926V112",
      [620, 88, "query read rail", 132], "compute");
    b += edge(rootId, "M348 133H372V350H1066V268H1052",
      [720, 348, "normalizer read rail", 166], "compute");
    b += edge(rootId, "M1052 150H1066V407H1052", null, "gather");
    b += edge(rootId, ortho(926, 306, 926, 368), null, "gather");
    b += edge(rootId, ortho(926, 446, 926, 502), null, "gather");
    b += edge(rootId, ortho(562, 320, 572, 424), null, "orange");
    return baseSvg(rootId, "linear", 606, b,
      "Original 2020 kernelized linear attention with ELU plus one features, S and z states, and normalized readout");
  }

  function deltaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 374, 598, "SEPARATE TOKEN PATHS · CONV ONLY ON q / k / v", "compute");
    b += panel(416, 52, 374, 598, "TRANSPOSED FAST-WEIGHT STATE · F = S TRANSPOSE", "state");
    b += panel(808, 52, 268, 598, "READ, GATE & WRITE BACK", "gather");

    b += box(40, 284, 104, 64, M("x_t", "Input hidden"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "03");
    b += box(176, 92, 202, 68, M("q_t", "q path"),
      "WQ → causal ShortConv → SiLU → L2Norm", "compute", null, "01", { titleSize: 9.0, subSize: 7.9 });
    b += box(176, 162, 202, 68, M("k_t", "k path"),
      "WK → causal ShortConv → SiLU → L2Norm", "compute", null, "01", { titleSize: 9.0, subSize: 7.9 });
    b += box(176, 252, 202, 68, M("v_t", "v path"),
      "WV → causal ShortConv → SiLU", "compute", null, "01", { titleSize: 9.2, subSize: 8.1 });
    b += box(176, 354, 202, 62,
      M("\\alpha_t=\\exp[-\\operatorname{softplus}(\\cdot)]", "Direct scalar alpha"),
      "no ShortConv · per-head decay", "control", null, "03", { titleSize: 9.4 });
    b += box(176, 440, 202, 62, M("\\beta_t=\\sigma(x_tW_\\beta)", "Direct write beta"),
      "no ShortConv · write strength", "control", null, "03");
    b += box(176, 526, 202, 62, M("g_t=x_tW_g", "Direct output gate g"),
      "no ShortConv · SiLU at readout", "control", null, "03");

    b += cacheBox(436, 92, 148, 72, M("F_{t-1}=S_{t-1}^{\\mathsf T}", "Previous F state"),
      M("F\\in\\mathbb R^{d_v\\times d_k}", "transposed convention"), 2, "02");
    b += box(436, 180, 148, 68, M("F=S^{\\mathsf T}", "State convention"),
      M("Fk=\\widehat v,\\;Fq=o", "rows are value channels"), "cyan", null, "02");
    b += box(614, 92, 156, 72, M("\\widetilde F_{t-1}=\\alpha_tF_{t-1}", "1 · decay"),
      "scalar GDN retention", "state", 3, "02");
    b += box(614, 184, 156, 72, M("\\widehat v_t=\\widetilde F_{t-1}k_t", "2 · predict"),
      "old value at current key", "compute", 4, "02");
    b += box(614, 290, 156, 72, M("e_t=v_t-\\widehat v_t", "3 · error"),
      "write only what is missing", "cyan", 5, "02");
    b += box(604, 402, 176, 94,
      M("F_t=\\widetilde F_{t-1}+\\beta_te_tk_t^{\\mathsf T}", "4 · rank-1 write"),
      M("F_t\\in\\mathbb R^{d_v\\times d_k}", "updated recurrent state"), "gather", 6, "02",
      { titleSize: 9.5 });
    b += box(436, 536, 344, 72, "Training execution",
      "decay-aware chunkwise WY/UT; decode keeps F plus three ShortConv states",
      "orange", null, "04", { dashed: true, subSize: 8.0 });

    b += box(832, 402, 220, 94,
      M("o_t=F_t(q_t/\\sqrt{d_k})", "5 · read with q"),
      "query-dependent value output", "gather", 7, "02");
    b += box(832, 536, 220, 72,
      M("W_O[\\operatorname{RMSNorm}(o_t)\\odot\\operatorname{SiLU}(g_t)]", "Gate → WO → residual"),
      "Gated DeltaNet output", "gather", 8, "03", { titleSize: 9.2 });

    b += edge(rootId, ortho(144, 316, 176, 126), null, "compute");
    b += edge(rootId, ortho(144, 316, 176, 196), null, "compute");
    b += edge(rootId, ortho(144, 316, 176, 286), null, "compute");
    b += edge(rootId, ortho(144, 316, 176, 385), null, "control");
    b += edge(rootId, ortho(144, 316, 176, 471), null, "control");
    b += edge(rootId, ortho(144, 316, 176, 557), null, "control");
    b += edge(rootId, ortho(584, 128, 614, 128), null, "state");
    b += edge(rootId, ortho(378, 385, 614, 144, "x", 598), null, "control");
    b += edge(rootId, ortho(692, 164, 692, 184), null, "state");
    b += edge(rootId, "M378 196H410V166H596V220H614", null, "compute");
    b += edge(rootId, ortho(692, 256, 692, 290), null, "compute");
    b += edge(rootId, ortho(378, 286, 614, 326, "x", 594), null, "compute");
    b += edge(rootId, ortho(692, 362, 692, 402), null, "gather");
    b += edge(rootId, ortho(378, 471, 604, 460, "x", 590), null, "control");
    b += edge(rootId, ortho(692, 164, 604, 430, "x", 596), null, "state");
    b += edge(rootId, ortho(780, 449, 832, 449), null, "gather");
    b += edge(rootId, "M277 92V72H942V402",
      [620, 82, "q read rail", 108], "compute");
    b += edge(rootId, ortho(942, 496, 942, 536), null, "gather");
    b += edge(rootId, "M277 588V630H806V572H832",
      [610, 628, "output-gate rail", 144], "control");
    b += edge(rootId, ortho(692, 496, 608, 536, "y", 516),
      [690, 522, "TRAINING · chunk transform", 218], "orange", true);
    b += edge(rootId, ortho(510, 164, 510, 180), null, "cyan");
    return baseSvg(rootId, "gated-delta", 678, b,
      "Gated DeltaNet with separate q k v ShortConv paths, direct gates, transposed state, and decay predict error write read order");
  }

  function kdaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 374, 598, "SEPARATE TOKEN PATHS · CONV ONLY ON q / k / v", "compute");
    b += panel(416, 52, 374, 598, "CHANNEL DECAY + DELTA UPDATE · F = S TRANSPOSE", "state");
    b += panel(808, 52, 268, 598, "READOUT & CHECKPOINT PLACEMENT", "gather");

    b += box(40, 284, 104, 64, M("x_t", "Input hidden"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "03");
    b += box(176, 92, 202, 68, M("q_t", "q path"),
      "WQ → causal ShortConv → SiLU → L2Norm", "compute", null, "01", { titleSize: 9.0, subSize: 7.9 });
    b += box(176, 162, 202, 68, M("k_t", "k path"),
      "WK → causal ShortConv → SiLU → L2Norm", "compute", null, "01", { titleSize: 9.0, subSize: 7.9 });
    b += box(176, 252, 202, 68, M("v_t", "v path"),
      "WV → causal ShortConv → SiLU", "compute", null, "01", { titleSize: 9.2, subSize: 8.1 });
    b += box(176, 354, 202, 62,
      M("\\alpha_t\\in(0,1)^{d_k}", "Direct channel alpha"),
      M("W_\\alpha^\\downarrow\\to\\operatorname{SiLU}\\to W_\\alpha^\\uparrow\\to\\log\\text{-decay}",
        "direct low-rank gate · no ShortConv"), "control", null, "03", { titleSize: 9.3, subSize: 8.0 });
    b += box(176, 440, 202, 62, M("\\beta_t=\\sigma(x_tW_\\beta)", "Direct write beta"),
      "no ShortConv · scalar write gate", "control", null, "03");
    b += box(176, 526, 202, 62,
      M("g_t=W_g^\\uparrow\\operatorname{SiLU}(x_tW_g^\\downarrow)", "Direct low-rank gate g"),
      "no ShortConv · sigmoid at readout", "control", null, "03", { titleSize: 9.1 });

    b += cacheBox(436, 92, 148, 72, M("F_{t-1}=S_{t-1}^{\\mathsf T}", "Previous F state"),
      M("F\\in\\mathbb R^{d_v\\times d_k}", "transposed convention"), 2, "02");
    b += box(614, 92, 156, 72,
      M("\\widetilde F_{t-1}=F_{t-1}\\operatorname{Diag}(\\alpha_t)", "1 · channel decay"),
      "right-multiply in F convention", "state", 3, "02", { titleSize: 9.4 });
    b += box(614, 184, 156, 72, M("\\widehat v_t=\\widetilde F_{t-1}k_t", "2 · predict"),
      "old value at current key", "compute", 4, "02");
    b += box(614, 290, 156, 72, M("e_t=v_t-\\widehat v_t", "3 · error"),
      "post-decay prediction error", "cyan", 5, "02");
    b += box(604, 402, 176, 94,
      M("F_t=\\widetilde F_{t-1}+\\beta_te_tk_t^{\\mathsf T}", "4 · rank-1 write"),
      M("F_t\\in\\mathbb R^{d_v\\times d_k}", "updated KDA state"), "gather", 6, "02",
      { titleSize: 9.5 });
    b += box(436, 522, 344, 92,
      M("S_t=(I-\\beta_tk_tk_t^{\\mathsf T})D_tS_{t-1}+\\beta_tk_tv_t^{\\mathsf T}",
        "Exact column-state DPLR order"),
      M("D_t=\\operatorname{Diag}(\\alpha_t)", "rank-1 factor acts after channel decay"),
      "cyan", null, "02", { titleSize: 9.0, subSize: 7.8 });

    b += box(832, 402, 220, 94,
      M("o_t=F_t(q_t/\\sqrt{d_k})", "5 · read with q"),
      "same as S-transpose q", "gather", 7, "02");
    b += box(832, 522, 220, 92,
      M("W_O[\\operatorname{RMSNorm}(o_t)\\odot\\sigma(g_t)]", "Gate → WO → residual"),
      "KDA block output", "gather", 8, "03", { titleSize: 9.2 });
    b += box(40, 674, 220, 70, "NoPE global attention",
      "checkpoint MLA variant · no positional encoding", "compute", null, "04", { subSize: 8.0 });
    b += box(286, 674, 766, 70,
      M("(\\mathrm{KDA}\\times3\\to\\mathrm{MLA\\!\\!-NoPE})\\times6\\;\\to\\;\\mathrm{KDA}\\times2\\to\\mathrm{MLA\\!\\!-NoPE}",
        "(KDA×3→MLA-NoPE)×6 → KDA×2→MLA-NoPE"),
      "checkpoint tail · layerwise sequence, not mixed heads", "orange", 9, "05", { titleSize: 10.2 });

    b += edge(rootId, ortho(144, 316, 176, 126), null, "compute");
    b += edge(rootId, ortho(144, 316, 176, 196), null, "compute");
    b += edge(rootId, ortho(144, 316, 176, 286), null, "compute");
    b += edge(rootId, ortho(144, 316, 176, 385), null, "control");
    b += edge(rootId, ortho(144, 316, 176, 471), null, "control");
    b += edge(rootId, ortho(144, 316, 176, 557), null, "control");
    b += edge(rootId, ortho(584, 128, 614, 128), null, "state");
    b += edge(rootId, ortho(378, 385, 614, 144, "x", 598), null, "control");
    b += edge(rootId, ortho(692, 164, 692, 184), null, "state");
    b += edge(rootId, "M378 196H410V166H596V220H614", null, "compute");
    b += edge(rootId, ortho(692, 256, 692, 290), null, "compute");
    b += edge(rootId, ortho(378, 286, 614, 326, "x", 594), null, "compute");
    b += edge(rootId, ortho(692, 362, 692, 402), null, "gather");
    b += edge(rootId, ortho(378, 471, 604, 460, "x", 590), null, "control");
    b += edge(rootId, ortho(692, 164, 604, 430, "x", 596), null, "state");
    b += edge(rootId, ortho(780, 449, 832, 449), null, "gather");
    b += edge(rootId, "M277 92V72H942V402",
      [620, 82, "q read rail", 108], "compute");
    b += edge(rootId, ortho(942, 496, 942, 522), null, "gather");
    b += edge(rootId, "M277 588V638H806V566H832",
      [610, 636, "output-gate rail", 144], "control");
    b += edge(rootId, ortho(692, 496, 608, 522, "y", 508), null, "cyan");
    b += edge(rootId, "M942 614V650H669V674", [820, 650, "checkpoint layer order", 196], "orange");
    b += edge(rootId, ortho(260, 709, 286, 709), null, "orange");
    return baseSvg(rootId, "kda", 772, b,
      "KDA with separate q k v ShortConv paths, direct channel gates, exact DPLR order, and checkpoint-tail layer sequence");
  }

  function key(config) {
    if (config.type === "heads") return config.mode;
    if (config.type === "compressed") return config.mode;
    if (config.type === "latent") return "mla";
    if (config.type === "sparse") return "dsa";
    if (config.type === "linear") return "linear";
    if (config.type === "delta") return "delta";
    if (config.type === "kda") return "kda";
    return config.type;
  }

  function build(config) {
    var k = key(config);
    var svg = "";
    buildSerial += 1;
    var rootId = "attention-diagram-" + escapeText(k).replace(/[^A-Za-z0-9_-]/g, "-") +
      "-" + buildSerial;
    if (k === "mha") svg = mhaDiagram(rootId);
    if (k === "mqa") svg = mqaDiagram(rootId);
    if (k === "gqa") svg = gqaDiagram(rootId);
    if (k === "mla") svg = mlaDiagram(rootId);
    if (k === "dsa") svg = dsaDiagram(rootId);
    if (k === "csa") svg = csaDiagram(rootId);
    if (k === "hca") svg = hcaDiagram(rootId);
    if (k === "linear") svg = linearDiagram(rootId);
    if (k === "delta") svg = deltaDiagram(rootId);
    if (k === "kda") svg = kdaDiagram(rootId);
    validateStaticGeometry(svg, k);
    return {
      svg: svg,
      notes: guides[k] || [],
      memory: memories[k] || "",
      badges: [
        "BLUE · COMPUTE",
        "GREEN · CONTROL",
        "ROSE · CACHE/STATE",
        "LAVENDER · GATHER/WRITE",
        "DASHED · TRAINING/OPTIONAL"
      ]
    };
  }

  window.AttentionDiagrams = { build: build };
})();
