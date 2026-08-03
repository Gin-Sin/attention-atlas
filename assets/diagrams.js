(function () {
  "use strict";

  var R = String.raw;

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
      ["原论文入口", R`词嵌入先乘 \(\sqrt{d_{\mathrm{model}}}\)，再加固定正弦位置编码；Q/K/V 投影发生在其后。`],
      ["独立多头投影", R`每个头有独立 \(W_h^Q\) / \(W_h^K\) / \(W_h^V\)，\(H_{kv}=H_q=H\)，三路形状均为 \([B,H,L,d_h]\)。`],
      ["精确注意力", R`\(S_h=Q_hK_h^{\mathsf T}/\sqrt{d_h}+M\)；causal mask 只出现在 decoder，encoder 自注意力省略 \(M\)；softmax 后精确乘 \(V_h\)。`],
      ["写回与残差", R`\(\operatorname{Concat}(O_h)\,W^O\) 之后才做 Add & Norm；这是 2017 的 post-LN 顺序，残差取自 Q/K/V 投影之前的 \(X\)。`],
      ["现代缓存只是叠加层", R`2017 训练图没有 KV cache；虚线玫瑰框表示现代增量解码把 \(K_{1:t}\) / \(V_{1:t}\) 追加进缓存，softmax 本身不变。`],
      ["视觉语义", "蓝=计算，绿=控制，玫瑰=缓存/状态，薰衣草=聚合/写回；虚线只表示可选或现代叠加路径。"]
    ],
    mqa: [
      ["论文剖面", "Shazeer 2019 基线使用 learned input positions；MQA 的创新只是共享一套 K/V，不是新的位置编码。"],
      ["多 Q 单 KV", R`保留 \(H_q\) 个独立 query 头 \([B,H_q,L,d_h]\)；K/V 只有一份，形状 \(2\times[B,1,L,d_h]\)。`],
      ["广播不是复制", R`绿色 broadcast 框表示 \(H_q\) 个头 stride-0 逻辑读取同一份 K/V；高效实现不 materialize repeat。`],
      ["精确 softmax", R`\(S_h=Q_hK^{\mathsf T}/\sqrt{d_h}+M\) 逐头计算；共享 K/V 只减少搬运，不引入任何近似。`],
      ["缓存收益", R`玫瑰色缓存只有两份 \([B,1,L,d_h]\)；解码时历史搬运量随 \(H_{kv}=1\) 大幅下降。`],
      ["视觉语义", "蓝=计算，绿=控制，玫瑰=缓存/状态，薰衣草=聚合/写回；橙色标注论文年代边界。"]
    ],
    gqa: [
      ["T5 论文剖面", R`原 GQA uptraining 继承 T5 的按 query head 相对位置偏置 \(b_{h,\operatorname{bucket}(t-s)}\)，而不是现代 Llama RoPE。`],
      ["显式组映射", R`\(r=H_q/H_{kv}\)，\(g(h)=\lfloor h/r\rfloor\)；每个 Q 头只逻辑读取所属组的 K/V，不做物理 repeat。`],
      ["分组打分与读取", R`\(S_h=Q_hK_{g(h)}^{\mathsf T}/\sqrt{d_h}+b_h+M\)；softmax 后乘 \(V_{g(h)}\)，每个 query 头一份输出。`],
      ["缓存形状", R`分组缓存为 \(2\times[B,H_{kv},L,d_h]\)，容量介于 MHA 与 MQA 之间。`],
      ["Uptraining 配方", "组内 K/V 权重均值池化初始化，再继续约 5% 预训练；虚线 inset 只属训练，不是推理路径。"],
      ["视觉语义", "蓝=计算，绿=控制，玫瑰=缓存/状态，薰衣草=聚合/写回；虚线只表示训练或可选路径。"]
    ],
    mla: [
      ["Decode 主路径", R`缓存只含 \(c^{KV}\) 与 \(k^R\)；图中主干就是吸收式解码，历史 \(k^C\) / \(v\) 从不显式重建。`],
      ["两次下投影", R`\(c^Q=W^{DQ}h_t\) 与 \(c^{KV}=W^{DKV}h_t\)；解耦 RoPE 键 \(k^R=\operatorname{RoPE}(W^{KR}h_t)\) 直接来自 \(h_t\)，是全头共享的位置切片。`],
      ["吸收技巧", R`内容分数把 \(W^{UK}\) 吸收到 query 侧：\(\widetilde q_i=(W_i^{UK})^{\mathsf T}q_i^C\)；输出侧把 \(W^{UV}\) 吸收进写回，一次投影完成。`],
      ["正确缩放", R`即使内容 query 落在 \(d_c\) 维 latent 空间，分数仍按完整 head 宽度 \(\sqrt{d_h+d_h^R}\) 缩放。`],
      ["可选 RMSNorm", R`DeepSeek 检查点会对 \(c^Q\) / \(c^{KV}\) 加 RMSNorm；那是实现配方而非 MLA 定义，图中保持原始下投影、此处说明。`],
      ["重建仅为等价解释", R`虚线橙框 \(k^C=W^{UK}c\)、\(v=W^{UV}c\) 是训练/概念等价视角，不是 decode 路径，也不进缓存。`],
      ["视觉语义", "玫瑰=持久 latent 缓存，蓝=投影/attention，薰衣草=聚合/写回；橙色虚线=明确可选的概念视角。"]
    ],
    dsa: [
      ["两条清晰车道", R`上方 Indexer 生成 \(q^I\)、\(k^I\)、\(w^I\)、全历史 logits 与 TopK；官方实现中 \(q^I\) 由共享的 MLA query latent \(c^Q\) 投影，\(k^I\) 先过 LayerNorm。下方从 MLA latent cache gather 后运行候选 MLA。`],
      ["低精度对称路径", R`\(q^I\) 与 \(k^I\) 都经过 partial RoPE、Hadamard 与 FP8；Hadamard 服务数值范围，不是位置编码。`],
      ["没有固定局部窗", "DSA 原型由内容 TopK 选择候选；图中不添加 local-window 捷径。"],
      ["候选内精确注意力", R`gather 出的原始 \(c^{KV}\) / \(k^R\) 交给高维 MLA query；softmax 只在选中集合内重新归一化，随后 \(W^O\) 写回残差。`],
      ["训练监督", "teacher full logits 与 KL 仅通过带标签虚线连接，detach 后不属于推理图。"],
      ["视觉语义", "绿=TopK/路由，玫瑰=历史 cache，薰衣草=Gather；青/橙为精度和训练注释。"]
    ],
    csa: [
      ["双压缩器、双缓存", R`core compressor 产生 \(C^{\mathrm{Comp}}\)；独立 index compressor 产生 \(K^{I\mathrm{Comp}}\)（\(\theta_I\ne\theta_C\)），参数与缓存职责不能合并。`],
      ["重叠压缩内部", R`两路 a/b 投影对 \(2m\) 重叠窗口做 per-channel softmax 加权求和，得到一个压缩条目。`],
      ["地址到内容", R`\(K^{I\mathrm{Comp}}\) 只负责打分与 TopK（V4-Flash / Pro 配置）；地址下传给 Gather，从 \(C^{\mathrm{Comp}}\) cache 取候选内容。`],
      ["三路进入同一核心", "选中全局摘要、独立 SWA lane 与 query lane 汇入一次 shared-KV MQA；sink、global、SWA 在同一个 softmax 中归一化。"],
      ["输出坐标", R`partial-RoPE 值混合后先 inverse \(\operatorname{RoPE}(-t)\) 回到 query 坐标，再按组 \(W^{OA}\to\operatorname{Concat}\to W^{OB}\) 写回残差。`],
      ["视觉语义", "蓝=计算，绿=选择，玫瑰=两类 cache，薰衣草=Gather/写回；青/橙为位置与因果注释。"]
    ],
    hca: [
      ["只压缩、不索引", R`非重叠重压缩器（\(m'=128\)）只发布已完成块到 \(C^{\mathrm{Comp}}\) cache；HCA 没有 indexer 或 TopK。`],
      ["压缩器内部", R`块内 per-channel \(\operatorname{softmax}(Z+B)\odot C\) 加权求和，把 128 个位置压成一个摘要条目。`],
      ["因果发布门", R`绿色 gate 保证只有 \(m'(i+1)\le t\) 的已关闭块可见，避免泄露未来信息。`],
      ["Dense 读取短历史", R`全部 completed \(C^{\mathrm{Comp}}\)、独立 SWA lane 与 query lane 进入 shared-KV dense MQA；sink、全部压缩摘要、SWA 一次归一化。`],
      ["位置与输出", R`压缩与局部 entry 使用 partial RoPE；输出先 inverse \(\operatorname{RoPE}(-t)\)，再按组 \(W^{OA}\to\operatorname{Concat}\to W^{OB}\) 写回。`],
      ["视觉语义", "蓝=计算，绿=因果完成控制，玫瑰=cache，薰衣草=汇合/写回；青/橙为位置和边界注释。"]
    ],
    linear: [
      ["原始特征映射", R`2020 Linear Transformer 使用 \(\phi(x)=\operatorname{ELU}(x)+1\)，使核可结合且非负；\(\phi\) 同时作用在 q 与 k。`],
      ["一个循环单元", R`按原论文的 RNN 视角，整个历史折进固定大小状态：\(S\) 累积 \(\phi(k)v^{\mathsf T}\)（\(r\times d_v\)），\(z\) 累积 \(\phi(k)\)（\(r\) 维）。`],
      ["单一反馈环", R`右侧自环表示 \(S_{t-1}\) / \(z_{t-1}\) 进入下一步；解码状态大小与序列长度无关。`],
      ["读取即归一化", R`\(y_t=\phi(q)^{\mathsf T}S_t/(\phi(q)^{\mathsf T}z_t+\varepsilon)\)：分子读值、分母归一化，再经 \(W^O\) 写回残差。`],
      ["训练表述保持克制", "原论文给出因果递推及自定义 CUDA 实现；图不把后来的并行 scan / chunk kernel 冒充官方实现。"],
      ["视觉语义", "蓝=投影/读取，玫瑰=递推状态单元与反馈环，薰衣草=归一化写回；橙色注释标明执行边界。"]
    ],
    delta: [
      ["参数路径必须分开", R`只有 q/k/v 经过 causal ShortConv（q/k 再 SiLU + L2Norm，v 只 SiLU）；\(\alpha\)、\(\beta\)、output gate \(g\) 直接由当前 hidden 投影。`],
      ["转置状态约定", R`图用 \(F=S^{\mathsf T}\)，形状 \(d_v\times d_k\)；因此预测和读取写成 \(Fk\)、\(Fq\)。`],
      ["单一更新方程", R`\(F_t=\alpha_tF_{t-1}(I-\beta_tk_tk_t^{\mathsf T})+\beta_tv_tk_t^{\mathsf T}\)：先 decay 再 delta 纠写，与论文块图一致。`],
      ["五步展开", R`等价展开为 decay → predict → error → write → read：\(e_t=v_t-(\alpha_tF_{t-1})k_t\)，误差基于衰减后的状态，顺序决定语义。`],
      ["读取与门控", R`\(o_t=F_t(q_t/\sqrt{d_k})\)；输出 \(W_O[\operatorname{RMSNorm}(o_t)\odot\operatorname{SiLU}(g_t)]\)，\(g\) 是 direct 投影门。`],
      ["训练执行", R`训练用 decay-aware chunkwise WY/UT 变换；解码只保留 \(F\) 和三份 ShortConv 状态。`],
      ["视觉语义", "蓝=特征计算，绿=gate/control，玫瑰=F 状态单元与反馈环，薰衣草=读取/写回。"]
    ],
    kda: [
      ["逐通道衰减", R`\(\alpha\) 是 \(d_k\) 维 direct 低秩门（\(W\) 降维 → SiLU → \(W\) 升维 → log-decay）；在 \(F=S^{\mathsf T}\) 约定下 decay 写成 \(F\operatorname{Diag}(\alpha)\)。`],
      ["单一更新方程", R`\(F_t=F_{t-1}\operatorname{Diag}(\alpha_t)(I-\beta_tk_tk_t^{\mathsf T})+\beta_tv_tk_t^{\mathsf T}\)：通道 decay 在 rank-1 纠写之前。`],
      ["DPLR 次序不可交换", R`列向量原式严格为 \(S_t=(I-\beta kk^{\mathsf T})\operatorname{Diag}(\alpha)S_{t-1}+\beta kv^{\mathsf T}\)；rank-1 因子作用在 decay 之后。`],
      ["参数路径", R`q/k/v 各走 causal ShortConv；\(\alpha\)、\(\beta\)、\(g\) 直接投影且不接卷积支路；\(g\) 为低秩、读出处用 sigmoid。`],
      ["读取与门控", R`\(o_t=F_t(q_t/\sqrt{d_k})\)；输出 \(W_O[\operatorname{RMSNorm}(o_t)\odot\sigma(g_t)]\) 写回残差。`],
      ["Checkpoint 尾部", "官方层序为 (KDA×3 → MLA-NoPE)×6 → KDA×2 → MLA-NoPE；MLA 变体不带位置编码，逐层交错而非混头。"],
      ["视觉语义", "蓝=计算，绿=逐通道控制，玫瑰=状态单元与反馈环，薰衣草=写回；橙色标注精确层序。"]
    ]
  };

  var memories = {
    mha: "2017 MHA：缩放 embedding 加正弦位置后投影多头，精确注意力写回，再做 post Add & Norm。",
    mqa: "MQA：很多独立 Q 逻辑广播到唯一共享 K/V；减少历史搬运，不近似 softmax。",
    gqa: R`GQA：T5 query head 用 \(g(h)\) 找组内 K/V；MHA checkpoint 先组内均值再 uptrain。`,
    mla: R`MLA decode：直接用缓存 \(c^{KV}\)+\(k^R\) 做吸收式打分与读出；\(k^C\)/\(v\) 重建只是等价解释。`,
    dsa: "DSA：低维 FP8 Indexer 选地址，Gather 再把原始 MLA latent 交给精确候选 attention。",
    csa: R`CSA：\(K^{I\mathrm{Comp}}\) 负责找地址，\(C^{\mathrm{Comp}}\) 提供内容；再与 SWA 一起进入唯一的 MQA+sink 核心。`,
    hca: "HCA：只缓存已完成的重压缩块，不做 TopK；全部摘要与 SWA 一起 dense 读取。",
    linear: R`Linear Transformer：\(\operatorname{ELU}+1\) 把历史折进 \(S\)/\(z\) 循环单元，query 用分子除以分母读取固定状态。`,
    delta: R`GDN：q/k/v 走 ShortConv，direct gates 控制单一循环单元 \(F_t=\alpha F(I-\beta kk^{\mathsf T})+\beta vk^{\mathsf T}\)，再 \(Fq\) 读出。`,
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

  // Static SVG-coordinate guard:
  //  1. every connector is axis-aligned and may touch node boundaries only at
  //     its endpoints; no segment may traverse a node interior;
  //  2. no two different edges may share a rail: collinear (or nearly
  //     collinear, within RAIL_GAP px) parallel segments from different edges
  //     must not overlap for more than OVERLAP_LIMIT px. The only exception is
  //     shared source fan-out: the first segments of two edges that leave the
  //     exact same start point may run together;
  //  3. dashed edges must carry a label, and every data-code-block id must
  //     exist in the chapter's implementation blocks.
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

    var edges = [];
    var edgeMatch;
    var edgePattern = /<path d="([^"]+)"[^>]*marker-end/g;
    while ((edgeMatch = edgePattern.exec(svg))) {
      edges.push({ d: edgeMatch[1], points: pathPoints(edgeMatch[1]) });
    }

    edges.forEach(function (item) {
      for (var i = 1; i < item.points.length; i += 1) {
        for (var j = 0; j < boxes.length; j += 1) {
          if (crossesInterior(item.points[i - 1], item.points[i], boxes[j])) {
            throw new Error(diagramKey + ": connector traverses node: " + item.d);
          }
        }
      }
    });

    // Collinear rail-overlap check between different edges.
    var RAIL_GAP = 3.5;
    var OVERLAP_LIMIT = 6;
    function sharedSpan(a1, a2, b1, b2) {
      return Math.min(Math.max(a1, a2), Math.max(b1, b2)) -
        Math.max(Math.min(a1, a2), Math.min(b1, b2));
    }
    for (var ei = 0; ei < edges.length; ei += 1) {
      for (var ej = ei + 1; ej < edges.length; ej += 1) {
        var A = edges[ei].points;
        var B = edges[ej].points;
        var sharedSource = A[0].x === B[0].x && A[0].y === B[0].y;
        for (var si = 1; si < A.length; si += 1) {
          for (var sj = 1; sj < B.length; sj += 1) {
            if (sharedSource && si === 1 && sj === 1) continue;
            var a0 = A[si - 1];
            var a1 = A[si];
            var b0 = B[sj - 1];
            var b1 = B[sj];
            var overlap = -1;
            if (a0.x === a1.x && b0.x === b1.x &&
                Math.abs(a0.x - b0.x) <= RAIL_GAP) {
              overlap = sharedSpan(a0.y, a1.y, b0.y, b1.y);
            } else if (a0.y === a1.y && b0.y === b1.y &&
                Math.abs(a0.y - b0.y) <= RAIL_GAP) {
              overlap = sharedSpan(a0.x, a1.x, b0.x, b1.x);
            }
            if (overlap > OVERLAP_LIMIT) {
              throw new Error(diagramKey + ": overlapping rails between edges: " +
                edges[ei].d + " | " + edges[ej].d);
            }
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

    b += box(470, 84, 160, 60, M("E_{\\mathrm{token}}", "Token embedding"),
      M("[B,L,d_{\\mathrm{model}}]", "[B,L,dmodel]"), "compute", 1, "03");
    b += box(470, 176, 160, 60, M("\\sqrt{d_{\\mathrm{model}}}\\,E", "Scale embedding"),
      null, "compute", 2, "03");
    b += box(240, 176, 170, 60, M("\\operatorname{PE}_{\\sin/\\cos}(p)", "Sinusoidal PE"),
      null, "compute", 3, "01");
    b += box(470, 268, 160, 64, M("X=\\sqrt d\\,E+\\operatorname{PE}", "X = scaled E + PE"),
      null, "gather", 4, "03");

    b += box(250, 384, 180, 64, M("Q_h=XW_h^Q", "Q heads"),
      M("h=1,\\ldots,H", "h = 1…H"), "compute", null, "04");
    b += box(470, 384, 160, 64, M("K_h=XW_h^K", "K heads"),
      M("H_{kv}=H", "Hkv = H"), "compute", null, "04");
    b += box(680, 384, 180, 64, M("V_h=XW_h^V", "V heads"),
      M("[B,H,L,d_h]", "[B,H,L,dh]"), "compute", null, "04");
    b += cacheBox(880, 384, 140, 64, "KV cache",
      M("K_{1:t},V_{1:t}", "modern decode"), null, "05", { dashed: true, titleSize: 10.6 });

    b += box(60, 500, 180, 68, M("M_{\\mathrm{causal}}", "Causal mask"),
      "decoder only", "control", null, "06", { dashed: true });
    b += box(400, 500, 300, 68,
      M("S_h=Q_hK_h^{\\mathsf T}/\\sqrt{d_h}+M", "Scaled dot-product scores"),
      null, "compute", 5, "06");
    b += box(400, 610, 300, 64,
      M("O_h=\\operatorname{softmax}(S_h)V_h", "Exact softmax × V"),
      null, "compute", 6, "06");
    b += box(400, 716, 190, 60, M("\\operatorname{Concat}(O_h)W^O", "Concat → WO"),
      null, "gather", 7, "07", { titleSize: 10.2 });
    b += box(640, 716, 160, 60, "Add & Norm",
      "post-norm · 2017", "gather", 8, "07");

    b += edge(rootId, ortho(550, 144, 550, 176), null, "compute");
    b += edge(rootId, ortho(550, 236, 550, 268), null, "compute");
    b += edge(rootId, "M410 206H440V300H470", null, "compute");
    b += edge(rootId, "M510 332V358H340V384", null, "compute");
    b += edge(rootId, ortho(550, 332, 550, 384), null, "compute");
    b += edge(rootId, "M590 332V358H770V384", null, "compute");
    b += edge(rootId, "M340 448V474H460V500", null, "compute");
    b += edge(rootId, ortho(550, 448, 550, 500), null, "compute");
    b += edge(rootId, ortho(240, 534, 400, 534),
      [320, 552, "OPTIONAL · decoder mask", 150], "control", true);
    b += edge(rootId, ortho(550, 568, 550, 610), null, "compute");
    b += edge(rootId, "M770 448V642H700", null, "compute");
    b += edge(rootId, ortho(495, 674, 495, 716), null, "gather");
    b += edge(rootId, ortho(590, 746, 640, 746), null, "gather");
    b += edge(rootId, "M630 300H1040V746H800",
      [900, 286, "residual → Add & Norm", 190], "gather");
    b += edge(rootId, ortho(860, 416, 880, 416),
      [940, 366, "OPTIONAL · V append", 150], "state", true);
    b += edge(rootId, "M610 448V470H950V448",
      [780, 486, "OPTIONAL · K append", 150], "state", true);
    return baseSvg(rootId, "mha", 810, b,
      "2017 MHA with scaled embedding, sinusoidal positions, exact multi-head attention, post Add and Norm, and optional modern cache overlay");
  }

  function mqaDiagram(rootId) {
    var b = "";

    b += box(470, 84, 160, 60, M("E_t", "Token embedding"),
      M("[B,L,d]", "[B,L,d]"), "compute", 1, "03");
    b += box(240, 176, 190, 60, M("P_t^{\\mathrm{learned}}", "Learned position"),
      "2019 baseline PE", "compute", 2, "04");
    b += box(470, 176, 160, 60, M("X_t=E_t+P_t", "Add position"),
      null, "gather", 3, "03");

    b += box(240, 280, 190, 68, M("Q_{1:H_q}", "Many Q heads"),
      M("[B,H_q,L,d_h]", "[B,Hq,L,dh]"), "compute", 4, "03");
    b += box(620, 280, 190, 68, M("K,V", "One shared K/V"),
      M("2\\times[B,1,L,d_h]", "2 × [B,1,L,dh]"), "compute", 5, "03");
    b += cacheBox(620, 392, 190, 64, "Shared KV cache",
      M("H_{kv}=1", "Hkv = 1"), 6, "05");
    b += box(620, 500, 190, 64, "Logical broadcast",
      "stride-0 · no repeat", "control", 7, "06");

    b += box(240, 500, 300, 68,
      M("S_h=Q_hK^{\\mathsf T}/\\sqrt{d_h}+M", "Per-Q-head exact scores"),
      null, "compute", 8, "06");
    b += box(240, 612, 300, 64,
      M("O_h=\\operatorname{softmax}(S_h)V", "Softmax × shared V"),
      null, "compute", 9, "06");
    b += box(240, 720, 300, 60,
      M("\\operatorname{Concat}(O_h)W^O", "Concat heads → WO"),
      null, "gather", 10, "07");
    b += box(760, 612, 260, 60, "2019 scope",
      "sharing change · not a new PE", "orange", null, "01");

    b += edge(rootId, ortho(550, 144, 550, 176), null, "compute");
    b += edge(rootId, ortho(430, 206, 470, 206), null, "compute");
    b += edge(rootId, "M510 236V254H335V280", null, "compute");
    b += edge(rootId, "M590 236V254H715V280", null, "compute");
    b += edge(rootId, ortho(335, 348, 335, 500), null, "compute");
    b += edge(rootId, ortho(715, 348, 715, 392), null, "state");
    b += edge(rootId, ortho(715, 456, 715, 500), null, "state");
    b += edge(rootId, ortho(620, 532, 540, 532), null, "control");
    b += edge(rootId, "M715 564V644H540", null, "control");
    b += edge(rootId, ortho(390, 568, 390, 612), null, "compute");
    b += edge(rootId, ortho(390, 676, 390, 720), null, "gather");
    return baseSvg(rootId, "mqa", 810, b,
      "2019 MQA with learned input positions, many query heads, one shared KV head, and logical broadcasting");
  }

  function gqaDiagram(rootId) {
    var b = "";
    b += panel(74, 500, 952, 150, "TRAINING ONLY · MHA → GQA UPTRAINING", "orange", true);

    b += box(40, 240, 130, 64, M("X_t", "T5 hidden"),
      M("[B,L,d]", "[B,L,d]"), "compute", 1, "03");
    b += box(230, 100, 190, 68, M("Q_h=XW_h^Q", "Hq query heads"),
      M("[B,H_q,L,d_h]", "[B,Hq,L,dh]"), "compute", 2, "03");
    b += box(460, 100, 210, 68, "T5 relative bias",
      M("b_{h,\\operatorname{bucket}(t-s)}", "by query head"), "compute", 3, "04");
    b += box(460, 240, 210, 68, M("g(h)=\\lfloor h/r\\rfloor", "Head-to-group map"),
      M("r=H_q/H_{kv}", "r = Hq / Hkv"), "control", 4, "06");
    b += box(230, 380, 190, 68, M("K_g,V_g", "Hkv grouped K/V"),
      M("g=0,\\ldots,H_{kv}-1", "group index g"), "compute", 5, "03");
    b += cacheBox(460, 380, 210, 68, "Grouped KV cache",
      M("2\\times[B,H_{kv},L,d_h]", "K_g and V_g"), 6, "05");

    b += box(730, 100, 310, 76,
      M("S_h=Q_hK_{g(h)}^{\\mathsf T}/\\sqrt{d_h}+b_h+M", "Grouped score + T5 bias"),
      null, "compute", 7, "07", { titleSize: 9.8 });
    b += box(730, 240, 310, 68,
      M("O_h=\\operatorname{softmax}(S_h)V_{g(h)}", "Exact groupwise read"),
      null, "compute", 8, "07");
    b += box(730, 352, 310, 60,
      M("\\operatorname{Concat}(O_h)W^O", "Concat → WO"),
      null, "gather", 9, "08");

    b += box(100, 536, 180, 68, M("W_{1:H_q}^{K,V}", "MHA K/V weights"),
      "pretrained checkpoint", "state", null, "01");
    b += box(330, 536, 210, 68,
      M("\\bar W_g=r^{-1}\\!\\sum_{h:g(h)=g}W_h", "Groupwise mean pool"),
      null, "gather", null, "01", { titleSize: 9.6 });
    b += box(590, 536, 180, 68, M("W_{1:H_{kv}}^{K,V}", "GQA K/V init"),
      null, "state", null, "02");
    b += box(820, 536, 180, 68, "Continue pretraining",
      "about 5% of budget", "control", null, "02");

    b += edge(rootId, "M170 256H200V134H230", null, "compute");
    b += edge(rootId, "M170 288H200V414H230", null, "compute");
    b += edge(rootId, "M325 100V76H860V100", null, "compute");
    b += edge(rootId, ortho(670, 134, 730, 134), null, "compute");
    b += edge(rootId, "M670 274H685V176", null, "control");
    b += edge(rootId, "M670 398H715V176", null, "state");
    b += edge(rootId, "M670 430H700V274H730", null, "state");
    b += edge(rootId, ortho(420, 414, 460, 414), null, "state");
    b += edge(rootId, ortho(885, 176, 885, 240), null, "compute");
    b += edge(rootId, ortho(885, 308, 885, 352), null, "gather");
    b += edge(rootId, ortho(280, 570, 330, 570),
      [305, 632, "TRAINING · pool within g", 180], "orange", true);
    b += edge(rootId, ortho(540, 570, 590, 570),
      [555, 514, "TRAINING · initialize", 170], "orange", true);
    b += edge(rootId, ortho(770, 570, 820, 570),
      [795, 514, "TRAINING · uptrain", 160], "orange", true);
    return baseSvg(rootId, "gqa", 680, b,
      "GQA with T5 relative position bias, explicit query-to-KV grouping, and groupwise mean-pool uptraining");
  }

  function mlaDiagram(rootId) {
    var b = "";

    b += box(470, 84, 160, 60, M("h_t", "Input hidden"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "03");

    b += box(150, 190, 200, 64, M("c_t^Q=W^{DQ}h_t", "Query down-projection"),
      M("c_t^Q\\in\\mathbb R^{d_q}", "raw query latent"), "compute", 2, "04");
    b += box(450, 190, 200, 64, M("c_t^{KV}=W^{DKV}h_t", "KV down-projection"),
      M("c_t^{KV}\\in\\mathbb R^{d_c}", "raw joint KV latent"), "compute", 3, "05");
    b += box(750, 190, 200, 64, M("k_t^R=R_tW^{KR}h_t", "Decoupled RoPE key"),
      M("d_h^R\\;\\text{shared}", "shared positional slice"), "compute", 4, "06");

    b += box(150, 300, 200, 72, M("q_i^C,\\;q_i^R", "Up-project queries"),
      M("W_i^{UQ}c_t^Q,\\;R_tW_i^{QR}c_t^Q", "content + RoPE query"),
      "compute", 5, "06", { subSize: 8.2 });
    b += cacheBox(450, 300, 200, 64, M("c_{1:t}^{KV}", "Latent KV cache"),
      "the only content cache", 6, "07");
    b += cacheBox(750, 300, 200, 64, M("k_{1:t}^{R}", "RoPE key cache"),
      "the only positional cache", 7, "06");

    b += box(150, 420, 200, 64,
      M("\\widetilde q_i=(W_i^{UK})^{\\mathsf T}q_i^C", "Absorbed query"),
      "decode main path", "compute", 8, "08", { titleSize: 9.8 });
    b += box(430, 420, 300, 72,
      M("s_{i,s}=\\frac{\\widetilde q_i^{\\mathsf T}c_s^{KV}+q_i^{R\\mathsf T}k_s^R}{\\sqrt{d_h+d_h^R}}+M",
        "Absorbed score + mask"),
      null, "compute", 9, "09", { titleSize: 9.4 });
    b += box(430, 532, 300, 68,
      M("m_i=\\textstyle\\sum_s a_{i,s}c_s^{KV}", "Latent value read"),
      M("a_i=\\operatorname{softmax}_s(s_{i,s})", "softmax over history"),
      "gather", 10, "10");
    b += box(430, 640, 300, 64,
      M("u_t=W^O\\operatorname{Concat}_i(W_i^{UV}m_i)", "Absorbed output write"),
      null, "gather", 11, "11", { titleSize: 9.8 });

    b += box(780, 540, 260, 72,
      M("k^C=W^{UK}c,\\;v=W^{UV}c", "Conceptual reconstruction"),
      "training-equivalent · never cached", "orange", null, "08",
      { dashed: true, titleSize: 9.8, subSize: 8.2 });

    b += edge(rootId, "M510 144V166H250V190", null, "compute");
    b += edge(rootId, ortho(550, 144, 550, 190), null, "compute");
    b += edge(rootId, "M590 144V166H850V190", null, "compute");
    b += edge(rootId, ortho(250, 254, 250, 300), null, "compute");
    b += edge(rootId, ortho(550, 254, 550, 300), null, "state");
    b += edge(rootId, ortho(850, 254, 850, 300), null, "state");
    b += edge(rootId, ortho(250, 372, 250, 420), null, "compute");
    b += edge(rootId, ortho(350, 452, 430, 452), null, "compute");
    b += edge(rootId, "M350 336H400V436H430", null, "compute");
    b += edge(rootId, ortho(550, 364, 550, 420), null, "state");
    b += edge(rootId, "M850 364V390H650V420", null, "state");
    b += edge(rootId, "M450 332H410V566H430", null, "state");
    b += edge(rootId, ortho(580, 492, 580, 532), null, "compute");
    b += edge(rootId, ortho(580, 600, 580, 640), null, "gather");
    b += edge(rootId, "M650 348H740V576H780",
      [870, 510, "OPTIONAL · conceptual view", 210], "orange", true);
    return baseSvg(rootId, "mla", 744, b,
      "MLA following the DeepSeek-V2 block: h_t, two down-projections, highlighted latent caches, absorbed decode attention, and a clearly optional conceptual reconstruction");
  }

  function dsaDiagram(rootId) {
    var b = "";
    b += panel(24, 52, 1052, 256, "UPPER LANE · LIGHTNING INDEXER SCORES FULL HISTORY", "control");
    b += panel(24, 340, 1052, 260, "LOWER LANE · MLA LATENT CACHE → GATHER → CANDIDATE MLA", "gather");

    b += box(40, 104, 112, 62, M("h_t", "Query hidden"),
      "detached index input", "compute", 1, "04");
    b += box(40, 214, 112, 62, M("h_{1:L}", "History hidden"),
      "index-key source", "compute", null, "04");
    b += box(184, 88, 178, 82, M("q_{t,j}^I", "Indexer qI heads"),
      "pRoPE → Hadamard → FP8", "compute", 2, "04");
    b += cacheBox(184, 202, 178, 82, M("k_{1:L}^I", "Indexer kI cache"),
      "pRoPE → Hadamard → FP8", 3, "04");
    b += box(398, 74, 132, 64, M("w_{t,j}^I", "Head weights wI"),
      "direct projection", "control", null, "04");
    b += box(398, 170, 214, 94,
      M("I_{t,s}=\\sum_jw_{t,j}^I\\operatorname{ReLU}(q_{t,j}^{I\\mathsf T}k_s^I)",
        "Full-history Indexer logits"),
      M("[B,L_q,L_k]", "every history position"), "compute", 4, "04", { titleSize: 8.8 });
    b += box(646, 176, 166, 80, M("\\mathcal I_t=\\operatorname{TopK}_s(I_{t,s},k)", "TopK addresses"),
      "indices only", "control", 5, "06", { titleSize: 9.8 });

    b += box(844, 68, 202, 72, "Teacher full MLA logits",
      "detach · dense warm-up", "orange", null, "05", { dashed: true });
    b += box(844, 194, 202, 72,
      M("D_{\\mathrm{KL}}(p_t\\Vert\\operatorname{softmax}I_t)", "Indexer KL loss"),
      "training only", "orange", null, "02", { dashed: true });

    b += cacheBox(328, 402, 250, 86, "MLA latent cache",
      M("\\{c_s^{KV},k_s^R\\}_{s=1}^{L}", "full history · original latent"), 6, "07");
    b += box(646, 390, 166, 88,
      M("\\operatorname{Gather}(\\{c^{KV},k^R\\},\\mathcal I_t)", "Gather candidates"),
      null, "gather", 7, "07", { titleSize: 9.6 });
    b += box(646, 508, 166, 64, M("q_t^C,q_t^R", "MLA query"),
      "high-dimensional", "compute", null, "07");
    b += box(844, 386, 202, 108,
      M("o_t=\\operatorname{softmax}_{\\mathcal I_t}\\!(S_t)V", "Candidate-only exact MLA"),
      "renormalize in selected set", "compute", 8, "08", { titleSize: 9.8 });
    b += box(844, 524, 202, 58, M("W^O\\to u_t", "Output projection"),
      "residual write", "gather", 9, "09");

    b += edge(rootId, "M152 135H168V129H184", null, "compute");
    b += edge(rootId, "M152 245H168V243H184", null, "state");
    b += edge(rootId, "M96 104V68H464V74", null, "control");
    b += edge(rootId, "M362 129H380V196H398", null, "compute");
    b += edge(rootId, ortho(362, 243, 398, 243), null, "state");
    b += edge(rootId, ortho(464, 138, 464, 170), null, "control");
    b += edge(rootId, ortho(612, 217, 646, 217), null, "control");
    b += edge(rootId, ortho(729, 256, 729, 390),
      [776, 326, "selected addresses", 174], "control");
    b += edge(rootId, "M578 445H612V434H646", null, "gather");
    b += edge(rootId, ortho(812, 440, 844, 440), null, "gather");
    b += edge(rootId, "M812 540H828V470H844", null, "compute");
    b += edge(rootId, ortho(945, 494, 945, 524), null, "gather");
    b += edge(rootId, ortho(945, 140, 945, 194),
      [1008, 165, "TRAINING · teacher p", 180], "orange", true);
    b += edge(rootId, "M612 180H626V154H826V230H844",
      [736, 140, "TRAINING · full logits I", 190], "orange", true);
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
      M("C^{\\mathrm{Comp}}=\\mathcal C_{\\mathrm{overlap}}(H;\\theta_C)", "Core compressor"),
      "overlap a/b · softmax over 2m", "compute", 2, "02", { titleSize: 9.8, subSize: 8.2 });
    b += box(184, 344, 168, 106,
      M("K^{I\\mathrm{Comp}}=\\mathcal C_{\\mathrm{overlap}}(H;\\theta_I)", "Index compressor"),
      M("\\theta_I\\ne\\theta_C", "independent parameters"), "compute", 3, "02", { titleSize: 9.8 });

    b += cacheBox(410, 92, 160, 88, M("C^{\\mathrm{Comp}}_{1:L/m}", "CComp cache"),
      "core content entries", 4, "06");
    b += cacheBox(410, 242, 160, 88, M("K^{I\\mathrm{Comp}}_{1:L/m}", "KIComp cache"),
      "index keys only", 5, "04");
    b += box(410, 382, 160, 70, M("q_t^I,w_t^I", "Indexer query"),
      "independent path", "compute", null, "04");
    b += box(594, 222, 146, 86,
      M("I_t=\\operatorname{Index}(q_t^I,w_t^I,K^{I\\mathrm{Comp}})", "Score KIComp"),
      null, "compute", 6, "04", { titleSize: 8.8 });
    b += box(594, 346, 146, 70, M("\\mathcal J_t=\\operatorname{TopK}(I_t)", "TopK addresses"),
      null, "control", 7, "04", { titleSize: 9.8 });
    b += box(594, 468, 146, 78,
      M("\\operatorname{Gather}(C^{\\mathrm{Comp}},\\mathcal J_t)", "Gather CComp"),
      null, "gather", 8, "06", { titleSize: 9.6 });

    b += cacheBox(800, 82, 116, 84, "SWA cache",
      "recent raw · pRoPE", 9, "07", { subSize: 8.2 });
    b += box(938, 82, 116, 84, "Query lane",
      M("q_t\\cdot\\operatorname{pRoPE}_{64}", "Hq heads"),
      "compute", 10, "05", { subSize: 8.1 });
    b += box(800, 222, 254, 112,
      M("\\operatorname{MQA}_{K=V}(q_t;\\mathrm{sink},C_{\\mathcal J_t},C_{\\mathrm{SWA}})",
        "One shared-KV MQA + sink"),
      "one softmax: sink + global + SWA", "compute", 11, "08",
      { titleSize: 9.2, subSize: 8.2 });
    b += box(800, 374, 254, 66, M("\\operatorname{RoPE}^{-1}_{64}(-t)", "Inverse partial RoPE"),
      null, "compute", 12, "01");
    b += box(800, 478, 254, 70,
      M("W^{OA}_{\\mathrm{group}}\\to\\operatorname{Concat}\\to W^{OB}", "Grouped output projection"),
      null, "gather", 13, "09", { titleSize: 9.6 });
    b += box(800, 584, 254, 48, M("u_t", "Output hidden"),
      M("[B,L,d]", "[B,L,d]"), "gather", 14, "09");

    b += edge(rootId, "M152 284H168V147H184", null, "compute");
    b += edge(rootId, "M152 284H168V397H184", null, "compute");
    b += edge(rootId, ortho(352, 147, 410, 136), null, "state");
    b += edge(rootId, ortho(352, 397, 410, 286), null, "state");
    b += edge(rootId, ortho(570, 286, 594, 286), null, "compute");
    b += edge(rootId, "M570 417H582V308", null, "compute");
    b += edge(rootId, ortho(667, 308, 667, 346), null, "control");
    b += edge(rootId, ortho(667, 416, 667, 468),
      [720, 442, "addresses", 112], "control");
    b += edge(rootId, "M410 160H400V507H594", null, "gather");
    b += edge(rootId, "M740 507H770V294H800", null, "gather");
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
      "m'=128 · per-channel softmax",
      "compute", 2, "02", { titleSize: 9.2, subSize: 8.2 });
    b += box(184, 86, 176, 72, "Causal publish gate",
      "only closed blocks visible", "control", 3, "06");

    b += cacheBox(426, 116, 304, 92, "Completed CComp cache",
      M("\\{C_i^{\\mathrm{Comp}}:m'(i+1)\\le t\\}", "closed summaries · pRoPE"),
      4, "06");
    b += cacheBox(426, 356, 138, 88, "SWA cache",
      "recent raw · w=128", 5, "07", { subSize: 8.1 });
    b += box(592, 356, 138, 88, "Query lane",
      M("q_t\\cdot\\operatorname{pRoPE}_{64}", "Hq heads"),
      "compute", 6, "04", { subSize: 8.1 });

    b += box(800, 188, 252, 118,
      M("\\operatorname{MQA}_{K=V}(q_t;\\mathrm{sink},C_{\\mathrm{all}},C_{\\mathrm{SWA}})",
        "Dense shared-KV MQA + sink"),
      "one softmax: sink + all CComp + SWA", "compute", 7, "08",
      { titleSize: 9.1, subSize: 8.2 });
    b += box(800, 346, 252, 66, M("\\operatorname{RoPE}^{-1}_{64}(-t)", "Inverse partial RoPE"),
      null, "compute", 8, "05");
    b += box(800, 452, 252, 70,
      M("W^{OA}_{\\mathrm{group}}\\to\\operatorname{Concat}\\to W^{OB}", "Grouped output projection"),
      null, "gather", 9, "09", { titleSize: 9.6 });
    b += box(800, 552, 252, 48, M("u_t", "Output hidden"),
      M("[B,L,d]", "[B,L,d]"), "gather", 10, "09");

    b += edge(rootId, "M152 300H168V263H184", null, "compute");
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

    b += box(470, 84, 160, 60, M("x_t", "Input token"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "04");

    b += box(220, 190, 180, 68, M("\\phi(q_t)", "Query feature"),
      M("\\operatorname{ELU}(x_tW_Q)+1", "ELU(q)+1"), "compute", 2, "01");
    b += box(460, 190, 180, 68, M("\\phi(k_t)", "Key feature"),
      M("\\operatorname{ELU}(x_tW_K)+1", "ELU(k)+1"), "compute", 3, "01");
    b += box(700, 190, 180, 68, M("v_t=x_tW_V", "Value"),
      M("\\mathbb R^{d_v}", "dv channels"), "compute", 4, "04");

    b += box(400, 310, 300, 96,
      M("S_t=S_{t-1}+\\phi(k_t)v_t^{\\mathsf T}", "Recurrent cell · update S"),
      M("z_t=z_{t-1}+\\phi(k_t)", "and normalizer z"),
      "state", 5, "03", { titleSize: 10.4 });

    b += box(400, 470, 300, 80,
      M("y_t=\\phi(q_t)^{\\mathsf T}S_t/(\\phi(q_t)^{\\mathsf T}z_t+\\varepsilon)",
        "Read: numerator / denominator"),
      null, "gather", 6, "03", { titleSize: 9.6 });
    b += box(400, 594, 300, 56, M("u_t=W^Oy_t", "Output hidden"),
      null, "gather", 7, "04");

    b += box(760, 470, 280, 80, "2020 execution boundary",
      "causal recurrence · custom CUDA kernel", "orange", null, "03",
      { dashed: true, subSize: 8.2 });

    b += edge(rootId, "M510 144V166H310V190", null, "compute");
    b += edge(rootId, ortho(550, 144, 550, 190), null, "compute");
    b += edge(rootId, "M590 144V166H790V190", null, "compute");
    b += edge(rootId, ortho(550, 258, 550, 310), null, "state");
    b += edge(rootId, "M790 258V284H660V310", null, "state");
    b += edge(rootId, "M700 340H744V380H700",
      [855, 360, M("S_{t-1},z_{t-1}\\;\\text{carry}", "carry S, z to t+1"), 170], "state");
    b += edge(rootId, "M310 258V510H400", null, "compute");
    b += edge(rootId, ortho(550, 406, 550, 470), null, "state");
    b += edge(rootId, ortho(550, 550, 550, 594), null, "gather");
    return baseSvg(rootId, "linear", 680, b,
      "Original 2020 kernelized linear attention drawn as one recurrent cell with a single S and z feedback loop and normalized readout");
  }

  function deltaDiagram(rootId) {
    var b = "";

    b += box(470, 84, 160, 56, M("x_t", "Input hidden"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "03");

    b += box(50, 180, 150, 76, M("q_t", "q path"),
      "ShortConv→SiLU→L2", "compute", null, "01", { subSize: 8.0 });
    b += box(220, 180, 150, 76, M("k_t", "k path"),
      "ShortConv→SiLU→L2", "compute", null, "01", { subSize: 8.0 });
    b += box(390, 180, 150, 76, M("v_t", "v path"),
      "ShortConv→SiLU", "compute", null, "01", { subSize: 8.0 });
    b += box(560, 180, 150, 76, M("\\alpha_t", "Decay gate"),
      "direct · per-head", "control", null, "03", { subSize: 8.0 });
    b += box(730, 180, 150, 76, M("\\beta_t", "Write gate"),
      "direct · sigmoid", "control", null, "03", { subSize: 8.0 });
    b += box(900, 180, 150, 76, M("g_t", "Output gate"),
      "direct · no conv", "control", null, "03", { subSize: 8.0 });

    b += box(300, 320, 500, 96,
      M("F_t=\\alpha_tF_{t-1}(I-\\beta_tk_tk_t^{\\mathsf T})+\\beta_tv_tk_t^{\\mathsf T}",
        "Recurrence: decay, then delta write"),
      M("F=S^{\\mathsf T}\\in\\mathbb R^{d_v\\times d_k}", "F = S^T in R^{dv×dk}"),
      "state", 2, "02", { titleSize: 10.0 });

    b += box(300, 470, 240, 72,
      M("o_t=F_t(q_t/\\sqrt{d_k})", "Read with q"),
      null, "gather", 3, "02", { titleSize: 10.4 });
    b += box(600, 470, 260, 72,
      M("\\operatorname{RMSNorm}(o_t)\\odot\\operatorname{SiLU}(g_t)", "Norm ⊙ output gate"),
      null, "gather", 4, "03", { titleSize: 9.6 });
    b += box(600, 586, 260, 56, M("W_O\\to u_t", "WO → residual"),
      null, "gather", 5, "03");

    b += box(60, 586, 300, 56, "Training execution",
      "chunkwise WY/UT · decode keeps F + conv states", "orange", null, "04",
      { dashed: true, subSize: 8.0 });

    b += edge(rootId, "M480 140V154H125V180", null, "compute");
    b += edge(rootId, "M508 140V162H295V180", null, "compute");
    b += edge(rootId, "M536 140V170H465V180", null, "compute");
    b += edge(rootId, "M564 140V170H635V180", null, "control");
    b += edge(rootId, "M592 140V162H805V180", null, "control");
    b += edge(rootId, "M620 140V154H975V180", null, "control");
    b += edge(rootId, "M295 256V288H400V320", null, "compute");
    b += edge(rootId, ortho(465, 256, 465, 320), null, "compute");
    b += edge(rootId, ortho(635, 256, 635, 320), null, "control");
    b += edge(rootId, "M805 256V288H700V320", null, "control");
    b += edge(rootId, "M800 344H844V392H800",
      [860, 414, M("F_{t-1}\\;\\text{carry}", "carry F to t+1"), 150], "state");
    b += edge(rootId, ortho(420, 416, 420, 470), null, "state");
    b += edge(rootId, "M125 256V506H300", null, "compute");
    b += edge(rootId, ortho(540, 506, 600, 506), null, "gather");
    b += edge(rootId, "M975 256V506H860", null, "control");
    b += edge(rootId, ortho(730, 542, 730, 586), null, "gather");
    return baseSvg(rootId, "gated-delta", 678, b,
      "Gated DeltaNet following the paper block figure: parallel conv and direct-gate projections, one recurrence cell with a single feedback loop, gated readout");
  }

  function kdaDiagram(rootId) {
    var b = "";

    b += box(470, 84, 160, 56, M("x_t", "Input hidden"),
      M("[B,1,d]", "[B,1,d]"), "compute", 1, "03");

    b += box(50, 180, 150, 76, M("q_t", "q path"),
      "ShortConv→SiLU→L2", "compute", null, "01", { subSize: 8.0 });
    b += box(220, 180, 150, 76, M("k_t", "k path"),
      "ShortConv→SiLU→L2", "compute", null, "01", { subSize: 8.0 });
    b += box(390, 180, 150, 76, M("v_t", "v path"),
      "ShortConv→SiLU", "compute", null, "01", { subSize: 8.0 });
    b += box(560, 180, 150, 76, M("\\alpha_t\\in(0,1)^{d_k}", "Channel decay"),
      "low-rank direct", "control", null, "03", { titleSize: 9.8, subSize: 8.0 });
    b += box(730, 180, 150, 76, M("\\beta_t", "Write gate"),
      "direct · sigmoid", "control", null, "03", { subSize: 8.0 });
    b += box(900, 180, 150, 76, M("g_t", "Output gate"),
      "low-rank direct", "control", null, "03", { subSize: 8.0 });

    b += box(300, 320, 500, 96,
      M("F_t=F_{t-1}\\operatorname{Diag}(\\alpha_t)(I-\\beta_tk_tk_t^{\\mathsf T})+\\beta_tv_tk_t^{\\mathsf T}",
        "Recurrence: channel decay, then delta write"),
      M("F=S^{\\mathsf T}\\in\\mathbb R^{d_v\\times d_k}", "F = S^T in R^{dv×dk}"),
      "state", 2, "02", { titleSize: 9.4 });

    b += box(300, 470, 240, 72,
      M("o_t=F_t(q_t/\\sqrt{d_k})", "Read with q"),
      null, "gather", 3, "02", { titleSize: 10.4 });
    b += box(600, 470, 260, 72,
      M("\\operatorname{RMSNorm}(o_t)\\odot\\sigma(g_t)", "Norm ⊙ output gate"),
      null, "gather", 4, "03", { titleSize: 9.8 });
    b += box(600, 586, 260, 56, M("W_O\\to u_t", "WO → residual"),
      null, "gather", 5, "03");

    b += box(60, 676, 220, 60, "MLA · NoPE",
      "global attention · no PE", "compute", null, "04", { subSize: 8.2 });
    b += box(320, 676, 740, 60,
      "(KDA×3 → MLA-NoPE)×6 → KDA×2 → MLA-NoPE",
      "checkpoint layer order · layerwise, not mixed heads", "orange", 6, "05",
      { titleSize: 10.4, subSize: 8.2 });

    b += edge(rootId, "M480 140V154H125V180", null, "compute");
    b += edge(rootId, "M508 140V162H295V180", null, "compute");
    b += edge(rootId, "M536 140V170H465V180", null, "compute");
    b += edge(rootId, "M564 140V170H635V180", null, "control");
    b += edge(rootId, "M592 140V162H805V180", null, "control");
    b += edge(rootId, "M620 140V154H975V180", null, "control");
    b += edge(rootId, "M295 256V288H400V320", null, "compute");
    b += edge(rootId, ortho(465, 256, 465, 320), null, "compute");
    b += edge(rootId, ortho(635, 256, 635, 320), null, "control");
    b += edge(rootId, "M805 256V288H700V320", null, "control");
    b += edge(rootId, "M800 344H844V392H800",
      [860, 414, M("F_{t-1}\\;\\text{carry}", "carry F to t+1"), 150], "state");
    b += edge(rootId, ortho(420, 416, 420, 470), null, "state");
    b += edge(rootId, "M125 256V506H300", null, "compute");
    b += edge(rootId, ortho(540, 506, 600, 506), null, "gather");
    b += edge(rootId, "M975 256V506H860", null, "control");
    b += edge(rootId, ortho(730, 542, 730, 586), null, "gather");
    b += edge(rootId, ortho(730, 642, 730, 676),
      [880, 656, "checkpoint layer order", 190], "orange");
    b += edge(rootId, ortho(280, 706, 320, 706), null, "orange");
    return baseSvg(rootId, "kda", 768, b,
      "KDA following the Kimi Linear block figure: parallel conv and direct-gate projections, one channel-decay delta recurrence cell with a single feedback loop, gated readout, and the exact checkpoint layer order");
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
