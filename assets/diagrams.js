(function () {
  "use strict";

  var P = {
    ink: "var(--ink)",
    muted: "var(--muted)",
    rule: "var(--rule)",
    dense: "var(--dense)",
    denseSoft: "var(--dense-soft)",
    sparse: "var(--sparse)",
    sparseSoft: "var(--sparse-soft)",
    linear: "var(--linear)",
    linearSoft: "var(--linear-soft)",
    hybrid: "var(--hybrid)",
    hybridSoft: "var(--hybrid-soft)",
    cache: "var(--cache)",
    cacheInk: "var(--orange-ink)",
    cacheSoft: "var(--cache-soft)",
    paper: "var(--paper)",
    paper2: "var(--off-white)"
  };

  var guides = {
    mha: [
      ["投影不是复制", "同一个 hidden state 分别经过 WQ、WK、WV；MHA 的每个 Q/K/V 头都有独立投影。"],
      ["位置机制要区分年代", "原始 MHA 在 embedding 端加入正弦位置编码；图中的 Q/K RoPE 是现代 causal-LLM profile，V 不旋转。"],
      ["真正缓存的是 K/V", "decode 时历史 Q 不再需要；缓存宽度由 KV 头数决定。"],
      ["核心仍是精确 softmax", "缩放点积、因果 mask、softmax、再乘 V；FlashAttention 只改变执行方式。"],
      ["多头最后必须写回", "各头输出先 concat，再经 WO 回到 residual stream 的 d 维。"]
    ],
    mqa: [
      ["Q 仍然多头", "WQ 产生 Hq 个问题；MQA 不是单头注意力。"],
      ["K/V 只投影一份", "所有 Q 头共享同一 K 与 V，KV cache 的头维从 Hq 降为 1。"],
      ["广播是逻辑关系", "高效 kernel 不应真的复制共享 K/V；图中的扇出表示读取关系。"],
      ["softmax 没有近似", "每个 Q 头仍独立计算分数与输出，减少的是历史数据搬运；RoPE 只是现代实现 profile。"],
      ["类比", "多位分析员各提不同问题，但共用一套档案目录和原始材料。"]
    ],
    gqa: [
      ["先把 Q 头分组", "Hq 个 Q 头被划成 Hkv 组，每组共享一个 K/V 头。"],
      ["组内共享、组间独立", "它保留比 MQA 更多的 K/V 子空间，同时少于 MHA。"],
      ["映射由 g(h) 决定", "kernel 根据 query head 找所属 KV group，不需要复制缓存。"],
      ["缓存旋钮", "Hkv=1 是 MQA；Hkv=Hq 是 MHA；中间值就是 GQA。"],
      ["输出路径未变", "每个 Q 头仍产生一个 head output，最后 concat→WO；原始 GQA/T5 使用 relative bias，现代 Llama profile 才常见 RoPE。"]
    ],
    mla: [
      ["两条低秩主干", "query 侧先得到 cQ；KV 侧先得到 cKV。两者在下投影后做 RMSNorm，再进入上投影。"],
      ["内容与位置解耦", "qC/kC 来自低秩 latent；qR/kR 单独应用 RoPE，再分别 concat。"],
      ["缓存只保留两块", "推理保存 cKV 与共享 kR；斜线纹理表示随历史累积的状态。"],
      ["多头表达仍存在", "不同 WUK/WUV 从同一 latent 恢复各头内容 key/value，不等同于 MQA。"],
      ["矩阵吸收", "decode 可把 key 上投影吸收到 query 侧，避免为全部历史显式重建 K。"],
      ["类比", "缓存压缩源文件和位置印章，各注意力头按自己的模板读取。"]
    ],
    dsa: [
      ["Indexer 是独立检索支路", "低维 qI 与缓存的 kI 扫描全部历史；两者含独立 partial RoPE，kI 再做 Hadamard rotation 以支持 FP8。"],
      ["多头 ReLU 分数", "每个 indexer query head 做 ReLU 点积，再由 wt 加权汇总为位置分数。"],
      ["top-k 输出索引", "离散选择只产生地址；随后从 MLA cache gather 原始 latent，而非使用 indexer value。"],
      ["core attention 仍完整", "被选中的 cKV/kR 进入高维 MLA core，候选集合内继续做精确 softmax。"],
      ["训练有专门监督", "虚线 KL 路径把 indexer 分布对齐到主注意力目标，输入从主图 detach；推理时整条 KL 支路关闭。"],
      ["类比", "先用目录卡找书号，再到书库取原书精读。"]
    ],
    csa: [
      ["两路重叠压缩", "a 路看当前 m-token 块，b 路看前一块；两路合并可减轻块边界损失。"],
      ["逐维而非整向量加权", "Za/Zb 加位置 bias 后，在 2m 个位置上逐 channel softmax。"],
      ["core 与 indexer 各有摘要", "同一压缩算子分别生成 CComp 和 IComp；后者仅用于检索。"],
      ["先缩短、再 top-k", "候选池先从 L 变为 L/m，Lightning Indexer 再选 k 个摘要。"],
      ["局部窗口保留原件", "最近 w 个 token 的未压缩 KV 与选中的全局摘要一起进入同一个 core MQA。"],
      ["V4 的位置与输出细节", "compressed/local KV 与 query 使用 trailing-64 partial RoPE；core 含 attention sink，输出再 inverse RoPE(-t) 并经 WOA→WOB。"],
      ["类比", "相邻页先做重叠摘要，再用摘要索引找章节，同时把桌面上的最近几页原文一起读。"]
    ],
    hca: [
      ["单路非重叠压缩", "每 m′ 个 token 通过逐维 softmax 汇成一个 CComp；没有 CSA 的 a/b 重叠。"],
      ["压缩率足够大", "m′=128 时历史只剩 L/128 个条目，因此可直接读取全部摘要。"],
      ["没有 Lightning Indexer", "HCA 不做 top-k，也不存在选择漏召回；主要误差来自重压缩。"],
      ["全局摘要 + 局部原文", "dense compressed branch 与 w=128 的 sliding-window branch 在 core MQA 汇合。"],
      ["输出仍需分组写回", "shared KV core 含 attention sink；输出 inverse RoPE(-t)，再分组经 WOA 与 WOB 回到 d 维。"],
      ["位置编码不是装饰", "query、compressed KV 与 local KV 都只旋转 trailing 64 维；inverse RoPE 在输出侧撤销位置旋转。"],
      ["类比", "完整阅读一份高度浓缩的总目录，同时保留手边最近一页原文。"]
    ],
    linear: [
      ["先做特征映射", "q/k 经过 φ 后，核相似度可分解，矩阵乘法顺序才可以交换；原始 Linear Transformer 使用 ELU(x)+1。"],
      ["状态有两部分", "S 累积 key-value 外积；z 累积 key，用于分母归一化。"],
      ["decode 是真正递推", "每来一个 token 只更新固定大小的 S、z，不保存逐 token KV。"],
      ["训练不是逐步 Python 循环", "并行/chunk kernel 用 prefix scan 或块矩阵乘法重排同一递推。"],
      ["读取仍然 query-dependent", "φ(q) 左乘 S 得分子，左乘 z 得分母，二者相除后经 WO 输出。"],
      ["类比", "不保存每张票据，而维护可被不同查询读取的统计台账。"]
    ],
    delta: [
      ["2024 大模型参数化", "WQ/WK/WV 投影后，q/k/v 分别经过 causal ShortConv 与 SiLU；q/k 再做 L2Norm。"],
      ["q/k 做归一化", "单位 key 使 rank-1 擦除的几何意义稳定，也便于精确覆盖证明。"],
      ["α 是全局遗忘", "Gated DeltaNet 先把旧状态整体乘 α，快速清理不相关上下文。"],
      ["β 控制定点改写", "先擦除状态对 k 的旧预测，再写入 βkv^T；不是无条件累加。"],
      ["读写共用快权重状态", "更新后的 S 被 q 读取；Gated DeltaNet 再经 RMSNorm、独立 SiLU output gate 与 WO。"],
      ["训练与推理两张脸", "训练用 chunkwise WY/scan；decode 只保留 recurrent S 和短卷积缓存。"]
    ],
    kda: [
      ["逐通道 α", "低秩 gate 投影产生 dk 维 α；状态每一行可以拥有不同记忆半衰期。"],
      ["受约束 DPLR", "转移是 (I−βkk^T)Diag(α)，即 diagonal decay 加绑定到 k 的 rank-1 修正。"],
      ["完整输入支路", "WQ/WK/WV 后分别做 ShortConv；另有 channel gate α、write gate β 和低秩 output gate g。"],
      ["完整输出支路", "o=S^Tq 后做 RMSNorm，与 sigmoid(g) 相乘，再经 WO 写回 residual stream。"],
      ["chunkwise 不改变数学", "UT/WY 把一个 chunk 的 rank-1 更新打包成 matmul，只是训练执行图不同。"],
      ["Kimi Linear 是层级混合", "模型按 KDA→KDA→KDA→全局 MLA(NoPE) 周期堆叠，不是在单层内混合 heads。"]
    ]
  };

  var memories = {
    mha: "每个 Q 头都有自己的 K/V 档案柜：最完整，也最占缓存。",
    mqa: "很多个 Q 提问题，只共用一套 K/V 档案：多问、一库。",
    gqa: "把 Q 头分成若干小组，每组共用一套 K/V：组内共享、组间独立。",
    mla: "历史 token 只存一张低维提货单 cKV 和一枚位置印章 kR，各头按需读取。",
    dsa: "Lightning Indexer 先查目录选 top-k，真正的 MLA 再取原始 latent 精读。",
    csa: "先把每 4 个 token 做重叠智能摘要，再 top-k；最近 128 个 token 保留原文。",
    hca: "每 128 个 token 压成一份摘要，摘要足够少所以全部阅读，同时保留近期原文。",
    linear: "不保存逐 token KV，只维护一张键值统计表 S 和一个分母计数器 z。",
    delta: "先问记忆里这个 key 已经写了什么，再只写入预测误差；GDN 还会先整体褪色。",
    kda: "让记忆矩阵每个通道以不同速度褪色，再做 delta 纠写；每 3 层 KDA 插 1 层全局 MLA。"
  };

  function defs(id) {
    return (
      '<defs>' +
      '<marker id="' + id + '-arrow" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="6.6" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="' + P.muted + '"/></marker>' +
      '<pattern id="' + id + '-cache" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="9" height="9" fill="' + P.cacheSoft + '"/><line x1="0" y1="0" x2="0" y2="9" stroke="' + P.cache + '" stroke-opacity=".42" stroke-width="2"/></pattern>' +
      '</defs>'
    );
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

  function textLabel(x, y, value, fontSize, color, weight) {
    return '<text x="' + x + '" y="' + (y + fontSize * 0.34) + '" text-anchor="middle" font-family="JetBrains Mono" font-size="' + fontSize + '" font-weight="' + (weight || 400) + '" fill="' + color + '">' + escapeText(value) + '</text>';
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

  function zone(x, y, w, h, title, color) {
    return (
      '<g><rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="14" fill="none" stroke="' + color + '" stroke-opacity=".64" stroke-width="1.2" stroke-dasharray="7 6"/>' +
      '<rect x="' + (x + 12) + '" y="' + (y - 9) + '" width="' + Math.max(120, title.length * 9) + '" height="20" fill="' + P.paper2 + '"/>' +
      '<text x="' + (x + 20) + '" y="' + (y + 5) + '" font-family="JetBrains Mono" font-size="11" font-weight="600" fill="' + color + '">' + escapeText(title) + '</text></g>'
    );
  }

  function box(x, y, w, h, title, sub, color, fill, n, cachePatternId) {
    var number = n
      ? '<circle cx="' + (x + 13) + '" cy="' + (y + 13) + '" r="9" fill="' + P.ink + '"/>' + textLabel(x + 13, y + 13, n, 9, P.paper, 700)
      : "";
    var titleY = y + h / 2 - (sub ? 5 : 0);
    return (
      '<g><rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="9" fill="' + (cachePatternId ? 'url(#' + cachePatternId + '-cache)' : fill) + '" stroke="' + color + '" stroke-opacity=".9" stroke-width="1.25"/>' +
      number +
      labelMarkup(x + w / 2, titleY, w - 20, 30, title, 12, P.ink, 600) +
      (sub ? labelMarkup(x + w / 2, y + h / 2 + 15, w - 18, 22, sub, 9, P.muted, 400) : "") +
      '</g>'
    );
  }

  function cacheBox(id, x, y, w, h, title, sub, n) {
    var number = n
      ? '<circle cx="' + (x + 13) + '" cy="' + (y + 13) + '" r="9" fill="' + P.cacheInk + '"/>' + textLabel(x + 13, y + 13, n, 9, P.paper, 700)
      : "";
    return (
      '<g><rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="9" fill="url(#' + id + '-cache)" stroke="' + P.cache + '" stroke-width="1.35"/>' +
      number +
      labelMarkup(x + w / 2, y + h / 2 - 5, w - 20, 30, title, 12, P.ink, 600) +
      labelMarkup(x + w / 2, y + h / 2 + 15, w - 18, 22, sub, 9, P.cache, 500) + '</g>'
    );
  }

  function edge(id, d, label, color, dashed) {
    return (
      '<g><path d="' + d + '" fill="none" stroke="' + (color || P.muted) + '" stroke-width="1.5" ' +
      (dashed ? 'stroke-dasharray="6 5" ' : "") +
      'marker-end="url(#' + id + '-arrow)"/>' +
      (label ? labelMarkup(label[0], label[1], label[3] || 180, 22, label[2], 9, color || P.muted, 500) : "") +
      '</g>'
    );
  }

  function baseSvg(id, height, body, label) {
    return (
      '<svg viewBox="0 0 1100 ' + height + '" role="img" aria-label="' + escapeText(label) + '" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono">' +
      defs(id) +
      '<rect width="1100" height="' + height + '" fill="' + P.paper2 + '"/>' +
      body +
      '</svg>'
    );
  }

  function fullHeadAttention(mode) {
    var id = "full-" + mode;
    var routing = mode === "mha"
      ? M("\\text{one-to-one}\\;\\cdot\\;H_{kv}=H_q", "一一对应 · Hkv = Hq")
      : mode === "mqa"
        ? M("\\text{all Q share one KV}\\;\\cdot\\;H_{kv}=1", "全部 Q 读取同一 KV · Hkv = 1")
        : M("\\text{grouped KV}\\;\\cdot\\;1<H_{kv}<H_q", "组内共享 KV · 1 < Hkv < Hq");
    var body = "";
    body += zone(18, 58, 344, 540, "PROJECT & RESHAPE", P.dense);
    body += zone(382, 58, 344, 540, "POSITION, CACHE & ROUTING", P.cache);
    body += zone(746, 58, 336, 540, "SCORE, MIX & WRITE BACK", P.hybrid);

    body += box(34, 270, 120, 66, M("X\\,/\\,\\operatorname{Norm}(X)", "X / Norm(X)"), M("[B,L_q,d]", "[B,Lq,d] · profile-dependent"), P.ink, P.paper, 1);
    body += box(188, 92, 152, 68, M("W_Q\\to Q\\;\\text{heads}", "WQ → Q heads"), M("[B,H_q,L_q,D]", "[B,Hq,Lq,D]"), P.dense, P.denseSoft, 2);
    body += box(188, 270, 152, 68, M("W_K\\to K_{\\mathrm{new}}", "WK → Knew"), M("[B,H_{kv},L_q,D]", "[B,Hkv,Lq,D]"), P.dense, P.denseSoft, 2);
    body += box(188, 448, 152, 68, M("W_V\\to V_{\\mathrm{new}}", "WV → Vnew"), M("[B,H_{kv},L_q,D]", "[B,Hkv,Lq,D]"), P.dense, P.denseSoft, 2);

    body += box(402, 92, 142, 68, M("\\operatorname{Position}(Q)", "Position(Q)"), "RoPE · modern profile", P.hybrid, P.hybridSoft, 3);
    body += box(402, 244, 142, 68, M("\\operatorname{Position}(K)", "Position(K)"), "RoPE · modern profile", P.hybrid, P.hybridSoft, 3);
    body += cacheBox(id, 568, 226, 138, 116, "Append KV cache", M("2\\times[B,H_{kv},L_k,D]", "2×[B,Hkv,Lk,D]"), 4);
    body += box(402, 420, 304, 74, "Logical head routing", routing, P.dense, P.denseSoft, 5);

    body += box(766, 106, 296, 94, M("S=QK^{\\mathsf T}/\\sqrt D+M+B_{\\mathrm{pos}}", "S = QK^T/√D + mask + position bias"), M("[B,H_q,L_q,L_k]", "[B,Hq,Lq,Lk] · before softmax"), P.hybrid, P.hybridSoft, 6);
    body += box(766, 246, 296, 76, M("A=\\operatorname{softmax}_{k}(S)", "A = softmax over key axis"), "exact attention weights", P.hybrid, P.paper, 6);
    body += box(766, 366, 296, 70, M("O_{\\mathrm{heads}}=AV", "Oheads = A · V"), M("[B,H_q,L_q,D]", "[B,Hq,Lq,D]"), P.linear, P.linearSoft, 7);
    body += box(766, 474, 142, 72, M("\\operatorname{Concat}\\to W_O", "Concat → WO"), M("H_qD\\to d", "Hq·D → d"), P.linear, P.paper, 8);
    body += box(930, 474, 132, 72, "Residual output", M("[B,L_q,d]", "[B,Lq,d]"), P.ink, P.paper, 9);

    body += edge(id, "M154 303C172 303 172 126 188 126");
    body += edge(id, "M154 303H188");
    body += edge(id, "M154 303C172 303 172 482 188 482");
    body += edge(id, "M340 126H402");
    body += edge(id, "M340 304C370 304 382 278 402 278");
    body += edge(id, "M544 278H568");
    body += edge(id, "M340 482C520 482 520 326 568 326");
    body += edge(id, "M473 160C473 350 510 390 554 420");
    body += edge(id, "M637 342V420");
    body += edge(id, "M706 457C738 430 742 190 766 153");
    body += edge(id, "M914 200V246");
    body += edge(id, "M914 322V366");
    body += edge(id, "M706 284C740 300 744 394 766 401", [734, 290, M("\\text{cached }V", "cached V")]);
    body += edge(id, "M914 436C914 456 860 456 837 474");
    body += edge(id, "M908 510H930");
    return baseSvg(id, 630, body, mode.toUpperCase() + " 完整 causal self-attention block");
  }

  function mlaDiagram() {
    var id = "full-mla";
    var b = "";
    b += zone(18, 74, 492, 548, "QUERY CONSTRUCTION", P.dense);
    b += zone(528, 74, 390, 548, "JOINT KV LATENT & CACHE", P.cache);
    b += zone(936, 74, 146, 548, "ATTENTION CORE", P.hybrid);

    b += box(340, 642, 420, 54, M("\\text{Input hidden }h_t", "Input hidden h_t"), M("[B,1,d]", "[B,1,d] · shared source"), P.ink, P.paper, 1);

    b += box(138, 526, 252, 66, M("W^{DQ}\\to\\operatorname{RMSNorm}\\to c_t^Q", "W^DQ → RMSNorm → latent c_t^Q"), M("d_q\\;\\text{query bottleneck}", "dq · query low-rank bottleneck"), P.dense, P.denseSoft, 2);
    b += box(42, 394, 186, 64, M("W^{UQ}\\to q^C\\;\\text{heads}", "W^UQ → q^C heads"), M("H\\times d_h", "content query · H × dh"), P.dense, P.paper, 3);
    b += box(264, 394, 220, 64, M("W^{QR}\\to\\operatorname{RoPE}(q^R)", "W^QR → RoPE(q^R)"), M("H\\times d_R", "position query · H × dR"), P.hybrid, P.hybridSoft, 4);
    b += box(92, 258, 350, 68, M("\\operatorname{concat}[q^C;q^R]", "concat [q^C ; q^R]"), "complete multi-head query", P.dense, P.denseSoft, 5);

    b += cacheBox(id, 598, 526, 270, 66, M("W^{DKV}\\to\\operatorname{RMSNorm}\\to c_t^{KV}", "W^DKV → RMSNorm → latent c_t^KV"), M("d_c\\;\\text{joint K/V bottleneck}", "CACHED · dc · joint K/V bottleneck"), 2);
    b += box(544, 394, 156, 64, M("W^{UK}\\to k^C\\;\\text{heads}", "W^UK → k^C heads"), M("H\\times d_h", "content key · H × dh"), P.cache, P.paper, 3);
    b += box(724, 394, 156, 64, M("W^{UV}\\to v\\;\\text{heads}", "W^UV → v heads"), M("H\\times d_v", "value · H × dv"), P.cache, P.paper, 3);
    b += cacheBox(id, 724, 286, 156, 72, M("W^{KR}h_t\\to\\operatorname{RoPE}(k^R)", "W^KR(h_t) → RoPE(k^R)"), M("d_R\\;\\text{shared}", "CACHED · shared dR"), 4);
    b += box(544, 258, 156, 68, M("\\operatorname{concat}[k^C;k^R]", "concat [k^C ; k^R]"), "complete key heads", P.cache, P.cacheSoft, 5);

    b += box(948, 112, 122, 86, M("QK^{\\mathsf T}/\\sqrt D", "Scaled QK^T"), M("+\\;M_{\\mathrm{causal}}", "+ causal mask"), P.hybrid, P.hybridSoft, 6);
    b += box(948, 238, 122, 64, "Softmax", "over cached history", P.hybrid, P.paper, 6);
    b += box(948, 350, 122, 68, M("A\\,V", "A · V"), "per-head output", P.hybrid, P.hybridSoft, 7);
    b += box(948, 474, 122, 68, M("\\operatorname{Concat}\\to W_O\\to u_t", "Concat → WO → u_t"), "write residual", P.ink, P.paper, 8);

    b += edge(id, "M460 642C410 624 330 610 264 592");
    b += edge(id, "M640 642C690 624 760 610 733 592");
    b += edge(id, "M264 526C220 500 165 482 135 458");
    b += edge(id, "M264 526C320 500 360 482 374 458");
    b += edge(id, "M135 394C135 360 190 344 220 326");
    b += edge(id, "M374 394C374 360 330 344 310 326");
    b += edge(id, "M733 526C690 500 640 482 622 458");
    b += edge(id, "M733 526C780 500 800 482 802 458");
    b += edge(id, "M700 394C700 360 650 344 622 326");
    b += edge(id, "M760 669C920 669 920 322 880 322", [914, 520, "direct position-key branch"]);
    b += edge(id, "M802 286C760 270 720 270 700 292");
    b += edge(id, "M442 292C650 214 850 158 948 155");
    b += edge(id, "M700 292C800 230 880 180 948 155");
    b += edge(id, "M1009 198V238");
    b += edge(id, "M1009 302V350");
    b += edge(id, "M880 426C916 426 930 384 948 384", [918, 414, M("V", "V")]);
    b += edge(id, "M1009 418V474");
    return baseSvg(id, 720, b, "MLA 完整低秩、解耦 RoPE 与推理缓存结构");
  }

  function dsaDiagram() {
    var id = "full-dsa";
    var b = "";
    b += zone(18, 48, 390, 544, "LIGHTNING INDEXER · CHEAP SEARCH", P.sparse);
    b += zone(426, 48, 310, 544, "MLA CACHE & GATHER", P.cache);
    b += zone(754, 48, 328, 544, "HIGH-DIMENSION CORE", P.hybrid);
    b += box(34, 488, 126, 60, M("h_t", "h_t"), "query token", P.ink, P.paper, 1);
    b += box(188, 420, 92, 54, M("\\text{shared }c^Q", "shared c^Q"), M("W^{DQ}+\\operatorname{RMSNorm}", "MLA W^DQ + RMSNorm"), P.sparse, P.sparseSoft, 2);
    b += box(298, 420, 92, 54, M("W^{IUQ}\\to q^I", "W^IUQ → q^I"), M("64\\times128", "64 × 128"), P.sparse, P.paper, 2);
    b += box(188, 314, 92, 54, M("W_w", "Ww"), "head weights", P.sparse, P.sparseSoft, 2);
    b += box(298, 314, 92, 54, M("w^I", "w^I"), "64 weights", P.sparse, P.paper, 2);
    b += cacheBox(id, 34, 168, 160, 72, M("W^{IK}(h_{1:L})\\to k^I", "W^IK(h_1…h_L) → k^I"), "LN + pRoPE + Hadamard · FP8", 3);
    b += box(220, 168, 170, 72, M("\\sum_j w_j\\operatorname{ReLU}(q_j\\cdot k_s)", "Σ_j w_j ReLU(q_j·k_s)"), "one score per position", P.sparse, P.sparseSoft, 4);
    b += box(126, 70, 188, 64, M("\\operatorname{TopK}", "Top-k selector"), M("\\mathcal I_t,\\;k=2048", "indices I_t · k=2048"), P.sparse, P.paper, 5);
    b += cacheBox(id, 446, 344, 270, 92, "MLA latent cache", M("c^{KV}+k^R\\;\\text{over full history}", "cKV + kR · full history"), 3);
    b += box(446, 188, 270, 76, M("\\operatorname{Gather}(\\mathrm{cache},\\mathcal I_t)", "Gather cache[I_t]"), "selected latent entries only", P.cache, P.cacheSoft, 6);
    b += box(774, 382, 132, 64, "MLA query", M("q^C+q^R", "qC + qR"), P.dense, P.denseSoft, 6);
    b += box(926, 270, 136, 82, "Sparse MLA core", "exact softmax on k", P.hybrid, P.hybridSoft, 7);
    b += box(926, 416, 136, 64, M("W_O\\to\\text{output}", "WO → output"), "residual write", P.ink, P.paper, 8);
    b += box(774, 82, 288, 70, "KL index loss · training only", "align index scores to main-attention target", P.sparse, P.paper, 9);
    b += edge(id, "M160 518C174 518 174 447 188 447");
    b += edge(id, "M160 518C174 518 174 341 188 341");
    b += edge(id, "M280 447H298");
    b += edge(id, "M280 341H298");
    b += edge(id, "M344 420V240");
    b += edge(id, "M344 314V240");
    b += edge(id, "M194 204H220");
    b += edge(id, "M305 168V134");
    b += edge(id, "M314 102C380 102 400 226 446 226", [390, 130, "indices"]);
    b += edge(id, "M581 344V264");
    b += edge(id, "M716 226C780 226 805 311 926 311");
    b += edge(id, "M160 518C500 570 690 414 774 414");
    b += edge(id, "M906 414C920 414 920 352 926 331");
    b += edge(id, "M994 352V416");
    b += edge(id, "M305 70C460 20 760 20 840 82", [600, 23, "detach + KL"], P.sparse, true);
    return baseSvg(id, 620, b, "DSA Lightning Indexer 与 MLA core 完整 block");
  }

  function csaDiagram() {
    var id = "full-csa";
    var b = "";
    b += zone(18, 42, 482, 558, "OVERLAPPING SEQUENCE COMPRESSOR", P.sparse);
    b += zone(518, 42, 272, 558, "INDEX & CACHE", P.cache);
    b += zone(808, 42, 274, 558, "LOCAL + GLOBAL CORE", P.hybrid);
    b += box(34, 492, 120, 60, "Hidden H", M("[B,L,d]", "[B,L,d]"), P.ink, P.paper, 1);
    b += box(184, 434, 126, 58, M("W_a^{KV}\\,/\\,W_b^{KV}", "WaKV / WbKV"), "two value streams", P.sparse, P.sparseSoft, 2);
    b += box(184, 522, 126, 48, M("W_a^Z\\,/\\,W_b^Z", "WaZ / WbZ"), "channel gates", P.sparse, P.sparseSoft, 2);
    b += box(336, 434, 144, 58, M("C^a\\,/\\,C^b", "Ca / Cb"), "current + previous block", P.sparse, P.paper, 3);
    b += box(336, 522, 144, 48, M("Z^a+B^a\\,/\\,Z^b+B^b", "Za+Ba / Zb+Bb"), "position-biased logits", P.sparse, P.paper, 3);
    b += box(184, 312, 296, 76, M("\\operatorname{softmax}_{2m}\\;\\text{per channel}", "row-softmax over 2m positions"), M("\\text{weighted merge}\\;\\cdot\\;\\operatorname{stride}=m", "per-channel weighted merge · stride m"), P.sparse, P.sparseSoft, 4);
    b += box(184, 186, 132, 68, M("C^{\\mathrm{Comp}}", "CComp"), "core compressed KV", P.sparse, P.paper, 5);
    b += box(348, 186, 132, 68, M("I^{\\mathrm{Comp}}", "IComp"), "compressed index key", P.sparse, P.paper, 5);
    b += cacheBox(id, 538, 180, 232, 82, "compressed pools", M("\\operatorname{RMSNorm}+\\operatorname{pRoPE}\\;\\cdot\\;L/m", "RMSNorm + block pRoPE · L/m"), 6);
    b += box(538, 354, 108, 58, M("q^I,w^I", "qI,wI"), "low-rank query", P.sparse, P.sparseSoft, 6);
    b += box(662, 354, 108, 58, M("\\operatorname{TopK}", "Top-k"), "k · checkpoint config", P.sparse, P.paper, 7);
    b += box(538, 476, 232, 64, M("\\operatorname{Gather}(C^{\\mathrm{Comp}})", "Gather selected CComp"), "global summaries", P.cache, P.cacheSoft, 7);
    b += cacheBox(id, 828, 448, 112, 78, "SWA cache", M("w=128\\;\\cdot\\;\\operatorname{pRoPE}", "recent w=128 · pRoPE"), 6);
    b += box(958, 448, 104, 78, "query q", M("H_q\\;\\text{heads}\\;\\cdot\\;\\operatorname{pRoPE}", "Hq heads · pRoPE"), P.dense, P.denseSoft, 6);
    b += box(828, 284, 234, 86, M("K=V\\;\\text{ shared MQA + sink}", "shared K=V MQA + sink"), M("\\text{summaries}\\cup\\text{local raw KV}", "selected summaries ∪ local raw KV"), P.hybrid, P.hybridSoft, 8);
    b += box(828, 220, 234, 48, M("\\operatorname{RoPE}^{-1}(-t)", "inverse RoPE(-t)"), "output trailing 64 dims", P.hybrid, P.paper, 9);
    b += box(828, 140, 234, 54, M("W^{OA}\\to\\operatorname{concat}\\to W^{OB}", "Grouped W^OA → concat → W^OB"), M("\\text{low-rank groups}\\to d", "low-rank head groups → d"), P.linear, P.linearSoft, 9);
    b += box(828, 68, 234, 54, "Output hidden", M("[B,L,d]", "[B,L,d]"), P.ink, P.paper, 10);
    b += edge(id, "M154 522H184");
    b += edge(id, "M154 522C164 522 174 546 184 546");
    b += edge(id, "M310 463H336");
    b += edge(id, "M310 546H336");
    b += edge(id, "M408 434V388");
    b += edge(id, "M408 522V388");
    b += edge(id, "M332 312C280 290 250 274 250 254");
    b += edge(id, "M420 312V254");
    b += edge(id, "M250 186C330 150 460 150 538 210");
    b += edge(id, "M414 186C470 160 500 180 538 210");
    b += edge(id, "M654 262C654 318 592 326 592 354", [634, 310, M("\\operatorname{scan}(L/m)", "scan L/m")]);
    b += edge(id, "M646 383H662");
    b += edge(id, "M716 412V476");
    b += edge(id, "M770 508C800 508 800 327 828 327");
    b += edge(id, "M940 487C950 430 900 410 890 370");
    b += edge(id, "M1010 448V370");
    b += edge(id, "M945 284V268");
    b += edge(id, "M945 220V194");
    b += edge(id, "M945 140V122");
    return baseSvg(id, 630, b, "CSA 重叠压缩、Indexer、局部窗口与 core MQA 完整 block");
  }

  function hcaDiagram() {
    var id = "full-hca";
    var b = "";
    b += zone(18, 46, 466, 532, "HEAVY NON-OVERLAPPING COMPRESSOR", P.sparse);
    b += zone(502, 46, 286, 532, "COMPRESSED GLOBAL CACHE", P.cache);
    b += zone(806, 46, 276, 532, "DENSE COMPRESSED CORE", P.hybrid);
    b += box(34, 454, 120, 60, "Hidden H", M("[B,L,d]", "[B,L,d]"), P.ink, P.paper, 1);
    b += box(184, 404, 112, 56, M("W^{KV}", "WKV"), "content stream", P.sparse, P.sparseSoft, 2);
    b += box(184, 494, 112, 46, M("W^Z+B", "WZ + B"), "channel logits", P.sparse, P.sparseSoft, 2);
    b += box(326, 404, 138, 56, "C block", M("m'=128\\;\\text{tokens}", "m'=128 tokens"), P.sparse, P.paper, 3);
    b += box(326, 494, 138, 46, "Z block", "non-overlap", P.sparse, P.paper, 3);
    b += box(184, 276, 280, 76, M("\\operatorname{softmax}_{m'}\\;\\text{per channel}", "row-softmax over m′"), "per-channel weighted compression", P.sparse, P.sparseSoft, 4);
    b += box(254, 154, 210, 70, M("C^{\\mathrm{Comp}}", "CComp"), "one summary / 128 tokens", P.sparse, P.paper, 5);
    b += cacheBox(id, 522, 154, 246, 88, "all compressed entries", M("\\operatorname{RMSNorm}+\\operatorname{pRoPE}\\;\\cdot\\;L/m'", "RMSNorm + block pRoPE · L/m′"), 6);
    b += cacheBox(id, 522, 388, 118, 82, "SWA cache", M("w=128\\;\\cdot\\;\\operatorname{pRoPE}", "recent 128 raw · pRoPE"), 6);
    b += box(658, 388, 110, 82, "query q", M("H_q\\;\\text{heads}\\;\\cdot\\;\\operatorname{pRoPE}", "Hq heads · pRoPE"), P.dense, P.denseSoft, 6);
    b += box(826, 304, 236, 90, M("K=V\\;\\text{ dense MQA + sink}", "Dense shared K=V MQA + sink"), M("C^{\\mathrm{Comp}}\\cup\\text{local SWA}", "all CComp ∪ local SWA"), P.hybrid, P.hybridSoft, 7);
    b += box(826, 250, 236, 42, M("\\operatorname{RoPE}^{-1}(-t)", "inverse RoPE(-t)"), "output trailing 64 dims", P.hybrid, P.paper, 8);
    b += box(826, 168, 236, 56, M("W^{OA}\\to\\operatorname{concat}\\to W^{OB}", "Grouped W^OA → concat → W^OB"), M("\\text{low-rank groups}\\to d", "low-rank head groups → d"), P.linear, P.linearSoft, 8);
    b += box(826, 76, 236, 56, "Output hidden", M("[B,L,d]", "[B,L,d]"), P.ink, P.paper, 9);
    b += edge(id, "M154 484C170 484 170 432 184 432");
    b += edge(id, "M154 484C170 484 170 517 184 517");
    b += edge(id, "M296 432H326");
    b += edge(id, "M296 517H326");
    b += edge(id, "M395 404V352");
    b += edge(id, "M395 494V352");
    b += edge(id, "M324 276V224");
    b += edge(id, "M464 189H522");
    b += edge(id, "M768 198C800 198 800 349 826 349");
    b += edge(id, "M640 429C760 429 780 370 826 370");
    b += edge(id, "M768 429C792 429 800 349 826 349");
    b += edge(id, "M944 304V292");
    b += edge(id, "M944 250V224");
    b += edge(id, "M944 168V132");
    return baseSvg(id, 610, b, "HCA 重压缩、全局 dense 摘要与局部窗口完整 block");
  }

  function linearDiagram() {
    var id = "full-linear";
    var b = "";
    b += zone(18, 46, 410, 518, "TOKEN PROJECTIONS", P.linear);
    b += zone(446, 46, 360, 518, "ASSOCIATIVE MEMORY", P.cache);
    b += zone(824, 46, 258, 518, "READ & OUTPUT", P.hybrid);
    b += box(34, 258, 110, 60, M("x_t", "x_t"), M("[B,1,d]", "[B,1,d]"), P.ink, P.paper, 1);
    b += box(176, 82, 100, 54, M("W_Q", "WQ"), "query", P.linear, P.linearSoft, 2);
    b += box(176, 252, 100, 54, M("W_K", "WK"), "key", P.linear, P.linearSoft, 2);
    b += box(176, 422, 100, 54, M("W_V", "WV"), "value", P.linear, P.linearSoft, 2);
    b += box(306, 82, 102, 54, M("\\phi(q_t)", "φ(q_t)"), "feature map", P.linear, P.paper, 3);
    b += box(306, 252, 102, 54, M("\\phi(k_t)", "φ(k_t)"), "feature map", P.linear, P.paper, 3);
    b += box(306, 422, 102, 54, M("v_t", "v_t"), M("d_v", "dv"), P.linear, P.paper, 3);
    b += cacheBox(id, 466, 128, 150, 92, M("S_{t-1}", "S_t-1"), M("r\\times d_v\\;\\text{state}", "r × dv state"), 4);
    b += cacheBox(id, 636, 128, 150, 92, M("z_{t-1}", "z_t-1"), "r normalizer", 4);
    b += box(466, 300, 150, 82, M("S_t=S_{t-1}+\\phi(k_t)v_t^{\\mathsf T}", "S_t=S_t-1+φ(k_t)v_t^T"), "outer-product update", P.cache, P.cacheSoft, 5);
    b += box(636, 300, 150, 82, M("z_t=z_{t-1}+\\phi(k_t)", "z_t=z_t-1+φ(k_t)"), "normalizer update", P.cache, P.cacheSoft, 5);
    b += box(844, 126, 218, 74, M("\\phi(q_t)^{\\mathsf T}S_t", "φ(q_t)^T S_t"), "numerator", P.hybrid, P.hybridSoft, 6);
    b += box(844, 256, 218, 74, M("\\phi(q_t)^{\\mathsf T}z_t+\\varepsilon", "φ(q_t)^T z_t + ε"), "denominator", P.hybrid, P.hybridSoft, 6);
    b += box(844, 384, 218, 66, M("\\operatorname{normalize}\\to W_O", "normalize → WO"), M("y_t+\\text{residual}", "y_t + residual"), P.linear, P.linearSoft, 7);
    b += box(844, 488, 218, 54, "Output hidden", M("[B,1,d]", "[B,1,d]"), P.ink, P.paper, 8);
    b += edge(id, "M144 288C160 288 160 109 176 109");
    b += edge(id, "M144 288H176");
    b += edge(id, "M144 288C160 288 160 449 176 449");
    b += edge(id, "M276 109H306");
    b += edge(id, "M276 279H306");
    b += edge(id, "M276 449H306");
    b += edge(id, "M408 279C440 279 440 341 466 341");
    b += edge(id, "M408 449C440 449 440 341 466 341");
    b += edge(id, "M541 220V300");
    b += edge(id, "M408 279C560 250 620 341 636 341");
    b += edge(id, "M711 220V300");
    b += edge(id, "M616 341C800 341 800 163 844 163");
    b += edge(id, "M786 341C820 341 820 293 844 293");
    b += edge(id, "M408 109C650 40 780 163 844 163");
    b += edge(id, "M408 109C690 60 780 293 844 293");
    b += edge(id, "M953 200V256");
    b += edge(id, "M953 330V384");
    b += edge(id, "M953 450V488");
    b += textLabel(670, 524, "TRAIN: chunk / prefix-scan parallelization", 10, P.muted, 500);
    b += mathLabel(670, 544, 360, 22, M("\\text{DECODE: recurrent }S,z\\text{ only}\\;\\cdot\\;\\text{constant state}", "DECODE: recurrent S,z only · constant state"), 10, P.cache, 500);
    return baseSvg(id, 600, b, "Kernelized Linear Attention 完整投影、状态更新与归一化 block");
  }

  function deltaDiagram() {
    var id = "full-delta";
    var b = "";
    b += zone(18, 54, 330, 540, "1 · PARAMETERIZE CURRENT TOKEN", P.linear);
    b += zone(366, 54, 374, 540, "2 · PREDICT, ERASE & REWRITE", P.cache);
    b += zone(758, 54, 324, 540, "3 · READOUT", P.hybrid);

    b += box(38, 480, 116, 58, M("\\text{Input }x_t", "Input x_t"), M("[B,1,d]", "[B,1,d]"), P.ink, P.paper, 1);
    b += box(176, 470, 152, 78, "Parallel projections", M("W_Q/W_K/W_V/W_\\alpha/W_\\beta/W_g", "WQ/WK/WV/Wα/Wβ/Wg"), P.linear, P.linearSoft, 2);
    b += box(58, 330, 124, 72, M("q/k\\;\\text{paths}", "q/k paths"), M("\\operatorname{ShortConv}\\to\\operatorname{SiLU}\\to\\operatorname{L2Norm}", "ShortConv→SiLU→L2Norm"), P.linear, P.paper, 3);
    b += box(204, 330, 124, 72, M("v,\\beta\\;\\text{paths}", "v and β paths"), M("\\operatorname{ShortConv}\\to\\operatorname{SiLU}/\\sigma", "ShortConv→SiLU / sigmoid"), P.linear, P.paper, 3);
    b += box(58, 186, 124, 82, M("\\text{GDN scalar }\\alpha_t", "GDN scalar α"), M("W_\\alpha\\to\\log\\text{-decay}", "Wα→log-decay · DeltaNet=1"), P.hybrid, P.hybridSoft, 4);
    b += box(204, 186, 124, 82, M("\\text{output gate }g_t", "output gate g"), M("W_g\\to\\operatorname{SiLU}", "Wg→SiLU · GDN profile"), P.hybrid, P.hybridSoft, 4);

    b += cacheBox(id, 482, 96, 142, 76, M("S_{t-1}", "S_t-1"), M("[B,H,d_k,d_v]", "[B,H,dk,dv]"), 5);
    b += box(398, 236, 310, 118, M("S_t=\\alpha_tS_{t-1}+\\beta_tk_t\\!\\left(v_t-(\\alpha_tS_{t-1})^{\\mathsf T}k_t\\right)^{\\mathsf T}", "S_t = αS_t-1 + βk(v-(αS_t-1)^T k)^T"), "decay → predict old value → error-controlled rewrite", P.cache, P.cacheSoft, 6);
    b += cacheBox(id, 482, 442, 142, 76, M("S_t", "S_t"), "fixed recurrent state", 7);

    b += box(788, 112, 264, 72, M("o_t=S_t^{\\mathsf T}(q_t/\\sqrt{d_k})", "o_t = S_t^T(q_t/√dk)"), "query-dependent state read", P.hybrid, P.hybridSoft, 8);
    b += box(788, 236, 264, 72, M("\\operatorname{RMSNorm}(o_t)\\times\\operatorname{SiLU}(g_t)", "RMSNorm(o_t) × SiLU(g)"), "Gated DeltaNet output gate", P.hybrid, P.paper, 9);
    b += box(788, 360, 264, 64, M("W_O+\\text{residual}", "WO + residual"), "attention-block output", P.ink, P.paper, 10);
    b += box(788, 488, 264, 66, "TRAIN: decay-aware UT / WY", "chunk boundary state + intra-chunk parallel read", P.linear, P.linearSoft);

    b += edge(id, "M154 509H176");
    b += edge(id, "M252 470C220 444 152 426 120 402");
    b += edge(id, "M252 470C285 444 288 426 266 402");
    b += edge(id, "M252 470C220 390 152 304 120 268");
    b += edge(id, "M252 470C300 390 290 304 266 268");
    b += edge(id, "M553 172V236");
    b += edge(id, "M182 366C300 366 340 300 398 300", [286, 356, M("q_t,k_t", "q,k")]);
    b += edge(id, "M328 366C360 350 380 322 398 312", [354, 340, M("v_t,\\beta_t", "v,β")]);
    b += edge(id, "M182 227C300 190 350 246 398 270", [300, 194, M("\\alpha_t", "α")]);
    b += edge(id, "M553 354V442");
    b += edge(id, "M624 480C720 480 730 148 788 148");
    b += edge(id, "M920 184V236");
    b += edge(id, "M920 308V360");
    b += edge(id, "M920 424V488", [938, 466, "training schedule"], P.linear, true);
    return baseSvg(id, 630, b, "Gated DeltaNet 的预测误差改写、标量遗忘与输出门");
  }

  function kdaDiagram() {
    var id = "full-kda";
    var b = "";
    b += zone(18, 54, 330, 540, "1 · PARAMETERIZE CURRENT TOKEN", P.linear);
    b += zone(366, 54, 374, 540, "2 · UPDATE CHANNEL-WISE STATE", P.cache);
    b += zone(758, 54, 324, 540, "3 · READ & PLACE IN HYBRID STACK", P.hybrid);

    b += box(38, 480, 116, 58, M("\\text{Input }x_t", "Input x_t"), M("[B,1,d]", "[B,1,d]"), P.ink, P.paper, 1);
    b += box(176, 470, 152, 78, "Parallel projections", M("W_Q/W_K/W_V/W_\\alpha/W_\\beta/W_g", "WQ/WK/WV/Wα/Wβ/Wg"), P.linear, P.linearSoft, 2);
    b += box(58, 330, 124, 72, M("q/k\\;\\text{paths}", "q/k paths"), M("\\operatorname{ShortConv}\\to\\operatorname{SiLU}\\to\\operatorname{L2Norm}", "ShortConv→SiLU→L2Norm"), P.linear, P.paper, 3);
    b += box(204, 330, 124, 72, M("v,\\beta\\;\\text{paths}", "v and β paths"), M("\\operatorname{ShortConv}\\to\\operatorname{SiLU}/\\sigma", "ShortConv→SiLU / sigmoid"), P.linear, P.paper, 3);
    b += box(58, 186, 124, 82, M("\\text{channel }\\alpha_t", "channel α"), M("W_\\alpha^\\downarrow\\to W_\\alpha^\\uparrow\\to\\log\\text{-decay}", "W↓α→W↑α→log-decay"), P.hybrid, P.hybridSoft, 4);
    b += box(204, 186, 124, 82, M("\\text{output gate }g_t", "output gate g"), M("W_g^\\downarrow\\to W_g^\\uparrow\\to\\sigma", "W↓g→W↑g→sigmoid"), P.hybrid, P.hybridSoft, 4);

    b += cacheBox(id, 482, 96, 142, 76, M("S_{t-1}", "S_t-1"), M("[B,H,d_k,d_v]", "[B,H,dk,dv]"), 5);
    b += box(398, 236, 310, 118, M("(I-\\beta_tk_tk_t^{\\mathsf T})\\operatorname{Diag}(\\alpha_t)S_{t-1}+\\beta_tk_tv_t^{\\mathsf T}", "(I-βkk^T)Diag(α)S_t-1 + βkv^T"), "channel decay → directional erase → delta write", P.cache, P.cacheSoft, 6);
    b += cacheBox(id, 482, 442, 142, 76, M("S_t", "S_t"), "fixed recurrent state", 7);

    b += box(788, 112, 264, 72, M("o_t=S_t^{\\mathsf T}(q_t/\\sqrt{d_k})", "o_t = S_t^T(q_t/√dk)"), "query-dependent state read", P.hybrid, P.hybridSoft, 8);
    b += box(788, 236, 264, 72, M("\\operatorname{RMSNorm}(o_t)\\times\\sigma(g_t)", "RMSNorm(o_t) × sigmoid(g)"), "low-rank output gate", P.hybrid, P.paper, 9);
    b += box(788, 360, 264, 64, M("W_O+\\text{residual}", "WO + residual"), "KDA block output", P.ink, P.paper, 10);
    b += box(788, 488, 56, 66, "KDA", "L1", P.linear, P.linearSoft);
    b += box(852, 488, 56, 66, "KDA", "L2", P.linear, P.linearSoft);
    b += box(916, 488, 56, 66, "KDA", "L3", P.linear, P.linearSoft);
    b += box(980, 488, 72, 66, "MLA", "L4 · NoPE", P.dense, P.denseSoft);

    b += edge(id, "M154 509H176");
    b += edge(id, "M252 470C220 444 152 426 120 402");
    b += edge(id, "M252 470C285 444 288 426 266 402");
    b += edge(id, "M252 470C220 390 152 304 120 268");
    b += edge(id, "M252 470C300 390 290 304 266 268");
    b += edge(id, "M553 172V236");
    b += edge(id, "M182 366C300 366 340 300 398 300", [286, 356, M("q_t,k_t", "q,k")]);
    b += edge(id, "M328 366C360 350 380 322 398 312", [354, 340, M("v_t,\\beta_t", "v,β")]);
    b += edge(id, "M182 227C300 190 350 246 398 270", [300, 194, M("\\alpha_t", "α")]);
    b += edge(id, "M553 354V442");
    b += edge(id, "M624 480C720 480 730 148 788 148");
    b += edge(id, "M920 184V236");
    b += edge(id, "M920 308V360");
    b += edge(id, "M920 424V488");
    b += mathLabel(920, 466, 280, 24, M("\\text{KIMI LINEAR}\\;\\cdot\\;\\mathrm{KDA}\\to\\mathrm{KDA}\\to\\mathrm{KDA}\\to\\mathrm{MLA(NoPE)}", "KIMI LINEAR · KDA → KDA → KDA → MLA(NoPE)"), 10, P.hybrid, 600);
    return baseSvg(id, 630, b, "KDA 参数生成、通道级 DPLR 更新、输出门与 3:1 层栈");
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
    if (k === "mha" || k === "mqa" || k === "gqa") svg = fullHeadAttention(k);
    if (k === "mla") svg = mlaDiagram();
    if (k === "dsa") svg = dsaDiagram();
    if (k === "csa") svg = csaDiagram();
    if (k === "hca") svg = hcaDiagram();
    if (k === "linear") svg = linearDiagram();
    if (k === "delta") svg = deltaDiagram();
    if (k === "kda") svg = kdaDiagram();
    return {
      svg: svg,
      notes: guides[k] || [],
      memory: memories[k] || "",
      badges: ["TECHNICAL-REPORT VIEW", "STRIPED = CACHED", "SOLID = COMPUTE", "DASHED = TRAINING / BOUNDARY"]
    };
  }

  window.AttentionDiagrams = { build: build };
})();
