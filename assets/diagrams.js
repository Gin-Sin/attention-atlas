(function () {
  "use strict";

  var P = {
    ink: "#10243b",
    muted: "#718096",
    rule: "#d9d6ce",
    dense: "#285f8f",
    denseSoft: "#eaf2f8",
    sparse: "#8b5d12",
    sparseSoft: "#f8f0df",
    linear: "#39704f",
    linearSoft: "#eaf3ed",
    hybrid: "#684b91",
    hybridSoft: "#f0ebf7",
    cache: "#b6531b",
    cacheSoft: "#fbecdc",
    paper: "#ffffff",
    paper2: "#faf9f5"
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
      ["β 控制定点改写", "先擦除状态对 k 的旧预测，再写入 βkvᵀ；不是无条件累加。"],
      ["读写共用快权重状态", "更新后的 S 被 q 读取；Gated DeltaNet 再经 RMSNorm、独立 SiLU output gate 与 WO。"],
      ["训练与推理两张脸", "训练用 chunkwise WY/scan；decode 只保留 recurrent S 和短卷积缓存。"]
    ],
    kda: [
      ["逐通道 α", "低秩 gate 投影产生 dk 维 α；状态每一行可以拥有不同记忆半衰期。"],
      ["受约束 DPLR", "转移是 (I−βkkᵀ)Diag(α)，即 diagonal decay 加绑定到 k 的 rank-1 修正。"],
      ["完整输入支路", "WQ/WK/WV 后分别做 ShortConv；另有 channel gate α、write gate β 和低秩 output gate g。"],
      ["完整输出支路", "o=Sᵀq 后做 RMSNorm，与 sigmoid(g) 相乘，再经 WO 写回 residual stream。"],
      ["chunkwise 不改变数学", "UT/WY 把一个 chunk 的 rank-1 更新打包成 matmul，只是训练执行图不同。"],
      ["Kimi Linear 是层级混合", "模型按 KDA→KDA→KDA→全局 MLA(NoPE) 周期堆叠，不是在单层内混合 heads。"]
    ]
  };

  function defs(id) {
    return (
      '<defs>' +
      '<marker id="' + id + '-arrow" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="6.6" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#718096"/></marker>' +
      '<pattern id="' + id + '-cache" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#fbecdc"/><line x1="0" y1="0" x2="0" y2="8" stroke="#b6531b" stroke-width="3"/></pattern>' +
      '</defs>'
    );
  }

  function zone(x, y, w, h, title, color) {
    return (
      '<g><rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="10" fill="none" stroke="' + color + '" stroke-width="1.4" stroke-dasharray="7 5"/>' +
      '<rect x="' + (x + 12) + '" y="' + (y - 9) + '" width="' + Math.max(120, title.length * 9) + '" height="20" fill="' + P.paper2 + '"/>' +
      '<text x="' + (x + 20) + '" y="' + (y + 5) + '" font-size="11" font-weight="600" fill="' + color + '">' + title + '</text></g>'
    );
  }

  function box(x, y, w, h, title, sub, color, fill, n, cached) {
    var number = n
      ? '<circle cx="' + (x + 13) + '" cy="' + (y + 13) + '" r="9" fill="' + color + '"/><text x="' + (x + 13) + '" y="' + (y + 17) + '" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">' + n + '</text>'
      : "";
    return (
      '<g><rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="6" fill="' + (cached ? 'url(#diagram-cache)' : fill) + '" stroke="' + color + '" stroke-width="1.4"/>' +
      number +
      '<text x="' + (x + w / 2) + '" y="' + (y + h / 2 - (sub ? 3 : -4)) + '" text-anchor="middle" font-size="12" font-weight="600" fill="' + P.ink + '">' + title + '</text>' +
      (sub ? '<text x="' + (x + w / 2) + '" y="' + (y + h / 2 + 15) + '" text-anchor="middle" font-size="9" fill="' + P.muted + '">' + sub + '</text>' : "") +
      '</g>'
    );
  }

  function cacheBox(id, x, y, w, h, title, sub, n) {
    var number = n
      ? '<circle cx="' + (x + 13) + '" cy="' + (y + 13) + '" r="9" fill="' + P.cache + '"/><text x="' + (x + 13) + '" y="' + (y + 17) + '" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">' + n + '</text>'
      : "";
    return (
      '<g><rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="6" fill="url(#' + id + '-cache)" stroke="' + P.cache + '" stroke-width="1.5"/>' +
      number +
      '<text x="' + (x + w / 2) + '" y="' + (y + h / 2 - 3) + '" text-anchor="middle" font-size="12" font-weight="600" fill="' + P.ink + '">' + title + '</text>' +
      '<text x="' + (x + w / 2) + '" y="' + (y + h / 2 + 15) + '" text-anchor="middle" font-size="9" fill="' + P.cache + '">' + sub + '</text></g>'
    );
  }

  function edge(id, d, label, color, dashed) {
    return (
      '<g><path d="' + d + '" fill="none" stroke="' + (color || P.muted) + '" stroke-width="1.5" ' +
      (dashed ? 'stroke-dasharray="6 5" ' : "") +
      'marker-end="url(#' + id + '-arrow)"/>' +
      (label ? '<text x="' + label[0] + '" y="' + label[1] + '" text-anchor="middle" font-size="9" fill="' + (color || P.muted) + '">' + label[2] + '</text>' : "") +
      '</g>'
    );
  }

  function baseSvg(id, height, body, label) {
    return (
      '<svg viewBox="0 0 1100 ' + height + '" role="img" aria-label="' + label + '" xmlns="http://www.w3.org/2000/svg">' +
      defs(id) +
      '<rect width="1100" height="' + height + '" fill="' + P.paper2 + '"/>' +
      body +
      '</svg>'
    );
  }

  function fullHeadAttention(mode) {
    var id = "full-" + mode;
    var kv = mode === "mha" ? "Hkv = Hq" : mode === "mqa" ? "Hkv = 1" : "1 < Hkv < Hq";
    var mapping = mode === "mha" ? "一一对应" : mode === "mqa" ? "全部 Q 读取同一 KV" : "组内共享 KV";
    var body = "";
    body += zone(20, 44, 430, 540, "PROJECTION & POSITION PATH", P.dense);
    body += zone(472, 44, 310, 540, "ATTENTION CORE", P.hybrid);
    body += zone(804, 44, 276, 540, "MERGE & WRITE BACK", P.linear);
    body += box(42, 276, 120, 62, "X / Norm(X)", "profile-dependent [B,L,d]", P.ink, P.paper, 1);
    body += box(204, 82, 110, 58, "WQ", "d → Hq·dh", P.dense, P.denseSoft, 2);
    body += box(204, 252, 110, 58, "WK", "d → Hkv·dh", P.dense, P.denseSoft, 2);
    body += box(204, 422, 110, 58, "WV", "d → Hkv·dv", P.dense, P.denseSoft, 2);
    body += box(342, 82, 88, 58, "Q heads", "Hq × dh", P.dense, P.paper, 3);
    body += box(342, 252, 88, 58, "K heads", kv, P.dense, P.paper, 3);
    body += box(342, 422, 88, 58, "V heads", kv, P.dense, P.paper, 3);
    body += box(492, 82, 116, 58, "Position(Q)", "RoPE · modern profile", P.hybrid, P.hybridSoft, 4);
    body += box(492, 252, 116, 58, "Position(K)", "RoPE · modern profile", P.hybrid, P.hybridSoft, 4);
    body += cacheBox(id, 626, 230, 132, 104, "KV cache", "K′ + V · decode", 5);
    body += box(492, 392, 266, 58, "Head mapping", mapping, P.dense, P.denseSoft, 5);
    body += box(492, 478, 266, 74, "A=softmax(QKᵀ/√dh+M+Bpos)", "[B,Hq,Lq,Lk]", P.hybrid, P.hybridSoft, 6);
    body += box(824, 182, 236, 68, "A · V", "per-query-head outputs", P.linear, P.linearSoft, 7);
    body += box(824, 292, 236, 58, "Concat heads", "[B,L,Hq·dv]", P.linear, P.paper, 8);
    body += box(824, 392, 236, 58, "WO", "Hq·dv → d", P.linear, P.linearSoft, 8);
    body += box(824, 492, 236, 58, "Output + residual", "[B,L,d]", P.ink, P.paper, 9);
    body += edge(id, "M162 307C182 307 182 111 204 111");
    body += edge(id, "M162 307H204");
    body += edge(id, "M162 307C182 307 182 451 204 451");
    body += edge(id, "M314 111H342");
    body += edge(id, "M314 281H342");
    body += edge(id, "M314 451H342");
    body += edge(id, "M430 111H492");
    body += edge(id, "M430 281H492");
    body += edge(id, "M608 111C650 111 650 462 625 478", [654, 104, "Q′"]);
    body += edge(id, "M608 281H626", [617, 271, "K′"]);
    body += edge(id, "M430 451C520 451 560 334 626 316", [532, 434, "V"]);
    body += edge(id, "M758 282C780 282 780 507 758 515");
    body += edge(id, "M625 450V478");
    body += edge(id, "M758 515C790 515 790 216 824 216");
    body += edge(id, "M758 282C790 282 790 216 824 216");
    body += edge(id, "M942 250V292");
    body += edge(id, "M942 350V392");
    body += edge(id, "M942 450V492");
    return baseSvg(id, 620, body, mode.toUpperCase() + " 完整 attention block");
  }

  function mlaDiagram() {
    var id = "full-mla";
    var b = "";
    b += zone(18, 52, 492, 548, "QUERY PATH", P.dense);
    b += zone(528, 52, 390, 548, "JOINT KV LATENT PATH", P.cache);
    b += zone(936, 52, 146, 548, "CORE", P.hybrid);
    b += box(34, 500, 180, 62, "Input hidden hₜ", "[B,1,d]", P.ink, P.paper, 1);
    b += box(246, 448, 112, 56, "WᴰQ", "down-project + RMSNorm", P.dense, P.denseSoft, 2);
    b += box(384, 448, 108, 56, "latent cₜQ", "dq", P.dense, P.paper, 2);
    b += box(246, 314, 112, 56, "WᵁQ", "up-project H heads", P.dense, P.denseSoft, 3);
    b += box(384, 314, 108, 56, "qᶜ heads", "H × dh", P.dense, P.paper, 3);
    b += box(246, 180, 112, 56, "WQR", "RoPE branch", P.hybrid, P.hybridSoft, 4);
    b += box(384, 180, 108, 56, "RoPE(qᴿ)", "H × dR", P.hybrid, P.paper, 4);
    b += box(246, 82, 246, 62, "concat [qᶜ ; qᴿ]", "full query heads", P.dense, P.denseSoft, 5);
    b += box(548, 448, 112, 56, "WᴰKV", "down-project + RMSNorm", P.cache, P.cacheSoft, 2);
    b += cacheBox(id, 686, 438, 206, 76, "latent cₜKV", "CACHED · dc", 3);
    b += box(548, 314, 112, 56, "WᵁK", "content keys", P.cache, P.cacheSoft, 4);
    b += box(686, 314, 96, 56, "kᶜ heads", "H × dh", P.cache, P.paper, 4);
    b += box(796, 382, 96, 44, "WᵁV", "value up-project", P.cache, P.cacheSoft, 4);
    b += box(796, 314, 96, 56, "v heads", "H × dv", P.cache, P.paper, 4);
    b += box(548, 180, 112, 56, "WKR", "shared RoPE key", P.hybrid, P.hybridSoft, 4);
    b += cacheBox(id, 686, 170, 206, 76, "RoPE(kᴿ)", "CACHED · shared dR", 4);
    b += box(548, 82, 246, 62, "concat [kᶜ ; kᴿ]", "full key heads", P.cache, P.cacheSoft, 5);
    b += box(948, 108, 122, 88, "Scaled QKᵀ", "+ causal mask", P.hybrid, P.hybridSoft, 6);
    b += box(948, 238, 122, 68, "Softmax", "over history", P.hybrid, P.paper, 6);
    b += box(948, 348, 122, 68, "A · V", "head outputs", P.hybrid, P.hybridSoft, 7);
    b += box(948, 458, 122, 68, "WO → uₜ", "write residual", P.ink, P.paper, 8);
    b += edge(id, "M214 531C230 531 230 476 246 476");
    b += edge(id, "M358 476H384");
    b += edge(id, "M384 476C360 476 330 410 302 370");
    b += edge(id, "M358 342H384");
    b += edge(id, "M438 448C438 270 380 250 358 208");
    b += edge(id, "M358 208H384");
    b += edge(id, "M438 180V144");
    b += edge(id, "M438 314V144");
    b += edge(id, "M214 531H548");
    b += edge(id, "M660 476H686");
    b += edge(id, "M686 476C620 476 604 400 604 370");
    b += edge(id, "M660 342H686");
    b += edge(id, "M789 438C820 438 844 438 844 426");
    b += edge(id, "M844 382V370");
    b += edge(id, "M214 531C520 531 520 208 548 208");
    b += edge(id, "M660 208H686");
    b += edge(id, "M734 314V144");
    b += edge(id, "M794 113H948");
    b += edge(id, "M492 113H948");
    b += edge(id, "M1009 196V238");
    b += edge(id, "M1009 306V348");
    b += edge(id, "M892 342H948", [919, 332, "V"]);
    b += edge(id, "M1009 416V458");
    return baseSvg(id, 630, b, "MLA 完整低秩与解耦 RoPE attention block");
  }

  function dsaDiagram() {
    var id = "full-dsa";
    var b = "";
    b += zone(18, 48, 390, 544, "LIGHTNING INDEXER · CHEAP SEARCH", P.sparse);
    b += zone(426, 48, 310, 544, "MLA CACHE & GATHER", P.cache);
    b += zone(754, 48, 328, 544, "HIGH-DIMENSION CORE", P.hybrid);
    b += box(34, 488, 126, 60, "hₜ", "query token", P.ink, P.paper, 1);
    b += box(188, 420, 92, 54, "shared cQ", "MLA WᴰQ + RMSNorm", P.sparse, P.sparseSoft, 2);
    b += box(298, 420, 92, 54, "WᴵUQ → qᴵ", "64 × 128", P.sparse, P.paper, 2);
    b += box(188, 314, 92, 54, "Ww", "head weights", P.sparse, P.sparseSoft, 2);
    b += box(298, 314, 92, 54, "wᴵ", "64 weights", P.sparse, P.paper, 2);
    b += cacheBox(id, 34, 168, 160, 72, "WᴵK(h₁…hₗ) → kᴵ", "LN+pRoPE+Hadamard · FP8", 3);
    b += box(220, 168, 170, 72, "Σ wⱼ ReLU(qⱼ·kₛ)", "one score per position", P.sparse, P.sparseSoft, 4);
    b += box(126, 70, 188, 64, "Top-k selector", "indices 𝓘ₜ · k=2048", P.sparse, P.paper, 5);
    b += cacheBox(id, 446, 344, 270, 92, "MLA latent cache", "cKV + kR · full history", 3);
    b += box(446, 188, 270, 76, "Gather cache[𝓘ₜ]", "selected latent entries only", P.cache, P.cacheSoft, 6);
    b += box(774, 382, 132, 64, "MLA query", "qC + qR", P.dense, P.denseSoft, 6);
    b += box(926, 270, 136, 82, "Sparse MLA core", "exact softmax on k", P.hybrid, P.hybridSoft, 7);
    b += box(926, 416, 136, 64, "WO → output", "residual write", P.ink, P.paper, 8);
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
    b += box(34, 492, 120, 60, "Hidden H", "[B,L,d]", P.ink, P.paper, 1);
    b += box(184, 434, 126, 58, "WaKV / WbKV", "two value streams", P.sparse, P.sparseSoft, 2);
    b += box(184, 522, 126, 48, "WaZ / WbZ", "channel gates", P.sparse, P.sparseSoft, 2);
    b += box(336, 434, 144, 58, "Ca / Cb", "current + previous block", P.sparse, P.paper, 3);
    b += box(336, 522, 144, 48, "Za+Ba / Zb+Bb", "position-biased logits", P.sparse, P.paper, 3);
    b += box(184, 312, 296, 76, "row-softmax over 2m positions", "per-channel weighted merge · stride m", P.sparse, P.sparseSoft, 4);
    b += box(184, 186, 132, 68, "CComp", "core compressed KV", P.sparse, P.paper, 5);
    b += box(348, 186, 132, 68, "IComp", "compressed index key", P.sparse, P.paper, 5);
    b += cacheBox(id, 538, 180, 232, 82, "compressed pools", "RMSNorm + block pRoPE · L/m", 6);
    b += box(538, 354, 108, 58, "qᴵ,wᴵ", "low-rank query", P.sparse, P.sparseSoft, 6);
    b += box(662, 354, 108, 58, "Top-k", "k · checkpoint config", P.sparse, P.paper, 7);
    b += box(538, 476, 232, 64, "Gather selected CComp", "global summaries", P.cache, P.cacheSoft, 7);
    b += cacheBox(id, 828, 448, 112, 78, "SWA cache", "recent w=128 · pRoPE", 6);
    b += box(958, 448, 104, 78, "query q", "Hq heads · pRoPE", P.dense, P.denseSoft, 6);
    b += box(828, 284, 234, 86, "shared K=V MQA + sink", "selected summaries ∪ local raw KV", P.hybrid, P.hybridSoft, 8);
    b += box(828, 220, 234, 48, "inverse RoPE(−t)", "output trailing 64 dims", P.hybrid, P.paper, 9);
    b += box(828, 140, 234, 54, "Grouped WᴼA → concat → WᴼB", "low-rank head groups → d", P.linear, P.linearSoft, 9);
    b += box(828, 68, 234, 54, "Output hidden", "[B,L,d]", P.ink, P.paper, 10);
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
    b += edge(id, "M654 262V354", [675, 310, "scan L/m"]);
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
    b += box(34, 454, 120, 60, "Hidden H", "[B,L,d]", P.ink, P.paper, 1);
    b += box(184, 404, 112, 56, "WKV", "content stream", P.sparse, P.sparseSoft, 2);
    b += box(184, 494, 112, 46, "WZ + B", "channel logits", P.sparse, P.sparseSoft, 2);
    b += box(326, 404, 138, 56, "C block", "m′=128 tokens", P.sparse, P.paper, 3);
    b += box(326, 494, 138, 46, "Z block", "non-overlap", P.sparse, P.paper, 3);
    b += box(184, 276, 280, 76, "row-softmax over m′", "per-channel weighted compression", P.sparse, P.sparseSoft, 4);
    b += box(254, 154, 210, 70, "CComp", "one summary / 128 tokens", P.sparse, P.paper, 5);
    b += cacheBox(id, 522, 154, 246, 88, "all compressed entries", "RMSNorm + block pRoPE · L/m′", 6);
    b += cacheBox(id, 522, 388, 118, 82, "SWA cache", "recent 128 raw · pRoPE", 6);
    b += box(658, 388, 110, 82, "query q", "Hq heads · pRoPE", P.dense, P.denseSoft, 6);
    b += box(826, 304, 236, 90, "Dense shared K=V MQA + sink", "all CComp ∪ local SWA", P.hybrid, P.hybridSoft, 7);
    b += box(826, 250, 236, 42, "inverse RoPE(−t)", "output trailing 64 dims", P.hybrid, P.paper, 8);
    b += box(826, 168, 236, 56, "Grouped WᴼA → concat → WᴼB", "low-rank head groups → d", P.linear, P.linearSoft, 8);
    b += box(826, 76, 236, 56, "Output hidden", "[B,L,d]", P.ink, P.paper, 9);
    b += edge(id, "M154 484H184");
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
    b += box(34, 258, 110, 60, "xₜ", "[B,1,d]", P.ink, P.paper, 1);
    b += box(176, 82, 100, 54, "WQ", "query", P.linear, P.linearSoft, 2);
    b += box(176, 252, 100, 54, "WK", "key", P.linear, P.linearSoft, 2);
    b += box(176, 422, 100, 54, "WV", "value", P.linear, P.linearSoft, 2);
    b += box(306, 82, 102, 54, "φ(qₜ)", "feature map", P.linear, P.paper, 3);
    b += box(306, 252, 102, 54, "φ(kₜ)", "feature map", P.linear, P.paper, 3);
    b += box(306, 422, 102, 54, "vₜ", "dv", P.linear, P.paper, 3);
    b += cacheBox(id, 466, 128, 150, 92, "Sₜ₋₁", "r × dv state", 4);
    b += cacheBox(id, 636, 128, 150, 92, "zₜ₋₁", "r normalizer", 4);
    b += box(466, 300, 150, 82, "Sₜ=Sₜ₋₁+φ(k)vᵀ", "outer-product update", P.cache, P.cacheSoft, 5);
    b += box(636, 300, 150, 82, "zₜ=zₜ₋₁+φ(k)", "normalizer update", P.cache, P.cacheSoft, 5);
    b += box(844, 126, 218, 74, "φ(q)ᵀSₜ", "numerator", P.hybrid, P.hybridSoft, 6);
    b += box(844, 256, 218, 74, "φ(q)ᵀzₜ + ε", "denominator", P.hybrid, P.hybridSoft, 6);
    b += box(844, 384, 218, 66, "normalize → WO", "yₜ + residual", P.linear, P.linearSoft, 7);
    b += box(844, 488, 218, 54, "Output hidden", "[B,1,d]", P.ink, P.paper, 8);
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
    b += '<text x="520" y="524" font-size="10" fill="' + P.muted + '">TRAIN: chunk / prefix-scan parallelization</text><text x="520" y="544" font-size="10" fill="' + P.cache + '">DECODE: recurrent S,z only · constant state</text>';
    return baseSvg(id, 600, b, "Kernelized Linear Attention 完整投影、状态更新与归一化 block");
  }

  function deltaDiagram() {
    var id = "full-delta";
    var b = "";
    b += zone(18, 46, 394, 544, "LOCAL MIXING & PROJECTIONS", P.linear);
    b += zone(430, 46, 374, 544, "GATED DELTA FAST WEIGHT", P.cache);
    b += zone(822, 46, 260, 544, "READOUT", P.hybrid);
    b += box(34, 474, 120, 58, "Input xₜ", "[B,1,d]", P.ink, P.paper, 1);
    b += box(184, 474, 206, 58, "WQ/WK/WV/Wα/Wβ/Wg", "parallel projections", P.linear, P.linearSoft, 2);
    b += box(184, 338, 92, 54, "q/k ShortConv", "SiLU", P.linear, P.linearSoft, 3);
    b += box(298, 338, 92, 54, "v ShortConv", "SiLU", P.linear, P.linearSoft, 3);
    b += box(184, 228, 92, 54, "L2 norm", "q̂,k̂", P.linear, P.paper, 4);
    b += box(298, 228, 92, 54, "β = sigmoid", "write gate / head", P.linear, P.paper, 4);
    b += box(184, 118, 206, 54, "GDN: Wα → log-decay", "scalar α / head · optional in DeltaNet", P.linear, P.linearSoft, 4);
    b += cacheBox(id, 450, 110, 160, 88, "Sₜ₋₁", "dk × dv fast weight", 5);
    b += box(634, 104, 150, 100, "α(I−βkkᵀ)Sₜ₋₁", "forget + directional erase", P.cache, P.cacheSoft, 6);
    b += box(522, 286, 188, 74, "+ β k vᵀ", "write corrected value", P.cache, P.cacheSoft, 6);
    b += cacheBox(id, 522, 438, 188, 78, "updated Sₜ", "recurrent decode state", 7);
    b += box(842, 128, 220, 66, "oₜ = Sₜᵀqₜ", "query-dependent read", P.hybrid, P.hybridSoft, 8);
    b += box(842, 274, 220, 72, "RMSNorm × SiLU(Wg x)", "Gated DeltaNet output gate", P.hybrid, P.paper, 9);
    b += box(842, 426, 220, 62, "WO + residual", "output hidden", P.ink, P.paper, 10);
    b += edge(id, "M154 503H184");
    b += edge(id, "M287 474C287 430 230 430 230 392");
    b += edge(id, "M287 474C320 440 344 430 344 392");
    b += edge(id, "M230 338V282");
    b += edge(id, "M390 503C420 503 420 255 390 255");
    b += edge(id, "M287 474V172");
    b += edge(id, "M610 154H634");
    b += edge(id, "M390 145C500 40 700 40 709 104");
    b += edge(id, "M276 255C430 230 520 300 522 323");
    b += edge(id, "M390 255C470 255 490 323 522 323");
    b += edge(id, "M344 392C430 392 470 340 522 323");
    b += edge(id, "M709 204V286");
    b += edge(id, "M616 360V438");
    b += edge(id, "M710 477C790 477 790 161 842 161");
    b += edge(id, "M276 255C600 210 760 161 842 161");
    b += edge(id, "M952 194V274");
    b += edge(id, "M390 503C640 590 760 390 842 310", [690, 525, "Wg output branch"], P.hybrid);
    b += edge(id, "M952 346V426");
    b += '<text x="452" y="560" font-size="10" fill="' + P.muted + '">TRAIN: WY / chunkwise scan packs rank-1 updates</text>';
    return baseSvg(id, 620, b, "Gated DeltaNet 完整局部卷积、门控状态更新与输出 block");
  }

  function kdaDiagram() {
    var id = "full-kda";
    var b = "";
    b += zone(18, 42, 400, 548, "KDA PARAMETERIZATION", P.linear);
    b += zone(436, 42, 366, 548, "CHANNEL-WISE DPLR STATE", P.cache);
    b += zone(820, 42, 262, 548, "READOUT & HYBRID STACK", P.hybrid);
    b += box(34, 486, 120, 56, "Input xₜ", "[B,1,d]", P.ink, P.paper, 1);
    b += box(182, 486, 216, 56, "WQ/WK/WV/Wα/Wβ/Wg", "parallel projections", P.linear, P.linearSoft, 2);
    b += box(182, 366, 96, 52, "q/k ShortConv", "SiLU + L2Norm", P.linear, P.linearSoft, 3);
    b += box(302, 366, 96, 52, "v ShortConv + β", "SiLU / sigmoid", P.linear, P.linearSoft, 3);
    b += box(182, 246, 216, 66, "W↓α → W↑α", "low-rank channel gate", P.hybrid, P.hybridSoft, 4);
    b += box(182, 128, 216, 66, "αₜ ∈ [0,1]ᵈᵏ", "one decay per key channel", P.hybrid, P.paper, 4);
    b += cacheBox(id, 456, 104, 142, 84, "Sₜ₋₁", "dk × dv", 5);
    b += box(622, 92, 160, 108, "Diag(αₜ)Sₜ₋₁", "fine-grained decay", P.cache, P.cacheSoft, 6);
    b += box(456, 272, 326, 92, "(I−βkkᵀ) · decayed S", "Householder-style erase", P.cache, P.cacheSoft, 6);
    b += box(506, 430, 226, 70, "+ β k vᵀ", "delta write", P.cache, P.cacheSoft, 7);
    b += cacheBox(id, 506, 522, 226, 56, "Sₜ", "decode recurrent state", 7);
    b += box(840, 96, 222, 66, "oₜ = Sₜᵀqₜ", "state read", P.hybrid, P.hybridSoft, 8);
    b += box(840, 220, 222, 66, "RMSNorm × sigmoid(g)", "Wg↓ → Wg↑ output gate", P.hybrid, P.paper, 9);
    b += box(840, 344, 222, 58, "WO + residual", "output hidden", P.ink, P.paper, 10);
    b += box(840, 468, 52, 68, "KDA", "L1", P.linear, P.linearSoft);
    b += box(900, 468, 52, 68, "KDA", "L2", P.linear, P.linearSoft);
    b += box(960, 468, 52, 68, "KDA", "L3", P.linear, P.linearSoft);
    b += box(1020, 468, 52, 68, "MLA", "L4 · NoPE", P.dense, P.denseSoft);
    b += edge(id, "M154 514H182");
    b += edge(id, "M290 486C290 450 230 450 230 418");
    b += edge(id, "M290 486C350 450 350 436 350 418");
    b += edge(id, "M290 486V312");
    b += edge(id, "M290 246V194");
    b += edge(id, "M398 161C500 40 690 40 702 92");
    b += edge(id, "M598 146H622");
    b += edge(id, "M702 200V272");
    b += edge(id, "M278 392C420 360 430 318 456 318");
    b += edge(id, "M398 392C440 392 470 465 506 465");
    b += edge(id, "M619 364V430");
    b += edge(id, "M619 500V522");
    b += edge(id, "M732 550C810 550 810 129 840 129");
    b += edge(id, "M278 392C650 230 760 129 840 129");
    b += edge(id, "M951 162V220");
    b += edge(id, "M398 514C650 590 760 360 840 253", [690, 523, "low-rank Wg"], P.hybrid);
    b += edge(id, "M951 286V344");
    b += '<text x="840" y="448" font-size="10" font-weight="600" fill="' + P.hybrid + '">KIMI LINEAR · LAYERWISE 3 : 1</text>';
    return baseSvg(id, 620, b, "KDA 完整通道门控状态更新、输出门与 Kimi Linear 混合层栈");
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
      badges: ["TECHNICAL-REPORT VIEW", "STRIPED = CACHED", "SOLID = COMPUTE", "DASHED = TRAINING / BOUNDARY"]
    };
  }

  window.AttentionDiagrams = { build: build };
})();
