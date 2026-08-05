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

  var STYLE_VARS = [
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
      ["先分瓶颈再读图", "左车道是训练/Prefill 的 compute-bound 执行图，右车道是逐 token Decode 的 memory-bound 执行图；先确定阶段瓶颈，再决定沿哪条车道读。"],
      ["共享投影", R`同一组权重供两条车道使用：\(c^{KV}=W^{DKV}h_t\)（512 维共享源）、每头 query 的 128+64 两段，以及共享 \(k^R=R_tW^{KR}h_t\)（64 维）。`],
      ["Prefill 车道 · MHA 形态", R`\(k_i^C=W_i^{UK}c^{KV}\)、\(v_i=W_i^{UV}c^{KV}\) 显式展开成每头 128 维，再做标准多头打分与读取；展开量只服务本次并行计算，不作为跨请求的持久缓存。`],
      ["Decode 车道 · MQA 形态", R`\(\widetilde q_i=(W_i^{UK})^{\mathsf T}q_i^C\) 直接对缓存 latent 打分，softmax 后先聚合 512 维 latent，再经吸收后的 \(W_i^{O}W_i^{UV}\) 一次写回；历史多头 K/V 从不重建。`],
      ["精确桥梁", "中央绿桥标注 same weights · exact linear reassociation：NoPE 内容通道的线性结合律允许在“先展开”与“先吸收”之间切换，两条车道给出相同结果，不是可选近似。"],
      ["Partial RoPE 支路", R`64 维逐头 \(q^R\) 与一份共享 \(k^R\) 负责相对位置，同时进入两条车道的分数；内容主干保持 NoPE，吸收才可能成立。`],
      ["条件化近优", "在 full softmax、线性投影、给定两阶段预算且 Decode 确实 memory-bound 的限定下，MLA 同时逼近 Prefill 想要的 MHA 与 Decode 想要的共享 MQA；超出假设不自动成立。"],
      ["视觉语义", "蓝=计算，玫瑰=持久缓存/状态，薰衣草=聚合/写回，绿=执行图切换桥；两条车道等宽等重，没有谁是“主路径”。悬停或键盘聚焦任一节点，可只保留它的上下游数据通路。"]
    ],
    mfa: [
      ["共享三投影", R`\(S_q,S_k,S_v\) 把每个 token 投到同一 C 维空间；共享 k/v 形状为 \([B,T,C]\)，没有 head 轴。`],
      ["逐头 C×C 镜头", R`每个 head 用完整 \(Q_c\in\mathbb R^{C\times C}\) 改写匹配方式；\([B,m,T,C]\) 的逐头 query 只存在于计算中，不进缓存。`],
      ["缓存与 head 数解耦", R`玫瑰色缓存每 token 只有 \(2C\) 个元素（KR 为 \(C\)）；增加 head 数只增加权重与算力，不复制历史 KV。`],
      ["标准 RoPE", R`旋转只作用于逐头 \(q_{t,c}\) 与共享 \(k_s\)，value 不旋转；论文 common settings 使用 base 500,000，ALiBi 为单独消融。`],
      ["KR 变体", R`橙色虚线为 MFA-KR：\(v_s=k_s(I+\operatorname{diag}(\alpha)N)\)，\(\alpha\) 零初始化，初始时 value 精确等于 key，缓存从 2C 降到 C。`],
      ["视觉语义", "蓝=计算，玫瑰=缓存/状态，薰衣草=聚合/写回；橙色虚线表示可选的 key-reuse 变体。"]
    ],
    tpa: [
      ["六路因子流", R`每个 token 生成 \(A_{Q/K/V}\)（head 轴）与 \(B_{Q/K/V}\)（channel 轴）六路因子；A 为 \([B,T,R,h]\)，B 为 \([B,T,R,d_h]\)。`],
      ["逐行 RoPE", R`旋转只作用于 \(B_Q\)、\(B_K\) 的每一行：\(R_t(A^\top B)=A^\top R_t(B)\)；A 因子与 \(B_V\) 保持不旋转。`],
      ["因子缓存", R`玫瑰色缓存保存 \(A_K,\widetilde B_K,A_V,B_V\)，每 token \((R_K+R_V)(h+d_h)\) 个元素；\(\widetilde B_K\) 已预旋转。`],
      ["因子域收缩", R`score 由 \(R_Q\times R_K\) 个 channel 内积经 A 因子逐头加权得到，从不物化完整历史 K；value 聚合同理直接收缩 \(A_V,B_V\)。`],
      ["重建仅为验证", "橙色虚线路径显式重建 Q/K/V 以验证代数等价；先重建再调普通 MHA kernel 数值不变，但丢掉 FlashTPA 式 decode 收益。"],
      ["视觉语义", "蓝=计算，玫瑰=缓存/状态，薰衣草=聚合/写回；橙色虚线表示仅供验证的重建路径。"]
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
    mla: "MLA：Prefill 把 512 维 latent 展开成每头 128 维 MHA；Decode 把展开矩阵吸收到两侧，按 512 维共享 MQA 读取；64 维 RoPE 支路只负责位置。",
    mfa: R`MFA：共享 C 维 K/V 只存一份，逐头 \(Q_c\)/\(O_c\) 在权重里提供高秩视角；KR 再把 value 折进 key。`,
    tpa: R`TPA：token 现场生成 A/B 因子，RoPE 只转 \(B_Q\)/\(B_K\)；缓存低秩因子并在因子域直接收缩出 score 与输出。`,
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

  function panel(x, y, w, h, title, tone, dashed, options) {
    options = options || {};
    var stroke = toneStroke(tone);
    /* titlePos "bottom" frees the panel's top border for entry ports. */
    var bandY = options.titlePos === "bottom" ? y + h - 9 : y - 9;
    var textY = options.titlePos === "bottom" ? y + h + 5 : y + 5;
    return (
      '<g class="diagram-panel"><rect x="' + x + '" y="' + y + '" width="' + w +
      '" height="' + h + '" rx="14" fill="' + toneFill(tone) +
      '" fill-opacity=".22" stroke="' + stroke + '" stroke-opacity=".62" stroke-width="1.2" ' +
      (dashed ? 'stroke-dasharray="7 6" ' : "") + '/>' +
      '<rect x="' + (x + 12) + '" y="' + bandY + '" width="' +
      Math.max(124, title.length * 7.2 + 20) + '" height="20" fill="' + P.canvas + '"/>' +
      '<text x="' + (x + 20) + '" y="' + textY +
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
    /* options.flow opts the node into the hover/keyboard path preview. */
    var flowAttr = options.flow
      ? ' data-flow-node="' + escapeText(options.flow) + '"'
      : "";
    return (
      '<g class="diagram-code-node" data-code-block="' + blockId + '"' +
      flowAttr + ' role="button" tabindex="0" aria-label="' + aria + '">' +
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

  function edge(rootId, d, label, tone, dashed, flow, marker) {
    tone = tone || "muted";
    if (dashed && !label) {
      throw new Error("Dashed diagram edge requires an explicit label");
    }
    if (flow && (!flow.from || !flow.to)) {
      throw new Error("Flow edge requires both from and to endpoints: " + d);
    }
    var color = tone === "muted" ? P.muted : toneStroke(tone);
    var open = flow
      ? '<g class="diagram-flow-edge" data-flow-from="' + escapeText(flow.from) +
        '" data-flow-to="' + escapeText(flow.to) + '">'
      : "<g>";
    /* marker === false draws an undirected connector (e.g. an equality
       bridge); data-plain-edge keeps it visible to the geometry validator. */
    var tip = marker === false
      ? 'data-plain-edge="1"'
      : 'marker-end="url(#' + rootId + '-arrow-' + tone + ')"';
    return (
      open +
      '<path d="' + d + '" fill="none" stroke="' + (color || P.muted) + '" stroke-width="1.5" ' +
      (dashed ? 'stroke-dasharray="6 5" ' : "") +
      'stroke-linecap="square" stroke-linejoin="round" ' + tip + '/>' +
      (label ? labelMarkup(label[0], label[1], label[3] || 190, 24, label[2],
        label[4] || 8.8, color, 600) : "") +
      '</g>'
    );
  }

  function baseSvg(rootId, diagramKey, height, body, label) {
    return (
      '<svg viewBox="0 0 1100 ' + height + '" role="img" aria-label="' + escapeText(label) + '" data-diagram-key="' + escapeText(diagramKey) + '" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono">' +
      '<g style="' + STYLE_VARS + '">' + defs(rootId) +
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
  //     exist in the chapter's implementation blocks;
  //  4. flow-preview metadata must be closed: data-flow-node ids are unique
  //     and every data-flow-from/to endpoint names an existing flow node.
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
    var edgePattern = /<path d="([^"]+)"[^>]*(?:marker-end|data-plain-edge)/g;
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
    var dashedPattern = /<g(?: [^>]*)?><path\b[^>]*stroke-dasharray="[^"]+"[^>]*\/>([\s\S]*?)<\/g>/g;
    while ((dashedMatch = dashedPattern.exec(svg))) {
      if (!/<(?:text|switch)\b/.test(dashedMatch[1])) {
        throw new Error(diagramKey + ": dashed connector lacks a label");
      }
    }

    // Flow-preview graph metadata: unique node ids, resolvable endpoints.
    var flowNodeIds = {};
    var flowMatch;
    var flowNodePattern = /data-flow-node="([^"]+)"/g;
    while ((flowMatch = flowNodePattern.exec(svg))) {
      if (flowNodeIds[flowMatch[1]]) {
        throw new Error(diagramKey + ": duplicate flow node id " + flowMatch[1]);
      }
      flowNodeIds[flowMatch[1]] = true;
    }
    var flowEdgePattern = /<g class="diagram-flow-edge"[^>]*>/g;
    while ((flowMatch = flowEdgePattern.exec(svg))) {
      var flowAttrs = attributes(flowMatch[0]);
      var flowFrom = flowAttrs["data-flow-from"];
      var flowTo = flowAttrs["data-flow-to"];
      if (!flowFrom || !flowTo) {
        throw new Error(diagramKey + ": flow edge is missing a from/to endpoint");
      }
      if (!flowNodeIds[flowFrom] || !flowNodeIds[flowTo]) {
        throw new Error(diagramKey + ": flow edge references unknown node " +
          flowFrom + " -> " + flowTo);
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
    /* Titles sit on the bottom border so lane tops stay free for ports. */
    b += panel(24, 460, 426, 480,
      "TRAINING / PREFILL · COMPUTE-BOUND · MHA-LIKE", "compute", false,
      { titlePos: "bottom" });
    b += panel(650, 460, 426, 480,
      "TOKEN-BY-TOKEN DECODE · MEMORY-BOUND · MQA-LIKE", "state", false,
      { titlePos: "bottom" });

    /* Shared spine down the center: input, projections, cache, RoPE term. */
    b += box(470, 56, 160, 56, M("h_t", "Input hidden"),
      M("[B,T,d]", "[B,T,d]"), "compute", 1, "03", { flow: "input" });
    b += box(60, 152, 250, 64, M("q_i^C,\\;q_i^R", "Per-head queries"),
      M("W_i^{UQ}c_t^Q,\\;R_tW_i^{QR}c_t^Q", "content 128 + RoPE 64"),
      "compute", 2, "04", { subSize: 8.2, flow: "queries" });
    b += box(440, 152, 220, 64, M("c_t^{KV}=W^{DKV}h_t", "Joint KV latent"),
      M("512\\text{-d shared source}", "512-d shared source"),
      "compute", 3, "05", { subSize: 8.4, flow: "latent" });
    b += box(810, 152, 230, 64, M("k_t^R=R_tW^{KR}h_t", "Decoupled RoPE key"),
      M("64\\text{-d, shared by all heads}", "64-d shared by all heads"),
      "compute", 4, "06", { subSize: 8.4, flow: "rope-key" });

    b += cacheBox(440, 256, 220, 64,
      M("c_{1:t}^{KV},\\;k_{1:t}^{R}", "Persistent cache"),
      M("512+64=576\\ /\\ \\text{token}", "512+64=576 per token"), 5, "07",
      { flow: "cache" });
    b += box(470, 360, 160, 56,
      M("(q_i^{R})^{\\mathsf T}k_s^{R}", "Shared RoPE score"),
      "one term · both lanes", "compute", 6, "09",
      { subSize: 8.2, flow: "rope-score" });

    /* Prefill lane: one expansion box covers per-head K and V, then score,
       read, write. The read consumes the expanded values from that box. */
    b += box(194, 496, 240, 64,
      M("k_{s,i}^{C},\\;v_{s,i}", "Expand per-head K and V"),
      M("W_i^{UK}c_s^{KV},\\;W_i^{UV}c_s^{KV}\\ \\cdot\\ 128\\text{-d}",
        "WUK c, WUV c · 128-d per head"),
      "compute", null, "08", { subSize: 8.2, flow: "prefill-expand" });
    b += box(94, 640, 340, 76,
      M("a_i=\\operatorname{softmax}\\!\\Big(\\tfrac{q_i^{C\\mathsf T}k_{s,i}^{C}+q_i^{R\\mathsf T}k_s^R}{\\sqrt{192}}+M\\Big)",
        "Standard multi-head score"),
      null, "compute", null, "09", { titleSize: 8.8, flow: "prefill-score" });
    b += box(94, 760, 340, 72,
      M("o_i=\\textstyle\\sum_s a_{i,s}v_{s,i}", "Per-head value read"),
      M("v_{s,i}\\ \\text{from the expansion above · compute-only}",
        "v from the expansion above, compute-only"),
      "gather", null, "10", { subSize: 8.2, flow: "prefill-read" });
    b += box(94, 866, 340, 56,
      M("u_t=\\operatorname{Concat}_i(o_i)\\,W^O", "Concat → WO"),
      null, "gather", null, "11", { titleSize: 10.2, flow: "prefill-out" });

    /* Decode lane: absorbed query, latent score, latent read, folded write.
       The row-1 box leaves a western corridor for the latent port below. */
    b += box(766, 496, 240, 64,
      M("\\widetilde q_i=(W_i^{UK})^{\\mathsf T}q_i^{C}", "Absorbed query"),
      M("512\\text{-d, folded into query}", "512-d folded into query"),
      "compute", null, "08",
      { titleSize: 9.6, subSize: 8.2, flow: "decode-absorb" });
    b += box(666, 640, 340, 76,
      M("a_i=\\operatorname{softmax}\\!\\Big(\\tfrac{\\widetilde q_i^{\\mathsf T}c_s^{KV}+q_i^{R\\mathsf T}k_s^R}{\\sqrt{192}}+M\\Big)",
        "Latent score, K = latent"),
      null, "compute", null, "09", { titleSize: 8.8, flow: "decode-score" });
    b += box(666, 760, 340, 72,
      M("m_i=\\textstyle\\sum_s a_{i,s}c_s^{KV}", "Shared latent read"),
      M("512\\text{-d, }K=V=c_s", "512-d, K = V = latent"),
      "gather", null, "10", { subSize: 8.4, flow: "decode-read" });
    b += box(666, 866, 340, 56,
      M("u_t=\\textstyle\\sum_i(W_i^{O}W_i^{UV})m_i", "Absorbed output write"),
      null, "gather", null, "11", { titleSize: 9.8, flow: "decode-out" });

    /* Central exact bridge: equal paths, not an optional approximation. */
    b += box(470, 640, 160, 76, "SAME WEIGHTS",
      "EXACT LINEAR REASSOCIATION", "control", null, "08",
      { titleSize: 9.8, subSize: 7.6, flow: "bridge" });

    /* Distribution bus from the input into the three projections. */
    b += edge(rootId, "M510 112V132H185V152", null, "compute", false,
      { from: "input", to: "queries" });
    b += edge(rootId, "M550 112V152", null, "compute", false,
      { from: "input", to: "latent" });
    b += edge(rootId, "M590 112V132H925V152", null, "compute", false,
      { from: "input", to: "rope-key" });

    /* Both cached quantities converge on the one persistent cache. */
    b += edge(rootId, "M520 216V256", null, "state", false,
      { from: "latent", to: "cache" });
    b += edge(rootId, "M925 216V232H580V256", null, "state", false,
      { from: "rope-key", to: "cache" });

    /* Cached RoPE keys feed the single shared positional score. */
    b += edge(rootId, "M550 320V360", null, "state", false,
      { from: "cache", to: "rope-score" });

    /* One labeled latent port per lane. Prefill: into the K/V expansion,
       whose products reach the read through the lane. Decode: into the
       score; the score→read arrow already carries the cached latent. */
    b += edge(rootId, "M455 320V336H210V496",
      [270, 320, M("c_s^{KV}", "shared latent"), 90, 8.6], "state", false,
      { from: "cache", to: "prefill-expand" });
    b += edge(rootId, "M645 320V336H700V640",
      [746, 328, M("c_s^{KV}", "shared latent"), 80, 8.6], "state", false,
      { from: "cache", to: "decode-score" });

    /* Queries: straight drop into prefill, one perimeter rail to decode. */
    b += edge(rootId, "M150 216V640",
      [198, 384, M("q_i^{C},q_i^{R}", "per-head q"), 84, 8.6], "compute",
      false, { from: "queries", to: "prefill-score" });
    b += edge(rootId, "M120 152V36H1060V528H1006",
      [706, 22, M("q_i^{C}\\ \\to\\ \\text{decode lane}", "qC to decode lane"),
        190, 8.6], "compute", false,
      { from: "queries", to: "decode-absorb" });
    b += edge(rootId, "M270 216V240H430V388H470",
      [352, 226, M("q_i^{R}", "qR"), 60, 8.6], "compute", false,
      { from: "queries", to: "rope-score" });

    /* The shared RoPE term enters each lane score once, symmetrically. */
    b += edge(rootId, "M510 416V440H462V656H434", null, "compute", false,
      { from: "rope-score", to: "prefill-score" });
    b += edge(rootId, "M590 416V440H638V656H666", null, "compute", false,
      { from: "rope-score", to: "decode-score" });

    /* Lane-local flow stays vertical inside each panel. */
    b += edge(rootId, ortho(314, 560, 314, 640), null, "compute", false,
      { from: "prefill-expand", to: "prefill-score" });
    b += edge(rootId, ortho(264, 716, 264, 760), null, "compute", false,
      { from: "prefill-score", to: "prefill-read" });
    b += edge(rootId, ortho(264, 832, 264, 866), null, "gather", false,
      { from: "prefill-read", to: "prefill-out" });
    b += edge(rootId, ortho(886, 560, 886, 640), null, "compute", false,
      { from: "decode-absorb", to: "decode-score" });
    b += edge(rootId, ortho(836, 716, 836, 760), null, "compute", false,
      { from: "decode-score", to: "decode-read" });
    b += edge(rootId, ortho(836, 832, 836, 866), null, "gather", false,
      { from: "decode-read", to: "decode-out" });

    /* Bridge: equality is not a one-way data flow, so the short symmetric
       connectors are drawn without arrowheads. */
    b += edge(rootId, "M470 678H434", null, "control", false,
      { from: "bridge", to: "prefill-score" }, false);
    b += edge(rootId, "M630 678H666", null, "control", false,
      { from: "bridge", to: "decode-score" }, false);
    return baseSvg(rootId, "mla", 970, b,
      "MLA dual execution graphs: a central shared spine with the input, per-head queries, joint 512-d latent, decoupled 64-d RoPE key, one persistent cache, and one shared RoPE score; a compute-bound MHA-like prefill lane expanding per-head 128-d keys and values and reading those values; a memory-bound MQA-like decode lane scoring and reading the shared latent with absorbed query and output projections; one labeled latent port entering each lane; and a green same-weights exact-reassociation bridge joining the two score rows with undirected connectors");
  }

  function mfaDiagram(rootId) {
    var b = "";

    b += box(470, 84, 160, 60, M("x_t", "Input hidden"),
      M("[B,T,d]", "[B,T,d]"), "compute", 1, "03");

    b += box(150, 190, 200, 64, M("\\bar q_t=x_tS_q", "Shared query features"),
      M("\\bar q_t\\in\\mathbb R^{C}", "never cached"), "compute", 2, "03");
    b += box(450, 190, 200, 64, M("k_t=x_tS_k", "Shared key"),
      M("k_t\\in\\mathbb R^{C}", "one per token"), "compute", 3, "03");
    b += box(750, 190, 200, 64, M("v_t=x_tS_v", "Shared value"),
      M("v_t\\in\\mathbb R^{C}", "standard variant"), "compute", 4, "03");

    b += box(150, 300, 200, 68, M("q_{t,c}=\\bar q_tQ_c", "Per-head Qc expansion"),
      M("[B,m,T,C]", "full C×C per head"), "compute", 5, "04");
    b += cacheBox(450, 300, 200, 64, M("k_{1:t},v_{1:t}", "Shared KV cache"),
      M("2C\\ \\text{(KR: }C\\text{)/token}", "no head axis"), 6, "05");
    b += box(750, 420, 220, 68, M("v_s=k_sM", "MFA-KR key reuse"),
      M("M=I+\\operatorname{diag}(\\alpha)N", "alpha zero-init"), "orange",
      null, "06", { dashed: true, subSize: 8.2 });

    b += box(150, 420, 200, 64, M("R_tq_{t,c},\\;R_sk_s", "Standard RoPE"),
      "value stays unrotated", "compute", 7, "07", { subSize: 8.2 });
    b += box(430, 420, 300, 72,
      M("a_{t,s}^{(c)}=\\operatorname{softmax}_s\\!\\left(\\frac{q_{t,c}k_s^\\top}{\\sqrt C}+M\\right)",
        "Per-head scores + causal mask"),
      null, "compute", 8, "08", { titleSize: 9.4 });
    b += box(430, 532, 300, 68,
      M("m_{t,c}=\\textstyle\\sum_s a_{t,s}^{(c)}v_s", "Shared value read"),
      M("[B,m,T,C]", "[B,m,T,C]"), "gather", 9, "09");
    b += box(430, 640, 300, 60,
      M("o_t=\\textstyle\\sum_c m_{t,c}O_c^\\top", "Head-specific Oc write-back"),
      null, "gather", 10, "09");
    b += box(430, 740, 300, 56, M("u_t=W^Oo_t", "Output projection"),
      M("[B,T,d]", "[B,T,d]"), "gather", 11, "10");

    b += edge(rootId, "M510 144V166H250V190", null, "compute");
    b += edge(rootId, ortho(550, 144, 550, 190), null, "compute");
    b += edge(rootId, "M590 144V166H850V190", null, "compute");
    b += edge(rootId, ortho(250, 254, 250, 300), null, "compute");
    b += edge(rootId, ortho(550, 254, 550, 300), null, "state");
    b += edge(rootId, "M850 254V332H650", null, "state");
    b += edge(rootId, ortho(250, 368, 250, 420), null, "compute");
    b += edge(rootId, "M450 332H400V436H350", null, "state");
    b += edge(rootId, ortho(350, 468, 430, 468), null, "compute");
    b += edge(rootId, "M620 364V400H745V550H730", null, "state");
    b += edge(rootId, "M650 348H740V452H750",
      [700, 388, "OPTIONAL · MFA-KR", 150], "orange", true);
    b += edge(rootId, "M860 488V582H730",
      [886, 612, M("v=kM\\ \\text{(KR)}", "OPTIONAL · v = k M"), 150],
      "orange", true);
    b += edge(rootId, ortho(580, 492, 580, 532), null, "compute");
    b += edge(rootId, ortho(580, 600, 580, 640), null, "gather");
    b += edge(rootId, ortho(580, 700, 580, 740), null, "gather");
    return baseSvg(rootId, "mfa", 830, b,
      "MFA with shared C-dimensional key/value features, per-head C by C query and output transforms, a head-count-independent shared KV cache, standard RoPE, and an optional key-reuse value path");
  }

  function tpaDiagram(rootId) {
    var b = "";

    b += box(470, 84, 160, 56, M("x_t", "Input hidden"),
      M("[B,T,d]", "[B,T,d]"), "compute", 1, "03");

    b += box(40, 180, 160, 72, M("A_Q(x_t)", "AQ head factor"),
      M("[B,T,R_Q,h]", "[B,T,RQ,h]"), "compute", 2, "03");
    b += box(212, 180, 160, 72, M("B_Q(x_t)", "BQ channel factor"),
      M("[B,T,R_Q,d_h]", "[B,T,RQ,dh]"), "compute", 3, "03");
    b += box(384, 180, 160, 72, M("A_K(x_t)", "AK head factor"),
      M("[B,T,R_K,h]", "[B,T,RK,h]"), "compute", 4, "03");
    b += box(556, 180, 160, 72, M("B_K(x_t)", "BK channel factor"),
      M("[B,T,R_K,d_h]", "[B,T,RK,dh]"), "compute", 5, "03");
    b += box(728, 180, 160, 72, M("A_V(x_t)", "AV head factor"),
      M("[B,T,R_V,h]", "[B,T,RV,h]"), "compute", 6, "03");
    b += box(900, 180, 160, 72, M("B_V(x_t)", "BV channel factor"),
      M("[B,T,R_V,d_h]", "[B,T,RV,dh]"), "compute", 7, "03");

    b += box(212, 300, 160, 64, M("R_t(B_Q)", "Row-wise RoPE"),
      "each rank row rotated", "compute", 8, "04", { subSize: 8.2 });
    b += box(556, 300, 160, 64, M("R_t(B_K)", "Row-wise RoPE"),
      "pre-rotate before cache", "compute", 9, "04", { subSize: 8.2 });

    b += box(140, 410, 280, 84,
      M("s_{t,s,i}=\\tfrac{1}{R_QR_K}\\sum_{p,r}A_QA_K\\,(B_Q[p]\\!\\cdot\\!B_K[r])",
        "Factor-domain score contraction"),
      M("\\text{no full historical }K", "never rebuilds full K"),
      "compute", 11, "07", { titleSize: 8.8, subSize: 8.2 });
    b += cacheBox(556, 410, 340, 80,
      M("A_K,\\widetilde B_K,A_V,B_V", "Factor KV cache"),
      M("(R_K+R_V)(h+d_h)\\ \\text{/token}", "568 for T6-XL"), 10, "05");

    b += box(140, 540, 280, 68,
      M("a=\\operatorname{softmax}(s/\\sqrt{d_h}+M)", "Causal mask + softmax"),
      null, "compute", 12, "08", { titleSize: 10 });
    b += box(620, 540, 300, 72,
      M("Q,K,V=\\tfrac1RA^\\top B", "Reference reconstruction"),
      "materializes full history · verify only", "orange", null, "06",
      { dashed: true, subSize: 8.2 });

    b += box(140, 650, 280, 72,
      M("o_{t,i}=\\tfrac1{R_V}\\sum_{s,r}a\\,A_V[r,i]\\,B_V[r]",
        "Factor-domain value aggregation"),
      null, "gather", 13, "09", { titleSize: 9.2 });
    b += box(140, 760, 280, 60,
      M("\\operatorname{Concat}\\to W^O", "Concat heads → WO"),
      M("\\text{inner width }h\\,d_h\\to d", "h·dh back to d_model"),
      "gather", 14, "10", { subSize: 8.2 });

    b += edge(rootId, "M480 140V154H120V180", null, "compute");
    b += edge(rootId, "M505 140V161H292V180", null, "compute");
    b += edge(rootId, "M530 140V168H464V180", null, "compute");
    b += edge(rootId, "M555 140V168H636V180", null, "compute");
    b += edge(rootId, "M580 140V161H808V180", null, "compute");
    b += edge(rootId, "M605 140V154H980V180", null, "compute");
    b += edge(rootId, ortho(292, 252, 292, 300), null, "compute");
    b += edge(rootId, ortho(636, 252, 636, 300), null, "compute");
    b += edge(rootId, ortho(636, 364, 636, 410), null, "state");
    b += edge(rootId, "M464 252V430H556", null, "state");
    b += edge(rootId, ortho(808, 252, 808, 410), null, "state");
    b += edge(rootId, "M980 252V450H896", null, "state");
    b += edge(rootId, "M120 252V426H140", null, "compute");
    b += edge(rootId, ortho(292, 364, 292, 410), null, "compute");
    b += edge(rootId, ortho(556, 470, 420, 470), null, "state");
    b += edge(rootId, ortho(280, 494, 280, 540), null, "compute");
    b += edge(rootId, ortho(770, 490, 770, 540),
      [820, 515, "OPTIONAL · rebuild full K/V", 190], "orange", true);
    b += edge(rootId, ortho(620, 576, 420, 576),
      [520, 562, "verification only", 140], "orange", true);
    b += edge(rootId, ortho(280, 608, 280, 650), null, "compute");
    b += edge(rootId, "M580 490V686H420", null, "state");
    b += edge(rootId, ortho(280, 722, 280, 760), null, "gather");
    return baseSvg(rootId, "tpa", 850, b,
      "TPA with six contextual A and B factor streams, row-wise RoPE on the Q and K channel factors, a factorized KV cache, factor-domain score and value contraction, and an optional reconstruction reference path");
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

  /* Most chapters share one legend; MLA's green tone marks the exact
     same-weights bridge between its two equal lanes, not a control path. */
  var badgeOverrides = {
    mla: [
      "BLUE · COMPUTE",
      "ROSE · PERSISTENT CACHE / MEMORY-BOUND LANE",
      "LAVENDER · GATHER/WRITE",
      "GREEN · SAME-WEIGHTS EXACT BRIDGE"
    ]
  };

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
    if (k === "mfa") svg = mfaDiagram(rootId);
    if (k === "tpa") svg = tpaDiagram(rootId);
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
      badges: badgeOverrides[k] || [
        "BLUE · COMPUTE",
        "GREEN · CONTROL",
        "ROSE · CACHE/STATE",
        "LAVENDER · GATHER/WRITE",
        "DASHED · TRAINING/OPTIONAL"
      ]
    };
  }

  /* ==================================================================== *
   * Config explorer builder · window.AttentionDiagrams.buildConfig
   *
   * A compact, data-driven parameter/model-structure map used by the
   * interactive attentionConfig explorer. It is deliberately separate
   * from the hand-drawn chapter diagrams above: modules carry
   * data-config-module (never data-code-block), so the two interaction
   * layers cannot collide.
   *
   * Schema:
   *   buildConfig({
   *     key: "mla",                       // stable diagram key
   *     title: "…",                       // accessible <title> text
   *     description: "…",                 // accessible <desc> text
   *     lanes: [{ id, label, tone }],     // optional horizontal lanes
   *     nodes: [{                         // array, or an { id: node } map
   *       id: "wq",                       // required, stable module id
   *       label: "W^Q" | M(tex, fb),      // display label (text or math)
   *       value: "[d, H·dh]",             // short summary/value line
   *                                       //   (alias: summary)
   *       tone: "compute",                // palette tone key
   *       lane: "proj", col: 0,           // optional lane / column hints
   *       x, y, w, h,                     // optional explicit geometry
   *       dashed: false,                  // optional-path styling
   *       selected: false,                // initial aria-pressed state
   *       detail: "wq-detail"             // detail-panel linkage id
   *     }],
   *     edges: [{ from, to, label, tone, dashed, back }]
   *   })
   *
   * When x/y are omitted, modules are placed on left-to-right columns
   * derived from edge topology (edges marked back:true are excluded from
   * ranking and routed on a return rail). Lanes stack as horizontal
   * bands sharing the same columns. Returns
   * { svg, key, modules, width, height }; every module <g> carries
   * data-config-module, role="button", tabindex="0" and aria-pressed so
   * an external controller can wire selection without inline scripts.
   * ==================================================================== */

  var CONFIG_LAYOUT = {
    nodeW: 156,
    nodeH: 58,
    colGap: 54,
    rowGap: 24,
    margin: 30,
    lanePadX: 20,
    lanePadY: 22,
    laneGap: 38
  };

  /* Only these tones have arrow markers in defs(); others fall back. */
  var CONFIG_MARKER_TONES = {
    compute: 1, control: 1, state: 1, gather: 1, cyan: 1, orange: 1, muted: 1
  };

  function configMarkerTone(tone) {
    return CONFIG_MARKER_TONES[tone] ? tone : "muted";
  }

  function normalizeConfigNode(id, spec, order) {
    return {
      id: id,
      label: spec.label != null ? spec.label : id,
      value: spec.value != null ? spec.value
        : (spec.summary != null ? spec.summary : null),
      tone: spec.tone || "compute",
      lane: spec.lane != null ? String(spec.lane) : null,
      col: typeof spec.col === "number" ? spec.col : null,
      x: typeof spec.x === "number" ? spec.x : null,
      y: typeof spec.y === "number" ? spec.y : null,
      w: typeof spec.w === "number" ? spec.w : CONFIG_LAYOUT.nodeW,
      h: typeof spec.h === "number" ? spec.h : CONFIG_LAYOUT.nodeH,
      dashed: !!spec.dashed,
      selected: !!spec.selected,
      detail: spec.detail != null ? String(spec.detail) : null,
      order: order
    };
  }

  function resolveLaneId(node, lanes, laneIndex) {
    if (node.lane && laneIndex[node.lane]) return node.lane;
    return lanes[0].id;
  }

  /* Assign x/y to every module that lacks explicit geometry. Columns come
     from longest-path ranks over forward edges; an explicit col wins. */
  function layoutConfigNodes(nodes, edges, lanes) {
    var byId = {};
    nodes.forEach(function (node) { byId[node.id] = node; });

    var rank = {};
    nodes.forEach(function (node) {
      rank[node.id] = node.col != null ? node.col : 0;
    });
    for (var pass = 0; pass < nodes.length; pass += 1) {
      var changed = false;
      edges.forEach(function (e) {
        if (e.back || e.from === e.to || byId[e.to].col != null) return;
        if (rank[e.to] < rank[e.from] + 1) {
          rank[e.to] = rank[e.from] + 1;
          changed = true;
        }
      });
      if (!changed) break;
    }

    var auto = nodes.filter(function (node) {
      return node.x == null || node.y == null;
    });
    if (!auto.length) return;

    /* Compress the rank values in use into consecutive columns. */
    var used = {};
    auto.forEach(function (node) { used[rank[node.id]] = true; });
    var ordered = Object.keys(used).map(Number).sort(function (a, b) {
      return a - b;
    });
    var colOf = {};
    ordered.forEach(function (value, index) { colOf[value] = index; });

    var columns = ordered.map(function () { return []; });
    auto.forEach(function (node) {
      columns[colOf[rank[node.id]]].push(node);
    });
    columns.forEach(function (column) {
      column.sort(function (a, b) { return a.order - b.order; });
    });

    var L = CONFIG_LAYOUT;
    var colW = columns.map(function (column) {
      return column.reduce(function (w, node) {
        return Math.max(w, node.w);
      }, L.nodeW);
    });
    var colX = [];
    var cursor = L.margin + (lanes.length ? L.lanePadX : 0);
    colW.forEach(function (w, index) {
      colX[index] = cursor;
      cursor += w + L.colGap;
    });

    function stack(cell, cellH, top, c) {
      var y = top;
      cell.forEach(function (node) {
        node.x = colX[c] + (colW[c] - node.w) / 2;
        node.y = y;
        y += node.h + L.rowGap;
      });
    }

    function cellHeight(cell) {
      return cell.reduce(function (h, node) { return h + node.h; }, 0) +
        Math.max(0, cell.length - 1) * L.rowGap;
    }

    if (!lanes.length) {
      var colH = columns.map(cellHeight);
      var maxH = colH.reduce(function (a, b) { return Math.max(a, b); }, 0);
      columns.forEach(function (column, c) {
        stack(column, colH[c], L.margin + (maxH - colH[c]) / 2, c);
      });
      return;
    }

    /* Lane layout: one horizontal band per lane, columns shared across
       lanes so cross-lane edges stay aligned. */
    var laneIndex = {};
    lanes.forEach(function (lane) { laneIndex[lane.id] = lane; });
    var laneY = L.margin + 14;
    lanes.forEach(function (lane) {
      var cells = columns.map(function (column) {
        return column.filter(function (node) {
          return resolveLaneId(node, lanes, laneIndex) === lane.id;
        });
      });
      var cellH = cells.map(cellHeight);
      var contentH = Math.max(L.nodeH, cellH.reduce(function (a, b) {
        return Math.max(a, b);
      }, 0));
      cells.forEach(function (cell, c) {
        stack(cell, cellH[c], laneY + L.lanePadY + (contentH - cellH[c]) / 2, c);
      });
      laneY += contentH + 2 * L.lanePadY + L.laneGap;
    });
  }

  /* Default orthogonal route between two resolved module geometries.
     backIndex offsets stacked return rails so they do not overlap. */
  function routeConfigEdge(a, b, backIndex) {
    if (a === b) {
      var loopX = a.x + a.w;
      var loopY = a.y + a.h / 2;
      return {
        d: "M" + loopX + " " + (loopY - 9) + "H" + (loopX + 20) +
          "V" + (loopY + 9) + "H" + loopX,
        lx: loopX + 34, ly: loopY - 20, lw: 96,
        maxX: loopX + 20, maxY: loopY + 9
      };
    }
    var sx;
    var sy;
    var tx;
    var ty;
    if (b.x >= a.x + a.w + 8) {
      sx = a.x + a.w;
      sy = a.y + a.h / 2;
      tx = b.x;
      ty = b.y + b.h / 2;
      return {
        d: ortho(sx, sy, tx, ty, "x"),
        lx: (sx + tx) / 2,
        ly: (sy === ty ? sy : (sy + ty) / 2) - 11,
        lw: Math.max(72, tx - sx - 8),
        maxX: tx, maxY: Math.max(sy, ty)
      };
    }
    if (a.x >= b.x + b.w + 8) {
      sx = a.x + a.w / 2;
      tx = b.x + b.w / 2;
      var rail = Math.max(a.y + a.h, b.y + b.h) + 18 + backIndex * 14;
      return {
        d: "M" + sx + " " + (a.y + a.h) + "V" + rail + "H" + tx +
          "V" + (b.y + b.h),
        lx: (sx + tx) / 2, ly: rail - 10,
        lw: Math.max(88, Math.abs(sx - tx) - 16),
        maxX: Math.max(sx, tx), maxY: rail
      };
    }
    /* Same column: connect the facing horizontal borders. */
    sx = a.x + a.w / 2;
    tx = b.x + b.w / 2;
    if (b.y >= a.y + a.h) {
      sy = a.y + a.h;
      ty = b.y;
    } else {
      sy = a.y;
      ty = b.y + b.h;
    }
    return {
      d: ortho(sx, sy, tx, ty, "y"),
      lx: Math.max(sx, tx) + 58, ly: (sy + ty) / 2, lw: 104,
      maxX: Math.max(sx, tx), maxY: Math.max(sy, ty)
    };
  }

  function configModule(node) {
    var fill = toneFill(node.tone);
    var stroke = toneStroke(node.tone);
    var cx = node.x + node.w / 2;
    var titleY = node.y + node.h / 2 - (node.value != null ? 7 : 0);
    var aria = "配置模块 " + fallbackLabel(node.label) +
      (node.value != null ? "：" + fallbackLabel(node.value) : "");
    return (
      '<g class="config-module" data-config-module="' + escapeText(node.id) + '"' +
      (node.detail ? ' data-config-detail="' + escapeText(node.detail) + '"' : "") +
      ' role="button" tabindex="0" aria-pressed="' +
      (node.selected ? "true" : "false") +
      '" aria-label="' + escapeText(aria) + '">' +
      '<rect class="config-module-box" x="' + node.x + '" y="' + node.y +
      '" width="' + node.w + '" height="' + node.h +
      '" rx="9" fill="' + fill + '" stroke="' + stroke +
      '" stroke-width="1.35" ' +
      (node.dashed ? 'stroke-dasharray="6 5" ' : "") + '/>' +
      labelMarkup(cx, titleY, node.w - 18, 30, node.label, 10.4, P.ink, 600) +
      (node.value != null
        ? labelMarkup(cx, node.y + node.h / 2 + 14, node.w - 16, 22,
            node.value, 8.4, P.muted, 500)
        : "") +
      '</g>'
    );
  }

  function configEdgeMarkup(rootId, route, e) {
    var color = e.tone === "muted" ? P.muted : toneStroke(e.tone);
    return (
      '<g class="config-edge" data-config-edge-from="' + escapeText(e.from) +
      '" data-config-edge-to="' + escapeText(e.to) + '">' +
      '<path d="' + route.d + '" fill="none" stroke="' + color +
      '" stroke-width="1.4" ' +
      (e.dashed ? 'stroke-dasharray="6 5" ' : "") +
      'stroke-linecap="square" stroke-linejoin="round" marker-end="url(#' +
      rootId + '-arrow-' + e.tone + ')"/>' +
      (e.label != null
        ? labelMarkup(route.lx, route.ly, route.lw, 22, e.label, 8.2, color, 600)
        : "") +
      '</g>'
    );
  }

  function buildConfig(config) {
    if (!config || typeof config !== "object") {
      throw new Error("buildConfig: a config object is required");
    }
    var key = String(config.key || config.id || config.type || "config");
    buildSerial += 1;
    var rootId = "attention-config-" +
      key.replace(/[^A-Za-z0-9_-]/g, "-") + "-" + buildSerial;

    var rawNodes = config.nodes || config.modules;
    var nodes = [];
    if (Array.isArray(rawNodes)) {
      rawNodes.forEach(function (spec, index) {
        if (!spec || spec.id == null) {
          throw new Error(key + ": config node #" + index + " is missing an id");
        }
        nodes.push(normalizeConfigNode(String(spec.id), spec, index));
      });
    } else if (rawNodes && typeof rawNodes === "object") {
      Object.keys(rawNodes).forEach(function (id, index) {
        nodes.push(normalizeConfigNode(id, rawNodes[id] || {}, index));
      });
    }
    if (!nodes.length) {
      throw new Error(key + ": config.nodes must declare at least one module");
    }
    var byId = {};
    nodes.forEach(function (node) {
      if (byId[node.id]) {
        throw new Error(key + ": duplicate config module id " + node.id);
      }
      byId[node.id] = node;
    });

    var edges = (config.edges || []).map(function (spec, index) {
      if (!spec || spec.from == null || spec.to == null) {
        throw new Error(key + ": config edge #" + index + " needs from and to");
      }
      var from = String(spec.from);
      var to = String(spec.to);
      if (!byId[from] || !byId[to]) {
        throw new Error(key + ": config edge references unknown module " +
          from + " -> " + to);
      }
      return {
        from: from,
        to: to,
        label: spec.label != null ? spec.label : null,
        tone: configMarkerTone(spec.tone || "muted"),
        dashed: !!spec.dashed,
        back: !!spec.back
      };
    });

    var lanes = [];
    if (Array.isArray(config.lanes)) {
      config.lanes.forEach(function (lane, index) {
        if (!lane || lane.id == null) {
          throw new Error(key + ": config lane #" + index + " is missing an id");
        }
        lanes.push({
          id: String(lane.id),
          label: lane.label != null ? String(lane.label) : String(lane.id),
          tone: lane.tone || "paper"
        });
      });
    } else {
      var seenLanes = {};
      nodes.forEach(function (node) {
        if (node.lane && !seenLanes[node.lane]) {
          seenLanes[node.lane] = true;
          lanes.push({ id: node.lane, label: node.lane, tone: "paper" });
        }
      });
    }

    layoutConfigNodes(nodes, edges, lanes);

    var L = CONFIG_LAYOUT;
    var maxRight = 0;
    var maxBottom = 0;
    nodes.forEach(function (node) {
      maxRight = Math.max(maxRight, node.x + node.w);
      maxBottom = Math.max(maxBottom, node.y + node.h);
    });

    /* Lane panels sit behind the modules; band extents come from the
       members' bounding boxes, so explicit-geometry modules are covered. */
    var laneBody = "";
    if (lanes.length) {
      var laneIndex = {};
      lanes.forEach(function (lane) { laneIndex[lane.id] = lane; });
      lanes.forEach(function (lane) {
        var top = Infinity;
        var bottom = -Infinity;
        nodes.forEach(function (node) {
          if (resolveLaneId(node, lanes, laneIndex) !== lane.id) return;
          top = Math.min(top, node.y);
          bottom = Math.max(bottom, node.y + node.h);
        });
        if (top === Infinity) return;
        var panelX = L.margin;
        var panelY = top - L.lanePadY;
        var panelW = maxRight + L.lanePadX - panelX;
        var panelH = bottom - top + 2 * L.lanePadY;
        maxRight = Math.max(maxRight, panelX + panelW);
        maxBottom = Math.max(maxBottom, panelY + panelH);
        laneBody += panel(panelX, panelY, panelW, panelH, lane.label, lane.tone);
      });
    }

    var nodeBody = nodes.map(configModule).join("");

    var edgeBody = "";
    var backCount = 0;
    edges.forEach(function (e) {
      var a = byId[e.from];
      var b = byId[e.to];
      var isBack = a !== b && a.x >= b.x + b.w + 8;
      var route = routeConfigEdge(a, b, isBack ? backCount : 0);
      if (isBack) backCount += 1;
      maxRight = Math.max(maxRight, route.maxX);
      maxBottom = Math.max(maxBottom, route.maxY);
      if (e.label != null) {
        maxRight = Math.max(maxRight, route.lx + route.lw / 2);
        maxBottom = Math.max(maxBottom, route.ly + 12);
      }
      edgeBody += configEdgeMarkup(rootId, route, e);
    });

    var width = Math.ceil(maxRight + L.margin);
    var height = Math.ceil(maxBottom + L.margin);
    var titleId = rootId + "-title";
    var descId = rootId + "-desc";
    var title = config.title != null
      ? String(config.title)
      : key + " 参数与模块结构";
    var description = config.description != null
      ? String(config.description)
      : "包含 " + nodes.length + " 个可选模块的参数结构图；聚焦任一模块可查看对应配置详情。";

    var svg =
      '<svg viewBox="0 0 ' + width + " " + height +
      '" class="attention-config-svg" role="group" aria-labelledby="' +
      titleId + " " + descId + '" data-config-key="' + escapeText(key) +
      '" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono">' +
      '<title id="' + titleId + '">' + escapeText(title) + '</title>' +
      '<desc id="' + descId + '">' + escapeText(description) + '</desc>' +
      '<g style="' + STYLE_VARS + '">' + defs(rootId) +
      '<rect width="' + width + '" height="' + height + '" fill="' +
      P.canvas + '"/>' +
      laneBody + edgeBody + nodeBody +
      '</g></svg>';

    if (/<script\b/i.test(svg)) {
      throw new Error(key + ": config svg must not contain inline scripts");
    }

    return {
      svg: svg,
      key: key,
      modules: nodes.map(function (node) { return node.id; }),
      width: width,
      height: height
    };
  }

  window.AttentionDiagrams = { build: build, buildConfig: buildConfig };
})();
