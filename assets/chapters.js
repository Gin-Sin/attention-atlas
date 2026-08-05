(function () {
  "use strict";

  var R = String.raw;

  window.ATTENTION_CHAPTERS = [
    {
      id: "mha",
      order: 0,
      title: "MHA",
      fullTitle: "Multi-Head Attention",
      zhTitle: "多头注意力：所有分支的共同祖先",
      year: "2017",
      category: "dense",
      difficulty: "基础",
      report: "Attention Is All You Need",
      deck: "先建立一把统一的“成本尺”。2017 Transformer 在编码器自注意力、带因果掩码的解码器自注意力和编码器—解码器交叉注意力中都使用 MHA；自回归解码时必须为相关层保存历史 K/V。",
      takeaway: "MHA 的关键不是“有很多头”，而是每个头都拥有一套独立的读地址 K 与读内容 V；这同时给出高容量与最大的 KV cache。",
      motivation: [
        "循环网络必须按时间步串行传播状态。编码器自注意力可读取全部源位置；解码器自注意力只能读取当前及过去位置；交叉注意力则让解码器 query 读取编码器输出。",
        "不同头使用不同投影子空间：一个头可学习局部搭配，另一个头可学习指代或远程依赖；最终经拼接和输出投影写回残差流。",
        "它后来成为所有 KV 共享、低秩压缩、稀疏选择和线性状态方法的参照系。理解后续架构，本质上是在回答：MHA 的哪部分可以共享、压缩或省略？"
      ],
      constraints: [
        { label: "训练算力", title: "注意力图是二次的", body: R`长度为 \(L\) 时要形成 \(H\) 个 \(L\times L\) 分数图，核心算量约为 \(O(L^2d)\)。` },
        { label: "推理存储", title: "KV 随上下文线性增长", body: "每层每 token 缓存 H 组 K 和 V；长上下文与大 batch 会迅速吃满显存。" },
        { label: "解码带宽", title: "常常不是算不动，而是搬不快", body: "每生成一个 token，都要从 HBM 读取历史 KV；小 batch 解码常受内存带宽限制。" }
      ],
      intuitions: [
        { label: "Query", title: "我正在找什么？", body: "每个头产生一个不同的问题。" },
        { label: "Key", title: "我能被怎样索引？", body: "历史 token 为每个头提供独立地址。" },
        { label: "Value", title: "找到后读出什么？", body: "匹配权重对内容向量做加权汇总。" }
      ],
      diagram: { type: "heads", mode: "mha", caption: "MHA：每个 Q 头都配有独立 K/V 头；同一算子可用于编码器自注意力、解码器因果自注意力和交叉注意力。" },
      derivations: [
        {
          title: "从投影到加权读取",
          body: R`令 \(X_Q\) 提供 query、\(X_{KV}\) 提供 key/value；自注意力时二者相同，交叉注意力时分别来自解码器与编码器。对第 \(h\) 个头，
            \[
            Q_h=X_QW_h^Q,\quad K_h=X_{KV}W_h^K,\quad V_h=X_{KV}W_h^V,
            \]
            \[
            O_h=\operatorname{softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right)V_h,\qquad
            Y=\operatorname{Concat}(O_1,\ldots,O_H)W^O.
            \]
            缩放因子 \(\sqrt{d_h}\) 抑制点积方差。\(M\) 在解码器自注意力中含因果掩码；编码器自注意力和交叉注意力没有“未来目标 token”这一掩码，但仍可含 padding mask。`
        },
        {
          title: "统一的 KV cache 计量公式",
          body: R`设 batch 为 \(B\)、已缓存长度为 \(L\)、层数为 \(N\)、KV 头数为 \(H_{kv}\)、头维为 \(d_h\)、每元素 \(b\) 字节，则
            \[
            \mathrm{KVBytes}=2BNLH_{kv}d_hb.
            \]
            系数 2 来自 K 与 V。对 MHA，\(H_{kv}=H_q=H\)。该公式只算注意力 KV，不含权重、激活、碎片和运行时工作区。`
        }
      ],
      warning: "“MHA”只描述多头注意力算子，不等于“因果自注意力”。原始 Transformer 的编码器、解码器和交叉注意力掩码/输入来源不同；“头数减半”也只有在 KV 头数随之减少时才会减小 KV cache。",
      exercises: [
        {
          q: R`某模型有 \(N=32\) 层、\(H=32\)、\(d_h=128\)，以 BF16 缓存一条 \(L=4096\) 的序列。忽略 batch 维，MHA KV cache 多大？`,
          hint: R`代入 \(2NLHd_hb\)，BF16 的 \(b=2\)。`,
          answer: R`\(2\times32\times4096\times32\times128\times2=2{,}147{,}483{,}648\) 字节，约 \(2.0\) GiB。`
        },
        {
          q: R`若 \(q_i,k_j\) 的各维独立、零均值、方差为 1，为什么点积要除以 \(\sqrt{d_h}\)？`,
          hint: "先算未缩放点积的方差。",
          answer: R`\(\operatorname{Var}(q_i^\top k_j)=d_h\)。除以 \(\sqrt{d_h}\) 后方差回到约 1，避免 softmax 过早饱和。`
        }
      ],
      sources: [
        { label: "Vaswani et al. (2017), Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762" },
        { label: "Dao et al. (2022), FlashAttention：精确注意力的 IO-aware 实现", url: "https://arxiv.org/abs/2205.14135" }
      ]
    },
    {
      id: "mqa",
      order: 1,
      title: "MQA",
      fullTitle: "Multi-Query Attention",
      zhTitle: "多查询注意力：只保留一套 K/V",
      year: "2019",
      category: "dense",
      difficulty: "基础",
      report: "Fast Transformer Decoding",
      deck: "MQA 没有改变查询头数量，也没有近似 softmax；它让全部 Q 头共享一套 K/V。原论文在 WMT 编码器—解码器的三类注意力层都替换为 MQA，并另测 decoder-only 语言模型。",
      takeaway: R`把“提出多少个问题”和“保存多少份历史索引”解耦：保留多 Q 头，令 \(H_{kv}=1\)。`,
      motivation: [
        "增量解码一次只处理一个新 token，矩阵乘法很窄，GPU 计算单元难以吃满；反而从 HBM 反复加载历史 K/V 成为主要瓶颈。",
        "Shazeer 的核心观察是：查询头需要多样性，但历史 token 不一定要为每个查询头保存独立地址与内容。",
        "因此 MQA 保留 H 个 Q 投影与 H 个输出通道，只把 K/V 投影压成一个共享头。论文把它用于编码器自注意力、解码器自注意力与交叉注意力；主要速度动机和收益仍来自增量解码。"
      ],
      constraints: [
        { label: "目标", title: "优化解码带宽", body: "训练阶段仍需计算全部 Q 头分数；主要收益发生在增量解码。原论文的结构范围并不只限 decoder self-attention。" },
        { label: "存储", title: R`KV 缩小约 \(H\) 倍`, body: R`相同头维下，\(H_{kv}\) 从 \(H\) 变为 1；实际收益还取决于布局、量化和 kernel。` },
        { label: "容量", title: "共享可能损失质量", body: "所有 Q 头只能在同一 K/V 表示上读取，独立的键值子空间容量下降。" }
      ],
      intuitions: [
        { label: "类比", title: "多位读者，一本目录", body: "每位读者提出不同问题，但共用同一本索引和资料。" },
        { label: "保留", title: "Q 头仍然独立", body: "不是把 H 个注意力头变成 1 个头。" },
        { label: "牺牲", title: "K/V 多样性下降", body: "同一 token 只有一种可寻址方式和内容编码。" }
      ],
      diagram: { type: "heads", mode: "mqa", caption: "MQA：多个 Q 头共同连接到唯一的 K/V 头。" },
      derivations: [
        {
          title: "共享 K/V 的定义",
          body: R`令 \(Q_h=X_QW_h^Q\)，但对同一层只计算一套
            \[
            K=X_{KV}W^K,\qquad V=X_{KV}W^V.
            \]
            第 \(h\) 个输出仍为
            \[
            O_h=\operatorname{softmax}\!\left(\frac{Q_hK^\top}{\sqrt{d_h}}+M\right)V.
            \]
            因此 softmax 仍精确，查询仍多头；变化只发生在 K/V 的头轴。自注意力有 \(X_Q=X_{KV}\)，交叉注意力则分别来自解码器和编码器。`
        },
        {
          title: "带宽收益的上界",
          body: R`MHA 与 MQA 的理想 KV 比值为
            \[
            \frac{\mathrm{KV}_{MHA}}{\mathrm{KV}_{MQA}}
            =\frac{H_q}{1}=H_q.
            \]
            这是 KV 数据量的上界比，不等同于端到端加速比。权重读取、通信、调度和非注意力层不会按 H 倍缩小。`
        }
      ],
      warning: "MQA 的“显著更快且只有轻微质量损失”来自论文中的 TPU、batch、WMT/Billion Word 与特定等参数配方；它不是所有模型和硬件上的定理。论文还明确把三类 encoder-decoder attention 都替换成 MQA。",
      exercises: [
        {
          q: "沿用上一章 32 层、32 头、头维 128、4096 长度、BF16 的例子，MQA KV cache 多大？",
          hint: R`把 \(H_{kv}\) 从 32 改为 1。`,
          answer: "约 64 MiB；相对 MHA 理论缩小 32 倍。"
        },
        {
          q: "MQA 是否把注意力计算从二次复杂度变成线性复杂度？",
          hint: R`考察长度 \(L\) 对 \(QK^{\mathsf T}\) 的影响。`,
          answer: R`否。prefill 仍需要对 \(L\times L\) 的位置对计算分数，关于序列长度仍为二次；MQA 主要减少 KV 存储与解码时的数据搬运。`
        }
      ],
      sources: [
        { label: "Shazeer (2019), Fast Transformer Decoding: One Write-Head Is All You Need", url: "https://arxiv.org/abs/1911.02150" }
      ]
    },
    {
      id: "gqa",
      order: 2,
      title: "GQA",
      fullTitle: "Grouped-Query Attention",
      zhTitle: "分组查询注意力：共享与容量的旋钮",
      year: "2023",
      category: "dense",
      difficulty: "进阶",
      report: "GQA: Training Generalized Multi-Query Transformer Models",
      deck: R`GQA 把 MHA 与 MQA 放在同一条离散分组轴上：每组 Q 头共享一个 K/V 头，常见实现从 \(H_q\) 的整除因子中选择 \(H_{kv}\)。`,
      takeaway: R`GQA 不是连续可微的“组数旋钮”，而是离散头布局：\(H_{kv}=1\) 为 MQA，\(H_{kv}=H_q\) 为 MHA，中间整除值给出等大小分组。`,
      motivation: [
        "MQA 的缓存最小，但单一 K/V 头可能成为表达瓶颈；MHA 表达充足，却为每个 Q 头重复保存历史。",
        "GQA 把查询头分为 G 组，每组共享一套 K/V。模型设计者可以根据目标硬件和质量预算选择中间点。",
        "原论文还给出从 MHA checkpoint uptrain 到 GQA 的办法：组内 K/V 投影做均值池化，再用约原预训练算力 5% 的继续训练恢复能力。实验只改 T5.1.1 的 decoder self-attention 与 cross-attention，不改 encoder self-attention；5% 也只是该配方。"
      ],
      constraints: [
        { label: "结构约束", title: "通常要求整除", body: R`常见实现要求 \(H_q\) 能被 \(H_{kv}\) 整除，每个 KV 头服务 \(r=H_q/H_{kv}\) 个 Q 头。` },
        { label: "硬件约束", title: "布局与 kernel 要匹配", body: "理论缓存下降不保证 kernel 自动高效；广播和并行切分会影响实际吞吐。" },
        { label: "迁移成本", title: "可从 MHA uptrain", body: "均值池化只是初始化，仍需继续训练适应共享后的表示。" }
      ],
      intuitions: [
        { label: "类比", title: "小组共用资料员", body: "每位成员有问题；每组有一位资料员维护索引。" },
        { label: "旋钮", title: R`\(H_{kv}\) 控制折中`, body: "越大越接近 MHA，越小越接近 MQA。" },
        { label: "分组", title: "共享发生在组内", body: "组间仍可学习不同 K/V 子空间。" }
      ],
      diagram: { type: "heads", mode: "gqa", caption: "GQA：离散示例中 8 个 Q 头按 2 个一组，共享 4 个 K/V 头。" },
      derivations: [
        {
          title: "头到组的映射",
          body: R`设 \(H_q\) 个查询头、\(H_{kv}=G\) 个 K/V 头，组大小
            \[
            r=H_q/H_{kv},\qquad g(h)=\left\lfloor h/r\right\rfloor.
            \]
            则
            \[
            O_h=\operatorname{softmax}\!\left(
            \frac{Q_hK_{g(h)}^\top}{\sqrt{d_h}}+M\right)V_{g(h)}.
            \]
            \(G=1\) 得 MQA；\(G=H_q\) 得 MHA。`
        },
        {
          title: "缓存与组数线性相关",
          body: R`\[
            \mathrm{KVBytes}_{GQA}=2BNLGd_hb,\qquad
            \frac{\mathrm{KV}_{MHA}}{\mathrm{KV}_{GQA}}=\frac{H_q}{G}=r.
            \]
            所以 32 个 Q 头、8 个 KV 头的 GQA 理论上把 KV cache 压到 MHA 的 \(1/4\)。`
        }
      ],
      warning: "“GQA 速度与 MQA 相当、质量接近 MHA”来自 T5.1.1 encoder-decoder 的特定 uptraining；论文只改 decoder self/cross-attention。服务框架、序列长度和并行策略改变后，最优离散组数不一定相同。",
      exercises: [
        {
          q: R`\(H_q=48\)、\(H_{kv}=8\) 时，每个 KV 头服务多少个 Q 头？相对 MHA 的 KV 缩减倍数是多少？`,
          hint: R`都等于 \(H_q/H_{kv}\)。`,
          answer: "每组 6 个 Q 头；KV cache 理论缩小 6 倍。"
        },
        {
          q: R`把 MHA checkpoint 转成 GQA-2。某一组原 K 投影为 \(W_1^K,W_2^K,W_3^K,W_4^K\)，论文初始化怎样做？`,
          hint: "组内 mean pooling。",
          answer: R`令共享投影 \(W_g^K=(W_1^K+W_2^K+W_3^K+W_4^K)/4\)，V 同理，再继续预训练。`
        }
      ],
      sources: [
        { label: "Ainslie et al. (2023), GQA", url: "https://aclanthology.org/2023.emnlp-main.298/" },
        { label: "Llama 2 technical report：开源模型中的 GQA 实例", url: "https://arxiv.org/abs/2307.09288" }
      ]
    },
    {
      id: "mla",
      order: 3,
      title: "MLA",
      fullTitle: "Multi-Head Latent Attention",
      zhTitle: "一套权重，两种执行图：Prefill 像 MHA，Decode 像 MQA",
      year: "2024",
      category: "dense",
      difficulty: "高阶",
      report: "DeepSeek-V2 Technical Report",
      deck: R`理解 MLA，先不要从“低秩压缩”开始，而要先问同一层在两个阶段分别怕什么：Prefill 怕大规模矩阵计算，Decode 怕反复搬运历史 KV。MLA 用同一份 512 维 latent 和两侧线性投影，让内容通道在 Prefill 展开成每头 128 维的 MHA 形态，在 Decode 吸收成共享 512 维的 MQA 形态。`,
      takeaway: R`MLA 最关键的不是“缓存一个 latent”，而是改变计算括号：Prefill 先把 \(c^{KV}\) 展开为多头 K/V，保留 MHA-128 的并行计算形态；Decode 把 K 上投影吸收到 query、把 V 上投影吸收到输出，直接把同一份 \(c^{KV}\) 当作共享 K=V。小型 decoupled-RoPE 支路负责位置。换句话说，MLA 把 Prefill 的计算旋钮 \(d\) 与 Decode 的缓存旋钮 \(C\) 解耦。`,
      motivation: [
        "同一个 Attention 层面对两种完全不同的工作负载。Prefill 一次处理整段 prompt，主要成本是形成并计算大批量 score；逐 token Decode 每步只有一个 query，却要反复读取全部历史 KV，通常受显存带宽与缓存容量约束。",
        "在固定 num_heads 与计算预算下，Prefill 希望每头内容维较小且 K/V 彼此独立：MHA-128 限制最少、矩阵乘也更轻。把内容 head dim 直接放大到 512，会把主要 score 计算近似放大 4 倍。",
        R`在固定 KV cache 宽度下，Decode 则希望把预算集中到一份共享表示：把所有历史信息装进 512 维 \(c^{KV}\)，让全部 query heads 读取同一份 K=V，正是缓存友好的 MQA-512 视角。`
      ],
      constraints: [
        { label: "恒等变换", title: "切换执行图不能改变函数", body: "MLA 依赖 NoPE 内容通道中的线性结合律：投影可以移到 query 侧和输出侧，但 softmax 前后的运算顺序必须严格保持。" },
        { label: "位置支路", title: "完整 RoPE 会阻断吸收", body: "位置相关旋转随历史位置变化，不能并入一个固定上投影；MLA 只给小型共享 key 支路保留 RoPE，让主要内容通道保持可吸收的 NoPE。" },
        { label: "结论边界", title: "“几乎最优”是条件化判断", body: "它依赖 full softmax、线性 K/V 上投影、Partial RoPE 足够有效，以及 Prefill/Decode 分别主要受计算与 KV 带宽约束；真实 TP、kernel、量化和参数预算会改变最优点。" }
      ],
      phaseComparison: {
        eyebrow: "One Layer · Two Workloads",
        title: "先选阶段，再选择同一组权重的执行图",
        intro: "MLA 不要求 Prefill 和 Decode 使用同一个 kernel 形态。两边计算完全等价，但各自把昂贵的维度放在更能承受的位置。",
        phases: [
          {
            label: "Training / Prefill",
            bottleneck: { label: "主要瓶颈", value: "大批量 Attention 计算" },
            preferred: R`MHA-like · \(H\) 个 128-d 内容头`,
            execution: R`先由 512-d latent 展开 \(K_i^{C}\)、\(V_i\)，再做标准多头 score/read。`,
            note: "prompt 内部可以一次并行材料化；写入长期 cache 的仍是 latent + RoPE key。"
          },
          {
            label: "Token-by-token Decode",
            bottleneck: { label: "主要瓶颈", value: "历史 KV 容量与 HBM 搬运" },
            preferred: R`MQA-like · 共享 512-d \(K=V\)`,
            execution: R`先吸收 \(W_i^{UK}/W_i^{UV}\)，再直接对历史 latent 打分与聚合。`,
            note: R`每步只产生一个 query，却会重复读全历史；不重建多头 KV 才有意义。MQA-512 是结构性/NoPE 执行视角，不是标准 MQA 层：分数仍按原口径除以 \(\sqrt{128+64}\)，64 维 RoPE 支路单独计算。`
          }
        ],
        bridge: {
          label: "Exact bridge",
          title: R`MLA 把 Prefill 的 \(d\) 与 Decode 的 \(C+r\) 解耦`,
          body: R`同一份 latent 和上投影在左侧表现为 \(H\) 个 \(d\) 维内容头，在右侧表现为一份 \(C\) 维共享 K=V；线性恒等式保证只换括号、不换函数。`
        }
      },
      sectionTitles: {
        motivation: "先把问题拆成 Prefill 与 Decode",
        constraints: "两种阶段为什么想要相反的 Attention",
        intuitions: "MLA 的关键：同一组权重，两种执行形态",
        diagram: "一张图看懂 MHA-128 ↔ MQA-512",
        position: "RoPE 为什么必须拆成小支路",
        derivations: "同等表达力下，MLA 为什么更省 KV Cache",
        exercises: "练习：KV Cache、Decode 算量与等价变换",
        sources: "权威来源与延伸阅读"
      },
      intuitions: [
        { label: "Prefill", title: "一次展开，多头并行", body: "像打开源文件后一次导出所有 128 维 K/V 头：本轮计算材料化多头激活，但长期缓存仍只保存 latent 与 RoPE key。" },
        { label: "Decode", title: "只搬源文件，不搬导出件", body: "每一步直接读取 512 维 latent；query 和输出权重现场解释它，不为历史 token 重建并搬运全部多头 K/V。" },
        { label: "双旋钮", title: R`\(d\) 管 Prefill，\(C\) 管 Decode`, body: R`普通 MHA/GQA 的每头宽度同时进入计算量与 KV cache；MLA 用 \(d\) 决定显式多头计算宽度，用 \(C\) 决定共享历史表示宽度。\(H\) 个 softmax 头和逐头上投影仍在，但不再为历史 token 复制 \(H\) 份缓存。` }
      ],
      diagram: { type: "latent", caption: R`MLA 的双执行图：同一份 \(c^{KV}\) 与同一组上投影，Prefill 展开成每头 128 维 K/V 做 MHA 形态计算；Decode 把投影吸收到 query/输出侧，直接以 512 维 latent 作为共享 K=V。64 维 decoupled-RoPE key 同时服务两条路径。` },
      derivations: [
        {
          title: "证明：同等表达力下，MLA 的 KV Cache 更小",
          body: R`**前提。** 在 LLM 推理中，先采用一个简化但关键的经验判断：相比增加 head 数量，增大每个 head 的有效工作维度 \(d\) 对模型表达力更重要。下面的结论只在这个前提下成立。

            暂时忽略位置支路。设 GQA 有 \(G\) 个 KV heads、每头维度为 \(d\)；MLA 的联合 KV latent 宽度为 \(C\)。每个历史 token 的内容缓存宽度分别是
            \[
            \mathrm{Cache}_{\mathrm{GQA}}=2Gd,\qquad
            \mathrm{Cache}_{\mathrm{MQA}}=2d,\qquad
            \mathrm{Cache}_{\mathrm{MLA}}=C.
            \]
            GQA/MQA 必须分别缓存 K 和 V，所以有系数 2；MLA 用同一个
            \(c^{KV}\in\mathbb R^C\) 同时生成 K 和 V，Decode 时再把两侧上投影吸收到 query 与输出，因此只缓存一份 \(C\) 维 latent，并以 MQA-\(C\) 的工作维度执行。

            在上述前提下，要与 MLA-\(C\) 具有同等级的表达力，MQA/GQA 也应取 \(d=C\)。代入即得
            \[
            \frac{\mathrm{Cache}_{\mathrm{MLA}}}{\mathrm{Cache}_{\mathrm{MQA}}}
            =\frac12,\qquad
            \frac{\mathrm{Cache}_{\mathrm{MLA}}}{\mathrm{Cache}_{\mathrm{GQA}}}
            =\frac{1}{2G}.
            \]
            这就是核心结论：**同等有效 head 维度下，MLA 的内容缓存是 MQA 的 \(1/2\)，是 \(G\)-head GQA 的 \(1/(2G)\)。**

            **具体例子。** GQA-2-128 的内容缓存为
            \(2\times2\times128=512\) 个元素；MLA-512 也缓存 512 个元素。
            固定缓存时二者一样大，但 MLA 的 Decode 工作维度是 512，而 GQA
            只有 128。若把表达力口径对齐到 512，GQA 必须变成 GQA-2-512，
            缓存增至 \(2\times2\times512=2048\) 个元素，是 MLA-512 的 4 倍；
            MQA-512 则需 1024 个元素，是 MLA-512 的 2 倍。

            DeepSeek-V2 还缓存一份 64 维共享 RoPE key。若各方案都按“512 维内容
            + 64 维位置 key”对齐，则 MLA、MQA、GQA-2 分别为 576、1088、2176
            个元素；位置支路缩小了理论倍数，但不改变 MLA 更省缓存的结论。`
        },
        {
          title: "代价：Decode 做更多计算，但瓶颈访存更少",
          body: R`吸收上投影后，MLA-512 的内容路径在 Decode 时等效于
            MQA-512：每个 query head 都要在 512 维空间中完成打分与加权读取。
            对 \(H\) 个 query heads、长度 \(L\) 的历史，内容计算量近似为
            \[
            \mathrm{FLOPs}_{\mathrm{decode}}=\Theta(2HLC).
            \]
            因此相对 GQA-2-128，MLA-512 的每头内容算术约为 4 倍。两者都缓存
            512 个内容元素，所以这个“固定缓存”对比表达的是：MLA 用更多算术换来
            4 倍工作维度，而没有增加内容缓存。

            若按同等表达力比较，正确基线是 GQA-2-512：它与 MLA-512 的每头工作
            维度相同，但每个历史 token 要读取 2048 个内容元素；MLA 只读取 512 个，
            访存量降为 \(1/4\)。与 MQA-512 相比则降为 \(1/2\)。

            Decode 通常每步只产生一个 token，却要扫描全部历史，瓶颈往往是 HBM
            带宽而不是乘加吞吐。因此在长上下文、小 batch 且有高效吸收式 kernel 时，
            增加算术强度通常比重复搬运 K/V 更划算。反之，短上下文、大 batch 或缺少
            专用 kernel 时，MLA 的额外计算不一定能被访存收益抵消。`
        }
      ],
      warning: "本节把“有效 head_dim 比 head 数量更影响表达力”作为分析前提，而不是普适定理；MQA-512 也是 MLA 吸收后的结构与执行口径，不等于无约束的标准 MQA-512 参数化。真实选型还受 tensor parallel、kernel、量化、参数预算、batch、上下文长度和硬件影响。",
      exercises: [
        {
          q: R`采用“有效 head_dim 比 head 数量更影响表达力”的前提。设 MLA latent 宽度为 \(C\)，GQA 有 \(G\) 个 KV heads。证明同等表达力下 MLA 相对 MQA、GQA 的内容缓存比例。`,
          hint: R`先令 MQA/GQA 的 \(d=C\)，再比较 \(C\)、\(2C\) 与 \(2GC\)。`,
          answer: R`同等表达力要求三者的有效工作维度均为 \(C\)。MLA 联合存储一份 \(C\) 维 latent；MQA 分别存一份 \(C\) 维 K 和 V，共 \(2C\)；GQA 为 \(G\) 个 KV heads 分别存 K/V，共 \(2GC\)。所以 MLA/MQA \(=1/2\)，MLA/GQA \(=1/(2G)\)。结论依赖给定的表达力前提，不能外推为无条件最优定理。`
        },
        {
          q: R`比较 GQA-2-128 与 MLA-512 的 Decode：(a) 内容缓存各有多少元素？(b) MLA 每个 query head 的内容算术约是前者多少倍？(c) 若改为同等表达力，正确的 GQA 基线是什么，此时 MLA 的访存量缩小多少？`,
          hint: "固定缓存时比较 128 与 512；固定表达力时把 GQA 的 head_dim 也提高到 512。",
          answer: R`(a) GQA-2-128 为 \(2\times2\times128=512\)，MLA-512 为 512，忽略位置支路时相同。(b) score 与 value read 都随每头工作维度线性增长，因此 MLA 的内容算术约为 \(512/128=4\) 倍。(c) 同等表达力应比较 GQA-2-512，其缓存与历史读取为 \(2\times2\times512=2048\) 个元素，MLA 只有 512，访存量缩小 4 倍。两个答案不矛盾：前两个固定缓存，最后一个固定表达力。`
        }
      ],
      sources: [
        { label: "DeepSeek-AI (2024), DeepSeek-V2 Technical Report（primary 架构来源）", url: "https://arxiv.org/abs/2405.04434" },
        { label: "DeepSeek-V2 official repository", url: "https://github.com/deepseek-ai/DeepSeek-V2" },
        { label: "苏剑林（2025），Transformer升级之路：21、MLA好在哪里？（下）· 解释性解读", url: "https://spaces.ac.cn/archives/11111" },
        { label: "苏剑林（2024），缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA · 解释性解读", url: "https://spaces.ac.cn/archives/10091" }
      ]
    },
    {
      id: "mfa",
      order: 4,
      title: "MFA",
      fullTitle: "Multi-matrix Factorization Attention",
      zhTitle: "多矩阵分解注意力：用共享 C 维 KV 支撑更多、更宽的头",
      year: "2024",
      category: "dense",
      difficulty: "高阶",
      report: "Multi-matrix Factorization Attention",
      deck: "在固定 KV cache 预算下，MFA 不再把 head dimension 绑定到 d_model / head 数。它缓存共享的 C 维 key/value，却为每个 attention head 配置 C×C 的 QK 与 VO 变换，用参数和计算换取更高的每头秩与总有效秩。",
      takeaway: R`MFA 的缓存只有 \(k_t=x_tS_k\) 与 \(v_t=x_tS_v\)，两者各 C 维；head-specific \(Q_c\)、\(O_c\) 只存在于权重中，因此增加 head 数不会复制历史 KV。MFA-KR 进一步把 value 重参数化为 key 的变换，将缓存从 2C 降到 C。`,
      motivation: [
        "MQA/GQA 通过共享 K/V 缩小缓存，但共享 latent subspace 与每头 factorization rank 也随缓存预算一起变窄；在严格 KV 约束下，表达能力可能先成为瓶颈。",
        R`MFA 从 QK/VO circuit 的矩阵分解出发：共享 \(S_q\)、\(S_k\)、\(S_v\) 把 token 投到 C 维，而每个 head 使用独立 \(Q_c,O_c\in\mathbb R^{C\times C}\)，让每头 factorization rank 提升到 C。`,
        "MFA-KR 令 value projection 成为 key projection 的零初始化可学习变换，只缓存 key；它把内存再减半，但论文实验也显示了相对 MFA 的小幅质量折损。"
      ],
      constraints: [
        { label: "缓存", title: "2C 与 head 数无关", body: "MFA 每 token 每层缓存一份 C 维 key 和一份 C 维 value；MFA-KR 只缓存 key。上下文长度因子仍然存在。" },
        { label: "参数与算力", title: "head-specific C×C 不是免费的", body: R`增加 head 数会线性增加 \(Q_c\)、\(O_c\) 权重和 score/output 计算；它只是不增加 KV cache。` },
        { label: "系统证据", title: "论文未给端到端部署加速", body: "主论文验证了 cache 与质量，但明确把系统级端到端推理影响留作未来工作。" }
      ],
      intuitions: [
        { label: "共享底片", title: "所有头读取同一份 C 维 K/V", body: "历史 token 只存一次共享底片，不为每个 head 复制缓存。" },
        { label: "逐头镜头", title: R`\(Q_c\) 与 \(O_c\) 提供独立视角`, body: "每个 head 用完整 C×C 矩阵改变匹配和写回方式，差异保存在权重而不是缓存。" },
        { label: "Key Reuse", title: "同一缓存生成 K 与 V", body: "MFA-KR 从 key 通过零初始化门控变换得到 value，进一步节省一半缓存。" }
      ],
      diagram: { type: "mfa", caption: R`MFA：token 只缓存共享 C 维 K/V；每个 head 通过独立 \(C\times C\) 的 \(Q_c\) 与 \(O_c\) 获得高秩 QK/VO circuit。MFA-KR 可从 key 重参数化 value。` },
      derivations: [
        {
          title: "共享 K/V、逐头矩阵的推理式",
          body: R`共享投影把每个 token 映到同一 C 维空间：
            \[
            k_s=x_sS_k,\qquad v_s=x_sS_v,\qquad k_s,v_s\in\mathbb R^{C}.
            \]
            第 \(c\) 个 head 只在 query 与输出侧拥有独立矩阵：
            \[
            q_{t,c}=x_tS_qQ_c\in\mathbb R^{C},\qquad
            a_{t,s}^{(c)}=\operatorname{softmax}_s\!\left(
            \frac{q_{t,c}k_s^\top}{\sqrt C}+M\right),
            \]
            \[
            o_t=\sum_{c=1}^{m}\left(\sum_{s}a_{t,s}^{(c)}v_s\right)O_c^\top.
            \]
            批量形状：q 为 \([B,m,T,C]\)，k/v 为 \([B,T,C]\)，scores 为 \([B,m,T,T]\)。所有 head 读取同一份 k/v，差异全部由 \(Q_c,O_c\in\mathbb R^{C\times C}\) 承担。`
        },
        {
          title: "缓存与 head count 解耦",
          body: R`MFA 每 token 每层缓存一份 C 维 key 与一份 C 维 value：
            \[
            \mathrm{Cache}_{MFA}=2BNLCb,\qquad
            \mathrm{Cache}_{MFA\text{-}KR}=BNLCb,
            \]
            对比 MHA 的 \(2BNLH_{kv}d_hb\)：MFA 把 \(H_{kv}d_h\) 换成与 head 数无关的 \(C\)。论文 24 层 BF16、\(C=256\) 的配置下，每 token 的全模型缓存为
            \[
            24\times2\times256\times2=24{,}576\ \text{字节}\approx24\ \text{KiB};
            \]
            MFA-KR 只缓存 key，约 \(12\) KiB。上下文长度因子 \(L\) 仍然存在。`
        }
      ],
      warning: "MFA 的“head dimension=256”是共享 latent 维度 C 和每头 factorization rank，不要求 hidden size 等于 head 数×256。论文的 6.9B/1.2B-activated 模型是研究实验，不是公开生产 checkpoint；24.6 KB/token 是 24 层 BF16 缓存总量，不是单层。",
      exercises: [
        {
          q: R`按论文 24 层、BF16、\(C=256\) 的配置，分别计算 MFA 与 MFA-KR 每 token 的全模型缓存字节数。`,
          hint: R`MFA 每层 \(2C\) 个元素、MFA-KR 每层 \(C\) 个元素，每元素 2 字节，再乘 24 层。`,
          answer: R`MFA：\(24\times2\times256\times2=24{,}576\) 字节，约 24 KiB。MFA-KR：\(24\times256\times2=12{,}288\) 字节，约 12 KiB。`
        },
        {
          q: R`把 head 数 \(m\) 从 18 加倍到 36，KV cache、head-specific 参数量与注意力计算分别如何变化？`,
          hint: R`缓存宽度是 \(2C\)，与 \(m\) 无关；\(Q_c\)、\(O_c\) 逐头存在。`,
          answer: R`KV cache 不变（仍是每层 \(2C\)）；head-specific \(Q_c\)、\(O_c\) 参数量约 \(2mC^2\)，随 \(m\) 加倍；score 与输出的逐头计算也大致加倍。MFA 用参数和算力换容量，不用缓存换。`
        }
      ],
      sources: [
        { label: "Multi-matrix Factorization Attention (Findings of ACL 2025)", url: "https://aclanthology.org/2025.findings-acl.1288/" },
        { label: "Multi-matrix Factorization Attention (arXiv:2412.19255)", url: "https://arxiv.org/abs/2412.19255" }
      ]
    },
    {
      id: "tpa",
      order: 5,
      title: "TPA",
      fullTitle: "Tensor Product Attention",
      zhTitle: "张量积注意力：把每个 token 的 head×channel 激活分解成少量秩一因子",
      year: "2025",
      category: "dense",
      difficulty: "高阶",
      report: "Tensor Product Attention Is All You Need",
      deck: R`TPA 不对静态权重做 LoRA 式分解，而是让每个 token 动态生成 head-axis 因子 \(a\in\mathbb R^{h}\) 与 channel-axis 因子 \(b\in\mathbb R^{d_h}\)，用少量外积重建 Q/K/V。缓存保存 K/V 因子而非完整 h×d_h 矩阵。`,
      takeaway: R`\(K_t=\frac1{R_K}A_K(x_t)^\top B_K(x_t)\)，V 同理；每 token KV cache 为 \((R_K+R_V)(h+d_h)\)，而不是 \(2hd_h\)。因子 A、B 都依赖当前 token，因此 rank-1 contextual TPA 不等同于固定共享模式的 MQA。`,
      motivation: [
        "MHA 保存完整 head×channel K/V；MQA/GQA 用固定 head-sharing mask 降缓存，能表达的跨 head 结构由预设分组限制。",
        "TPA 直接分解每个 token 的 Q/K/V 激活矩阵：A 决定一个因子如何分布到各 heads，B 决定它在 head channel 上写入什么模式。",
        "FlashTPA decoding 用 factorized einsum 顺序计算 score 与 value 聚合，避免为历史 token 物化完整 K/V；若先重建再调用普通 MHA kernel，会保留数值但丢掉主要效率收益。"
      ],
      constraints: [
        { label: "秩预算", title: R`\(R_K\)、\(R_V\) 同时控制质量与缓存`, body: R`更高 rank 能表达更多 head×channel 结构，但 cache 按 \(R_K+R_V\) 线性增长。` },
        { label: "Kernel", title: "不能依赖先重建再计算", body: "真正 decode 收益需要 factorized contraction 或 FlashTPA kernel；教学重建路径只用于验证。" },
        { label: "维度", title: R`\(h\times d_h\) 可以大于 \(d_{\text{model}}\)`, body: R`T6 为参数量对齐会扩展 attention inner width，再由 \(W^O\) 写回 d_model；不能假设 \(d_{\text{model}}=hd_h\)。` }
      ],
      intuitions: [
        { label: "秩一瓷砖", title: "一个外积铺满 head×channel 平面", body: "少量瓷砖相加组成当前 token 的 Q/K/V 激活矩阵。" },
        { label: "A 因子", title: "决定哪些 heads 被写入", body: R`\(A(x_t)\) 是 token-dependent head mixing，不是固定 group mask。` },
        { label: "B 因子", title: "承载 channel pattern 与 RoPE", body: R`RoPE 逐行作用于 \(B_Q\)/\(B_K\)，A 因子保持不变。` }
      ],
      diagram: { type: "tpa", caption: R`TPA：每个 token 动态生成 \(A_{Q/K/V}\) 与 \(B_{Q/K/V}\) 因子；RoPE 只旋转 Q/K 的 B 因子，KV cache 保存低秩因子，FlashTPA 直接收缩而不重建完整历史 K/V。` },
      derivations: [
        {
          title: "Contextual tensor factorization 与形状",
          body: R`每个 token 的因子由当前 token 生成：
            \[
            Q_t=\frac1{R_Q}A_Q(x_t)^\top B_Q(x_t),\qquad
            A_Q\in\mathbb R^{R_Q\times h},\quad
            B_Q\in\mathbb R^{R_Q\times d_h},\quad
            Q_t\in\mathbb R^{h\times d_h},
            \]
            K/V 同理使用 \(R_K,R_V\)。这里的 rank 是对每个 token 的激活矩阵 \(Q_t,K_t,V_t\) 的秩预算，而不是对静态权重矩阵做低秩分解。批量实现中 A 因子为 \([B,T,R,h]\)、B 因子为 \([B,T,R,d_h]\)。`
        },
        {
          title: "KV cache 公式与 T6-XL 实例",
          body: R`TPA 只缓存 K/V 因子：
            \[
            \mathrm{Cache}_{TPA}=(R_K+R_V)(h+d_h)\ \text{elements/token/layer}.
            \]
            T6-XL 的 \(h=78,d_h=64,R_K=R_V=2\) 给出
            \[
            (2+2)(78+64)=568,
            \]
            而完整 MHA 形状的缓存为 \(2\times78\times64=9984\)。比值 \(568/9984\approx5.69\%\)，约小 \(17.6\) 倍。`
        }
      ],
      warning: "TPA 论文与官方仓库提供的是最高 1.55B 的 T6 研究模型，而不是公开的超大生产 checkpoint。官方 XL config 使用 h=78、d_h=64，所以 attention inner width 为 4992，明显大于 d_model=1600；这是参数量配平选择，不是维度错误。",
      exercises: [
        {
          q: R`按 T6-XL 的 \(h=78,d_h=64,R_K=R_V=2\)，每层每 token 的因子缓存与完整 MHA 形状缓存各是多少元素？比值是多少？`,
          hint: R`分别代入 \((R_K+R_V)(h+d_h)\) 与 \(2hd_h\)。`,
          answer: R`因子缓存 \((2+2)(78+64)=568\) 个元素；完整 MHA 形状 \(2\times78\times64=9984\) 个元素。比值 \(568/9984\approx5.69\%\)，约小 17.6 倍。`
        },
        {
          q: R`给定 \(B,T,R_K,h,d_h\)，写出 \(A_K\)、\(B_K\) 因子张量与重建后完整 K 的形状。`,
          hint: "A 决定 head 轴，B 决定 channel 轴。",
          answer: R`\(A_K\) 为 \([B,T,R_K,h]\)，\(B_K\) 为 \([B,T,R_K,d_h]\)；逐 token 重建 \(K_t=A_K^\top B_K/R_K\in\mathbb R^{h\times d_h}\)，批量布局为 \([B,h,T,d_h]\)。`
        }
      ],
      sources: [
        { label: "Tensor Product Attention Is All You Need (arXiv:2501.06425)", url: "https://arxiv.org/abs/2501.06425" },
        { label: "Official tensorgi/TPA repository", url: "https://github.com/tensorgi/TPA" },
        { label: "Official T6-XL training config", url: "https://github.com/tensorgi/TPA/blob/main/config/train_T6_xl_adam_80g8.py" }
      ]
    },
    {
      id: "dsa",
      order: 6,
      title: "DSA",
      fullTitle: "DeepSeek Sparse Attention",
      zhTitle: "DeepSeek 稀疏注意力：先索引，再精读",
      year: "2025",
      category: "sparse",
      difficulty: "高阶",
      report: "DeepSeek-V3.2-Exp",
      deck: "DSA 把 core attention 拆成两阶段：Lightning Indexer 为每个 query 选 top-k 历史位置，随后 MQA-mode MLA 直接在这些位置对应的 latent KV entries 上计算 core attention。",
      takeaway: "DSA 的候选单位是 MLA 的 token-indexed latent entry；它没有论文定义中的额外局部窗口，短序列 masked-MHA 只是实现 DSA 的 kernel 路径。",
      motivation: [
        R`MLA 解决了每个 token 缓存过宽的问题，但 dense attention 仍要让每个 query 与所有历史位置做高维交互；上下文极长时，算量仍随 \(L^2\) 增长。`,
        R`DSA 引入 Lightning Indexer：用低维低精度路径估计相关性，为每个 query 选择 k 个历史位置；core 读取这些位置的 MLA latent \(c_s^{KV}\)，而非另造一套高维候选缓存。`,
        "DeepSeek-V3.2-Exp/V3.2 通过 continued training 加入 DSA；“质量基本持平”和部署成本曲线均属于官方 checkpoint/服务自报，报告也引用了部分独立长上下文评测。"
      ],
      constraints: [
        { label: "选择预算", title: "k 决定精度与成本", body: "k 太小会漏掉关键 token；太大则接近 dense attention。" },
        { label: "索引成本", title: "索引器本身也要扫描", body: R`若对所有历史位置打分，索引路径仍含 \(L^2\) 项，只是维度和精度更低。` },
        { label: "训练系统", title: "离散 top-k 难优化", body: "需要索引损失、稳定训练与专用稀疏 kernel；纸面稀疏不自动等于硬件高效。" }
      ],
      intuitions: [
        { label: "阶段 1", title: "目录检索", body: "Indexer 只看短摘要，快速列出候选页码。" },
        { label: "阶段 2", title: "打开原文", body: "Core attention 在候选 token 上做完整高维读取。" },
        { label: "不同于窗口", title: "远处也能被选中", body: "选择由内容相关性决定，不只依赖距离。" }
      ],
      diagram: { type: "sparse", caption: "DSA：Indexer 扫描历史并选 top-k token positions，core 以 MQA-mode MLA 读取对应 latent entries；没有额外滑窗分支。" },
      derivations: [
        {
          title: "Lightning Indexer 的正式评分",
          body: R`设 Indexer 有 \(H^I\) 个 query 头，共享低维 key；pRoPE 先作用于 q/k 的指定子维，FP8 路径再对两侧施同一正交 Hadamard 变换 \(\mathcal H\)：
            \[
            \bar q^I_{t,j}=\mathcal H\,\operatorname{pRoPE}_t(q^I_{t,j}),\qquad
            \bar k^I_s=\mathcal H\,\operatorname{pRoPE}_s(k^I_s).
            \]
            官方评分为
            \[
            I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}
            \operatorname{ReLU}\!\left((\bar q^I_{t,j})^\top\bar k^I_s\right),
            \qquad
            \mathcal I_t=\operatorname{TopK}_s(I_{t,s},k).
            \]
            同一正交变换在精确算术下保持点积；其作用是改善低精度数值分布，不是 PE。core 只取
            \(\{c_s^{KV}:s\in\mathcal I_t\}\)：
            \[
            u_t=\operatorname{Attn}_{\mathrm{MLA\text{-}MQA}}
            \!\left(h_t,\{c_s^{KV}:s\in\mathcal I_t\}\right).
            \]
            top-k 后 core softmax 在候选 latent entries 上重算。V3.2 官方公开配置为
            \(H^I=64,d^I=128,k=2048\)，Indexer QK 路径使用 FP8。`
        },
        {
          title: "复杂度要分两条路径看",
          body: R`若 indexer 有 \(H^I\) 头、每头维 \(d_I\)，core 有 \(H_q\) 个 query 头、其直接 latent 点积有效宽度记为 \(d_{\mathrm{core}}\)，则
            \[
            C_{\mathrm{index}}=\Theta(L^2H^Id_I),\qquad
            C_{\mathrm{core}}=\Theta(LkH_qd_{\mathrm{core}}).
            \]
            常把固定头数/宽度省略后才写成 \(O(L^2)\) 与 \(O(Lk)\)。DSA 的价值来自低维低精度 Indexer 和 \(k\ll L\)，不是把整个模块都变成严格 \(O(Lk)\)。`
        },
        {
          title: "Indexer 用独立对齐目标训练",
          body: R`Dense warm-up 保留 full core attention，先把其各头的**概率权重**求和并沿完整历史轴 L1 归一化成 teacher \(p_{t,:}\)，再用完整 Indexer logits 训练
            \[
            \mathcal L^I=\sum_t D_{\mathrm{KL}}\!\left(
            p_{t,:}\,\|\,\operatorname{Softmax}(I_{t,:})\right).
            \]
            不能把 teacher 写成“对主 logits 直接求和”，也不能在 warm-up 偷换成 top-k 归一化。稀疏阶段才把 teacher 与 logits 都限制到选中集合 \(\mathcal S_t\) 后对齐。官方设计将 Indexer 输入 detach：Indexer 由
            \(\mathcal L^I\) 优化，语言模型损失不穿过离散 top-k 直接反传。`
        }
      ],
      warning: "内容稀疏注意力存在召回风险：被 Indexer 漏掉的 latent entry 不会进入 core。DeepSeek 的“效率提升/质量持平”必须标作官方、checkpoint 与部署栈特定自报；DSA 论文并未定义一个可兜底的局部滑窗分支。",
      exercises: [
        {
          q: R`长度为 \(L\) 的因果序列中，Lightning Indexer 有 \(H^I\) 个 query heads、每头宽度 \(d_I\)，core 有 \(H_q\) 个 query heads、点积宽度 \(d_{\rm core}\)，每个 query 选择 \(k\) 个位置。分别推导 Indexer、稀疏 core 与 DSA 总体的时间复杂度，并与 dense core 比较。`,
          hint: R`因果位置对总数为 \(\sum_{t=1}^{L}t=\Theta(L^2)\)；选中后每个 query 只在 \(k\) 个位置执行 core。`,
          answer: R`Indexer 对每个因果位置对执行 \(H^I\) 次 \(d_I\) 维点积，因此 \(C_{\rm index}=\Theta(L^2H^Id_I)\)。稀疏 core 每个 query 只读取 \(k\) 个位置，故 \(C_{\rm core}=\Theta(LkH_qd_{\rm core})\)；总体为两者之和。dense core 则为 \(\Theta(L^2H_qd_{\rm core})\)，所以昂贵 core 的位置轴由 \(L\) 降到 \(k\)，理想缩减 \(L/k\) 倍。若把头数和维度视为常数，DSA 总体渐近阶仍含 \(O(L^2)\) 的低维 Indexer，而不是严格 \(O(Lk)\)；收益来自 \(H^Id_I\) 较小、FP8 路径以及 \(k\ll L\)。`
        },
        {
          q: "为什么 DSA 不能只用固定滑动窗口代替 indexer？",
          hint: "考虑跨文档引用和很远的定义。",
          answer: "固定窗口保证局部性但无法按内容选取任意远 token；DSA 的目标正是保留数据依赖的远程读取。"
        }
      ],
      sources: [
        { label: "DeepSeek-AI (2025), DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DSA", url: "https://github.com/deepseek-ai/DeepSeek-V3.2-Exp" },
        { label: "DeepSeek-AI (2025), DeepSeek-V3.2 formal technical report", url: "https://arxiv.org/abs/2512.02556" },
        { label: "FlashMLA official sparse kernels", url: "https://github.com/deepseek-ai/FlashMLA" }
      ]
    },
    {
      id: "csa",
      order: 7,
      title: "CSA",
      fullTitle: "Compressed Sparse Attention",
      zhTitle: "压缩稀疏注意力：先缩短历史，再检索摘要",
      year: "2026",
      category: "sparse",
      difficulty: "前沿",
      report: "DeepSeek-V4 Technical Report",
      deck: R`CSA 分别生成宽度 \(c\) 的 compressed KV pool \(C^{Comp}\) 与宽度 \(c^I\) 的 compressed indexer-key pool \(K^{IComp}\)，再用后者选 top-k、用前者做共享-KV MQA；滑窗保留近期原始条目。`,
      takeaway: R`CSA 压缩 token 轴：\(C^{Comp}\) 与 \(K^{IComp}\) 都从 \(L\) 变为约 \(L/m\)，但两者是不同宽度、不同用途的缓存，不能拿 MLA 的 \(d_c\) 代称。`,
      motivation: [
        "到百万 token，上下文条目数本身成为瓶颈：即使每条 KV 已被 MLA 压窄，逐 token 存储和检索仍很昂贵。",
        R`CSA 每 \(m\) 个 token 前进一次输出；每个输出以联合 \(2m\)-位置、逐通道 softmax 形成 \(C^{Comp}\)，并以同型但独立参数形成 \(K^{IComp}\)，同时降低 cache 长度与 core 读取量。`,
        "压缩块只能在闭合后被因果读取，所以每层并联未压缩 sliding-window 分支，覆盖当前块与近期依赖。"
      ],
      constraints: [
        { label: "信息瓶颈", title: "序列压缩不可逆", body: "多个 token 合成一个条目，旧历史的 token 级细节可能丢失。" },
        { label: "因果性", title: "块内 token 尚未完成", body: "只能读取已闭合的前序压缩块，因此需要滑窗补足当前块和近期细节。" },
        { label: "系统复杂度", title: "三条路径共同运行", body: "压缩器、indexer/core、滑窗及混合精度缓存都需要专用实现。" }
      ],
      intuitions: [
        { label: "Compress", title: "四页合成一条摘要", body: "候选池先从 L 缩短到约 L/4。" },
        { label: "Select", title: "从摘要中搜索", body: "Indexer 选择相关压缩条目，而非原始 token。" },
        { label: "SWA", title: "桌面上的原件", body: "最近 token 保持未压缩，保证局部细节和因果覆盖。" }
      ],
      diagram: { type: "compressed", mode: "csa", caption: R`CSA：独立压缩出 \(C^{\mathrm{Comp}}\)（宽 \(c\)）与 \(K^{I\mathrm{Comp}}\)（宽 \(c^I\)）；Indexer 用后者选 top-k，MQA 用前者读写，滑窗保留近期 token。` },
      derivations: [
        {
          title: "沿序列维做重叠学习式压缩",
          body: R`CSA 产生两路 KV 流 \(C^a,C^b\in\mathbb R^{L\times c}\) 与 logits \(Z^a,Z^b\)。第 \(i\) 个输出先把当前 a 块与前一 b 块的 logits（含位置 bias）拼接，在联合 \(2m\) 行上逐通道归一化：
            \[
            [S^a_i;S^b_i]=\operatorname{Softmax}_{\rm row}
            ([Z^a_{mi:m(i+1)}+B^a;Z^b_{m(i-1):mi}+B^b]).
            \]
            \[
            C_i^{Comp}=
            \sum_{j=mi}^{m(i+1)-1}S_j^a\odot C_j^a+
            \sum_{j=m(i-1)}^{mi-1}S_j^b\odot C_j^b.
            \]
            两半不是各自 softmax 后相加。Indexer key 另用独立投影和同样压缩操作得到
            \(K^{IComp}\in\mathbb R^{(L/m)\times c^I}\)；它不等于 \(C^{Comp}\)。步长为 \(m\)，故长度约为 \(L/m\)，不是 \(L/(2m)\)。`
        },
        {
          title: "Indexer、core 与 cache 要分别计算",
          body: R`CSA 对每个 query 扫描约 \(L/m\) 个 \(c^I\)-维 indexer candidates，再选择 \(k\) 个 \(c\)-维 compressed KV entries：
            \[
            C_{\mathrm{index}}\sim O(L^2H^Ic^I/m),\qquad
            C_{\mathrm{CSA,core}}\sim O(LkH_qc),
            \]
            \[
            \mathrm{Cache}_{CSA}\sim O((L/m)(c+c^I)+L_{\rm win}c).
            \]
            还需滑窗分支。固定 \(m\) 时 Indexer 渐近仍为平方项；\(c,c^I\) 是 V4 报告宽度，不是 MLA query bottleneck \(d_c'\)。`
        },
        {
          title: "core normalization、sink 与正式配置",
          body: R`core attention 前对每个 query head 与唯一 compressed-KV head 分别做 RMSNorm。若归一化 logit 为 \(\ell_{h,t,s}\)，每头 sink logit \(z'_h\) 令
            \[
            a_{h,t,s}=\frac{e^{\ell_{h,t,s}}}
            {\sum_u e^{\ell_{h,t,u}}+e^{z'_h}},
            \]
            所以真实条目的权重和可小于 1。官方 checkpoint 配置为 \(m=4,w=128\)，Indexer 64 头×128 维且 QK 用 FP4；Flash/Pro 的 top-k 分别为 512/1024。这些都是官方模型配置。`
        }
      ],
      warning: R`DeepSeek-V4 是 2026 预览报告。百万上下文 FLOPs/cache、FP4 Indexer、召回与吞吐数字都是官方且 checkpoint/实现特定的自报；结构描述还必须同时计 \(C^{\mathrm{Comp}}\)、\(K^{I\mathrm{Comp}}\)、滑窗、RMSNorm 与 sink。`,
      exercises: [
        {
          q: R`\(L=1{,}000{,}000\)，CSA 的 \(m=4\)、\(k=512\)。压缩池有多少条目？每个 query 的 core 只读其中多少比例？`,
          hint: "先算 L/m，再算 k/(L/m)。",
          answer: "压缩池约 250,000 条；core 读取约 0.2048%。此外还有局部窗口和 indexer。"
        },
        {
          q: "为什么 CSA 覆盖 2m 个输入位置，却只把序列缩短 m 倍？",
          hint: "区分窗口覆盖宽度与输出步长。",
          answer: "两路窗口覆盖当前块和前一块，但每经过 m 个新 token 产生一个输出；相邻输出重叠，所以压缩率是 m。"
        }
      ],
      sources: [
        { label: "DeepSeek-AI (2026), DeepSeek-V4 Technical Report", url: "https://arxiv.org/abs/2606.19348" },
        { label: "DeepSeek-V4 official model and inference code", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro" }
      ]
    },
    {
      id: "hca",
      order: 8,
      title: "HCA",
      fullTitle: "Heavily Compressed Attention",
      zhTitle: "重压缩注意力：用粗粒度摘要换全局稠密视野",
      year: "2026",
      category: "dense",
      difficulty: "前沿",
      report: "DeepSeek-V4 Technical Report §2.3.2",
      deck: "HCA 把每 m′ 个 token 压成一个宽度 c 的条目，取消 Indexer 与 top-k，并对所有因果可见的已完成压缩块做 dense MQA；它是 compressed-dense attention，不是 sparse attention。",
      takeaway: R`HCA 的“dense”发生在压缩后的 key 轴：query \(t\) 读取全部 \(\lfloor t/m'\rfloor\) 个已完成块。它没有选择漏召回，但有重压缩损失。`,
      motivation: [
        "CSA 保留较细的摘要并依赖 learned top-k，适合内容检索，但 Indexer 仍要扫描候选且可能漏掉相关块。",
        "HCA 选择相反的极端：把每个大块压成一个 KV 条目，使全局摘要短到无需检索器即可全部读取。",
        "DeepSeek-V4 交错使用 CSA 与 HCA：CSA 提供较高分辨率选择，HCA 提供便宜、稳定且无 top-k 漏召回的全局概览。"
      ],
      constraints: [
        { label: "压缩损失", title: "128 个 token → 1 条摘要", body: "极强的信息瓶颈难以保留符号、引用和 token 级精确细节。" },
        { label: "计算", title: "压缩后仍是 dense", body: R`query 轴未压缩，prefill 仍含 \(O(L^2/m')\) 项，并非严格线性。` },
        { label: "局部性", title: "必须并联滑窗", body: R`未闭合块与近期细节由 \(w=128\) 的原始 KV 窗口补足。` }
      ],
      intuitions: [
        { label: "Compress", title: "把整卷做成目录", body: "每 128 个 token 合成一个粗粒度全局条目。" },
        { label: "Read", title: "完整阅读短目录", body: "没有 Indexer，也没有 top-k 选择遗漏。" },
        { label: "Trade-off", title: "不漏目录，但目录会丢细节", body: "选择误差消失，压缩误差变成主要风险。" }
      ],
      diagram: { type: "compressed", mode: "hca", caption: R`HCA（compressed-dense）：对全部因果已完成的宽 \(c\) 压缩条目做 dense MQA；局部滑窗覆盖当前未闭合块与近期 token。` },
      derivations: [
        {
          title: "非重叠重压缩",
          body: R`令 \(C=HW^{KV},Z=HW^Z\)。对第 \(i\) 个不重叠块，
            \[
            S_{m'i:m'(i+1)-1}
            =\operatorname{Softmax}_{row}(Z_{m'i:m'(i+1)-1}+B),
            \]
            \[
            C_i^{Comp}=\sum_{j=m'i}^{m'(i+1)-1}S_j\odot C_j.
            \]
            每个条目宽度为报告记号 \(c\)。整段最终可形成 \(\lfloor L/m'\rfloor\) 个块，但位置 \(t\) 的 query 只能读取
            \(n_c(t)=\lfloor t/m'\rfloor\) 个已完成前序块。与 CSA 不同，HCA 没有两路重叠压缩。`
        },
        {
          title: "dense over compressed history",
          body: R`HCA 对位置 \(t\) 的全部已完成压缩条目与局部窗口做核心注意力：
            \[
            N_{\rm global}=\sum_{t=0}^{L-1}\left\lfloor\frac{t}{m'}\right\rfloor,\qquad
            C_{\mathrm{HCA}}=\Theta\!\left(H_qc\,[N_{\rm global}+Lw]\right),
            \]
            \[
            \mathrm{Cache}_{HCA}=\Theta((L/m')c+wc).
            \]
            当 \(L\gg m',w\) 时 \(N_{\rm global}=\Theta(L^2/m')\)。固定 \(m'\) 仍为 \(\Theta(L^2)\)；只有让 \(m'\) 随 \(L\) 增长等额外假设才可改写渐近阶。官方配置 \(m'=128,w=128\)。`
        }
      ],
      warning: "HCA 属于 compressed-dense attention（本站归入现有 dense 类），不是 sparse attention，也不是 mHC。它有 RMSNorm、partial/inverse RoPE、attention sink 与滑窗；V4 效率数字均为官方 checkpoint/系统特定自报。",
      exercises: [
        {
          q: R`当 \(L=1{,}048{,}576,m'=128,w=128\) 时，每个 HCA query 最多读取多少个全局摘要与局部 token？`,
          hint: "先算 L/m′。",
          answer: R`最后一个 query 的位置 \(t=L-1\)，只有 \(\lfloor(L-1)/128\rfloor=8191\) 个完整前序块可见；再加至多 128 个局部 token，共至多 8319 个输入条目。不能读取包含该 query 的尚未闭合块。`
        },
        {
          q: R`为什么 HCA 不能称为严格 \(O(L)\) attention？`,
          hint: "m′ 是固定常数，query 数仍为 L。",
          answer: R`严格 causal 计数为 \(\sum_t\lfloor t/m'\rfloor=\Theta(L^2/m')\)。固定 \(m'\) 只降低常数，不改变平方渐近阶；若明确令 \(m'=\Theta(L)\)，结论才会不同。`
        },
        {
          q: "为什么 V4 不全部使用 HCA？",
          hint: "比较压缩误差与检索分辨率。",
          answer: "HCA 的 128→1 压缩过于粗糙；交错 CSA 可用较细摘要和 top-k 恢复内容选择能力。"
        }
      ],
      sources: [
        { label: "DeepSeek-AI (2026), DeepSeek-V4 Technical Report §2.3.2", url: "https://arxiv.org/abs/2606.19348" },
        { label: "DeepSeek-V4-Pro official configuration", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json" }
      ]
    },
    {
      id: "linear",
      order: 9,
      title: "Linear Attention",
      fullTitle: "Kernelized Linear Attention",
      zhTitle: "线性注意力：把历史折叠进固定状态",
      year: "2020",
      category: "linear",
      difficulty: "高阶",
      report: "Transformers are RNNs",
      deck: "线性注意力通过核特征映射与矩阵乘法结合律，不再显式构造 L×L 注意力图；因果推理可写成固定大小状态的递推。",
      takeaway: R`先算 \(QK^{\mathsf T}\) 再乘 \(V\) 是二次的；若相似度可分解为 \(\phi(q)^{\mathsf T}\phi(k)\)，就能先累计 KV，再让 \(q\) 读取。`,
      motivation: [
        R`\(\operatorname{softmax}(QK^{\mathsf T})V\) 的计算顺序会显式形成长度平方的分数矩阵，极长序列训练和推理代价高。`,
        R`若相似度写成核内积 \(\phi(q)^{\mathsf T}\phi(k)\)，利用结合律可先计算 \(\sum\phi(k)v^{\mathsf T}\)，序列维被汇总到固定矩阵状态。`,
        "因果场景中该汇总可按 token 递推为常数状态。这个代数等价不等于声称 2020 原实现采用了后来流行的 prefix-scan 或 chunk 算法。"
      ],
      constraints: [
        { label: "核约束", title: "不再是精确 softmax", body: R`必须选择可分解特征映射 \(\phi\)；核的归纳偏置决定模型可表达的相似性。` },
        { label: "状态容量", title: "固定状态会发生干扰", body: "所有历史写入同一个矩阵，精确复制和多键检索常弱于 full attention。" },
        { label: "硬件现实", title: R`\(O(L)\) 不等于一定更快`, body: "短序列上，成熟的 FlashAttention 可能因更高算术强度而更快。" }
      ],
      intuitions: [
        { label: "Dense", title: "每次翻全部档案", body: "query 与每条历史逐一比较。" },
        { label: "Linear", title: "维护统计台账", body: "历史到来时写入固定状态，query 直接查台账。" },
        { label: "代价", title: "位置可解释性下降", body: "通常无法还原一个显式 L×L 注意力图。" }
      ],
      diagram: { type: "linear", caption: R`线性注意力：每个 \((k,v)\) 写入状态 \(S,z\)；\(q\) 从固定状态读取，不保存完整历史表。` },
      derivations: [
        {
          title: "从核分解到结合律",
          body: R`把非负相似度写成
            \[
            \operatorname{sim}(q_i,k_j)=\phi(q_i)^\top\phi(k_j).
            \]
            归一化注意力为
            \[
            y_i=\frac{\sum_j\phi(q_i)^\top\phi(k_j)v_j}
            {\sum_j\phi(q_i)^\top\phi(k_j)}
            =\frac{\phi(q_i)^\top\left(\sum_j\phi(k_j)v_j^\top\right)}
            {\phi(q_i)^\top\left(\sum_j\phi(k_j)\right)}.
            \]`
        },
        {
          title: "因果递推就是 RNN 状态",
          body: R`只累计 \(j\le t\)：
            \[
            S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
            z_t=z_{t-1}+\phi(k_t),
            \]
            \[
            y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t+\varepsilon}.
            \]
            若特征维为 \(r\)、value 维为 \(d_v\)，状态大小为 \(O(rd_v+r)\)，与上下文长度无关。`
        }
      ],
      warning: "“Linear Attention”是方法族。2020 论文给出因果递推，但不要把后来的 prefix-scan/chunk 实现倒写成其原实现；因果输出对顺序敏感，而无衰减加法状态对同一组写入的最终汇总可交换，这两点并不矛盾。",
      exercises: [
        {
          q: R`取标量 \(\phi(q)=q,\phi(k)=k\)，依次写入 \((k_1,v_1)=(1,2)\)、\((k_2,v_2)=(3,4)\)。忽略归一化，q=2 时输出多少？`,
          hint: "S=Σkv。",
          answer: R`\(S=1\times2+3\times4=14\)，输出 \(qS=28\)。`
        },
        {
          q: R`特征维 \(r=64\)、value 维 128，单头状态 \(S\) 有多少元素？若上下文从 4K 变到 1M，\(S\) 是否变大？`,
          hint: R`\(S\) 的形状是 \(r\times d_v\)。`,
          answer: "S 有 8192 个元素；上下文增长时状态尺寸不变，但有限状态中的信息干扰可能增加。"
        }
      ],
      sources: [
        { label: "Katharopoulos et al. (2020), Transformers are RNNs", url: "https://proceedings.mlr.press/v119/katharopoulos20a.html" },
        { label: "Choromanski et al. (2021), Rethinking Attention with Performers", url: "https://arxiv.org/abs/2009.14794" }
      ]
    },
    {
      id: "gated-delta",
      order: 10,
      title: "DeltaNet",
      fullTitle: "DeltaNet & Gated DeltaNet",
      zhTitle: "从累加记忆到在线纠错：为什么 Delta Rule 更会检索",
      year: "2024–2025",
      category: "linear",
      difficulty: "前沿",
      report: "Parallelizing Linear Transformers with the Delta Rule · NeurIPS 2024",
      deck: R`线性注意力把历史压进 \(d_k\times d_v\) 的 fast-weight 矩阵，省掉随长度增长的 KV cache，却会把相似 key 的 value 不断叠加成检索噪声。DeltaNet 不再盲目追加：它先读出当前预测，再只写入目标与预测之间的误差；现代架构再用 L2Norm、ShortConv、输出归一化和 chunkwise WY/UT 把这个在线学习器变成可训练的语言模型模块。`,
      takeaway: R`把状态看成一个在线回归器：线性注意力执行 \(S\leftarrow S+kv^\top\)，DeltaNet 则对 \(\frac12\|S^\top k-v\|^2\) 做一步梯度下降。单位范数 key、\(\beta=1\) 时，它会精确改写当前 key 的读出；但固定状态仍有容量上限，所以全局 attention 是重要的延伸方向。`,
      sectionTitles: {
        motivation: "为什么线性记忆会过载",
        constraints: "Delta rule 解决什么，又没有解决什么",
        intuitions: "把状态看成在线回归器",
        diagram: "从纠错写入到现代 DeltaNet Block",
        position: "现代 DeltaNet 的神经架构",
        derivations: "从纯 DeltaNet 到 Gated DeltaNet：纠错与遗忘",
        exercises: "练习：干扰、稳定、并行与混合",
        sources: "原论文、三篇作者博客与实现"
      },
      motivation: [
        R`先看线性注意力为什么诱人：结合律把全部历史折叠为 \(S_t=\sum_{i\le t}k_iv_i^\top\)，单步只更新/读取固定矩阵；当 \(L\gg d\) 时，Decode 不再反复搬运 \(O(Ld)\) 的 KV cache，矩阵状态还把 RNN 记忆从 \(O(d)\) 便宜地扩到 \(O(d^2)\)。`,
        R`再看代价：用单位 key \(k_j\) 读取时，\(S^\top k_j=v_j+\sum_{i\ne j}(k_i^\top k_j)v_i\)。只有互相正交的 key 才没有串扰，而 \(d_k\) 维空间最多容纳 \(d_k\) 个正交方向；序列继续增长时，“其他记忆”会成为检索误差。`,
        R`Delta rule 的动机不是多加一个 gate，而是让记忆可修改：它先算旧预测 \(\hat v_t=S_{t-1}^\top k_t\)，再写 \(\beta_tk_t(v_t-\hat v_t)^\top\)。已经记住的内容几乎不写，错误答案才被擦除并替换；Gated DeltaNet 的 \(\alpha_t\) 是后续加入的时间遗忘机制。`
      ],
      constraints: [
        { label: "容量上限", title: "纠错不等于无限无损记忆", body: "Delta update 减少同方向覆盖冲突，但相近 key 仍会互相影响；固定 \(d_k\times d_v\) 状态不可能精确保留任意长历史。" },
        { label: "训练执行", title: "可结合不代表朴素 scan 高效", body: "把每步 transition 当稠密矩阵做 parallel scan 会产生高昂矩阵乘和 \(O(Ld^2)\) 中间状态写回；实用路线是 chunk checkpoint + WY/UT。" },
        { label: "延伸性", title: "局部混合与全局回看解决不同问题", body: "ShortConv/滑窗增强局部顺序但仍受固定范围限制；少量 global-attention 层恢复显式远程读取，代价是重新引入部分 KV cache 与平方项。" }
      ],
      intuitions: [
        { label: "Fast weights", title: "状态就是在线训练的权重", body: R`token 内的 \(k_t\) 是训练样本，\(v_t\) 是目标，\(S\) 是上下文内不断更新的线性回归器；query \(q_t\) 用同一组 fast weights 读答案。` },
        { label: "Projection", title: "单位 key 定义一个定向橡皮擦", body: R`\(\beta=1,\|k\|=1\) 时 \(I-kk^\top\) 擦掉状态在 key 方向上的旧分量，却保留正交方向；随后 \(kv^\top\) 写入新关联。` },
        { label: "Extension", title: "固定状态负责概括，attention 负责精确回看", body: "DeltaNet 擅长用固定预算维护可修改摘要；当任务要求逐字复制、多针检索或任意远程引用时，稀疏插入 global attention 更稳妥。" }
      ],
      diagram: { type: "delta", caption: "DeltaNet 的完整因果链：图中省略 head 下标，按单个 head 展示；实际实现中多个 head 并行，每头维护独立的 Q/K/V、状态矩阵与 α/β。加法 fast-weight memory 会累积串扰；delta update 先预测、再沿 key 方向擦除/纠写。现代 block 用 Q/K L2Norm、ShortConv 与输出 RMSNorm 提升稳定和局部寻址，训练则用 chunk checkpoint + WY/UT 把递推改写成矩阵乘；固定状态上限仍可由少量 global attention 补足。" },
      derivations: [
        {
          title: "第一步：把记忆状态看成在线线性回归器",
          body: R`采用作者常用的转置状态
            \[
            F_t=S_t^\top\in\mathbb R^{d_v\times d_k},\qquad
            \hat v=F_tk.
            \]
            当前 token 给出一组在线训练样本 \(k_t\mapsto v_t\)：key \(k_t\) 是输入，value \(v_t\) 是目标，而 \(F\) 是在上下文内不断更新的 fast weights。更新之前，旧状态的预测和误差分别为
            \[
            \hat v_t=F_{t-1}k_t,\qquad
            e_t=v_t-\hat v_t=v_t-F_{t-1}k_t.
            \]
            这一步揭示了 DeltaNet 与加法线性 attention 的根本区别：如果旧预测已经正确，则 \(e_t\approx0\)，无需重复写入；只有预测错误的部分才需要进入状态。`
        },
        {
          title: "第二步：对瞬时 MSE 做一步梯度下降",
          body: R`为了让当前 key 的读出接近目标 value，定义瞬时损失
            \[
            \ell_t(F)=\frac12\|Fk_t-v_t\|^2.
            \]
            对矩阵 \(F\) 求梯度：
            \[
            \nabla_F\ell_t=(Fk_t-v_t)k_t^\top.
            \]
            在旧状态 \(F_{t-1}\) 上，以 \(\beta_t\) 为步长做一次梯度下降：
            \[
            \begin{aligned}
            F_t
            &=F_{t-1}-\beta_t\nabla_F\ell_t(F_{t-1})\\
            &=F_{t-1}
              +\beta_t\big(v_t-F_{t-1}k_t\big)k_t^\top.
            \end{aligned}
            \]
            因而纯 DeltaNet 的更新式是
            \[
            \boxed{
            F_t=F_{t-1}
            +\beta_t\big(v_t-F_{t-1}k_t\big)k_t^\top
            }.
            \]
            它不是直接追加 \(v_tk_t^\top\)，而是写入“目标减旧预测”的误差。`
        },
        {
          title: "第三步：把 Delta Rule 改写成定向擦除与写入",
          body: R`展开纯 DeltaNet 更新：
            \[
            \begin{aligned}
            F_t
            &=F_{t-1}
              -\beta_tF_{t-1}k_tk_t^\top
              +\beta_tv_tk_t^\top\\
            &=F_{t-1}(I-\beta_tk_tk_t^\top)
              +\beta_tv_tk_t^\top.
            \end{aligned}
            \]
            第一项沿 \(k_t\) 方向擦除旧关联，第二项沿同一方向写入新 value。若 \(\|k_t\|=1\)，令 \(\hat v_t=F_{t-1}k_t\)，更新后用同一 key 读取：
            \[
            F_tk_t
            =(1-\beta_t)\hat v_t+\beta_tv_t.
            \]
            因此 \(\beta_t=0\) 表示不更新，\(0<\beta_t<1\) 在旧预测与新目标之间插值，\(\beta_t=1\) 则精确得到 \(F_tk_t=v_t\)。\(\beta_t\) 控制的是当前 key 方向的改写强度。`
        },
        {
          title: "第四步：为什么要衰减旧记忆，并得到 Gated DeltaNet",
          body: R`纯 Delta Rule 只能修改与当前 \(k_t\) 重合的方向。若旧记忆位于方向 \(u\)，且 \(u^\top k_t=0\)，那么
            \[
            (I-\beta_tk_tk_t^\top)u=u.
            \]
            即使这段记忆已经过时，当前更新也完全碰不到它。随着序列增长，过时关联会长期占用固定容量并增加检索串扰；语言中的主题、实体与局部语境却会不断变化，因此状态还需要主动遗忘。

            Gated DeltaNet 先用每头标量 retention \(\alpha_t\in(0,1)\) 衰减旧状态：
            \[
            \widetilde F_{t-1}=\alpha_tF_{t-1}.
            \]
            再对衰减后的状态执行同一个 Delta update：
            \[
            \begin{aligned}
            F_t
            &=\widetilde F_{t-1}
              +\beta_t\big(v_t-\widetilde F_{t-1}k_t\big)k_t^\top\\
            &=\alpha_tF_{t-1}
              +\beta_t\big(v_t-\alpha_tF_{t-1}k_t\big)k_t^\top\\
            &=\boxed{
              \alpha_tF_{t-1}(I-\beta_tk_tk_t^\top)
              +\beta_tv_tk_t^\top
            }.
            \end{aligned}
            \]
            一段历史从位置 \(i\) 传到 \(t\) 时会累乘约
            \(\prod_{r=i+1}^{t}\alpha_r\)：\(\alpha\approx1\) 支持长期保留，\(\alpha\approx0\) 允许快速重置。两种门分工不同——\(\beta_t\) 负责沿当前 key 方向纠错，\(\alpha_t\) 负责让整个旧状态随上下文需要而遗忘。`
        },
        /*
         * Temporarily hidden: the previous section content is retained here for
         * later restoration after the DeltaNet-to-Gated-DeltaNet walkthrough.
        {
          title: "线性 fast-weight memory 的检索误差从哪里来",
          body: R`取本站状态约定
            \[
            S_t=\sum_{i\le t}k_iv_i^\top\in\mathbb R^{d_k\times d_v},
            \qquad o_t=S_t^\top q_t.
            \]
            若用单位 key \(q_t=k_j\) 查询已写入的第 \(j\) 项，则
            \[
            S_t^\top k_j
            =\sum_i v_i(k_i^\top k_j)
            =v_j+\underbrace{\sum_{i\ne j}(k_i^\top k_j)v_i}_{\text{retrieval error}}.
            \]
            误差由 key 的非正交性直接产生。\(d_k\) 维空间最多只有 \(d_k\) 个两两正交方向，因此增加序列长度而不增加状态维度，最终必然让关联发生碰撞。线性 attention 的加法写入还无法删除旧关联，所以相同 key 改绑新 value 时会把两个答案叠加。`
        },
        {
          title: "Delta rule 是一步在线 MSE 梯度下降",
          body: R`令状态 \(S\in\mathbb R^{d_k\times d_v}\)，希望 \(S^\top k_t\approx v_t\)。瞬时损失
            \[
            \ell_t(S)=\frac12\|S^\top k_t-v_t\|^2.
            \]
            其梯度为 \(k_t(S^\top k_t-v_t)^\top\)，做步长 \(\beta_t\) 的一步更新：
            \[
            S_t=S_{t-1}+\beta_t k_t\big(v_t-S_{t-1}^\top k_t\big)^\top.
            \]
            这既是在线回归，也是最简 TTT-linear 视角：状态不是静态摘要，而是上下文中被每个样本训练的 fast weights。对照之下，加法线性 attention 的 \(S_t=S_{t-1}+k_tv_t^\top\) 可视为对线性目标 \(-\langle S^\top k_t,v_t\rangle\) 做一步更新；梯度里没有“当前预测减目标”，所以无论旧关联是否已经正确都会继续追加。

            若 \(\|k_t\|=1,\beta_t=1\)，则
            \[
            S_t^\top k_t=S_{t-1}^\top k_t+
            (v_t-S_{t-1}^\top k_t)(k_t^\top k_t)=v_t,
            \]
            即当前 key 的读出被精确改写。`
        },
        {
          title: "固定状态的收益与混合模型的代价",
          body: R`单头 recurrent state 为 \(S\in\mathbb R^{d_k\times d_v}\)，与上下文长度 \(L\) 无关。逐 token Decode 的写入和读取均为
            \[
            \Theta(d_kd_v),\qquad
            M_{\rm state}=\Theta(d_kd_v),
            \]
            而 dense softmax Decode 每步读取 \(\Theta(Ld_v)\) 的历史 KV。前者在 \(L\gg d_k\) 时更适合 memory-bound generation，但固定状态也给精确回看设了上限。

            插入固定窗口 \(w\) 的 attention 仍保持 \(\Theta(Lw)\) 总位置交互，却看不到窗口外原文；插入 \(m\) 个 global-attention 层会增加约 \(\Theta(mL^2d_h)\) 训练算量和 \(\Theta(mLd_h)\) KV cache，但让少数层可精确访问任意历史位置。作者博客 Part III 的 1.3B/100B-token 对比中，平均 retrieval 从纯 DeltaNet 的 34.7 提升到滑窗混合的 39.6、两个 global 层的 47.9；同表 Transformer++ 为 41.8。关键延伸结论因此不是“纯 RNN 或纯 attention 二选一”，而是用少量 global 层购买精确检索。`
        }
        */
      ],
      warning: R`必须区分三层结论：原始 DeltaNet 只有 \(\beta_t\) 纠错写入；Gated DeltaNet 另加每头标量遗忘 \(\alpha_t\)；KDA 再把遗忘细化到 key channel。本站使用 \(S\in\mathbb R^{d_k\times d_v}\)，而作者博客常用转置状态 \(F=S^\top\)。现代 block 只有 q/k/v 经过 ShortConv，\(\alpha,\beta\) 走 direct projection。1.3B 公开材料对 head count/head dim 有冲突，因此不展示伪精确参数卡。`,
      exercises: [
        {
          q: R`线性 fast-weight state 已写入 \((k_1,v_1)\)、\((k_2,v_2)\)，且 \(\|k_1\|=\|k_2\|=1,\ k_1^\top k_2=\rho\)。用 \(q=k_1\) 读取时得到什么？何时无串扰？`,
          hint: R`展开 \(S^\top k_1\)，分别保留 \(i=1,2\) 两项。`,
          answer: R`\(S=k_1v_1^\top+k_2v_2^\top\)，所以 \(S^\top k_1=v_1+\rho v_2\)。只有 \(\rho=0\)，即两个 key 正交时，第二项不污染第一个 value。`
        },
        {
          q: R`若 \(\|k\|=1\)，推导 Delta update 后 \(S_t^\top k\) 与旧预测、目标 \(v\) 的关系；解释 \(\beta=0,1\) 两个端点。`,
          hint: R`令 \(\hat v=S_{t-1}^\top k\)，代入 \(S_t=S_{t-1}+\beta k(v-\hat v)^\top\)。`,
          answer: R`\(S_t^\top k=\hat v+\beta(v-\hat v)=(1-\beta)\hat v+\beta v\)。\(\beta=0\) 完全保留旧关联，\(\beta=1\) 将当前 key 的读出精确改写为 \(v\)；中间值是在线插值。`
        }
      ],
      sources: [
        { label: "Yang et al. (2024), Parallelizing Linear Transformers with the Delta Rule over Sequence Length", url: "https://arxiv.org/abs/2406.06484" },
        { label: "Songlin Yang, DeltaNet Explained (Part I): Model and motivation", url: "https://sustcsonglin.github.io/blog/2024/deltanet-1/" },
        { label: "Songlin Yang, DeltaNet Explained (Part II): Chunkwise WY/UT algorithm", url: "https://sustcsonglin.github.io/blog/2024/deltanet-2/" },
        { label: "Songlin Yang, DeltaNet Explained (Part III): Neural architecture and hybrids", url: "https://sustcsonglin.github.io/blog/2024/deltanet-3/" },
        { label: "Yang, Kautz & Hatamizadeh (2025), Gated Delta Networks", url: "https://arxiv.org/abs/2412.06464" },
        { label: "Flash Linear Attention official DeltaNet kernels", url: "https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/delta_rule" }
      ]
    },
    {
      id: "kda",
      order: 11,
      title: "KDA",
      fullTitle: "Kimi Delta Attention",
      zhTitle: "Kimi Delta Attention：逐通道控制记忆",
      year: "2025",
      category: "hybrid",
      difficulty: "前沿",
      report: "Kimi Linear Technical Report",
      deck: "KDA 把 Gated DeltaNet 每头一个标量遗忘率升级为逐 key 通道的对角 gate，并把转移限制为高效 diagonal-plus-rank-1；发布的 27 层 checkpoint 含 20 个 KDA 层和 7 个 MLA 层。",
      takeaway: R`KDA 用 \(\operatorname{Diag}(\alpha_t)\) 做逐通道衰减、用 rank-1 delta 改写；周期性 MLA 保留 token-indexed、全局 dense softmax，但其每 token KV 仍是低秩缓存，不是“未压缩 MLA”。`,
      motivation: [
        "Gated DeltaNet 的 αt 是每头标量，头内所有 key 通道共享同一遗忘速度；这限制了同时追踪短期句法与长期主题的能力。",
        R`KDA 使用对角 gate \(\operatorname{Diag}(\alpha_t)\)，每个 key 通道有独立衰减；\(\beta_t=\sigma(W^\beta x_t)\) 由直接线性路径产生，ShortConv 只作用于 q/k/v。`,
        "一般 DPLR 转移表达力强但 chunk 并行昂贵。KDA 约束低秩项与 key 绑定，减少高精度二级 chunk 与额外 matmul，兼顾表达力和硬件效率。"
      ],
      constraints: [
        { label: "数值", title: "细粒度累积易不稳定", body: "逐通道衰减在长 chunk 的乘除中会产生精度问题，需要专门 UT/WY 形式。" },
        { label: "算子", title: "收益依赖定制 kernel", body: "简单 Python 递推无法体现 KDA 的吞吐优势；官方开源 FLA kernel 与 vLLM 实现。" },
        { label: "容量", title: "仍需 token-indexed global softmax", body: "Kimi Linear 周期插入 NoPE MLA：它全局读取每个历史 token 的低秩 MLA cache，而不是取消 cache 或使用未压缩 MHA。" }
      ],
      intuitions: [
        { label: "Channel gate", title: "一排不同速度的沙漏", body: "每个通道有数据依赖衰减；恒定 gate 的半衰期只是一种局部诊断。" },
        { label: "Delta", title: "同地址定点覆盖", body: "rank-1 更新减少相似 key 之间的污染。" },
        { label: "Hybrid", title: "三次压缩记忆，一次全局翻档", body: "大部分层便宜递推，周期性 MLA 做精确全局纠偏。" }
      ],
      diagram: { type: "kda", caption: "KDA 的 diagonal gate + rank-1 delta；发布 checkpoint 为 27 层尾部调度，共 20 KDA + 7 NoPE MLA。" },
      derivations: [
        {
          title: "KDA 的核心递推",
          body: R`Kimi Linear 报告给出的列向量约定为
            \[
            S_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
            +\beta_tk_tv_t^\top,\qquad
            o_t=S_t^\top q_t.
            \]
            与 Gated DeltaNet 相比，标量 \(\alpha_t\) 被向量 gate 替代；状态转移是“对角 + rank-1”的受限 DPLR。`
        },
        {
          title: "逐通道半衰期",
          body: R`仅作 decay-only 诊断：忽略 rank-1 delta，且假设第 r 个 gate 恒定时，
            \[
            S_{t,r}\approx \alpha_{t,r}S_{t-1,r}.
            \]
            若 \(\alpha_{t,r}=\alpha_r\) 近似恒定，其半衰期为
            \[
            \tau_{1/2,r}=\frac{\log 0.5}{\log \alpha_r}.
            \]
            该式描述单独乘法衰减的 e-fold/half-life，不是完整 KDA 记忆的保证；真实 \(\alpha_{t,r}\) 随 token 变化，delta 写入与通道耦合也会改变可检索寿命。`
        },
        {
          title: "发布 checkpoint 的尾部调度与缓存",
          body: R`发布的 27 层 Kimi-Linear-48B-A3B 配置以 1-based 层号列出：
            \[
            \mathrm{MLA}=\{4,8,12,16,20,24,27\},
            \]
            其余 20 层为 KDA。前 24 层近似 3:1，末层追加 MLA；因此精确总数是 20 KDA + 7 MLA，而非恰好四分之一。KDA 保存固定状态，7 个 MLA 层保存随 L 增长的 token-indexed low-rank cache。`
        }
      ],
      warning: "“最高 6×/6.3× 吞吐、最多 75% KV cache 降低”是 Kimi 团队对指定 48B checkpoint、1M 上下文、硬件、batch、kernel 与 MLA 基线的官方自报。NoPE 只表示 MLA 层不显式加位置编码，不表示模型无顺序信息或 MLA 无 token 轴。",
      exercises: [
        {
          q: R`某通道 \(\alpha=0.999\)，近似半衰期是多少步？`,
          hint: R`用 \(\log(0.5)/\log(0.999)\)。`,
          answer: "decay-only、恒定 α 近似下约 693 步；α=0.9 时约 6.58 步。真实 KDA 的 α 随 token 变化且有 delta 写入，所以这些不是端到端可检索记忆寿命。"
        },
        {
          q: R`比较 \(A_t=(I-\beta kk^\top)\operatorname{Diag}(\alpha)\) 与一般 DPLR \(D-ab^\top\)。KDA 约束了什么？`,
          hint: "观察 rank-1 项左右向量与 k、α 的关系。",
          answer: "KDA 的低秩项不是自由 a、b，而与同一个 key k 及对角 gate 绑定；表达空间更受限，但 chunk 算法更稳定、高效。"
        },
        {
          q: "发布的 27 层 Kimi Linear checkpoint 如何实现约 3:1，它能否理解成层内 75% KDA、25% MLA？",
          hint: "检查 1-based full-attention 层号与最后一层。",
          answer: R`它是 layerwise：MLA 在 \(\{4,8,12,16,20,24,27\}\)，其余 20 层为 KDA。前 24 层重复 3 KDA + 1 MLA，末层再接 MLA；不是层内混合，精确总比为 20:7。`
        }
      ],
      sources: [
        { label: "Kimi Team (2025), Kimi Linear: An Expressive, Efficient Attention Architecture", url: "https://arxiv.org/abs/2510.26692" },
        { label: "MoonshotAI official Kimi-Linear repository", url: "https://github.com/MoonshotAI/Kimi-Linear" },
        { label: "Flash Linear Attention official KDA kernels", url: "https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda" },
        { label: "Released 48B checkpoint config (27 layers, 20 KDA + 7 MLA)", url: "https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/main/config.json" }
      ]
    }
  ];

  // Keep the base chapter records readable.  Detailed, cross-cutting teaching
  // material lives in this id-keyed pass so that its schema can evolve without
  // duplicating or rewriting the original chapter objects above.
  var chapterEnhancements = {
    mha: {
      attentionConfig: {
        model: "Transformer base · WMT14 English–German",
        scope: "《Attention Is All You Need》表 3 的 base 配置；数值属于该实验，不是 MHA 的固定常数。",
        items: [
          { label: "Hidden size", value: "512", note: "d_model" },
          { label: "Q heads", value: "8", note: "独立 query heads" },
          { label: "KV heads", value: "8", note: "每个 Q head 对应一套 K/V" },
          { label: "Q / K / V head dim", value: "64 / 64 / 64", note: "512 ÷ 8" },
          { label: "Position", value: "512-d sinusoidal PE", note: "与 token embedding 相加" },
          { label: "KV cache width", value: "1024 elements / token / layer", note: "由 2 × 8 × 64 推导" }
        ],
        sources: [
          { label: "Vaswani et al. (2017), §§3.2.2, 3.5 and Table 3", url: "https://arxiv.org/abs/1706.03762" }
        ],
        caveat: "论文没有把 KV cache width 作为超参数列出；1024 是按论文维度计算的 K 与 V 元素总数。"
      },
      positionEncoding: {
        title: "原始 Transformer：缩放 embedding + 加性正弦 PE",
        summary: R`《Attention Is All You Need》把 token embedding 乘 \(\sqrt{d_{\text{model}}}\)，再加固定正弦位置编码；learned PE 只是对照实验且结果接近。每个 encoder/decoder sublayer 采用 post Add&Norm。MHA 算子本身不强制这些外围选择。`,
        equation: R`\[
          x_p=\sqrt{d_{\text{model}}}\,E[\mathrm{token}_p]+\operatorname{PE}(p),
          \qquad
          \operatorname{PE}(p,2i)=\sin\!\left(p/10000^{2i/d_{\text{model}}}\right),\qquad
          \operatorname{PE}(p,2i+1)=\cos\!\left(p/10000^{2i/d_{\text{model}}}\right).
        \]
        \[
          \operatorname{SublayerOut}=\operatorname{LayerNorm}
          \bigl(x+\operatorname{Dropout}(\operatorname{Sublayer}(x))\bigr).
        \]`,
        steps: [
          { label: "Embedding", title: R`先乘 \(\sqrt{d_{\text{model}}}\)`, body: "原论文对 encoder/decoder 的 token embedding 都做幅度缩放后再加 PE；不能漏掉这个因子。" },
          { label: "作用域", title: "编码器、解码器、交叉注意力不同", body: "编码器 self-attention 双向读源序列；decoder self-attention 有 causal mask；cross-attention 的 Q 来自 decoder、K/V 来自 encoder。三者都可用 MHA。" },
          { label: "残差", title: "原论文是 post Add&Norm", body: "每个 attention/FFN sublayer 先计算子层并加残差，再做 LayerNorm；现代 pre-norm 不能倒写成 2017 结构。" }
        ],
        caveat: "正弦 PE、embedding 缩放与 post Add&Norm 属于原始 Transformer 配方，不是 MHA 数学定义；现代 pre-norm/RoPE 模型仍可使用 MHA。"
      },
      derivationSourceFallback: "Vaswani et al. (2017), §3.1（encoder/decoder 与 Add&Norm）、§3.2（attention）、§3.4（embedding scale）与 §3.5（PE）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "derivation", level: "foundation" }
      ],
      derivations: [
        {
          title: "稳定 softmax 不改变注意力概率",
          body: R`**原式。** 对一行分数 \(s\in\mathbb R^L\)，
            \[
            a_j=\frac{e^{s_j}}{\sum_{\ell=1}^{L}e^{s_\ell}}.
            \]
            **补全代数。** 取 \(m=\max_\ell s_\ell\)，分子分母同乘 \(e^{-m}\)：
            \[
            a_j=\frac{e^{s_j-m}}{\sum_\ell e^{s_\ell-m}}.
            \]
            因此减去行最大值与原式严格等价，同时所有指数输入不大于 0。**张量形状。** 若
            \(Q,K\in\mathbb R^{B\times H\times L\times d_h}\)，则
            \(S=QK^\top/\sqrt{d_h}+M\) 与 \(A=\operatorname{softmax}(S)\) 都是
            \(\mathbb R^{B\times H\times L\times L}\)。**直观。** 只平移每行 logits，不改变相对比值，却避免指数上溢。**边界。** 被 mask 的位置应在 softmax 前置为 \(-\infty\)（或足够小的有限值）；全被 mask 的行需单独处理，否则会产生 NaN。`,
          source: "Vaswani et al. (2017), §3.2.1；稳定化步骤是 softmax 的标准等价实现"
        },
        {
          title: "prefill 与单步 decode 的成本不是同一个式子",
          body: R`**原式。** 单头核心为 \(A=\operatorname{softmax}(QK^\top)V\)。**补全代数。**
            prefill 中 \(QK^\top\) 与 \(AV\) 各需约 \(L^2d_h\) 次乘加；跨 \(H\) 头得到
            \[
            C_{\text{prefill}}=\Theta(HL^2d_h)=\Theta(L^2d_{\text{model}}).
            \]
            单步 decode 只有一个新 query，故
            \[
            C_{\text{decode,step}}=\Theta(HLd_h)=\Theta(Ld_{\text{model}}).
            \]
            **张量形状。** prefill score 为 \([B,H,L,L]\)，单步 score 为 \([B,H,1,L]\)，缓存 K/V 为
            \([B,H,L,d_h]\)。**直观。** decode 不再创建完整方阵，但每一步仍读取全部历史 KV。**边界。**
            这些是核心 attention 的算术量；FlashAttention 改善 IO 和中间存储，不改变精确 dense attention 的位置对数量。`,
          source: "Vaswani et al. (2017), §3.2.1；Dao et al. (2022), §2–3"
        }
      ],
      exercises: [
        {
          kind: "counterexample",
          level: "intermediate",
          q: R`构造一个反例说明：两个注意力头输出完全相同，并不意味着它们的 \(W^Q,W^K,W^V\) 必须相同。`,
          hint: "利用输出投影可以忽略某个头，或利用投影的可逆基变换。",
          answer: R`例如令输出投影中第二个头对应的列全为 0，则第二头无论使用什么 \(W_2^Q,W_2^K,W_2^V\)，最终 \(Y\) 都不受它影响。故从最终函数相同不能反推出各头参数相同；参数化存在不可辨识性。`
        },
        {
          kind: "code-shape",
          level: "intermediate",
          q: R`实现中 \(X\) 为 \([B,L,d]\)，一次线性层得到 \([B,L,3Hd_h]\)。写出拆成 Q/K/V 并转为 attention 布局后的形状顺序。`,
          hint: R`先 reshape 为 \([B,L,3,H,d_h]\)，再移动 head 轴。`,
          answer: R`可执行 reshape \([B,L,3,H,d_h]\)，沿第 3 轴 unbind 得三份 \([B,L,H,d_h]\)，再 transpose 为 \([B,H,L,d_h]\)。必须在 transpose 后保证 contiguous 要求与所用 kernel 一致。`
        },
        {
          kind: "design",
          level: "advanced",
          q: "若服务受 KV 带宽限制，但训练质量不能明显下降，应优先减少 Q 头还是 KV 头？说明实验设计。",
          hint: "Q 头与 KV 头可以解耦；需要质量和端到端吞吐两类指标。",
          answer: "优先保持 Q 头并逐步减少 KV 头，形成 GQA/MQA 消融。固定参数量与训练 token，比较验证损失/下游质量、每 token KV 字节、目标 batch 与上下文下的 TPOT；不能用理论 cache 比值替代实际吞吐。"
        },
        {
          kind: "derivation",
          level: "advanced",
          q: R`证明给所有 score 加同一常数 \(c\) 不改变 softmax，并说明为什么给不同 key 加 \(c_j\) 会改变结果。`,
          hint: R`把公共因子 \(e^c\) 从分子分母约掉。`,
          answer: R`\(\operatorname{softmax}(s+c\mathbf1)_j=e^{s_j}e^c/(e^c\sum_\ell e^{s_\ell})=\operatorname{softmax}(s)_j\)。若偏移依赖 \(j\)，因子 \(e^{c_j}\) 不能作为公共项约掉，相对 odds 变为 \(e^{s_j-s_k+c_j-c_k}\)。`
        }
      ]
    },

    mqa: {
      attentionConfig: {
        model: "Multi-query Transformer · WMT14 English–German",
        scope: "Shazeer (2019) 的 encoder–decoder 代表实验；三类 attention 都替换为 MQA。",
        items: [
          { label: "Hidden size", value: "1024", note: "模型维度" },
          { label: "Q heads", value: "8", note: "query 仍保持多头" },
          { label: "KV heads", value: "1", note: "全部 Q heads 共享" },
          { label: "Q / K / V head dim", value: "128 / 128 / 128", note: "每头维度" },
          { label: "Position", value: "Learned absolute PE", note: "论文未明确给出 PE 向量维度" },
          { label: "KV cache width", value: "256 elements / token / layer", note: "由 1 × 128 K + 1 × 128 V 推导" }
        ],
        sources: [
          { label: "Shazeer (2019), §§3–4.1 and Table 1", url: "https://arxiv.org/abs/1911.02150" }
        ],
        caveat: "H_kv=1 是 MQA 定义；8 heads、128 head dim 与 learned PE 是该论文实验配置。"
      },
      positionEncoding: {
        title: "MQA 论文评测 learned absolute PE；机制仍与 PE 正交",
        summary: R`Shazeer 的 WMT 基线明确使用 learned positional embeddings，并把 encoder self-attention、decoder self-attention 和 cross-attention 全部替换为 MQA；decoder-only LM 也另行评测。MQA 定义只规定共享 K/V，不能虚构一个适用于所有 PE 的通用加性 \(B_{\rm pos}\)。`,
        equation: R`\[
          \text{absolute: }X_p=E[\mathrm{token}_p]+P_p,\quad
          S_h=\frac{Q_hK^\top}{\sqrt{d_h}}+M;
        \]
        \[
          \text{relative bias: }S_{h,t,s}=\frac{q_{h,t}^\top k_s}{\sqrt{d_h}}+b_{h,t,s}+M_{t,s};
        \]
        \[
          \text{RoPE: }S_{h,t,s}=\frac{(R_tq_{h,t})^\top(R_sk_s)}{\sqrt{d_h}}+M_{t,s}.
        \]`,
        steps: [
          { label: "原评测", title: "learned embedding 在 attention 前相加", body: "论文 §4.1 的 WMT encoder-decoder baseline 使用 learned positional embeddings；这与相对 score bias 不是同一种计算。" },
          { label: "论文范围", title: "三类 encoder-decoder attention 都替换", body: "MQA 并非只在 decoder self-attention 上实验；但带宽收益最突出的是增量 decoder。" },
          { label: "现代形式", title: "bias 与 RoPE 必须分别写", body: R`相对 bias 是 score 加项；RoPE 改写 Q/K 后再点积。二者不能统一伪装成一个没有来源与 head 轴定义的 \(B_{\rm pos}\)。` }
        ],
        caveat: "不能从“MQA”推断 RoPE base、缩放方法或是否使用相对 bias；这些都由具体 checkpoint 决定。"
      },
      derivationSourceFallback: "Shazeer (2019), §3（multi-query attention）与 §4.1（learned PE、三类 attention 的实验范围）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "counterexample", level: "foundation" }
      ],
      derivations: [
        {
          title: "共享 K/V 的广播无需真实复制",
          body: R`**原式。** \(O_h=\operatorname{softmax}(Q_hK^\top/\sqrt{d_h})V\)。**补全代数。**
            把全部 query 写成 \(Q\in\mathbb R^{B\times H_q\times L_q\times d_h}\)，共享 key 写成
            \(K\in\mathbb R^{B\times1\times L_k\times d_h}\)，批量乘法按 head 轴广播：
            \[
            S_{b,h,t,s}=\sum_{r=1}^{d_h}Q_{b,h,t,r}K_{b,1,s,r}.
            \]
            **张量形状。** \(S\in\mathbb R^{B\times H_q\times L_q\times L_k}\)，但物理 K/V 缓存仍是
            \([B,1,L_k,d_h]\)。**直观。** 多位读者使用同一目录，不必复制目录 \(H_q\) 次。**边界。**
            某些通用 kernel 会显式 repeat K/V 以适配 MHA 接口，这保持数值等价却会丢掉部分内存/带宽收益。`,
          source: "Shazeer (2019), multi-query attention 定义与增量解码讨论"
        },
        {
          title: "MQA 只消除 KV 头因子，不消除 query 头因子",
          body: R`**原式。** 单步 score 含 \(H_qL\) 个点积。**补全代数。**
            \[
            C_{\text{score}}=\Theta(BH_qLd_h),\qquad
            R_{\text{KV}}=2BLd_hb.
            \]
            MHA 的读取量近似 \(2BLH_qd_hb\)，故共享消除了读取式中的 \(H_q\)，但 score 计算仍要为每个 query 头执行。
            **张量形状。** query 为 \([B,H_q,1,d_h]\)，cache 为两份 \([B,1,L,d_h]\)，输出为
            \([B,H_q,1,d_h]\)。**直观。** 问题数量没有减少，只减少被反复搬运的资料副本。**边界。**
            端到端还包含投影、MLP、通信与 kernel 启动；因此 \(H_q\) 倍 KV 降幅不是 \(H_q\) 倍模型加速。`,
          source: "Shazeer (2019), arithmetic-intensity 与 decoding bandwidth 分析"
        }
      ],
      exercises: [
        {
          kind: "derivation",
          level: "intermediate",
          q: R`若把共享 key 显式 repeat 成 \(K'\in\mathbb R^{B\times H_q\times L\times d_h}\)，证明其 score 与广播实现相同，并比较额外元素数。`,
          hint: R`定义 \(K'_{b,h,s,:}=K_{b,1,s,:}\)。`,
          answer: R`逐元素有 \(Q_{b,h,t,:}^{\top}K'_{b,h,s,:}=Q_{b,h,t,:}^{\top}K_{b,1,s,:}\)，故 score 相同。repeat 后保存 \(BH_qLd_h\) 个 key 元素而非 \(BLd_h\)，额外 \(B(H_q-1)Ld_h\) 个。`
        },
        {
          kind: "code-shape",
          level: "intermediate",
          q: R`一个 MQA kernel 接收 Q 为 \([B,H,1,d_h]\)、K/V 为 \([B,L,d_h]\)。为了广播但不复制，应给 K/V 增加哪一轴？`,
          hint: "在 head 位置插入长度 1 的轴。",
          answer: R`变为 \([B,1,L,d_h]\)，让 head 轴从 1 广播到 \(H\)。若后端要求 stride 合法，应使用支持 broadcast/stride-0 的算子，而非 materialize 的 repeat。`
        },
        {
          kind: "design",
          level: "advanced",
          q: "什么场景下 MQA 的较小 cache 可能仍不比 GQA 快？",
          hint: "考虑 kernel、并行切分、batch 和非注意力耗时。",
          answer: "若 MQA kernel 需要显式复制 K/V、单 KV 头造成张量并行通信/负载不佳、上下文很短或 MLP 占主导，额外缩小 cache 可能无法转化为 TPOT。应在目标硬件和真实服务 batch 上测量。"
        },
        {
          kind: "complexity",
          level: "advanced",
          q: R`prefill 长度为 \(L\) 时，MQA 的 score 元素数和 KV 投影输出元素数分别如何随 \(L,H_q\) 缩放？`,
          hint: "score 仍有每个 query 头的完整位置对。",
          answer: R`score 为 \(BH_qL^2\) 个元素，仍是 \(\Theta(H_qL^2)\)；K 与 V 投影输出合计 \(2BLd_h\) 个元素，是 \(\Theta(Ld_h)\) 且无 \(H_q\) 因子。`
        }
      ]
    },

    gqa: {
      attentionConfig: {
        model: "GQA-8-XXL · T5.1.1-XXL uptraining",
        scope: "论文主实验的 decoder self-attention 与 cross-attention；encoder self-attention 仍为 64-head MHA。",
        items: [
          { label: "Hidden size", value: "4096", note: "T5.1.1-XXL emb_dim" },
          { label: "Q heads", value: "64", note: "decoder query heads" },
          { label: "KV heads / groups", value: "8", note: "GQA-8" },
          { label: "Q / K / V head dim", value: "64 / 64 / 64", note: "官方 T5X 配置" },
          { label: "Group ratio", value: "8 Q heads / KV head", note: "64 ÷ 8" },
          { label: "Position bias", value: "32 buckets · max distance 128", note: "T5 self-attention；cross-attention 不套用此项" },
          { label: "KV cache width", value: "1024 elements / token / layer", note: "由 2 × 8 × 64 推导" }
        ],
        sources: [
          { label: "Ainslie et al. (2023), §§2.2–3.1 and Table 1", url: "https://aclanthology.org/2023.emnlp-main.298/" },
          { label: "Official T5.1.1-XXL T5X config", url: "https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xxl.gin" }
        ],
        caveat: "GQA 只规定分组共享关系；G=8、64 个 Q heads 与 T5 relative bias 都是该代表模型的选择。"
      },
      positionEncoding: {
        title: "GQA 与位置机制正交",
        summary: "GQA 论文从 T5.1.1 checkpoint uptrain；其 self-attention 沿用按 query head 索引的桶化相对位置偏置。实验把 GQA/MQA 用于 decoder self-attention 与 cross-attention，不改 encoder self-attention；cross-attention 不应被凭空补上同一桶化 bias。",
        equation: R`\[
          S_{h,t,s}=\frac{q_{h,t}^{\top}k_{g(h),s}}{\sqrt{d_h}}
          +b_{h,\operatorname{bucket}(t-s)}+M_{t,s}.
        \]`,
        steps: [
          { label: "原论文", title: "T5 self-attention bias 按 query head 索引", body: R`位置项 \(b_{h,\operatorname{bucket}(t-s)}\) 保留 \(h\) 轴；均值池化的是 K/V heads，不是把 query-head position-bias 参数池化成 KV-group bias。该式描述有 T5 relative bias 的 self-attention。` },
          { label: "分组", title: "每个 KV 组仍保留位置索引", body: R`共享的是内容投影 \(W_g^K,W_g^V\)，不是把多个历史位置合并。` },
          { label: "实验范围", title: "decoder self/cross，不改 encoder self", body: "这是 §3.1 的 T5.1.1 实验选择；decoder-only GQA 是合理后续用法，但不是该 encoder-decoder 实验的同一范围。" }
        ],
        caveat: "同为 GQA 的模型可能使用 T5 bias、RoPE、ALiBi 或其他方案；不能跨 checkpoint 搬用位置参数。"
      },
      derivationSourceFallback: "Ainslie et al. (2023), §2.1（MQA）、§2.2（GQA）与 §3.1（T5.1.1 uptraining/decoder scope）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "code-shape", level: "intermediate" }
      ],
      derivations: [
        {
          title: "组内均值是最小平方共享初始化",
          body: R`**原式。** 一组有 \(r\) 个原投影 \(W_1,\ldots,W_r\)，用一个共享矩阵 \(W\) 初始化。**补全代数。**
            最小化
            \[
            J(W)=\sum_{i=1}^{r}\|W-W_i\|_F^2,\qquad
            \nabla_WJ=2rW-2\sum_iW_i.
            \]
            令梯度为零得
            \[
            W^\star=\frac1r\sum_{i=1}^{r}W_i.
            \]
            **张量形状。** 每个 \(W_i^K,W_i^V\in\mathbb R^{d_{\text{model}}\times d_h}\)，均值保持同形状；全体共享投影为
            \([d_{\text{model}},H_{kv}d_h]\)。**直观。** 均值是在参数 Frobenius 距离下最接近整组旧头的单一代表。**边界。**
            这不是函数输出的全局最优合并，因为 softmax 非线性且各头输入分布不同；论文因此还要继续 uptrain。`,
          source: "Ainslie et al. (2023), §3.1（T5.1.1 scope 与 mean-pooling uptraining）"
        },
        {
          title: "组映射的整除条件来自连续等大小分组",
          body: R`**原式。** \(g(h)=\lfloor h/r\rfloor\)，其中 \(r=H_q/H_{kv}\)。**补全代数。**
            当 \(H_q=rH_{kv}\) 时，每个 \(g\in\{0,\ldots,H_{kv}-1\}\) 的原像恰有
            \[
            \left|\{h:g(h)=g\}\right|=r
            \]
            个 query 头。**张量形状。** Q 为 \([B,H_q,L_q,d_h]\)，K/V 为
            \([B,H_{kv},L_k,d_h]\)，逻辑展开后的组索引为 \([H_q]\)。**直观。** 整除让每位资料员服务同样多的读者。**边界。**
            不等大小分组在数学上可行，但许多 fused kernel 与张量并行布局假设固定 \(r\)，所以工程接口通常要求整除。`,
          source: "Ainslie et al. (2023), §2.2（grouped-query attention）"
        }
      ],
      exercises: [
        {
          kind: "derivation",
          level: "advanced",
          q: R`证明组内均值 \(\bar W\) 使 \(\sum_i\|W-W_i\|_F^2\) 最小，并说明解是否唯一。`,
          hint: "对矩阵逐元素求导，或配方。",
          answer: R`有 \(\sum_i\|W-W_i\|_F^2=r\|W-\bar W\|_F^2+\sum_i\|W_i-\bar W\|_F^2\)。第二项与 \(W\) 无关，第一项在 \(W=\bar W\) 处唯一为零；当 \(r>0\) 时 Hessian 为 \(2rI\)，故解唯一。`
        },
        {
          kind: "counterexample",
          level: "intermediate",
          q: "给出一个反例说明“KV cache 更小的 GQA 一定更快”不成立。",
          hint: "让上下文很短，或让实现复制 KV。",
          answer: "若 L 很短且运行时没有原生 GQA kernel，框架先 repeat K/V 再调用 MHA，实际读写量可能与 MHA 相同并多一次复制；此时 GQA 可能更慢。"
        },
        {
          kind: "design",
          level: "advanced",
          q: R`计划把 \(H_q=64\) 的 MHA uptrain 为 GQA。如何选择 \(H_{kv}\) 候选并避免只看困惑度？`,
          hint: "候选应整除 64，并测服务端指标。",
          answer: R`可测试 \(H_{kv}\in\{1,2,4,8,16,32\}\)，固定继续训练 token 与超参；同时报告验证损失、长上下文/检索质量、KV 字节、目标硬件上的 prefill 吞吐与 TPOT，并检查张量并行是否整齐切分 KV 头。`
        },
        {
          kind: "complexity",
          level: "intermediate",
          q: R`若 \(H_q=64,H_{kv}=8,d_h=128,L=32768\)，每层每序列 BF16 KV cache 为多少？`,
          hint: R`计算 \(2LH_{kv}d_h\times2\) 字节。`,
          answer: R`\(2\times32768\times8\times128\times2=134{,}217{,}728\) 字节，即 128 MiB。对应 MHA 为 1 GiB，理论缩小 8 倍。`
        }
      ]
    },

    mla: {
      attentionConfig: {
        model: "DeepSeek-V2 · 236B",
        scope: "DeepSeek-V2 技术报告与官方 checkpoint 配置；同一组权重按阶段选择执行形态——Prefill 展开成每头 128 维 K/V，Decode 吸收后直读共享 latent，长期缓存始终是 latent + 共享 RoPE key。",
        items: [
          { label: "Hidden size", value: "5120", note: "模型残差宽度" },
          { label: "Q heads", value: "128", note: "两种执行形态共享的 query heads" },
          { label: "Q / K head dim", value: "192 = 128 + 64", note: "content + RoPE" },
          { label: "V head dim", value: "128", note: "value content" },
          { label: "Query latent rank", value: "1536", note: "不进入 KV cache" },
          { label: "KV latent rank", value: "512", note: "joint compressed KV width" },
          { label: "RoPE dim", value: "64", note: "共享 positional key" },
          { label: "Prefill content form", value: "128 heads × 128-d K/V", note: "低秩派生的 MHA-like 执行形态" },
          { label: "Decode content form", value: "128 heads read shared 512-d latent", note: "吸收后的 MQA-like 执行形态" },
          { label: "Persistent cache", value: "576 elements / token / layer", note: "512 latent + 64 shared RoPE key" }
        ],
        sources: [
          { label: "DeepSeek-V2 Technical Report, §§2.1.2–2.1.3 and §3.1.2", url: "https://arxiv.org/abs/2405.04434" },
          { label: "Official DeepSeek-V2 checkpoint config", url: "https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json" }
        ],
        caveat: "两种阶段形态在代数上严格等价。Prefill 展开的 128 组 K/V 只是本轮计算的中间量：既不是持久缓存，也不是独立无约束的 MHA 权重；持久缓存口径始终是 512-d latent 加 64-d 共享 RoPE key。"
      },
      positionEncoding: {
        title: "Decoupled RoPE：为位置付一小笔不可吸收的成本",
        summary: "两种执行图能切换，前提是主要内容通道保持 NoPE。MLA 只把 64 维位置子空间留在外面：query 逐头计算，key 全头共享并随 latent 一起缓存。",
        equation: R`\[
          s_{t,s}^{(i)}=
          \frac{(\widetilde q_{t,i})^{\top}c_s^{KV}
          +(q_{t,i}^{R})^{\top}R_{s-t}k_s^{R}}
          {\sqrt{128+64}},\qquad
          \widetilde q_{t,i}=(W_i^{UK})^{\top}q_{t,i}^{C}.
        \]`,
        steps: [
          { label: "内容主干", title: "512↔128 的切换发生在 NoPE 空间", body: R`内容分数里没有随位置变化的矩阵块，\(W_i^{UK}\) 才能作为固定矩阵移到 query 侧、\(W_i^{UV}\) 移到输出侧；这是两种执行图可以互换的前提。` },
          { label: "位置 key", title: R`只缓存一份共享 64-d \(k^R\)`, body: R`RoPE key 随位置逐 token 保存；若逐头保存，位置通道宽度变成 \(128\times64\)，一项就吃掉 MLA 的缓存收益，所以 key 侧做 MQA 式共享。` },
          { label: "位置 query", title: R`逐头 \(q_i^R\) 现算即弃`, body: R`query 侧不进缓存、每步现算，逐头化只花 \(O(Hd_h^R)\) 的一次性投影算量，却保留头间位置敏感度的多样性，且没有任何随序列增长的缓存。` }
        ],
        caveat: "Partial/decoupled RoPE 有实验支持和检索直觉，但“少量 RoPE 不逊于完整 RoPE”不是普适定理；具体维度必须通过目标模型消融。"
      },
      derivationSourceFallback: "DeepSeek-V2 Technical Report (2024), §2.1.2–§2.1.4 与 Appendix C；苏剑林（2024/2025）Scientific Spaces MLA 系列（解释性来源）",
      existingExerciseMeta: [
        { kind: "derivation", level: "intermediate" },
        { kind: "complexity", level: "intermediate" }
      ],
      derivations: [
        {
          title: "两种执行图为什么严格等价",
          body: R`**原式。** 对头 \(i\)，Prefill 形态先展开 \(k_{s,i}^{C}=W_i^{UK}c_s\)、\(v_{s,i}=W_i^{UV}c_s\) 再按标准多头计算；Decode 形态直接使用 \(c_s\)。**补全代数。** K 侧吸收：
            \[
            (q_{t,i}^{C})^\top k_{s,i}^{C}
            =(q_{t,i}^{C})^\top W_i^{UK}c_s
            =\big((W_i^{UK})^\top q_{t,i}^{C}\big)^\top c_s
            =\widetilde q_{t,i}^{\;\top}c_s.
            \]
            左端是 Prefill 车道的显式多头打分，右端是 Decode 车道对共享 latent 的打分，逐元素相等，softmax 权重 \(a_{t,s,i}\) 完全一致。V 侧吸收：
            \[
            o_{t,i}=\sum_sa_{t,s,i}v_{s,i}
            =\sum_sa_{t,s,i}W_i^{UV}c_s
            =W_i^{UV}m_{t,i},\qquad
            m_{t,i}=\sum_sa_{t,s,i}c_s,
            \]
            再把 \(W_i^{UV}\) 与输出投影分块合并：\(W_i^{O}o_{t,i}=(W_i^{O}W_i^{UV})m_{t,i}\)。左端仍是 Prefill 的“先重建 value 再加权”，右端是 Decode 的“先聚合 latent 再一次线性写回”。**张量形状。**
            \(c_s,m_{t,i}\in\mathbb R^{512}\)、\(k^{C},v,q^{C}\in\mathbb R^{128}\)、
            \(W_i^{UK},W_i^{UV}\in\mathbb R^{128\times512}\)。
            **直观。** 两条车道是同一线性表达式的两种括号顺序；权重只有一份，结果逐位相同。**边界。**
            恒等式只覆盖 NoPE 内容通道，且要求 softmax 权重在 value 变换之前确定；位置子空间必须留在支路里（见下一条）。`,
          source: "DeepSeek-V2 Technical Report (2024), §2.1.2 与 Appendix C；苏剑林（2025），《MLA好在哪里？（下）》（解释性来源）"
        },
        {
          title: "Partial RoPE 是切换执行图的通行证",
          body: R`**原式。** 若把完整 RoPE 加在内容 key 上，分数变为
            \(q_{t,i}^\top R_t^\top R_sW_i^{UK}c_s\)。**补全代数。**
            \(R_sW_i^{UK}\) 随历史位置 \(s\) 变化：除非 \(W_i^{UK}\) 与全部旋转可交换（一般不成立），否则不存在与位置无关的固定矩阵 \(\widetilde W_i\) 使
            \(q^\top R_t^\top R_sW_i^{UK}=(\widetilde W_iq)^\top\)——K 侧吸收被阻断，Decode 只能退回显式重建。MLA 的解法是拆分：
            \[
            s_{t,s}^{(i)}
            =\underbrace{\widetilde q_{t,i}^{\;\top}c_s}_{\text{NoPE 内容，可吸收}}
            +\underbrace{(q_{t,i}^{R})^\top R_{s-t}k_s^{R}}_{\text{64 维位置支路}}.
            \]
            支路的不对称是成本驱动：key 进缓存，共享一份 \(k^R\) 使持久宽度为
            \(512+64=576\)；若逐头则为 \(512+128\times64=8704\)，位置一项就吃回全部压缩收益。query 现算即弃，逐头 \(q_i^R\) 只花一次投影算量。**张量形状。**
            共享 \(k^R\) 缓存为 \([B,1,L,64]\)，stride-0 广播给 128 个头；逐头 \(q^R\) 为 \([B,128,L_q,64]\)。
            **直观。** 给位置一小块不可吸收的“专用车道”，换来内容主干在两种执行图之间自由切换。**边界。**
            Partial RoPE 的质量结论来自消融而非定理；64 维是 DeepSeek-V2 的选择，迁移到其他模型需重新消融。`,
          source: "DeepSeek-V2 Technical Report (2024), §2.1.3（decoupled RoPE）与 §2.1.4；苏剑林（2024），《从MHA、MQA、GQA到MLA》（解释性来源）"
        }
      ],
      exercises: [
        {
          kind: "derivation",
          level: "advanced",
          q: R`证明 K 侧与 V 侧的两个吸收恒等式，并明确指出哪些运算不能移动到 softmax 的另一侧。`,
          hint: "线性变换可以跨越点积与加权求和移动，但不能穿过逐元素的非线性。",
          answer: R`K 侧：\(q_i^\top W_i^{UK}c_s=((W_i^{UK})^\top q_i)^\top c_s\)，这是同一双线性形式的两种结合顺序，对任意 \(q_i,c_s\) 成立。V 侧：\(\sum_s a_sW_i^{UV}c_s=W_i^{UV}\sum_s a_sc_s\)，由矩阵乘对加权和的线性性成立，随后 \(W_i^{UV}\) 可与输出分块 \(W_i^{O}\) 合并成固定矩阵。不能移动的是跨越 softmax 的运算：softmax（连同 mask 与缩放）必须在吸收后的分数上原位执行——把 value 侧变换提前到 softmax 之前作用在分数上，或把位置相关旋转 \(R_s\) 折进固定矩阵，都会改变函数本身。`
        },
        {
          kind: "complexity",
          level: "intermediate",
          q: R`取内容维 \(C=512\)、共享位置 key 宽度 \(r=64\)、上下文 \(L=32768\)，缓存使用 BF16。按同等有效内容维度，分别求 MLA、MQA、GQA-2 的单层单序列 KV Cache。`,
          hint: R`每 token 元素数分别为 \(C+r\)、\(2C+r\)、\(2(2C+r)\)，最后乘 \(L\times2\) 字节。`,
          answer: R`每 token 分别为 576、1088、2176 个元素。乘 \(32768\times2\) 字节后，MLA 为 36 MiB，MQA 为 68 MiB，GQA-2 为 136 MiB。因此加入 64 维位置支路后，MLA 仍分别比同维 MQA、GQA-2 少 32 MiB 和 100 MiB；相对倍数从理想的 2×/4× 降为约 1.89×/3.78×。`
        },
        {
          kind: "counterexample",
          level: "advanced",
          q: R`用公式说明：为什么给内容 key 加完整 RoPE 会使“固定吸收后的 query 矩阵”不存在，而单独的共享 64 维 \(k^R\) 支路可行？`,
          hint: R`比较 \(R_sW^{UK}\) 与 \(R_{s-t}\) 对位置的依赖方式。`,
          answer: R`完整 RoPE 下分数为 \(q^\top R_t^\top R_sW^{UK}c_s\)。要吸收成 \((\widetilde Wq)^\top c_s\)，需要 \(R_t^\top R_sW^{UK}\) 与 \(s\) 无关；但 \(R_t^\top R_s=R_{s-t}\) 随 \(s\) 变化，除非 \(W^{UK}\) 与所有旋转可交换（一般不成立），固定 \(\widetilde W\) 不存在。而单独支路中 \(k_s^R=R_sW^{KR}h_s\) 在写入缓存时已经完成旋转，位置分数 \((R_tq^R)^\top k_s^R=(q^R)^\top R_{s-t}k_s^R\) 按原样计算、不需要任何吸收——位置项走专用小通道，内容项保持 NoPE 并保留吸收自由。`
        },
        {
          kind: "design",
          level: "advanced",
          q: "有人断言“MLA 永远是最优 attention”。请批判这个说法，并列出至少四个可能反转工程选型的条件。",
          hint: "回到本节采用的表达力前提，以及 Decode 的系统边界。",
          answer: R`该断言把条件化结论当成了定理。可能反转选型的条件包括：(1) tensor parallel 布局——TP 切分下 MLA 的 latent 复制读取或头维切分可能不如 GQA 均衡；(2) kernel 可用性——目标硬件若缺少吸收式/FlashMLA 类 kernel，理论带宽收益无法兑现；(3) 量化——latent 与 RoPE key 对量化误差的敏感度可能高于普通 KV；(4) batch 与上下文长度——短上下文、大 batch 时 Decode 未必 memory-bound；(5) 完整 RoPE 的质量——若任务对位置通道要求高，64 维 Partial RoPE 可能不够；(6) 参数量对齐——上投影参数换成更宽 FFN 或更多层也许收益更高；(7) hybrid/线性替代——KDA、GDN 等方案改变长上下文比较基线。任何一条都可能让“几乎最优”失效。`
        }
      ]
    },

    mfa: {
      attentionConfig: {
        model: "MFA · 6.9B MoE / 1.2B activated · 1T tokens",
        scope: "MFA 论文的规模化研究实验：24 层 MoE 模型、1T tokens 训练，用于验证固定 KV 预算下的缓存与质量；数值属于该实验配置。",
        items: [
          { label: "Hidden size", value: "2048", note: "d_model" },
          { label: "Layers", value: "24", note: "全模型缓存按 24 层合计" },
          { label: "Attention heads", value: "m = 18", note: "每层 MFA heads" },
          { label: "Shared latent / head rank", value: "C = 256", note: "共享 K/V 宽度 = 每头 factorization rank" },
          { label: "Shared K/V heads", value: "1 / 1", note: "全部 heads 读取同一份 C 维 K/V" },
          { label: "RoPE base", value: "500,000", note: "论文 common settings" },
          { label: "MFA cache per layer", value: "512 elements/token = 1 KiB BF16", note: "2 × C" },
          { label: "24-layer cache", value: "24,576 bytes/token ≈ 24 KiB", note: "24 × 2 × 256 × 2 字节" },
          { label: "MFA-KR cache", value: "256 elements/layer ≈ 12 KiB across 24 layers", note: "只缓存共享 key" }
        ],
        sources: [
          { label: "Multi-matrix Factorization Attention (Findings of ACL 2025)", url: "https://aclanthology.org/2025.findings-acl.1288/" },
          { label: "Multi-matrix Factorization Attention (arXiv:2412.19255)", url: "https://arxiv.org/abs/2412.19255" }
        ],
        caveat: "MFA 的“head dimension=256”是共享 latent 维度 C 和每头 factorization rank，不要求 hidden size 等于 head 数×256。论文的 6.9B/1.2B-activated 模型是研究实验，不是公开生产 checkpoint；24.6 KB/token 是 24 层 BF16 缓存总量，不是单层。"
      },
      positionEncoding: {
        title: "MFA 使用标准 RoPE：逐头 query 与共享 key 在同一 C 维空间旋转",
        summary: R`标准 RoPE 作用于每头 query \(q_{t,c}=x_tS_qQ_c\) 与共享 key \(k_t=x_tS_k\)；value 不旋转。共享 key 每 token 只需旋转一次即可服务全部 heads。`,
        equation: R`\[
          s_{t,s}^{(c)}
          =\frac{\bigl(R_tq_{t,c}\bigr)^\top\bigl(R_sk_s\bigr)}{\sqrt C}
          =\frac{q_{t,c}^\top R_{s-t}k_s}{\sqrt C}.
        \]`,
        steps: [
          { label: "共享 key", title: "每个 token 只旋转一次", body: R`共享 \(k_t\) 只有一份，按位置 \(t\) 旋转一次即可被全部 heads 读取；不为每个 head 复制旋转结果。` },
          { label: "逐头 query", title: "所有 query heads 在同一 C 维空间旋转", body: R`每个 head 的 \(q_{t,c}\) 都是 C 维向量，使用同一组 RoPE 频率；head 差异来自 \(Q_c\)，不来自位置编码。` },
          { label: "实验配方", title: "base 500,000，另测 ALiBi", body: "论文 common settings 使用 RoPE base 500,000，但 MFA 机制本身与位置编码兼容，ALiBi 是单独消融。" }
        ],
        caveat: "RoPE base 500,000 是论文实验选择，不是 MFA 定义；换用其他位置机制不改变共享 C 维 KV 与逐头 C×C 变换的结构。"
      },
      derivationSourceFallback: "Multi-matrix Factorization Attention (2024/2025)：MFA 与 MFA-KR 的定义、KV cache 计量与实验配置",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "architecture", level: "intermediate" }
      ],
      derivations: [
        {
          title: "总有效秩增加而缓存不变",
          body: R`**原式。** 论文把每个 head 的 QK circuit 视为经过共享 C 维空间的矩阵分解，其 factorization rank 至多为共享宽度 \(C\)；MFA 让每头达到
            \[
            \mathrm{FRH}=C.
            \]
            **补全代数。** 对 \(m\) 个 head 求和得到论文的总有效秩
            \[
            \mathrm{TER}=m\cdot\mathrm{FRH}=mC,
            \]
            而缓存宽度保持 \(2C\)，与 \(m\) 无关。固定缓存预算下，MFA 可以同时增加 head 数与每头秩。**张量形状。** \(S_q,S_k\in\mathbb R^{d\times C}\)、\(Q_c\in\mathbb R^{C\times C}\)；每头 QK circuit 是 \(d\times d\) 矩阵，秩受 \(C\) 限制。**直观。** 缓存决定共享底片的分辨率，权重决定有多少个高秩镜头同时读它。**边界。** FRH/TER 是论文用于比较容量的代理指标，不是“精度随 TER 单调上升”的定理；质量结论仍以论文实验为准。`
        },
        {
          title: "MFA-KR 的零初始化 key reuse",
          body: R`**原式。** MFA-KR 把 value 投影约束到 key 派生族。教学式写法为
            \[
            v_s=k_sM,\qquad M=I+\operatorname{diag}(\alpha)N,
            \]
            其中 \(N\in\mathbb R^{C\times C}\) 可学习、\(\alpha\in\mathbb R^{C}\) 零初始化。**补全代数。** 初始时
            \(\alpha=0\Rightarrow M=I\Rightarrow v_s=k_s\)：训练从“value 等于 key”出发逐步学习偏离量，缓存从 \(2C\) 降到 \(C\)。**张量形状。** \(k_s,v_s\in\mathbb R^{C}\)，\(M\in\mathbb R^{C\times C}\)。**直观。** 同一份缓存底片既当地址又当内容；零初始化 gate 保证重参数化不破坏训练起点。**边界。** 矩阵朝向遵循实现约定（此处按行向量右乘书写）；核心点是 value 投影被约束为 key 派生族，论文实验也显示相对 MFA 的小幅质量折损。`
        }
      ],
      exercises: [
        {
          kind: "code-shape",
          level: "intermediate",
          q: R`设 hidden size 2048、\(C=256\)、\(m=18\)。写出共享 query/key/value 特征、逐头展开 query、scores 与每层每 token 缓存的形状或元素数。`,
          hint: R`共享特征都是 C 维；只有 query 有 head 轴。`,
          answer: R`共享特征 \(x S_q, x S_k, x S_v\) 均为 \([B,T,256]\)；逐头 query 为 \([B,18,T,256]\)；scores 为 \([B,18,T,T]\)。每层每 token 缓存 \(2C=512\) 个元素（MFA-KR 为 256 个），与 \(m=18\) 无关。`
        },
        {
          kind: "counterexample",
          level: "advanced",
          q: "构造一个反例说明“每 token 缓存相同的两种注意力，容量也相同”不成立。",
          hint: R`比较 MFA（\(C=256\)、\(m\) 个 head）与 head dim 为 256 的单头注意力。`,
          answer: R`单头注意力取 \(d_h=256\)，每 token 缓存 \(2\times256\) 个元素，与 \(C=256\) 的 MFA 完全相同；但它每 token 只有一个 softmax 分布和一套读写变换，按论文的代理指标 TER 为 \(1\times C\)，而 MFA 为 \(mC\)，且 MFA 有 \(m\) 个独立的 \(C\times C\) QK/VO circuit。缓存口径相同不代表表达容量相同；TER 是容量代理，不是精度定理。`
        },
        {
          kind: "design",
          level: "advanced",
          q: "给定质量优先或显存受限两类约束，如何在 MFA 与 MFA-KR 之间选择？",
          hint: "比较缓存宽度与论文报告的质量差异。",
          answer: R`显存或带宽极紧时选 MFA-KR：缓存从 \(2C\) 减半到 \(C\)，但论文实验显示相对 MFA 的小幅质量折损。质量优先且 \(2C\) 预算可接受时选标准 MFA。决策应在目标上下文长度与 batch 下同时测缓存字节、吞吐与任务质量，而不是只看理论宽度。`
        },
        {
          kind: "derivation",
          level: "intermediate",
          q: "解释为什么 RoPE 施加在 MFA 的 query 与 key 上，而不施加在 value 上。",
          hint: R`相对位置性质来自 \(R_t^\top R_s=R_{s-t}\) 在点积中的配对消去。`,
          answer: R`score 中 \((R_tq)^\top(R_sk)=q^\top R_{s-t}k\)，两侧旋转配对后只留下相对位移，这正是需要位置信息的地方。value 不参与点积配对：若旋转 \(v_s\)，绝对位置相位会直接进入输出内容且无从消去。MFA-KR 中 value 由未旋转的 key 特征派生，同样保持 value 无旋转。`
        }
      ]
    },

    tpa: {
      attentionConfig: {
        model: "T6-XL · 1.55B · FineWeb-Edu-100B",
        scope: "官方仓库 train_T6_xl_adam_80g8.py 给出的最大 T6 研究模型配置；数值属于该 config，不是生产 checkpoint。",
        items: [
          { label: "Hidden size", value: "1600", note: "d_model" },
          { label: "Layers", value: "48", note: "n_layer" },
          { label: "Expanded Q heads", value: "78", note: "attention heads h" },
          { label: "Head dim", value: "64", note: "d_h" },
          { label: "Q rank", value: "R_Q = 6", note: "query 因子秩" },
          { label: "K / V rank", value: "R_K = 2 / R_V = 2", note: "决定缓存的秩预算" },
          { label: "Attention inner width", value: "4992", note: "78 × 64，大于 d_model=1600" },
          { label: "Factorized KV cache", value: "568 elements/token/layer", note: "(R_K+R_V)(h+d_h)" },
          { label: "Full-MHA-shaped cache", value: "9984 elements/token/layer", note: "2 × 78 × 64 对照" },
          { label: "Training block size", value: "1024", note: "官方 config 的上下文长度" },
          { label: "Position", value: "RoPE", note: "config 未给出 base 数值" }
        ],
        sources: [
          { label: "Tensor Product Attention Is All You Need (arXiv:2501.06425)", url: "https://arxiv.org/abs/2501.06425" },
          { label: "Official tensorgi/TPA repository", url: "https://github.com/tensorgi/TPA" },
          { label: "Official T6-XL training config", url: "https://github.com/tensorgi/TPA/blob/main/config/train_T6_xl_adam_80g8.py" }
        ],
        caveat: "TPA 论文与官方仓库提供的是最高 1.55B 的 T6 研究模型，而不是公开的超大生产 checkpoint。官方 XL config 使用 h=78、d_h=64，所以 attention inner width 为 4992，明显大于 d_model=1600；这是参数量配平选择，不是维度错误。"
      },
      positionEncoding: {
        title: "TPA 的 RoPE 逐行作用于 B_Q / B_K 因子",
        summary: R`旋转线性作用在 \(d_h\) 轴上，因此 \(R_t(A^\top B)=A^\top R_t(B)\)：对 \(B_Q\)、\(B_K\) 的每一行做 RoPE 就等价于旋转重建后的每头 Q/K。缓存里保存的是已旋转的 \(\widetilde B_K\)；A 因子与 \(B_V\) 都不旋转。`,
        equation: R`\[
          \widetilde B_K=R_t(B_K),\qquad
          \widetilde K_t=\frac1{R_K}A_K^\top\widetilde B_K,
          \qquad
          \bigl(R_tq_{t,i}\bigr)^\top\bigl(R_sk_{s,i}\bigr)
          =q_{t,i}^\top R_{s-t}k_{s,i}.
        \]`,
        steps: [
          { label: "A 因子", title: "token 依赖但不旋转", body: R`\(A_Q,A_K\) 决定因子在 head 轴上的分布，与位置无关；旋转只发生在 channel 轴。` },
          { label: "缓存", title: R`写入前预旋转 \(B_K\)`, body: R`按位置 \(t\) 旋转后再缓存 \(\widetilde B_K\)，score 的相对位置性质由两侧配对保持；解码时无需回头重旋历史因子。` },
          { label: "范围", title: R`只旋转 Q/K 的 B 因子，\(B_V\) 不旋转`, body: R`value 不参与旋转配对；对 \(B_V\) 施 RoPE 会把绝对位置相位注入输出内容。` }
        ],
        caveat: "TPA 与位置机制兼容；T6 使用 RoPE，官方 XL config 未暴露 base 数值，不能替它虚构一个。"
      },
      derivationSourceFallback: "Tensor Product Attention Is All You Need (2025)：contextual factorization、RoPE 兼容性与 FlashTPA decoding；官方 tensorgi/TPA 仓库",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "code-shape", level: "intermediate" }
      ],
      derivations: [
        {
          title: "score 可在因子域展开",
          body: R`**原式。** head \(i\) 的 score 需要 \(q_{t,i}\cdot k_{s,i}\)。**补全代数。** 代入因子分解：
            \[
            q_{t,i}\cdot k_{s,i}
            =\frac1{R_QR_K}\sum_{p=1}^{R_Q}\sum_{r=1}^{R_K}
            A_Q[p,i]\,A_K[r,i]\,\bigl(B_Q[p]\cdot B_K[r]\bigr).
            \]
            先算 \(R_Q\times R_K\) 个 channel 内积 \(B_Q[p]\cdot B_K[r]\)，再用 A 因子逐头加权，即得全部 \(h\) 个 head 的 score。**张量形状。** channel 内积为 \([B,T,S,R_Q,R_K]\)，与 head 数无关；score 为 \([B,h,T,S]\)。**直观。** 历史 token 只以低秩因子存在，score 是因子域收缩，从不物化 \([B,h,S,d_h]\) 的完整历史 K。**边界。** 这正是 FlashTPA decoding 顺序收缩的代数基础；若先重建 K 再调用普通 MHA kernel，数值不变但主要效率收益消失。`
        },
        {
          title: "RoPE 只作用 channel factor",
          body: R`**原式。** RoPE 对 head \(i\) 的 key 行向量施位置旋转 \(R_t\)。**补全代数。** \(K_t=A_K^\top B_K/R_K\) 的第 \(i\) 行是 B 行的线性组合
            \(\frac1{R_K}\sum_rA_K[r,i]B_K[r]\)，而 \(R_t\) 线性作用在 \(d_h\)（右侧）轴上，故
            \[
            R_t\!\left(A_K^\top B_K\right)=A_K^\top R_t(B_K),
            \]
            其中 \(R_t(B_K)\) 表示对 \(B_K\) 的每一行做同一旋转。因此可以在写入缓存前预旋转 \(\widetilde B_K=R_t(B_K)\)，score 中 \((R_tq)^\top(R_sk)=q^\top R_{s-t}k\) 的相对位置性质保持不变。**张量形状。** \(B_K\in\mathbb R^{R_K\times d_h}\)，旋转不改变形状。**直观。** 位置信息只写进 channel 因子；head-mixing 的 A 因子与位置无关。**边界。** \(B_V\) 不参与 score 的旋转配对，因此保持不旋转；对 value 施 RoPE 会把绝对位置相位注入输出内容。`
        }
      ],
      exercises: [
        {
          kind: "counterexample",
          level: "advanced",
          q: R`解释为什么 \(R_K=1\) 的 contextual TPA 不等同于 MQA，并给出一个区分它们的具体机制。`,
          hint: R`MQA 的 head-sharing 模式是固定的；TPA 的 \(A_K(x_t)\) 随 token 变化。`,
          answer: R`\(R_K=1\) 时 \(K_t=a_K(x_t)^\top b_K(x_t)\)：head 轴权重 \(a_K(x_t)\in\mathbb R^{h}\) 由当前 token 生成，不同 token 可以把同一 channel 模式按不同强度写入不同 heads。MQA 相当于把 \(a_K\) 固定为常向量（全部 head 等权共享一个 key 头），与输入无关。只要存在 \(x_1,x_2\) 使 \(a_K(x_1)\ne a_K(x_2)\)（可训练投影一般如此），这种 token-dependent head mixing 就不能被任何固定 head-sharing mask 复现。`
        },
        {
          kind: "code-shape",
          level: "advanced",
          q: R`已知 A 为 \([B,T,R,H]\)、B 为 \([B,T,R,D]\)。写出用 torch.einsum 重建 \([B,H,T,D]\) 激活的表达式。`,
          hint: R`沿秩轴 \(R\) 收缩，别忘了 \(1/R\) 因子。`,
          answer: "torch.einsum(\"btrh,btrd->bhtd\", A, B) / R。秩轴 r 被收缩，head 轴来自 A，channel 轴来自 B；除以 R 与论文的 1/R 归一化一致。"
        },
        {
          kind: "design",
          level: "advanced",
          q: R`如何为固定缓存预算选择 \(R_K,R_V\)？说明质量与缓存的权衡实验。`,
          hint: R`缓存按 \((R_K+R_V)(h+d_h)\) 线性增长。`,
          answer: R`更高的 rank 能表达更多 head×channel 结构，但缓存随 \(R_K+R_V\) 线性增长。应在固定 \((R_K+R_V)(h+d_h)\) 预算下网格比较不同 \((R_K,R_V)\) 组合，同时报告困惑度/下游质量、每 token 缓存字节与目标硬件上的 decode 吞吐；K 与 V 的最优秩不必相等，不能只用一个指标外推。`
        },
        {
          kind: "derivation",
          level: "intermediate",
          q: R`证明缓存前预旋转 \(B_K\) 是合法的，并说明为什么 \(B_V\) 不做同样处理。`,
          hint: R`利用 \(R_t(A^\top B)=A^\top R_t(B)\) 与 \(R_t^\top R_s=R_{s-t}\)。`,
          answer: R`旋转作用在 channel 轴上且是线性映射，故 \(R_s(A_K^\top B_K)=A_K^\top R_s(B_K)\)：缓存 \(\widetilde B_K=R_s(B_K)\) 与旋转重建后的 key 完全等价，score 中 \((R_tq)^\top(R_sk)=q^\top R_{s-t}k\) 只依赖相对位移。value 侧没有与 query 的旋转配对，\(B_V\) 若被旋转，输出会带上无法消去的绝对位置相位，因此 \(B_V\) 保持不旋转。`
        }
      ]
    },

    dsa: {
      attentionConfig: {
        model: "DeepSeek-V3.2-Exp · 671B",
        scope: "官方 671B 配置；DSA 在 MLA cache 之外增加低维 Indexer cache，并让 core 只读取 top-k entries。",
        items: [
          { label: "Hidden size", value: "7168", note: "dim" },
          { label: "Q heads", value: "128", note: "core MLA query heads" },
          { label: "Q / K head dim", value: "192 = 128 + 64", note: "non-RoPE + RoPE" },
          { label: "V head dim", value: "128", note: "core value width" },
          { label: "MLA cache", value: "576 elements / token / layer", note: "512 latent + 64 RoPE key" },
          { label: "Indexer heads", value: "64", note: "lightning indexer queries" },
          { label: "Indexer head / cache dim", value: "128 / 128", note: "共享 index key 每 token 128 维" },
          { label: "Selected entries", value: "top-k 2048", note: "每个 query 的 core 候选" },
          { label: "Local window", value: "None", note: "原型 DSA 没有额外 raw-token 滑窗" }
        ],
        sources: [
          { label: "Official DeepSeek-V3.2-Exp 671B config", url: "https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json" },
          { label: "DeepSeek-V3.2 report, §2.1", url: "https://arxiv.org/abs/2512.02556" }
        ],
        caveat: "64×128 的 Indexer 与 k=2048 是 V3.2-Exp 实例参数；DSA 方法本身只要求低成本索引器和 token-level sparse core。"
      },
      positionEncoding: {
        title: "DSA：core 继承 MLA，Indexer 两侧 pRoPE + Hadamard",
        summary: "DeepSeek-V3.2 的 core 以 MQA-mode MLA 读取选中的 latent entries。Lightning Indexer 对 q 与共享 k 的指定子维都施 pRoPE；FP8 实现再对两侧施同一正交 Hadamard rotation。只转一侧会改变点积。",
        equation: R`\[
          \widehat q_{t,j}^{I}=\mathcal H[R_tq_{t,j}^{I,R};q_{t,j}^{I,N}],\qquad
          \widehat k_s^{I}=\mathcal H[R_sk_s^{I,R};k_s^{I,N}],
          \quad
          (\widehat q_{t,j}^{I})^\top\widehat k_s^I
          =(q_{t,j}^{I})_{\rm pRoPE}^\top(k_s^I)_{\rm pRoPE}.
        \]`,
        steps: [
          { label: "核心路径", title: "稀疏选择不改 MLA 位置定义", body: "top-k 只缩小被 core MLA 读取的位置集合；选中后的内容/位置 score 仍按 MLA 计算。" },
          { label: "索引路径", title: "pRoPE 同时作用 q 与 k", body: R`只有指定子维旋转，其余维保持内容表示；所有 indexer heads 共享 key，但每头 query 独立。` },
          { label: "低精度", title: "Hadamard 必须在两侧配对", body: R`正交 \(\mathcal H\) 满足 \((\mathcal Hq)^\top(\mathcal Hk)=q^\top k\)；它服务 FP8 数值分布，不注入位置。BF16/FP32 端口可省略这对变换。` }
        ],
        caveat: "不同端口可能以 BF16 直接算 Indexer score 而省略 Hadamard/FP8 路径；只要投影、partial RoPE 与 score 定义一致，语义可保持。"
      },
      derivationSourceFallback: "DeepSeek-V3.2 Technical Report (2025), §2.1（DSA prototype and training）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "design", level: "foundation" }
      ],
      derivations: [
        {
          title: "KL 对齐把 Indexer 训练成分布拟合器",
          body: R`**原式。** dense warm-up 使用
            \[
            \mathcal L_t^I=D_{\mathrm{KL}}(p_t\|\widehat p_t),\qquad
            \widehat p_t=\operatorname{softmax}(I_t).
            \]
            **补全代数。**
            \[
            \mathcal L_t^I=\sum_sp_{t,s}\log p_{t,s}
            -\sum_sp_{t,s}I_{t,s}+\log\sum_ue^{I_{t,u}},
            \quad
            \frac{\partial\mathcal L_t^I}{\partial I_{t,s}}=\widehat p_{t,s}-p_{t,s}.
            \]
            \(p_t\) 由 full main-attention 的**概率权重**跨头求和后沿完整历史轴 L1 归一化；不是对未归一化 teacher logits 求和。warm-up 的 student softmax 也覆盖完整 \(I_{t,:}\)。稀疏阶段才把 teacher 与 Indexer logits 同时限制到
            \(\mathcal S_t\) 上再做式 (4) 对齐。**张量形状。** \(I,p,\widehat p\in\mathbb R^{B\times L_q\times L_k}\)，
            top-k 索引为 \([B,L_q,k]\)。**直观。** Indexer 学的是“主 attention 会把概率放在哪里”，不是直接回归 value。**边界。**
            top-k 是离散路由，且报告将 Indexer 输入 detach；语言模型损失不经该选择直接训练 Indexer，漏召回只能由其独立 KL 信号与稀疏训练适配缓解。`,
          source: "DeepSeek-V3.2 Technical Report (2025), §2.1, Eqs. (3)–(4)"
        }
      ],
      exercises: [
        {
          kind: "architecture",
          level: "intermediate",
          q: R`按解码时的数据流排列以下步骤：生成 Indexer query、扫描 Indexer key cache、top-k、读取 MLA latent entries、执行 core attention。哪些数据需要跨 token 持久缓存？`,
          hint: "Indexer 负责选位置，core 才读取这些位置对应的完整 MLA 条目。",
          answer: R`当前隐藏状态 \(h_t\) 先生成 \(q_{t,j}^I,w_{t,j}^I\)，与历史 \(k_{1:t}^I\) 计算 \(I_{t,:}\)；随后 top-k 得到位置集合 \(\mathcal I_t\)，按位置读取 \(\{c_s^{KV},k_s^R:s\in\mathcal I_t\}\)，最后执行 MQA-mode MLA core attention。跨 token 持久缓存的是每个位置的 \(c_s^{KV}\)、共享 RoPE key \(k_s^R\) 和共享 Indexer key \(k_s^I\)；当前 query、完整 score、top-k 索引和展开后的每头 K/V 都是临时量。`
        },
        {
          kind: "cache",
          level: "intermediate",
          q: R`设 MLA latent 宽度为 \(d_c\)，共享 RoPE key 宽度为 \(d_h^R\)，共享 Indexer key 宽度为 \(d_I\)。写出 DSA 每层、每 token 的持久 attention cache 组成与字节数公式。为什么不乘 query 头数 \(H_q\) 或 Indexer 头数 \(H^I\)？`,
          hint: "分别考虑 core MLA 与 Lightning Indexer；两条路径的 key 都跨 query heads 共享。",
          answer: R`持久 cache 由 \(c_s^{KV}\in\mathbb R^{d_c}\)、\(k_s^R\in\mathbb R^{d_h^R}\) 和 \(k_s^I\in\mathbb R^{d_I}\) 组成。若前两者每元素 \(b_{\rm core}\) 字节、Indexer key 每元素 \(b_I\) 字节，则每层每 token 为 \((d_c+d_h^R)b_{\rm core}+d_Ib_I\) 字节。core 采用 MQA-mode MLA，Indexer 也只有一个共享 key，因此都不乘头数；query、top-k 索引和展开 K/V 不属于跨 token KV cache。`
        },
        {
          kind: "code-shape",
          level: "advanced",
          q: R`Indexer query 为 \([B,L,H^I,d_I]\)，共享 key cache 为 \([B,S,d_I]\)，权重为 \([B,L,H^I]\)。写出 score 的目标形状和 head 聚合。`,
          hint: R`先得到 \([B,L,H^I,S]\)，ReLU 后乘权重并沿 \(H^I\) 求和。`,
          answer: R`点积得到 \(D_{b,t,h,s}=\langle q_{b,t,h},k_{b,s}\rangle\)，形状 \([B,L,H^I,S]\)；计算 \(\sum_h w_{b,t,h}\operatorname{ReLU}(D_{b,t,h,s})\) 后为 \([B,L,S]\)，再沿 S 做 top-k 得 \([B,L,k]\)。`
        },
        {
          kind: "complexity",
          level: "advanced",
          q: R`若 \(d_c=512,d_h^R=64\) 且二者以 BF16 缓存，\(d_I=128\) 且 Indexer key 以 FP8 缓存，40 层、长度 128K、batch 2 的 DSA attention cache 约多大？`,
          hint: R`每层每 token 为 \((512+64)\times2+128\times1\) 字节，再乘 \(B N L\)；不要乘头数，也不要给 latent 额外乘 K/V 的 2。`,
          answer: R`每层每 token 为 \(576\times2+128=1280\) 字节。总量为 \(2\times40\times131072\times1280=13{,}421{,}772{,}800\) 字节，即 12.5 GiB；其中 core MLA cache 为 11.25 GiB，Indexer key cache 为 1.25 GiB。该估算不含量化 scale/对齐、页表和运行时工作区。`
        }
      ]
    },

    csa: {
      attentionConfig: {
        model: "DeepSeek-V4-Pro · 1.6T",
        scope: "官方 V4-Pro 配置中的 CSA layers（compress ratio 4）；core 使用一个共享 compressed KV head。",
        items: [
          { label: "Hidden size", value: "7168", note: "模型残差宽度" },
          { label: "Q heads", value: "128", note: "compressed core queries" },
          { label: "Effective KV heads", value: "1", note: "共享 compressed KV entry" },
          { label: "Core head dim", value: "512", note: "同一 512-d entry 兼作 K/V" },
          { label: "RoPE slice", value: "64", note: "末 64 维 partial RoPE" },
          { label: "Compression", value: "m=4 · span 8 · stride 4", note: "重叠 2m receptive field" },
          { label: "Indexer", value: "64 heads × 128 dim", note: "compressed index keys 为 128-d" },
          { label: "Selected entries", value: "top-k 1024", note: "compressed candidates" },
          { label: "Raw local window", value: "128 tokens", note: "与 compressed core 并行" }
        ],
        sources: [
          { label: "DeepSeek-V4 Technical Report, §§2.3.1 and 2.3.3", url: "https://arxiv.org/abs/2606.19348" },
          { label: "Official DeepSeek-V4-Pro inference config", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/config.json" }
        ],
        caveat: "m=4、top-k=1024、window=128 与各维度属于 V4-Pro；CSA 的结构不变量是重叠压缩、索引选择和共享-KV core。"
      },
      positionEncoding: {
        title: "CSA 使用 partial RoPE，并对输出 inverse RoPE",
        summary: R`DeepSeek-V4 的 \(C_s^{Comp}\) 同时充当 key 与 value。报告只旋转 query/KV entry 的末 64 维；compressed entry 使用实现给定的位置 \(\pi_s\)，输出再按原 query 位置 \(t\) 施 inverse RoPE。core 前还做 per-head RMSNorm，并在 softmax 分母加入 sink。`,
        equation: R`\[
          R_{-t}\sum_{s\in\mathcal S_t}a_{t,s}R_{\pi_s}c_s^R
          =\sum_{s\in\mathcal S_t}a_{t,s}R_{\pi_s-t}c_s^R.
        \]`,
        steps: [
          { label: "Partial", title: "只旋转末 64 维", body: R`写成 \(c_s=[c_s^N;c_s^R]\)，仅对 \(c_s^R\in\mathbb R^{64}\) 应用 \(R_{\pi_s}\)，其余通道保持非旋转内容。` },
          { label: "Inverse", title: "输出按原 query 位置反旋转", body: R`对输出 RoPE slice 左乘 \(R_{-t}=R_t^\top\)，使条目 \(s\) 的贡献依赖 \(\pi_s-t\)，而不是擅自把 compressed index \(s\) 当 raw-token 位置。` },
          { label: "Core", title: "RMSNorm 与 sink 不能省略", body: "query heads 和唯一 compressed-KV head 在 core 前 RMSNorm；每头 sink logit 加入 softmax 分母，使真实 entry 权重和可小于 1。" }
        ],
        caveat: R`压缩索引 \(s\) 与 raw-token 位置不一一对应；必须使用实现规定的 \(\pi_s\) 与 RoPE scaling，不能擅自设成 \(s\)、块中心或块末端。`
      },
      derivationSourceFallback: "DeepSeek-V4 Technical Report (2026), §2.3.1（Eqs. 9–19）与 §2.3.3（RMSNorm、partial/inverse RoPE、sink Eq. 27）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "derivation", level: "intermediate" }
      ],
      derivations: [
        {
          title: "inverse RoPE 把 value 的绝对相位变成相对相位",
          body: R`**原式。** CSA 的旋转 value slice 产生
            \[
            o_t^R=\sum_{s\in\mathcal S_t}a_{t,s}R_{\pi_s}c_s^R.
            \]
            **补全代数。** 报告在 query 位置 \(t\) 应用逆旋转：
            \[
            \widetilde o_t^R=R_{-t}o_t^R
            =\sum_sa_{t,s}R_t^\top R_{\pi_s}c_s^R
            =\sum_sa_{t,s}R_{\pi_s-t}c_s^R.
            \]
            **张量形状。** 每个压缩 entry \(c_s\in\mathbb R^{c}\)，RoPE slice
            \(c_s^R,o_t^R\in\mathbb R^{64}\)，权重 \(a_t\in\mathbb R^k\)。**直观。**
            先随 compressed position \(\pi_s\) 旋转，汇总后再站到 raw query 位置 \(t\) 的坐标系观察，于是只剩相对位移。**边界。**
            该恒等式要求同频率旋转且 \(R_{-t}=R_t^\top\)；只对 Q/K 旋转、漏掉输出逆旋转，或给压缩条目使用不一致的位置表都会破坏它。`,
          source: "DeepSeek-V4 Technical Report (2026), §2.3.3（Partial Rotary Positional Embedding；Eq. 26 是 HCA core，不是 inverse-RoPE 恒等式）"
        }
      ],
      exercises: [
        {
          kind: "counterexample",
          level: "advanced",
          q: "说明为什么 4→1 压缩不可能对任意输入都无损，并给出最简单的维数反例。",
          hint: R`比较从 \(\mathbb R^{4d}\) 到 \(\mathbb R^d\) 的连续映射自由度。`,
          answer: R`四个任意 token 含 \(4d\) 个自由度，而一个 \(d\) 维 entry 只有 \(d\) 个自由度；线性压缩的核至少有 \(3d\) 维，所以存在不同四元组映到同一摘要。非线性网络也不能在无附加结构时对所有连续输入建立连续双射。`
        },
        {
          kind: "code-shape",
          level: "advanced",
          q: R`输入 H 为 \([B,L,d]\)，压缩率 \(m=4\)，报告宽度为 \(c\)、Indexer 宽度为 \(c^I\)。写出 \(C^a,C^b,Z^a,Z^b,C^{Comp},K^{IComp}\) 的典型形状。`,
          hint: "KV compressor 与 Indexer compressor 独立；前者逐通道联合归一化 2m 行。",
          answer: R`\(C^a,C^b,Z^a,Z^b\) 为 \([B,L,c]\)，联合 \(2m\)-row softmax 后 \(C^{Comp}\) 约为 \([B,\lfloor L/m\rfloor,c]\)。独立 Indexer compressor 产出 \(K^{IComp}\approx[B,\lfloor L/m\rfloor,c^I]\)。边界 carry 与 overlap 条目数按实现处理。`
        },
        {
          kind: "design",
          level: "advanced",
          q: R`如何联合选择压缩率 \(m\) 与 top-k \(k\)，避免只保持固定 \(k\) 导致召回预算失衡？`,
          hint: "压缩越强，每个 entry 覆盖越多 token，但细节越少。",
          answer: R`在固定系统预算下网格搜索 \((m,k)\)，同时测压缩重建/attention-mass recall、远程检索、局部细节与吞吐。可比较固定 entry 数 \(k\)、固定原 token 覆盖量 \(mk\)、固定 FLOPs 三种控制，区分“读得少”和“摘要太粗”两类误差。`
        },
        {
          kind: "complexity",
          level: "intermediate",
          q: R`若 \(L=1{,}048{,}576,m=4,k=1024,w=128\)，每个 query 的 core 与局部分支共读多少条，Indexer 扫描多少候选？`,
          hint: R`core 读 \(k\)，局部读 \(w\)，Indexer 扫 \(L/m\)。`,
          answer: "core 与局部分支合计约读 1152 条；Indexer 扫描 262,144 个压缩候选。前者不包含去重，后者仍是低维低精度路径。"
        }
      ]
    },

    hca: {
      attentionConfig: {
        model: "DeepSeek-V4-Pro · 1.6T",
        scope: "官方 V4-Pro 配置中的 HCA layers（compress ratio 128）；对所有已完成 compressed blocks 做 dense attention。",
        items: [
          { label: "Hidden size", value: "7168", note: "模型残差宽度" },
          { label: "Q heads", value: "128", note: "compressed-dense queries" },
          { label: "Effective KV heads", value: "1", note: "共享 compressed KV entry" },
          { label: "Core head dim", value: "512", note: "同一 512-d entry 兼作 K/V" },
          { label: "RoPE slice", value: "64", note: "末 64 维 partial RoPE" },
          { label: "Compression", value: "m′=128", note: "非重叠 128-to-1" },
          { label: "Global selection", value: "All completed blocks", note: "dense；没有 top-k/indexer" },
          { label: "Raw local window", value: "128 tokens", note: "覆盖当前未闭合块" }
        ],
        sources: [
          { label: "DeepSeek-V4 Technical Report, §§2.3.2–2.3.3", url: "https://arxiv.org/abs/2606.19348" },
          { label: "Official DeepSeek-V4-Pro inference config", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/config.json" }
        ],
        caveat: "HCA 没有 Indexer 或 top-k；m′=128、window=128 与 512-d compressed entry 是 V4-Pro 的实例选择。"
      },
      positionEncoding: {
        title: "HCA 与 CSA 共享 partial/inverse RoPE",
        summary: R`HCA 是 compressed-dense MQA，不是 sparse attention。宽度 \(c\) 的压缩 KV entry 兼作 value，末 64 维按 compressed position \(\pi_j\) 做 partial RoPE，输出同一 slice 按 raw query 位置 \(t\) inverse RoPE；core 同样使用 RMSNorm 与 sink。`,
        equation: R`\[
          \widetilde o_t^R
          =R_{-t}\sum_{j=0}^{\lfloor t/m'\rfloor-1}a_{t,j}R_{\pi_j}c_j^R
          =\sum_ja_{t,j}R_{\pi_j-t}c_j^R.
        \]`,
        steps: [
          { label: "压缩位置", title: R`使用实现定义的 \(\pi_j\)`, body: R`HCA 把 \(m'\) 个 token 合成一个 entry；其 RoPE position 不能简单写成压缩索引 \(j\)。` },
          { label: "因果 dense", title: R`只读 \(\lfloor t/m'\rfloor\) 个已完成块`, body: "dense 表示没有 top-k；当前未闭合块不进入全局分支，由滑窗覆盖。" },
          { label: "输出坐标", title: "逆旋转、RMSNorm 与 sink", body: R`用 \(R_{-t}\) 得到 \(\pi_j-t\)；core 前归一化 query/KV，sink logit 加在 softmax 分母。` }
        ],
        caveat: "HCA 还并联未压缩滑窗；全局压缩分支与局部分支的位置索引/频率配置必须按官方实现对齐。"
      },
      derivationSourceFallback: "DeepSeek-V4 Technical Report (2026), §2.3.2（HCA）、§2.3.3（RMSNorm/RoPE/sink/SWA）与 §2.3.4（efficiency）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "derivation", level: "intermediate" },
        { kind: "design", level: "intermediate" }
      ],
      derivations: [
        {
          title: "逐通道 softmax 让每个压缩维保持凸组合",
          body: R`**原式。** 对块 \(\mathcal B_i\) 和通道 \(r\)，
            \[
            s_{j,r}=\frac{\exp(z_{j,r}+b_{j,r})}
            {\sum_{\ell\in\mathcal B_i}\exp(z_{\ell,r}+b_{\ell,r})},
            \qquad
            c_{i,r}^{Comp}=\sum_{j\in\mathcal B_i}s_{j,r}c_{j,r}.
            \]
            **补全代数。** softmax 给出 \(s_{j,r}\ge0\) 且
            \(\sum_{j\in\mathcal B_i}s_{j,r}=1\)，故每个输出通道位于该块对应输入通道值的凸包内。
            **张量形状。** 块输入 \(C_i,Z_i\in\mathbb R^{m'\times c}\)，权重
            \(S_i\in\mathbb R^{m'\times c}\)，输出 \(c_i^{Comp}\in\mathbb R^{c}\)。
            **直观。** 不同通道可从块内不同 token 摘要信息，而不是全向量共用一个标量权重。**边界。**
            凸组合性质只针对投影后的 C 通道；前后线性层仍可产生块外数值范围，且 128→1 仍不可逆。`,
          source: "DeepSeek-V4 Technical Report (2026), §2.3.2, Eqs. (20)–(23)"
        },
        {
          title: "重压缩 dense attention 的精确位置对计数",
          body: R`**原式。** 0-based query \(t\) 只能看块号 \(j<\lfloor t/m'\rfloor\)，故已闭合块数
            \(n_c(t)=\lfloor t/m'\rfloor\)，另看至多 \(w\) 个局部 token。**补全代数。**
            \[
            N_{\text{pairs}}\le
            \sum_{t=0}^{L-1}\left(\left\lfloor\frac t{m'}\right\rfloor+w\right)
            =\Theta(L^2/m'+Lw).
            \]
            可用 padded 矩形上界 \(L\lfloor L/m'\rfloor+Lw\)，但它不是精确 causal count。**张量形状。**
            padded 全局 score 可写 \([B,H,L,\lfloor L/m'\rfloor]\)，mask 后每行有效宽度不同；局部 score 逻辑上为 \([B,H,L,w]\)。**直观。**
            HCA 完整读一份短目录，再查最近原文。**边界。** 当前未闭合块不能提前进入全局摘要；实际 causal 有效 pair 少于矩形上界，固定
            \(m'\) 时主项仍为 \(\Theta(L^2)\)，只有明确让 \(m'\) 随 L 增长才可改变渐近阶。`,
          source: "DeepSeek-V4 Technical Report (2026), §2.3.2（HCA）、§2.3.3（causal sliding window）与 §2.3.4（efficiency）"
        }
      ],
      exercises: [
        {
          kind: "counterexample",
          level: "advanced",
          q: "给出一个 HCA 全局摘要必然丢失精确顺序的块内任务。",
          hint: "让任务要求恢复 128 个独立符号的完整排列。",
          answer: "若一个块包含 128 个独立随机符号，查询要求输出它们的完整排列，而全局只保留一个固定维 entry，则不同排列会发生表示碰撞；离开局部窗口后不能保证精确恢复。"
        },
        {
          kind: "code-shape",
          level: "advanced",
          q: R`HCA 输入长度 1000、\(m'=128\)。若只缓存完整块，产生多少全局 entry，余下多少 token 应由 carry/局部分支覆盖？`,
          hint: "整数除法和余数。",
          answer: R`\(\lfloor1000/128\rfloor=7\) 个完整 entry，余数 \(1000-7\times128=104\) 个 token。不能把这 104 个未来未闭合内容提前压入可见全局 entry。`
        },
        {
          kind: "derivation",
          level: "advanced",
          q: R`证明对正交 RoPE 矩阵，输出逆旋转把 compressed position \(R_{\pi_j}\) 变成相对旋转 \(R_{\pi_j-t}\)。`,
          hint: R`使用 \(R_{-t}=R_t^\top\) 和旋转群 \(R_aR_b=R_{a+b}\)。`,
          answer: R`\(R_{-t}R_{\pi_j}=R_{\pi_j-t}\)，故 \(R_{-t}\sum_ja_jR_{\pi_j}c_j=\sum_ja_jR_{\pi_j-t}c_j\)。若把 \(\pi_j\) 擅自换成 compressed index j，或不同位置使用不一致频率，等价关系会被破坏。`
        }
      ]
    },

    linear: {
      attentionConfig: {
        omit: true,
        model: "Linear (ours) · autoregressive MNIST",
        scope: "Katharopoulos et al. (2020) 表 1 的代表模型；全 8 层均使用 kernelized causal linear attention。",
        items: [
          { label: "Hidden size", value: "256", note: "embedding / model width" },
          { label: "Attention heads", value: "8", note: "每层" },
          { label: "Head width", value: "32", note: "论文给出的 dimensions per head" },
          { label: "Feature map", value: "ELU(x) + 1", note: "正值 kernel feature" },
          { label: "Recurrent state", value: "S: C×M · z: C", note: "论文未分别给出数值 C、M" },
          { label: "ShortConv / gates", value: "None / None", note: "不是 2020 原始结构的一部分" }
        ],
        sources: [
          { label: "Katharopoulos et al. (2020), §3.2, §4.2.1 and Table 1", url: "https://proceedings.mlr.press/v119/katharopoulos20a.html" }
        ],
        caveat: "论文只统一报告每头 32 维，没有分别列出 d_k、d_v、C、M；状态数值形状不作额外推断。"
      },
      positionEncoding: {
        title: "核线性化不自动提供位置编码",
        summary: "《Transformers are RNNs》的核心是核分解与因果递推，不是 prefix-scan/chunk 实现。整个 causal 输出序列对顺序敏感，因为每个时刻看到的前缀不同；但给定同一组写入，无衰减加法状态的最终汇总对排列不敏感，限制了状态内部编码顺序的能力。",
        equation: R`\[
          S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
          y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t+\varepsilon}.
        \]`,
        steps: [
          { label: "原论文", title: "递推公式不等于 scan/chunk 声明", body: "论文可用逐步 recurrence 计算 causal attention；本站不把后来的并行 prefix-scan 或 chunk kernel 归于其 2020 原实现。" },
          { label: "因果顺序", title: "输出轨迹敏感，终态汇总可交换", body: R`交换 token 会改变各时刻的可见前缀和输出；但若只比较写入同一组 \((k,v)\) 后的终态，\(\sum_j\phi(k_j)v_j^\top\) 不记录排列。` },
          { label: "现代实现", title: "ShortConv/decay 是架构扩展", body: "后续 gated linear attention、DeltaNet 等用局部卷积和有序状态转移增强位置感；不应把这些倒写进 2020 原式。" }
        ],
        caveat: "直接给核特征套标准 RoPE 未必保持可结合的非负核与归一化性质；必须针对具体线性 attention 公式验证。"
      },
      derivationSourceFallback: "Katharopoulos et al. (2020), §3（linear attention and causal formulation）",
      existingExerciseMeta: [
        { kind: "derivation", level: "foundation" },
        { kind: "code-shape", level: "foundation" }
      ],
      derivations: [
        {
          title: "归一化分母也能用一个固定状态累计",
          body: R`**原式。**
            \[
            y_t=\frac{\sum_{j\le t}\phi(q_t)^\top\phi(k_j)v_j}
            {\sum_{j\le t}\phi(q_t)^\top\phi(k_j)}.
            \]
            **补全代数。** 将与 \(j\) 无关的 \(\phi(q_t)^\top\) 提到求和外：
            \[
            S_t=\sum_{j\le t}\phi(k_j)v_j^\top,\quad
            z_t=\sum_{j\le t}\phi(k_j),\quad
            y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}.
            \]
            **张量形状。** 若 \(\phi(q),\phi(k)\in\mathbb R^r\)、\(v\in\mathbb R^{d_v}\)，则
            \(S_t\in\mathbb R^{r\times d_v}\)、\(z_t\in\mathbb R^r\)、\(y_t\in\mathbb R^{d_v}\)。
            **直观。** 分子维护“按特征分类的 value 总账”，分母维护“每类共写入多少权重”。**边界。**
            要像概率平均一样解释，核相似度通常需非负；实现还要加 \(\varepsilon\) 防止分母接近 0，这会引入小的数值偏差。`,
          source: "Katharopoulos et al. (2020), §3.2–3.3, normalized and causal linear attention"
        },
        {
          title: "线性复杂度的隐藏维度条件",
          body: R`**原式。** 每步更新外积 \(S_t\leftarrow S_{t-1}+\phi(k_t)v_t^\top\)。**补全代数。**
            单步写入和读取分别为 \(\Theta(rd_v)\)，长度 L 总成本
            \[
            C_{\text{linear}}=\Theta(Lrd_v),\qquad
            M_{\text{state}}=\Theta(rd_v+r).
            \]
            dense attention 为 \(\Theta(L^2d_h)\)。**张量形状。** 批量 H 头状态为
            \([B,H,r,d_v]\)，与 L 无关；并行训练的 Q/K/V 仍为 \([B,H,L,\cdot]\)。**直观。**
            用固定宽统计量换掉位置轴。该复杂度来自 recurrence/结合律，不声称原实现用了 prefix scan 或 chunk。**边界。** “关于 L 线性”假设 \(r,d_v\) 不随 L 增长；若为逼近 softmax 而让 \(r=\Theta(L)\)，则成本重新变成平方级。`,
          source: "Katharopoulos et al. (2020), §3.4 与 complexity discussion"
        }
      ],
      exercises: [
        {
          kind: "complexity",
          level: "intermediate",
          q: R`比较 \(L=4096,d_h=d_v=r=128\) 时单头 dense 的 \(L^2d_h\) 与 linear 的 \(Lrd_v\) 乘法量级。`,
          hint: R`两者比值是 \(L/r\)。`,
          answer: R`比值约为 \(4096/128=32\)。这是核心乘法量级；linear 的递推/scan、归一化和 dense kernel 的 IO 优化会改变实际速度。`
        },
        {
          kind: "counterexample",
          level: "advanced",
          q: "构造两个不同历史顺序，使无衰减加法线性状态完全相同。",
          hint: "交换两组 key-value 写入。",
          answer: R`历史 A 依次写 \((k_1,v_1),(k_2,v_2)\)，历史 B 反序写入。两者最终 \(S=\phi(k_1)v_1^\top+\phi(k_2)v_2^\top\)、\(z=\phi(k_1)+\phi(k_2)\) 完全相同，说明最终状态本身不能辨别这两个写入顺序。`
        },
        {
          kind: "design",
          level: "advanced",
          q: "若需要线性成本又要强化局部词序，如何设计最小改动的消融？",
          hint: "比较无位置、ShortConv、显式 embedding 与小窗口混合。",
          answer: "固定主体和训练预算，比较纯递推、Q/K/V 前 causal depthwise ShortConv、输入 learned position、周期性小窗口 attention；分别测困惑度、局部顺序任务、长程检索、状态大小与吞吐，避免把卷积收益误归因于核映射。"
        },
        {
          kind: "derivation",
          level: "advanced",
          q: R`若 \(\phi(x)=\operatorname{elu}(x)+1\)，为什么分母非负？它能否保证严格大于 0？`,
          hint: R`\(\operatorname{elu}(x)+1>0\) 对有限 x 成立。`,
          answer: R`各维特征为正，因此 \(\phi(q)^\top\phi(k_j)>0\)，只要前缀至少有一个有限 key，求和分母严格为正。有限精度下可能下溢或输入异常，工程上仍加 \(\varepsilon\)。`
        }
      ]
    },

    "gated-delta": {
      attentionConfig: {
        omit: true,
        model: "Pure Gated DeltaNet · 1.3B",
        scope: "论文 100B-token 对照模型及官方仓库；公开材料对 head count/head dim 存在冲突，以下保留各自口径。",
        items: [
          { label: "Hidden size", value: "2400", note: "官方 1.3B config" },
          { label: "Layers", value: "16", note: "全部为 Gated DeltaNet layers" },
          { label: "Head count", value: "16 config / 9 implementation", note: "仓库 config 未把 n_head 传给 block；作者确认实验使用 9" },
          { label: "Head dim", value: "128 paper / 192 author clarification", note: "两份一级材料不一致" },
          { label: "ShortConv width", value: "4", note: "官方实现默认值；仅 Q/K/V 路径" },
          { label: "State per head", value: "d_v × d_k", note: "冲突未解，避免给出伪精确数值" },
          { label: "Gates", value: "scalar α and scalar β / head / token", note: "decay 与 update rate" }
        ],
        sources: [
          { label: "Yang et al. (2024), §§3.1–3.3 and Appendix Table S.1", url: "https://arxiv.org/abs/2412.06464" },
          { label: "Official NVlabs GatedDeltaNet config", url: "https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/config.py" },
          { label: "Author clarification on released experiment dimensions", url: "https://github.com/NVlabs/GatedDeltaNet/issues/10" }
        ],
        caveat: "这里不能把 16 heads、9 heads、128 dim 与 192 dim 拼成一个并不存在的统一 checkpoint，因此本章不展示代表参数卡。"
      },
      positionEncoding: {
        title: "现代 DeltaNet Block：内容寻址、局部位移与稳定读出",
        summary: R`作者博客 Part III 的现代化路线不是给 q/k 套 RoPE，而是把 token mixer 重构为 Q/K: Linear→ShortConv→SiLU→L2Norm，V: Linear→ShortConv→SiLU，\(\beta\): Linear→Sigmoid；delta 读出后做 head-wise RMSNorm 再投影。ShortConv 提供局部顺序 shortcut，有序 delta transition 提供全局顺序。`,
        equation: R`\[
          q_t=d_k^{-1/2}\operatorname{L2Norm}(\operatorname{SiLU}(\operatorname{ShortConv}(W^qx)_t)),
          \quad
          k_t=\operatorname{L2Norm}(\operatorname{SiLU}(\operatorname{ShortConv}(W^kx)_t)),
        \]
        \[
          v_t=\operatorname{SiLU}(\operatorname{ShortConv}(W^vx)_t),\qquad
          \beta_t=\sigma(W^\beta x_t),
        \]
        \[
          S_t=S_{t-1}+\beta_tk_t(v_t-S_{t-1}^\top k_t)^\top,\qquad
          y_t=W^O\operatorname{RMSNorm}(S_t^\top q_t).
        \]`,
        steps: [
          { label: "几何", title: "L2Norm 把 erase 变成方向投影", body: R`\(\|k_t\|=1\) 时 \(I-\beta_tk_tk_t^\top\) 只改变 key 方向的特征值；\(\beta=1\) 精确擦除该方向旧分量，正交方向保持不变。` },
          { label: "局部顺序", title: "ShortConv 是单层 induction shortcut", body: "width-4 depthwise causal convolution 让 q/k/v 在进入全局状态前混合最近 token，显著改善局部寻址与检索；它不是 KV cache，也不替代长程状态。" },
          { label: "稳定读出", title: "去掉正分母，改用输出 RMSNorm", body: "现代 DeltaNet 允许 SiLU 产生有符号特征，不再依赖正核归一化分母；对每头读出做 RMSNorm 后再写回残差流。" }
        ],
        caveat: "本式描述作者博客中的纯 DeltaNet block；Gated DeltaNet 会在纠错前额外加入标量 retention \u03b1。ShortConv、L2Norm、RMSNorm 与 chunk kernel 都是神经架构/执行配方，不应与 delta update 的数学定义混为一谈。"
      },
      derivationSourceFallback: "Yang et al. (2024), Parallelizing Linear Transformers with the Delta Rule；Songlin Yang, DeltaNet Explained Parts I–III；Yang et al. (2025), Gated Delta Networks",
      existingExerciseMeta: [
        { kind: "derivation", level: "intermediate" },
        { kind: "derivation", level: "intermediate" }
      ],
      derivations: [
        /*
         * Temporarily hidden together with the previous sixth-section content.
        {
          title: "L2Norm 为什么让 Delta update 具有清晰投影几何",
          body: R`把纯 DeltaNet 状态更新展开为
            \[
            S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
            \]
            令 \(M_t=I-\beta_tk_tk_t^\top\)。对任意与 \(k_t\) 正交的
            \(u\)，有 \(M_tu=u\)；沿 key 方向则
            \[
            M_tk_t=(1-\beta_t\|k_t\|^2)k_t.
            \]
            因此特征值为：key 方向 \(1-\beta_t\|k_t\|^2\)，其余
            \(d_k-1\) 个正交方向均为 1。若 L2Norm 保证 \(\|k_t\|=1\)
            且 \(0\le\beta_t\le1\)，所有特征值落在 \([0,1]\)；当
            \(\beta_t=1\) 时 \(I-k_tk_t^\top\) 正是擦除 key 方向的正交投影。

            这解释了 L2Norm 的双重作用：稳定 transition 的谱，同时把“erase old
            value”变成可视化的方向操作。作者也指出一个自然延伸：若令有效步长覆盖
            \([0,2]\)，key 方向特征值可进入 \([-1,1]\)，在保持谱半径不超过 1 的同时
            允许符号翻转，从而扩展状态跟踪能力；这是一条潜在改进，不是原始默认配置。`,
          source: "Songlin Yang, DeltaNet Explained Part III（Normalization on Queries and Keys）"
        },
        {
          title: "为什么朴素 parallel scan 失败，而 WY / UT chunk 可行",
          body: R`在作者常用的转置状态 \(F=S^\top\in\mathbb R^{d_v\times d_k}\)
            下，每步递推写成
            \[
            F_t=F_{t-1}M_t+X_t,\qquad
            M_t=I-\beta_tk_tk_t^\top,\quad X_t=\beta_tv_tk_t^\top.
            \]
            二元组合 \((M_i,X_i)\bullet(M_j,X_j)
            =(M_iM_j,\,X_iM_j+X_j)\) 确实满足结合律，但直接 scan 会不断合并
            \(d_k\times d_k\) transition，并把大量 \(d_v\times d_k\) 中间状态写入
            HBM。低秩项的数量随树层扩张，最终要么结构膨胀，要么退化为昂贵稠密乘法；
            “数学可 scan”不等于“GPU 上值得 scan”。

            WY 表示把一个 chunk 内的 transition 乘积压成
            \[
            \prod_{r=1}^{C}(I-\beta_rk_rk_r^\top)
            =I-W^\top K.
            \]
            递归的 \(W,U\) 可由严格下三角依赖矩阵
            \[
            A=\operatorname{tril}(-\operatorname{diag}(\beta)KK^\top,-1),
            \quad T=(I-A)^{-1},
            \]
            写成 \(W=T\operatorname{diag}(\beta)K,\,
            U=T\operatorname{diag}(\beta)V\)。实现使用单位下三角求解而不是通用求逆。
            于是只在 chunk 边界保存 \(F_{[i]}\)，chunk 内输出与边界更新主要变成
            \(QK^\top\)、\(WF^\top\)、\(U^\top K\) 等矩阵乘，既避免保存全部 states，
            又能利用 Tensor Cores。`,
          source: "Songlin Yang, DeltaNet Explained Part II（failed scan, WY representation and UT transform）"
        }
        */
      ],
      exercises: [
        {
          kind: "derivation",
          level: "advanced",
          q: R`求 \(M=I-\beta kk^\top\) 的特征值。若 \(\|k\|=1\)，分别讨论 \(\beta=0,1,2\) 的几何作用。`,
          hint: "分解为 k 方向与其正交补。",
          answer: R`沿 \(k\) 的特征值为 \(1-\beta\|k\|^2\)，正交补上的特征值均为 1。单位 key 下：\(\beta=0\) 为恒等映射；\(\beta=1\) 擦除 k 方向，是正交投影；\(\beta=2\) 在 k 方向取 -1、其余方向取 1，是 Householder reflection。`
        },
        {
          kind: "code-shape",
          level: "intermediate",
          q: R`批量实现中 Q/K 为 \([B,L,H,d_k]\)、V 为 \([B,L,H,d_v]\)。写出 state、单步 prediction/error/correction 的形状，并指出哪些量不含 L 轴。`,
          hint: R`prediction 是 \(S^\top k_t\)，correction 是 \(k_te_t^\top\)。`,
          answer: R`state 为 \([B,H,d_k,d_v]\)；单步 prediction 与 error 为 \([B,H,d_v]\)；correction 为 \([B,H,d_k,d_v]\)。持久 state 与单步量都不含 L 轴，只有训练时的 Q/K/V 序列和 chunk 中间量保留长度轴。`
        },
        {
          kind: "complexity",
          level: "advanced",
          q: "为什么把 DeltaNet 的结合递推直接做 parallel scan 通常不如 chunkwise WY/UT？至少从算量、HBM 中间状态和 Tensor Core 利用三个角度回答。",
          hint: "可结合只保证结果能重排，不保证重排后的对象仍小且硬件友好。",
          answer: R`直接 scan 组合的是 \(d_k\times d_k\) transition 与 \(d_v\times d_k\) update；低秩项会随层级扩张，稠密实现又引入高阶矩阵乘，并需物化大量 \([L,d_v,d_k]\) 级中间状态到 HBM。WY/UT 只保留 chunk 边界 state，把 chunk 内依赖压成 C×C 下三角求解，并将主要工作改写为 QKᵀ、UᵀK 等大矩阵乘，因此减少 HBM 流量且更能利用 Tensor Cores。`
        },
        {
          kind: "design",
          level: "advanced",
          q: "为 24 层 DeltaNet 设计一次混合消融：纯 DeltaNet、插入滑窗层、插入两个 global-attention 层。应固定什么、报告什么，才能判断“延伸性”而非只看困惑度？",
          hint: "同时控制参数/训练 token，并测精确检索、缓存和端到端速度。",
          answer: "固定模型宽度、总参数量、训练 token、数据与优化器；滑窗方案固定窗口并说明层位，global 方案可按作者博客放在第 2 层和中部附近。报告语言建模损失、常识任务、多针/逐字/长距离检索、训练吞吐、Decode TPOT、固定 state 大小及新增 KV cache。滑窗若只改善局部任务而 global 显著改善远程精确检索，就能区分局部 shortcut 与真正的全局回看能力。"
        }
      ]
    },

    kda: {
      attentionConfig: {
        model: "Kimi K3 · 2.8T / 104B activated",
        scope: "MoonshotAI 官方 Kimi K3 配置；同一 93 层骨干按 3:1 主节奏组合 KDA 与 Gated MLA，并以末层 Gated MLA 收尾。",
        items: [
          { label: "Total / activated parameters", value: "2.8T / 104B", note: "稀疏 MoE 总量与单 token 激活量" },
          { label: "Hidden size", value: "7168", note: "attention residual width" },
          { label: "Attention heads", value: "96", note: "KDA 与 Gated MLA 的代表头数" },
          { label: "K / V head dim", value: "128 / 128", note: "KDA q/k/v channel width" },
          { label: "State per head", value: "128 × 128", note: "固定大小 recurrent matrix" },
          { label: "ShortConv width", value: "4", note: "Q/K/V 路径" },
          { label: "KDA gate", value: "128-channel decay + scalar β", note: "每 head、每 token；输出另有 full-rank gate" },
          { label: "Layer mix", value: "69 KDA + 24 Gated MLA", note: "共 93 层；23×(3 KDA + 1 MLA) 后再接 1 个 MLA" },
          { label: "Context length", value: "1,048,576", note: "官方 1M 上下文" }
        ],
        sources: [
          { label: "Kimi K3 Technical Report (2026)", url: "https://arxiv.org/abs/2607.24653" },
          { label: "Official MoonshotAI/Kimi-K3 repository", url: "https://github.com/MoonshotAI/Kimi-K3" },
          { label: "Official FlashKDA kernel interface", url: "https://github.com/MoonshotAI/FlashKDA" }
        ],
        caveat: "Kimi K3 的 69:24 是层级混合，不是单层内按 head 混合。KDA 使用固定 128×128/头状态；Gated MLA 仍保存随上下文增长的 latent cache。报告未公开的 Gated MLA 细分 rank 不在此处补写；正文递推与教学实现仍以首发 Kimi Linear 的 KDA 定义为算子来源。"
      },
      positionEncoding: {
        title: "Kimi Linear 用 NoPE；KDA 自己承担位置感",
        summary: "Kimi Linear 报告明确让全局 MLA 层使用 NoPE：不添加显式 RoPE/absolute PE，但仍对每个历史 token 的 low-rank MLA cache 做 causal global softmax。位置与 recency 主要来自 KDA 的数据依赖逐通道 decay、有序转移和 q/k/v ShortConv。",
        equation: R`\[
          S_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
          +\beta_tk_tv_t^\top,\qquad
          \beta_t=\sigma(W^\beta x_t),
        \]
        \[
          \log\alpha_{t,h,r}
          =-\exp(A^{\log}_h)\,
          \operatorname{softplus}\!\left(
          [W_\alpha^\uparrow W_\alpha^\downarrow x_t]_{h,r}
          +b^{\Delta}_{h,r}\right).
        \]`,
        steps: [
          { label: "参数路径", title: R`q/k/v 卷积，\(\beta\) 直接线性`, body: R`q/k/v 使用 ShortConv+Swish（q/k 再 L2Norm）；\(\beta=\sigma(W^\beta x)\) 不经过 ShortConv。` },
          { label: "长程", title: "逐通道乘积形成可学习距离衰减", body: R`从位置 \(s\) 到 \(t\) 的某通道保留量包含 \(\prod_{u=s+1}^{t}\alpha_{u,r}\)，天然依赖经过的有序步数与内容。` },
          { label: "发布配置", title: "20 KDA + 7 NoPE MLA", body: "27 层 checkpoint 的 MLA 层为 1-based {4,8,12,16,20,24,27}；NoPE 表示这些层无显式 PE，不表示它们没有 causal token index 或 low-rank KV cache。" }
        ],
        caveat: "后续采用 KDA 的模型可以选择不同混合层位置方案；但不能把后续实现的 RoPE 反推为 Kimi Linear 原报告的 KDA 机制。"
      },
      derivationSourceFallback: "Kimi Linear Technical Report (2025), §2.2–§3 与 released Kimi-Linear-48B-A3B config/modeling code",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "derivation", level: "advanced" },
        { kind: "design", level: "intermediate" }
      ],
      derivations: [
        {
          title: "KDA 是受约束 DPLR，而不是任意 DPLR",
          body: R`**原式。**
            \[
            A_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t).
            \]
            **补全代数。** 因
            \(k_t^\top\operatorname{Diag}(\alpha_t)=(k_t\odot\alpha_t)^\top\)，
            \[
            A_t=\operatorname{Diag}(\alpha_t)
            -\beta_tk_t(k_t\odot\alpha_t)^\top
            =D_t-a_tb_t^\top,
            \]
            其中 \(D_t=\operatorname{Diag}(\alpha_t)\)、\(a_t=\beta_tk_t\)、
            \(b_t=k_t\odot\alpha_t\)。**张量形状。**
            \(D_t,A_t\in\mathbb R^{d_k\times d_k}\)，\(a_t,b_t,k_t,\alpha_t\in\mathbb R^{d_k}\)，
            \(S_t\in\mathbb R^{d_k\times d_v}\)。**直观。** 对角项给每个通道独立沙漏，rank-1 项沿 key 方向纠错。**边界。**
            一般 DPLR 允许自由 \(a_t,b_t\)；KDA 把二者绑定到同一 key 和 gate，减少表达自由度以换取更稳定、更少高精度子块计算的 chunk kernel。`,
          source: "Kimi Linear Technical Report (2025), §3, Eqs. (1)–(9)"
        }
      ],
      exercises: [
        {
          kind: "counterexample",
          level: "advanced",
          q: "给出一个一般 DPLR 转移无法由固定 KDA 参数化表示的自由度论证。",
          hint: R`一般 rank-1 的左右向量 \(a,b\) 可独立；KDA 要求 \(a\parallel k\) 且 \(b=k\odot\alpha\)。`,
          answer: R`在二维中取 \(a=(1,0)^\top,b=(0,1)^\top\)。KDA 若 \(a=\beta k\)，则 \(k\parallel(1,0)\)，于是 \(k\odot\alpha\) 的第二维必为 0，不可能平行于 \(b=(0,1)\)。故该一般 DPLR rank-1 项不在 KDA 约束族内。`
        },
        {
          kind: "code-shape",
          level: "advanced",
          q: R`KDA 每头 \(d_k=d_v=128\)，batch 8、32 头。递推状态有多少元素？与长度 1M 是否相关？`,
          hint: R`状态形状为 \([B,H,d_k,d_v]\)。`,
          answer: R`元素数为 \(8\times32\times128\times128=4{,}194{,}304\)。单层状态不随 1M 长度增长；训练时 chunk 中间量和周期性 MLA cache 另计。`
        },
        {
          kind: "derivation",
          level: "advanced",
          q: R`忽略 rank-1 delta，推导通道 r 从时刻 s 写入到 t 后的保留系数。`,
          hint: "连续展开对角 gate。",
          answer: R`若 \(S_u=\operatorname{Diag}(\alpha_u)S_{u-1}\)，则第 r 行满足 \(S_{t,r}=(\prod_{u=s+1}^{t}\alpha_{u,r})S_{s,r}\)。当 \(\alpha_{u,r}=\alpha_r\) 恒定时为 \(\alpha_r^{t-s}\)，从而产生通道特定的距离衰减。`
        }
      ]
    }
  };

  var attentionConfigMaps = {
    mha: {
      modules: [
        {
          id: "x", kind: "activation", tone: "cyan",
          x: 262, y: 566, w: 136, h: 46,
          label: "输入残差流",
          tex: R`X`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "X", shapeFallback: "[B,L,d_model]",
          description: "每个 token 以模型宽度进入三套独立的 Q/K/V 投影；head 轴尚未展开。",
          symbols: [
            { tex: R`X`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 512", note: "Transformer base 残差宽度（Table 3）" }
          ]
        },
        {
          id: "wq", kind: "weight", glyph: "down",
          x: 94, y: 474,
          label: "Query 投影权重",
          tex: R`W^{Q}`, texShape: R`[d_{\mathrm{model}},\,H_q d_h]`,
          fallback: "W^Q", shapeFallback: "[d,Hq·dh]",
          relates: ["q"],
          description: "8 个独立 query head 的合并投影；逐头拆开等价于 8 个 512×64 矩阵。",
          symbols: [
            { tex: R`W^{Q}`, shape: R`[d_{\mathrm{model}},\,H_q d_h]`, label: "Query projection", value: "512 × (8 × 64) = 512 × 512", note: "逐头等价于 8 个独立的 W_h^Q ∈ R^{512×64}" }
          ]
        },
        {
          id: "wk", kind: "weight", glyph: "down",
          x: 274, y: 474,
          label: "Key 投影权重",
          tex: R`W^{K}`, texShape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`,
          fallback: "W^K", shapeFallback: "[d,Hkv·dh]",
          relates: ["k"],
          description: "MHA 中 H_kv = H_q，每个 query head 拥有独立的一套 key 投影。",
          symbols: [
            { tex: R`W^{K}`, shape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`, label: "Key projection", value: "512 × 512（H_kv = 8）", note: "与 W^Q 形状相同、参数独立" }
          ]
        },
        {
          id: "wv", kind: "weight", glyph: "down",
          x: 454, y: 474,
          label: "Value 投影权重",
          tex: R`W^{V}`, texShape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`,
          fallback: "W^V", shapeFallback: "[d,Hkv·dh]",
          relates: ["v"],
          description: "value 投影与 K 同构；三套投影可在实现中合并成一次线性层。",
          symbols: [
            { tex: R`W^{V}`, shape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`, label: "Value projection", value: "512 × 512（H_kv = 8）", note: "实现常把 QKV 合并为 [512, 3·512] 一次投影" }
          ]
        },
        {
          id: "q", kind: "activation",
          x: 86, y: 380,
          label: "多头 Query 激活",
          tex: R`Q`, texShape: R`[B,\,H_q,\,L,\,d_h]`,
          fallback: "Q", shapeFallback: "[B,Hq,L,dh]",
          description: "8 个独立 query heads，每头 64 维；只存在于本步计算，不进缓存。",
          symbols: [
            { tex: R`Q`, shape: R`[B,\,H_q,\,L,\,d_h]`, label: "Query heads", value: "H_q = 8, d_h = 64", note: "reshape/transpose 自 [B,L,512]" }
          ]
        },
        {
          id: "k", kind: "activation",
          x: 266, y: 380,
          label: "多头 Key 激活",
          tex: R`K`, texShape: R`[B,\,H_{kv},\,L,\,d_h]`,
          fallback: "K", shapeFallback: "[B,Hkv,L,dh]",
          description: "每个 query head 对应一套独立 key；增量解码时逐步追加进 KV cache。",
          symbols: [
            { tex: R`K`, shape: R`[B,\,H_{kv},\,L,\,d_h]`, label: "Key heads", value: "H_kv = 8, d_h = 64", note: "MHA 的缓存宽度含完整 head 轴" }
          ]
        },
        {
          id: "v", kind: "activation",
          x: 446, y: 380,
          label: "多头 Value 激活",
          tex: R`V`, texShape: R`[B,\,H_{kv},\,L,\,d_h]`,
          fallback: "V", shapeFallback: "[B,Hkv,L,dh]",
          description: "value 与 key 同形；softmax 权重决定每个位置读取多少 value 内容。",
          symbols: [
            { tex: R`V`, shape: R`[B,\,H_{kv},\,L,\,d_h]`, label: "Value heads", value: "H_kv = 8, d_h = 64", note: "与 K 一起构成 KV cache 的另一半" }
          ]
        },
        {
          id: "cache", kind: "state",
          x: 626, y: 378,
          label: "完整 KV cache",
          tex: R`K_{1:t},V_{1:t}`, texShape: R`2\times[B,\,H_{kv},\,L,\,d_h]`,
          fallback: "K/V cache", shapeFallback: "2×[B,Hkv,L,dh]",
          relates: ["k", "v"],
          description: "现代增量解码的叠加层：为 8 个 heads 分别保存历史 K 与 V。",
          symbols: [
            { tex: R`K_{1:t},V_{1:t}`, shape: R`2\times[B,\,H_{kv},\,L,\,d_h]`, label: "Persistent state", value: "2 × 8 × 64 = 1024 elements/token/layer", note: "2017 训练图没有 cache；此为现代解码口径" }
          ]
        },
        {
          id: "attn", kind: "activation", titleSize: 8.8,
          x: 130, y: 286, w: 220, h: 50,
          label: "缩放点积注意力权重",
          tex: R`A=\operatorname{softmax}\!\big(\tfrac{QK^{\top}}{\sqrt{d_h}}+M\big)`,
          texShape: R`[B,\,H_q,\,L,\,L]`,
          fallback: "A = softmax(QK^T/√dh + M)", shapeFallback: "[B,Hq,L,L]",
          description: "每个 head 独立形成 L×L score 并做 softmax；mask M 只出现在 decoder 子层。",
          symbols: [
            { tex: R`A`, shape: R`[B,\,H_q,\,L,\,L]`, label: "Attention weights", value: "scale = 1/√64", note: "encoder 双向、decoder causal，取决于子层" }
          ]
        },
        {
          id: "mul", kind: "operator",
          x: 493, y: 294,
          label: "A·V 加权读取",
          tex: R`\otimes`, fallback: "⊗",
          description: "softmax 权重与 value 的批量矩阵乘：每个 head 独立读取历史内容。",
          symbols: [
            { tex: R`AV`, shape: R`[B,\,H_q,\,L,\,d_h]`, label: "Weighted read", value: "每头独立 L×L·L×64", note: "精确 attention，无任何近似" }
          ]
        },
        {
          id: "o", kind: "activation",
          x: 446, y: 196,
          label: "逐头输出",
          tex: R`O_h`, texShape: R`[B,\,H_q,\,L,\,d_h]`,
          fallback: "O_h", shapeFallback: "[B,Hq,L,dh]",
          description: "各 head 的读取结果，等待拼接后写回残差宽度。",
          symbols: [
            { tex: R`O_h`, shape: R`[B,\,H_q,\,L,\,d_h]`, label: "Per-head output", value: "8 × 64 = 512 拼接宽度", note: "concat 后进入输出投影" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 454, y: 112,
          label: "输出投影权重",
          tex: R`W^{O}`, texShape: R`[H_q d_h,\,d_{\mathrm{model}}]`,
          fallback: "W^O", shapeFallback: "[Hq·dh,d]",
          relates: ["y"],
          description: "把拼接后的 head 输出线性映射回残差流宽度。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[H_q d_h,\,d_{\mathrm{model}}]`, label: "Output projection", value: "512 × 512", note: "使不同 head 的信息重新混合" }
          ]
        },
        {
          id: "y", kind: "activation", tone: "gather",
          x: 446, y: 28,
          label: "层输出",
          tex: R`Y`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "Y", shapeFallback: "[B,L,d_model]",
          description: "attention 子层输出，随后进入 2017 原始的 post Add & Norm。",
          symbols: [
            { tex: R`Y=\operatorname{Concat}(O_h)W^{O}`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 512", note: "与残差流宽度一致" }
          ]
        }
      ],
      edges: [
        { from: "x", to: "wq", tone: "compute", bend: 540 },
        { from: "x", to: "wk", tone: "compute" },
        { from: "x", to: "wv", tone: "compute", bend: 540 },
        { from: "wq", to: "q", tone: "weight" },
        { from: "wk", to: "k", tone: "weight" },
        { from: "wv", to: "v", tone: "weight" },
        { from: "q", to: "attn", tone: "compute", toAt: 0.25, bend: 356 },
        { from: "k", to: "attn", tone: "compute", toAt: 0.75, bend: 348 },
        { from: "k", to: "cache", tone: "state", dashed: true, fromSide: "top", fromAt: 0.8, toSide: "top", bend: 364, label: "decode 追加", lx: 560, ly: 354 },
        { from: "v", to: "cache", tone: "state", dashed: true },
        { from: "v", to: "mul", tone: "compute" },
        { from: "attn", to: "mul", tone: "compute" },
        { from: "mul", to: "o", tone: "gather" },
        { from: "o", to: "wo", tone: "gather" },
        { from: "wo", to: "y", tone: "gather" }
      ]
    },
    mqa: {
      modules: [
        {
          id: "x", kind: "activation", tone: "cyan",
          x: 262, y: 566, w: 136, h: 46,
          label: "输入残差流",
          tex: R`X`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "X", shapeFallback: "[B,L,d_model]",
          description: "输入同时进入多头 query 投影和单份共享 K/V 投影。",
          symbols: [
            { tex: R`X`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 1024", note: "Shazeer 2019 WMT 代表模型；learned PE 在其前相加" }
          ]
        },
        {
          id: "wq", kind: "weight", glyph: "down",
          x: 94, y: 474,
          label: "多头 Query 投影",
          tex: R`W^{Q}`, texShape: R`[d_{\mathrm{model}},\,H_q d_h]`,
          fallback: "W^Q", shapeFallback: "[d,Hq·dh]",
          relates: ["q"],
          description: "query 仍保留 8 个独立 heads；MQA 的共享只发生在 K/V 一侧。",
          symbols: [
            { tex: R`W^{Q}`, shape: R`[d_{\mathrm{model}},\,H_q d_h]`, label: "Query projection", value: "1024 × (8 × 128)", note: "多头 query 保持读取分布的多样性" }
          ]
        },
        {
          id: "wk", kind: "weight", glyph: "down",
          x: 274, y: 474,
          label: "单头 Key 投影",
          tex: R`W^{K}`, texShape: R`[d_{\mathrm{model}},\,d_h]`,
          fallback: "W^K", shapeFallback: "[d,dh]",
          relates: ["k"],
          description: "整层只有一套 key 投影：这是 MQA 定义本身（H_kv = 1）。",
          symbols: [
            { tex: R`W^{K}`, shape: R`[d_{\mathrm{model}},\,d_h]`, label: "Shared key projection", value: "1024 × 128", note: "参数量为 MHA 对应投影的 1/H_q" }
          ]
        },
        {
          id: "wv", kind: "weight", glyph: "down",
          x: 454, y: 474,
          label: "单头 Value 投影",
          tex: R`W^{V}`, texShape: R`[d_{\mathrm{model}},\,d_h]`,
          fallback: "W^V", shapeFallback: "[d,dh]",
          relates: ["v"],
          description: "value 同样只有一份；全部 query heads 读取同一份内容。",
          symbols: [
            { tex: R`W^{V}`, shape: R`[d_{\mathrm{model}},\,d_h]`, label: "Shared value projection", value: "1024 × 128", note: "与 W^K 一起构成唯一的 KV head" }
          ]
        },
        {
          id: "q", kind: "activation",
          x: 86, y: 380,
          label: "多头 Query 激活",
          tex: R`Q`, texShape: R`[B,\,H_q,\,L_q,\,d_h]`,
          fallback: "Q", shapeFallback: "[B,Hq,Lq,dh]",
          description: "8 个 query heads 各自计算读取分布；query 不进入缓存。",
          symbols: [
            { tex: R`Q`, shape: R`[B,\,H_q,\,L_q,\,d_h]`, label: "Query heads", value: "H_q = 8, d_h = 128", note: "每个 head 独立投影" }
          ]
        },
        {
          id: "k", kind: "activation",
          x: 266, y: 380,
          label: "共享 Key 激活",
          tex: R`K`, texShape: R`[B,\,1,\,L_k,\,d_h]`,
          fallback: "K", shapeFallback: "[B,1,Lk,dh]",
          description: "head 轴长度为 1；广播读取无需物理复制。",
          symbols: [
            { tex: R`K`, shape: R`[B,\,1,\,L_k,\,d_h]`, label: "Shared key", value: "H_kv = 1", note: "stride-0 广播即可服务全部 query heads" }
          ]
        },
        {
          id: "v", kind: "activation",
          x: 446, y: 380,
          label: "共享 Value 激活",
          tex: R`V`, texShape: R`[B,\,1,\,L_k,\,d_h]`,
          fallback: "V", shapeFallback: "[B,1,Lk,dh]",
          description: "唯一一份 value；解码时被反复读取但只保存一次。",
          symbols: [
            { tex: R`V`, shape: R`[B,\,1,\,L_k,\,d_h]`, label: "Shared value", value: "H_kv = 1", note: "与 K 一起构成 256 元素/token 的缓存" }
          ]
        },
        {
          id: "cache", kind: "state",
          x: 626, y: 378,
          label: "共享 KV cache",
          tex: R`K_{1:t},V_{1:t}`, texShape: R`2\times[B,\,1,\,L,\,d_h]`,
          fallback: "shared KV cache", shapeFallback: "2×[B,1,L,dh]",
          relates: ["k", "v"],
          description: "每个历史 token 只持久化一份 128-d key 与一份 128-d value。",
          symbols: [
            { tex: R`K_{1:t},V_{1:t}`, shape: R`2\times[B,\,1,\,L,\,d_h]`, label: "Persistent state", value: "2 × 128 = 256 elements/token/layer", note: "相对 8-head MHA 理论缩小 8 倍" }
          ]
        },
        {
          id: "bcast", kind: "operator", glyph: "pill", tone: "control",
          x: 287, y: 297, w: 86, h: 28, titleSize: 8.4,
          label: "head 轴逻辑广播",
          tex: R`\operatorname{broadcast}`, fallback: "broadcast",
          description: "共享 K/V 沿 head 轴 stride-0 广播到 8 个 query heads；高效实现不做物理 repeat。",
          symbols: [
            { tex: R`K'`, shape: R`[B,\,1,\,L,\,d_h]\to[B,\,H_q,\,L,\,d_h]`, label: "Logical view", value: "零复制", note: "显式 repeat 数值等价但丢掉带宽收益" }
          ]
        },
        {
          id: "attn", kind: "activation", titleSize: 8.8,
          x: 80, y: 286, w: 180, h: 50,
          label: "逐 query-head 注意力权重",
          tex: R`A=\operatorname{softmax}\!\big(\tfrac{QK^{\top}}{\sqrt{d_h}}+M\big)`,
          texShape: R`[B,\,H_q,\,L_q,\,L_k]`,
          fallback: "A = softmax(QK^T/√dh + M)", shapeFallback: "[B,Hq,Lq,Lk]",
          description: "score 仍保留全部 8 个 query heads；共享只减少历史搬运，不近似 softmax。",
          symbols: [
            { tex: R`A`, shape: R`[B,\,H_q,\,L_q,\,L_k]`, label: "Attention weights", value: "scale = 1/√128", note: "共享 K/V 不消除 query-head 计算量" }
          ]
        },
        {
          id: "mul", kind: "operator",
          x: 493, y: 294,
          label: "A·V 加权读取",
          tex: R`\otimes`, fallback: "⊗",
          description: "8 个 head 的 softmax 权重共同读取同一份共享 value。",
          symbols: [
            { tex: R`AV`, shape: R`[B,\,H_q,\,L_q,\,d_h]`, label: "Weighted read", value: "共享 V 广播参与", note: "读取分布逐头不同，内容只有一份" }
          ]
        },
        {
          id: "o", kind: "activation",
          x: 446, y: 196,
          label: "逐头输出",
          tex: R`O_h`, texShape: R`[B,\,H_q,\,L_q,\,d_h]`,
          fallback: "O_h", shapeFallback: "[B,Hq,Lq,dh]",
          description: "各 query head 的读取结果，拼接后写回残差宽度。",
          symbols: [
            { tex: R`O_h`, shape: R`[B,\,H_q,\,L_q,\,d_h]`, label: "Per-head output", value: "8 × 128 = 1024 拼接宽度", note: "concat 后进入 W^O" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 454, y: 112,
          label: "输出投影权重",
          tex: R`W^{O}`, texShape: R`[H_q d_h,\,d_{\mathrm{model}}]`,
          fallback: "W^O", shapeFallback: "[Hq·dh,d]",
          relates: ["y"],
          description: "输出投影不受 K/V 共享影响，与 MHA 完全同构。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[H_q d_h,\,d_{\mathrm{model}}]`, label: "Output projection", value: "1024 × 1024", note: "head 信息在此重新混合" }
          ]
        },
        {
          id: "y", kind: "activation", tone: "gather",
          x: 446, y: 28,
          label: "层输出",
          tex: R`Y`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "Y", shapeFallback: "[B,L,d_model]",
          description: "MQA 层输出；结构收益体现在解码时的 KV 搬运量。",
          symbols: [
            { tex: R`Y`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 1024", note: "共享改变缓存宽度，不改变输出形状" }
          ]
        }
      ],
      edges: [
        { from: "x", to: "wq", tone: "compute", bend: 540 },
        { from: "x", to: "wk", tone: "compute" },
        { from: "x", to: "wv", tone: "compute", bend: 540 },
        { from: "wq", to: "q", tone: "weight" },
        { from: "wk", to: "k", tone: "weight" },
        { from: "wv", to: "v", tone: "weight" },
        { from: "q", to: "attn", tone: "compute", toAt: 7 / 18 },
        { from: "k", to: "bcast", tone: "control" },
        { from: "bcast", to: "attn", tone: "control" },
        { from: "k", to: "cache", tone: "state", dashed: true, fromSide: "top", fromAt: 0.8, toSide: "top", bend: 364, label: "仅存一份", lx: 560, ly: 354 },
        { from: "v", to: "cache", tone: "state", dashed: true },
        { from: "v", to: "mul", tone: "compute" },
        { from: "attn", to: "mul", tone: "compute", fromSide: "top", toSide: "top", bend: 266 },
        { from: "mul", to: "o", tone: "gather", fromSide: "right", toAt: 0.75 },
        { from: "o", to: "wo", tone: "gather" },
        { from: "wo", to: "y", tone: "gather" }
      ]
    },
    gqa: {
      modules: [
        {
          id: "x", kind: "activation", tone: "cyan",
          x: 262, y: 640, w: 136, h: 46,
          label: "Decoder 输入残差流",
          tex: R`X`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "X", shapeFallback: "[B,L,d_model]",
          description: "T5.1.1-XXL 的 decoder self/cross-attention 输入；64 个 query heads 分配给 8 个 KV groups。",
          symbols: [
            { tex: R`X`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 4096", note: "GQA-8-XXL 代表配置" }
          ]
        },
        {
          id: "wq", kind: "weight", glyph: "down",
          x: 94, y: 548,
          label: "Query 投影权重",
          tex: R`W^{Q}`, texShape: R`[d_{\mathrm{model}},\,H_q d_h]`,
          fallback: "W^Q", shapeFallback: "[d,Hq·dh]",
          relates: ["q"],
          description: "query 多样性完整保留：64 个 heads 全部独立投影。",
          symbols: [
            { tex: R`W^{Q}`, shape: R`[d_{\mathrm{model}},\,H_q d_h]`, label: "Query projection", value: "4096 × (64 × 64)", note: "H_q = 64 保持不变" }
          ]
        },
        {
          id: "wk", kind: "weight", glyph: "down",
          x: 274, y: 548,
          label: "分组 Key 投影",
          tex: R`W^{K}`, texShape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`,
          fallback: "W^K", shapeFallback: "[d,Hkv·dh]",
          relates: ["k"],
          description: "只保留 8 组 key 投影；uptraining 时由组内 MHA 权重均值池化初始化。",
          symbols: [
            { tex: R`W^{K}`, shape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`, label: "Grouped key projection", value: "4096 × (8 × 64)", note: "初始化 W̄_g = r⁻¹ Σ W_h（组内均值）" }
          ]
        },
        {
          id: "wv", kind: "weight", glyph: "down",
          x: 454, y: 548,
          label: "分组 Value 投影",
          tex: R`W^{V}`, texShape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`,
          fallback: "W^V", shapeFallback: "[d,Hkv·dh]",
          relates: ["v"],
          description: "与 W^K 同构的 8 组 value 投影；缓存宽度由这 8 组决定。",
          symbols: [
            { tex: R`W^{V}`, shape: R`[d_{\mathrm{model}},\,H_{kv} d_h]`, label: "Grouped value projection", value: "4096 × (8 × 64)", note: "H_kv = 8（GQA-8）" }
          ]
        },
        {
          id: "q", kind: "activation",
          x: 86, y: 454,
          label: "Query 激活",
          tex: R`Q`, texShape: R`[B,\,H_q,\,L_q,\,d_h]`,
          fallback: "Q", shapeFallback: "[B,Hq,Lq,dh]",
          description: "64 个 query heads，每头 64 维；每个 head 仍有独立 softmax。",
          symbols: [
            { tex: R`Q`, shape: R`[B,\,H_q,\,L_q,\,d_h]`, label: "Query heads", value: "H_q = 64, d_h = 64", note: "decoder query heads" }
          ]
        },
        {
          id: "k", kind: "activation",
          x: 266, y: 454,
          label: "分组 Key 激活",
          tex: R`K_g`, texShape: R`[B,\,H_{kv},\,L_k,\,d_h]`,
          fallback: "K_g", shapeFallback: "[B,Hkv,Lk,dh]",
          description: "8 组 key；每组被 8 个相邻 query heads 逻辑读取。",
          symbols: [
            { tex: R`K_g`, shape: R`[B,\,H_{kv},\,L_k,\,d_h]`, label: "Grouped keys", value: "H_kv = 8", note: "组容量介于 MHA 与 MQA 之间" }
          ]
        },
        {
          id: "v", kind: "activation",
          x: 446, y: 454,
          label: "分组 Value 激活",
          tex: R`V_g`, texShape: R`[B,\,H_{kv},\,L_k,\,d_h]`,
          fallback: "V_g", shapeFallback: "[B,Hkv,Lk,dh]",
          description: "8 组 value 与 key 配对；组内 query heads 共享读取内容。",
          symbols: [
            { tex: R`V_g`, shape: R`[B,\,H_{kv},\,L_k,\,d_h]`, label: "Grouped values", value: "H_kv = 8", note: "与 K_g 一起进入分组缓存" }
          ]
        },
        {
          id: "cache", kind: "state",
          x: 626, y: 452,
          label: "分组 KV cache",
          tex: R`K_{1:t},V_{1:t}`, texShape: R`2\times[B,\,H_{kv},\,L,\,d_h]`,
          fallback: "grouped KV cache", shapeFallback: "2×[B,Hkv,L,dh]",
          relates: ["k", "v"],
          description: "缓存宽度只乘 8 个 KV heads，而不是 64 个 query heads。",
          symbols: [
            { tex: R`K_{1:t},V_{1:t}`, shape: R`2\times[B,\,H_{kv},\,L,\,d_h]`, label: "Persistent state", value: "2 × 8 × 64 = 1024 elements/token/layer", note: "相对 64-head MHA 理论缩小 8 倍" }
          ]
        },
        {
          id: "gmap", kind: "operator", glyph: "pill", tone: "control",
          x: 274, y: 376, w: 112, h: 28, titleSize: 8.6,
          label: "head-to-group 映射",
          tex: R`g(h)=\lfloor h/r\rfloor`, fallback: "g(h)=⌊h/r⌋",
          description: "连续等大小分组：每个 query head 按 g(h) 找到自己的 KV 组，不做物理 repeat。",
          symbols: [
            { tex: R`g(h)`, shape: R`[H_q]\to[H_{kv}]`, label: "Group map", value: "r = H_q/H_kv = 8", note: "要求 H_q 可被 H_kv 整除" }
          ]
        },
        {
          id: "bias", kind: "weight", glyph: "down", titleSize: 8.8,
          x: 34, y: 368,
          label: "T5 相对位置偏置",
          tex: R`b_{h,\mathrm{bucket}(t-s)}`, texShape: R`[H_q,\,32]`,
          fallback: "b_{h,bucket}", shapeFallback: "[Hq,32]",
          relates: ["attn"],
          description: "T5 self-attention 的桶化相对位置偏置按 query head 索引；cross-attention 不套用此项。",
          symbols: [
            { tex: R`b_{h,\mathrm{bucket}(t-s)}`, shape: R`[H_q,\,32]`, label: "Relative bias table", value: "32 buckets · max distance 128", note: "位置参数保留 h 轴，未被池化成组" }
          ]
        },
        {
          id: "attn", kind: "activation", titleSize: 8.2,
          x: 130, y: 282, w: 220, h: 50,
          label: "分组注意力权重",
          tex: R`A=\operatorname{softmax}\!\big(\tfrac{QK_{g(h)}^{\top}}{\sqrt{d_h}}+b_h+M\big)`,
          texShape: R`[B,\,H_q,\,L_q,\,L_k]`,
          fallback: "A = softmax(QK_g^T/√dh + b + M)", shapeFallback: "[B,Hq,Lq,Lk]",
          description: "每个 query head 保留独立 softmax，只是读取所属组的 K/V；self-attention 另加 T5 桶化偏置。",
          symbols: [
            { tex: R`A`, shape: R`[B,\,H_q,\,L_q,\,L_k]`, label: "Attention weights", value: "scale = 1/√64", note: "64 个独立 score map" }
          ]
        },
        {
          id: "mul", kind: "operator",
          x: 493, y: 290,
          label: "A·V 分组读取",
          tex: R`\otimes`, fallback: "⊗",
          description: "每个 query head 用自己的权重读取所属组的 value。",
          symbols: [
            { tex: R`AV_{g(h)}`, shape: R`[B,\,H_q,\,L_q,\,d_h]`, label: "Groupwise read", value: "精确 attention", note: "组共享不引入任何近似" }
          ]
        },
        {
          id: "o", kind: "activation",
          x: 446, y: 196,
          label: "逐头输出",
          tex: R`O_h`, texShape: R`[B,\,H_q,\,L_q,\,d_h]`,
          fallback: "O_h", shapeFallback: "[B,Hq,Lq,dh]",
          description: "64 个 head 输出拼接后写回残差宽度。",
          symbols: [
            { tex: R`O_h`, shape: R`[B,\,H_q,\,L_q,\,d_h]`, label: "Per-head output", value: "64 × 64 = 4096 拼接宽度", note: "concat 后进入 W^O" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 454, y: 112,
          label: "输出投影权重",
          tex: R`W^{O}`, texShape: R`[H_q d_h,\,d_{\mathrm{model}}]`,
          fallback: "W^O", shapeFallback: "[Hq·dh,d]",
          relates: ["y"],
          description: "输出投影不参与分组，与 MHA 完全同构。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[H_q d_h,\,d_{\mathrm{model}}]`, label: "Output projection", value: "4096 × 4096", note: "不受 H_kv 影响" }
          ]
        },
        {
          id: "y", kind: "activation", tone: "gather",
          x: 446, y: 28,
          label: "层输出",
          tex: R`Y`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "Y", shapeFallback: "[B,L,d_model]",
          description: "GQA 层输出；分组只改变缓存与搬运量，不改变输出形状。",
          symbols: [
            { tex: R`Y`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 4096", note: "与残差流宽度一致" }
          ]
        }
      ],
      edges: [
        { from: "x", to: "wq", tone: "compute", bend: 614 },
        { from: "x", to: "wk", tone: "compute" },
        { from: "x", to: "wv", tone: "compute", bend: 614 },
        { from: "wq", to: "q", tone: "weight" },
        { from: "wk", to: "k", tone: "weight" },
        { from: "wv", to: "v", tone: "weight" },
        { from: "q", to: "attn", tone: "compute", toAt: 0.25, bend: 350 },
        { from: "k", to: "gmap", tone: "control" },
        { from: "gmap", to: "attn", tone: "control", toAt: 0.6, bend: 352 },
        { from: "bias", to: "attn", tone: "weight", toAt: 0.1, bend: 340 },
        { from: "k", to: "cache", tone: "state", dashed: true, fromSide: "top", fromAt: 0.8, toSide: "top", bend: 436, label: "8 组持久化", lx: 560, ly: 426 },
        { from: "v", to: "cache", tone: "state", dashed: true },
        { from: "v", to: "mul", tone: "compute" },
        { from: "attn", to: "mul", tone: "compute" },
        { from: "mul", to: "o", tone: "gather" },
        { from: "o", to: "wo", tone: "gather" },
        { from: "wo", to: "y", tone: "gather" }
      ]
    },
    mla: {
      groups: [
        { id: "prefill", label: "PREFILL 展开 · MHA-LIKE 中间激活", tone: "compute", titlePos: "bottom", members: ["wuk", "wuv", "kc", "vc"] },
        { id: "decode", label: "DECODE 直读 · MQA-LIKE", tone: "state", titlePos: "bottom", members: ["cache"] }
      ],
      modules: [
        {
          id: "h", kind: "activation", tone: "cyan",
          x: 382, y: 700, w: 136, h: 46,
          label: "输入残差流",
          tex: R`h_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "h_t", shapeFallback: "[B,L,d_model]",
          description: "同一 hidden state 生成 query latent、joint KV latent 与解耦 RoPE key 三条支路。",
          symbols: [
            { tex: R`h_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 5120", note: "DeepSeek-V2 代表模型" }
          ]
        },
        {
          id: "wdq", kind: "weight", glyph: "down",
          x: 114, y: 608,
          label: "Query 下投影",
          tex: R`W^{DQ}`, texShape: R`[d,\,d_c']`,
          fallback: "W^DQ", shapeFallback: "[d,dc']",
          relates: ["cq"],
          description: "query 先压缩到 1536 维 latent；低秩化只为省参数，不进缓存。",
          symbols: [
            { tex: R`W^{DQ}`, shape: R`[d_{\mathrm{model}},\,d_c']`, label: "Query down-projection", value: "5120 × 1536", note: "query latent rank d_c' = 1536" }
          ]
        },
        {
          id: "wdkv", kind: "weight", glyph: "down",
          x: 394, y: 608,
          label: "Joint KV 下投影",
          tex: R`W^{DKV}`, texShape: R`[d,\,d_c]`,
          fallback: "W^DKV", shapeFallback: "[d,dc]",
          relates: ["ckv"],
          description: "把 5120 维 hidden state 压成 512 维 joint latent——K 与 V 的共同来源。",
          symbols: [
            { tex: R`W^{DKV}`, shape: R`[d_{\mathrm{model}},\,d_c]`, label: "KV down-projection", value: "5120 × 512", note: "KV latent rank d_c = 512" }
          ]
        },
        {
          id: "wkr", kind: "weight", glyph: "down",
          x: 654, y: 608,
          label: "RoPE key 投影",
          tex: R`W^{KR}`, texShape: R`[d,\,d_h^{R}]`,
          fallback: "W^KR", shapeFallback: "[d,dhR]",
          relates: ["kr"],
          description: "位置支路的共享 key 投影：全部 128 个 heads 共用一份 64 维 RoPE key。",
          symbols: [
            { tex: R`W^{KR}`, shape: R`[d_{\mathrm{model}},\,d_h^{R}]`, label: "Decoupled RoPE key projection", value: "5120 × 64", note: "逐头保存会使缓存宽度膨胀到 8704" }
          ]
        },
        {
          id: "cq", kind: "activation",
          x: 106, y: 470,
          label: "Query latent",
          tex: R`c_t^{Q}`, texShape: R`[B,\,L,\,d_c']`,
          fallback: "c_t^Q", shapeFallback: "[B,L,dc']",
          description: "当前 token 的临时激活；不进入任何跨 token 缓存。",
          symbols: [
            { tex: R`c_t^{Q}`, shape: R`[B,\,L,\,d_c']`, label: "Query latent", value: "d_c' = 1536", note: "现算即弃" }
          ]
        },
        {
          id: "ckv", kind: "activation", tone: "control",
          x: 386, y: 470,
          label: "Joint KV latent",
          tex: R`c_t^{KV}`, texShape: R`[B,\,L,\,d_c]`,
          fallback: "c_t^KV", shapeFallback: "[B,L,dc]",
          description: "同一份 512 维表示同时派生 K 与 V；Prefill 展开、Decode 直读都从这里出发。",
          symbols: [
            { tex: R`c_t^{KV}`, shape: R`[B,\,L,\,d_c]`, label: "Joint latent", value: "d_c = 512", note: "NoPE 内容通道，吸收恒等式的前提" }
          ]
        },
        {
          id: "rope1", kind: "operator", tone: "cyan",
          x: 693, y: 552,
          label: "RoPE 旋转（key 侧）",
          tex: R`R_t`, fallback: "R_t",
          description: "共享 key 在写入缓存前按位置 t 完成旋转；每 token 只旋转一次。",
          symbols: [
            { tex: R`R_t`, shape: R`[d_h^{R},\,d_h^{R}]`, label: "Rotary operator", value: "64 维位置子空间", note: "位置项不可吸收，因此单独走支路" }
          ]
        },
        {
          id: "kr", kind: "activation",
          x: 646, y: 470,
          label: "共享 RoPE key",
          tex: R`k_t^{R}`, texShape: R`[B,\,1,\,L,\,d_h^{R}]`,
          fallback: "k_t^R", shapeFallback: "[B,1,L,dhR]",
          description: "全部 heads 共享的 64 维位置 key；MQA 式共享保住了缓存收益。",
          symbols: [
            { tex: R`k_t^{R}`, shape: R`[B,\,1,\,L,\,d_h^{R}]`, label: "Shared positional key", value: "d_h^R = 64", note: "stride-0 广播给 128 个 heads" }
          ]
        },
        {
          id: "wuq", kind: "weight", glyph: "up",
          x: 44, y: 382,
          label: "内容 query 上投影",
          tex: R`W^{UQ}`, texShape: R`[d_c',\,H d_h^{C}]`,
          fallback: "W^UQ", shapeFallback: "[dc',H·dhC]",
          relates: ["qc"],
          description: "从 query latent 展开 128 个 128 维内容 query heads。",
          symbols: [
            { tex: R`W^{UQ}`, shape: R`[d_c',\,H d_h^{C}]`, label: "Content query up-projection", value: "1536 × (128 × 128)", note: "H = 128, d_h^C = 128" }
          ]
        },
        {
          id: "wqr", kind: "weight", glyph: "up",
          x: 184, y: 382,
          label: "位置 query 上投影",
          tex: R`W^{QR}`, texShape: R`[d_c',\,H d_h^{R}]`,
          fallback: "W^QR", shapeFallback: "[dc',H·dhR]",
          relates: ["qr"],
          description: "逐头 64 维位置 query：现算即弃，保留头间位置敏感度多样性。",
          symbols: [
            { tex: R`W^{QR}`, shape: R`[d_c',\,H d_h^{R}]`, label: "Positional query up-projection", value: "1536 × (128 × 64)", note: "query 侧不进缓存" }
          ]
        },
        {
          id: "wuk", kind: "weight", glyph: "up",
          x: 324, y: 382,
          label: "Key 上投影（可吸收）",
          tex: R`W^{UK}`, texShape: R`[d_c,\,H d_h^{C}]`,
          fallback: "W^UK", shapeFallback: "[dc,H·dhC]",
          relates: ["kc", "cache"],
          description: "Prefill 显式展开每头 key；Decode 把它吸收进 query 侧（q̃ = (W^UK)^⊤ q^C）。",
          symbols: [
            { tex: R`W^{UK}`, shape: R`[d_c,\,H d_h^{C}]`, label: "Key up-projection", value: "512 × (128 × 128)", note: "两种执行图共用同一份权重" }
          ]
        },
        {
          id: "wuv", kind: "weight", glyph: "up",
          x: 464, y: 382,
          label: "Value 上投影（可吸收）",
          tex: R`W^{UV}`, texShape: R`[d_c,\,H d_h^{C}]`,
          fallback: "W^UV", shapeFallback: "[dc,H·dhC]",
          relates: ["vc", "cache"],
          description: "Prefill 显式重建每头 value；Decode 与 W^O 合并成固定输出矩阵。",
          symbols: [
            { tex: R`W^{UV}`, shape: R`[d_c,\,H d_h^{C}]`, label: "Value up-projection", value: "512 × (128 × 128)", note: "V 侧吸收：W^O W^UV 先行合并" }
          ]
        },
        {
          id: "qc", kind: "activation",
          x: 36, y: 286,
          label: "内容 query heads",
          tex: R`q_t^{C}`, texShape: R`[B,\,H,\,L,\,d_h^{C}]`,
          fallback: "q_t^C", shapeFallback: "[B,H,L,dhC]",
          description: "128 个 128 维内容 query；Decode 时先与 W^UK 折叠成 512 维 q̃。",
          symbols: [
            { tex: R`q_t^{C}`, shape: R`[B,\,H,\,L,\,d_h^{C}]`, label: "Content queries", value: "H = 128, d_h^C = 128", note: "吸收后 q̃ ∈ R^512 直接对 latent 打分" }
          ]
        },
        {
          id: "rope2", kind: "operator", tone: "cyan",
          x: 223, y: 341,
          label: "RoPE 旋转（query 侧）",
          tex: R`R_t`, fallback: "R_t",
          description: "逐头位置 query 的旋转；每步现算，不产生随序列增长的状态。",
          symbols: [
            { tex: R`R_t`, shape: R`[d_h^{R},\,d_h^{R}]`, label: "Rotary operator", value: "64 维子空间", note: "与 key 侧配对给出相对位移" }
          ]
        },
        {
          id: "qr", kind: "activation",
          x: 176, y: 286,
          label: "位置 query heads",
          tex: R`q_t^{R}`, texShape: R`[B,\,H,\,L,\,d_h^{R}]`,
          fallback: "q_t^R", shapeFallback: "[B,H,L,dhR]",
          description: "逐头 64 维位置 query，与共享 k^R 配对形成位置分数。",
          symbols: [
            { tex: R`q_t^{R}`, shape: R`[B,\,H,\,L,\,d_h^{R}]`, label: "Positional queries", value: "H = 128, d_h^R = 64", note: "只花一次投影算量" }
          ]
        },
        {
          id: "kc", kind: "activation", dashed: true,
          x: 316, y: 286,
          label: "Prefill 展开 key",
          tex: R`k_{s,i}^{C}`, texShape: R`[B,\,H,\,L,\,d_h^{C}]`,
          fallback: "k_si^C", shapeFallback: "[B,H,L,dhC]",
          description: "仅 Prefill 车道的中间激活：k^C = W^UK c^KV；不作为持久缓存。",
          symbols: [
            { tex: R`k_{s,i}^{C}=W_i^{UK}c_s^{KV}`, shape: R`[B,\,H,\,L,\,d_h^{C}]`, label: "Expanded keys", value: "128 heads × 128-d", note: "compute-only，算完即弃" }
          ]
        },
        {
          id: "vc", kind: "activation", dashed: true,
          x: 456, y: 286,
          label: "Prefill 展开 value",
          tex: R`v_{s,i}`, texShape: R`[B,\,H,\,L,\,d_h^{C}]`,
          fallback: "v_si", shapeFallback: "[B,H,L,dhC]",
          description: "仅 Prefill 车道的中间激活：v = W^UV c^KV。",
          symbols: [
            { tex: R`v_{s,i}=W_i^{UV}c_s^{KV}`, shape: R`[B,\,H,\,L,\,d_h^{C}]`, label: "Expanded values", value: "128 heads × 128-d", note: "Decode 从不重建历史多头 V" }
          ]
        },
        {
          id: "cache", kind: "state",
          x: 800, y: 470, w: 148, h: 46,
          label: "Latent + RoPE 持久缓存",
          tex: R`c_{1:t}^{KV},k_{1:t}^{R}`, texShape: R`[B,\,L,\,d_c{+}d_h^{R}]`,
          fallback: "c^KV + k^R cache", shapeFallback: "[B,L,dc+dhR]",
          relates: ["ckv", "kr"],
          description: "持久缓存只含 512 维 joint latent 与一份共享 64 维 RoPE key；没有 128 个 KV-head 副本。",
          symbols: [
            { tex: R`c_{1:t}^{KV},k_{1:t}^{R}`, shape: R`[B,\,L,\,d_c]+[B,\,L,\,d_h^{R}]`, label: "Persistent state", value: "512 + 64 = 576 elements/token/layer", note: "两种执行形态共享同一份缓存" }
          ]
        },
        {
          id: "score", kind: "activation", titleSize: 8.2,
          x: 120, y: 196, w: 240, h: 52,
          label: "内容 + 位置分数",
          tex: R`a=\operatorname{softmax}\!\big(\tfrac{q^{C\top}k^{C}+q^{R\top}k^{R}}{\sqrt{192}}+M\big)`,
          texShape: R`[B,\,H,\,L,\,L]`,
          fallback: "a = softmax((qC·kC + qR·kR)/√192)", shapeFallback: "[B,H,L,L]",
          description: "两条车道给出逐元素相同的 softmax：Prefill 用展开 key，Decode 用 q̃ 直接对 latent 打分。",
          symbols: [
            { tex: R`a`, shape: R`[B,\,H,\,L,\,L]`, label: "Attention weights", value: "scale = 1/√(128+64)", note: "同权重、精确线性重结合" }
          ]
        },
        {
          id: "mul", kind: "operator",
          x: 503, y: 205,
          label: "加权读取（value / latent）",
          tex: R`\otimes`, fallback: "⊗",
          description: "Prefill 读展开 value；Decode 先聚合 512 维 latent（m = Σ a·c^KV），再一次线性写回。",
          symbols: [
            { tex: R`o_{t,i}`, shape: R`[B,\,H,\,L,\,d_h^{C}]`, label: "Prefill read", value: "128 heads × 128-d", note: "o = Σ a·v（展开车道）" },
            { tex: R`m_{t,i}`, shape: R`[B,\,H,\,L,\,d_c]`, label: "Decode latent read", value: "512-d 共享工作空间", note: "m = Σ a·c^KV（吸收车道）" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 464, y: 124,
          label: "输出投影（可并吸收）",
          tex: R`W^{O}`, texShape: R`[H d_h^{C},\,d]`,
          fallback: "W^O", shapeFallback: "[H·dhC,d]",
          relates: ["u"],
          description: "Decode 形态把 W_i^O W_i^UV 合并成固定矩阵，直接作用于聚合后的 latent。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[H d_h^{C},\,d_{\mathrm{model}}]`, label: "Output projection", value: "(128 × 128) × 5120", note: "吸收：u = Σ (W_i^O W_i^UV) m_i" }
          ]
        },
        {
          id: "u", kind: "activation", tone: "gather",
          x: 456, y: 40,
          label: "层输出",
          tex: R`u_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "u_t", shapeFallback: "[B,L,d_model]",
          description: "两种执行图逐位相同的输出；权重只有一份。",
          symbols: [
            { tex: R`u_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 5120", note: "括号顺序不同、函数完全相同" }
          ]
        }
      ],
      edges: [
        { from: "h", to: "wdq", tone: "compute", bend: 674 },
        { from: "h", to: "wdkv", tone: "compute" },
        { from: "h", to: "wkr", tone: "compute", bend: 674 },
        { from: "wdq", to: "cq", tone: "weight" },
        { from: "wdkv", to: "ckv", tone: "weight" },
        { from: "wkr", to: "rope1", tone: "weight" },
        { from: "rope1", to: "kr", tone: "cyan" },
        { from: "cq", to: "wuq", tone: "compute", fromAt: 0.25, bend: 446 },
        { from: "cq", to: "wqr", tone: "compute", fromAt: 0.75, bend: 446 },
        { from: "ckv", to: "wuk", tone: "compute", dashed: true, fromAt: 0.2, bend: 452, label: "Prefill 展开", lx: 300, ly: 462 },
        { from: "ckv", to: "wuv", tone: "compute", dashed: true, fromAt: 0.8, bend: 452 },
        { from: "ckv", to: "cache", tone: "state", dashed: true, fromSide: "bottom", fromAt: 0.8, toSide: "bottom", bend: 544, label: "persist 512", lx: 640, ly: 534 },
        { from: "kr", to: "cache", tone: "state", dashed: true },
        { from: "wuq", to: "qc", tone: "weight" },
        { from: "wqr", to: "rope2", tone: "weight" },
        { from: "rope2", to: "qr", tone: "cyan" },
        { from: "wuk", to: "kc", tone: "weight" },
        { from: "wuv", to: "vc", tone: "weight" },
        { from: "qc", to: "score", tone: "compute", toAt: 0.25, bend: 266 },
        { from: "qr", to: "score", tone: "compute" },
        { from: "kc", to: "score", tone: "compute", dashed: true, toAt: 0.75, bend: 266, label: "Prefill", lx: 344, ly: 276 },
        { from: "vc", to: "mul", tone: "compute", dashed: true },
        { from: "score", to: "mul", tone: "compute" },
        { from: "cache", to: "score", tone: "state", dashed: true, fromSide: "top", toSide: "bottom", toAt: 0.9, bend: 258, label: "Decode 直读 latent", lx: 700, ly: 248 },
        { from: "kr", to: "score", tone: "cyan", fromSide: "top", toSide: "right", toAt: 0.1, label: "共享位置分数", lx: 560, ly: 190 },
        { from: "mul", to: "wo", tone: "gather" },
        { from: "wo", to: "u", tone: "gather" }
      ]
    },
    mfa: {
      modules: [
        {
          id: "x", kind: "activation", tone: "cyan",
          x: 322, y: 804, w: 136, h: 46,
          label: "输入残差流",
          tex: R`x_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "x_t", shapeFallback: "[B,L,d_model]",
          description: "输入被三个共享投影映射到同一个 C 维特征空间。",
          symbols: [
            { tex: R`x_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 2048", note: "MFA 规模化研究代表模型（24 层）" }
          ]
        },
        {
          id: "sq", kind: "weight", glyph: "down",
          x: 94, y: 734,
          label: "共享 query 投影",
          tex: R`S_q`, texShape: R`[d,\,C]`,
          fallback: "S_q", shapeFallback: "[d,C]",
          relates: ["qbar"],
          description: "把 token 投到共享 C 维空间；head 差异稍后由逐头 Q_c 提供。",
          symbols: [
            { tex: R`S_q`, shape: R`[d_{\mathrm{model}},\,C]`, label: "Shared query projection", value: "2048 × 256", note: "C = 256 同时是共享宽度与每头 rank" }
          ]
        },
        {
          id: "sk", kind: "weight", glyph: "down",
          x: 334, y: 734,
          label: "共享 key 投影",
          tex: R`S_k`, texShape: R`[d,\,C]`,
          fallback: "S_k", shapeFallback: "[d,C]",
          relates: ["kf"],
          description: "每个 token 只产生一份共享 key；head 数不复制历史缓存。",
          symbols: [
            { tex: R`S_k`, shape: R`[d_{\mathrm{model}},\,C]`, label: "Shared key projection", value: "2048 × 256", note: "K 只有一个共享 head" }
          ]
        },
        {
          id: "sv", kind: "weight", glyph: "down",
          x: 574, y: 734,
          label: "共享 value 投影",
          tex: R`S_v`, texShape: R`[d,\,C]`,
          fallback: "S_v", shapeFallback: "[d,C]",
          relates: ["vf"],
          description: "标准 MFA 的 value 投影；MFA-KR 变体改由 key 派生 value。",
          symbols: [
            { tex: R`S_v`, shape: R`[d_{\mathrm{model}},\,C]`, label: "Shared value projection", value: "2048 × 256", note: "MFA-KR 下此路径被 v = kM 取代" }
          ]
        },
        {
          id: "qbar", kind: "activation",
          x: 86, y: 658,
          label: "共享 query 特征",
          tex: R`\bar q_t`, texShape: R`[B,\,L,\,C]`,
          fallback: "q̄_t", shapeFallback: "[B,L,C]",
          description: "C 维共享 query 特征；从不进入缓存。",
          symbols: [
            { tex: R`\bar q_t=x_tS_q`, shape: R`[B,\,L,\,C]`, label: "Shared query features", value: "C = 256", note: "现算即弃" }
          ]
        },
        {
          id: "kf", kind: "activation",
          x: 326, y: 658,
          label: "共享 key",
          tex: R`k_t`, texShape: R`[B,\,L,\,C]`,
          fallback: "k_t", shapeFallback: "[B,L,C]",
          description: "每 token 一份共享 key，被全部 m 个 heads 读取。",
          symbols: [
            { tex: R`k_t=x_tS_k`, shape: R`[B,\,L,\,C]`, label: "Shared key", value: "C = 256", note: "没有 head 轴" }
          ]
        },
        {
          id: "vf", kind: "activation",
          x: 566, y: 658,
          label: "共享 value",
          tex: R`v_t`, texShape: R`[B,\,L,\,C]`,
          fallback: "v_t", shapeFallback: "[B,L,C]",
          description: "标准 MFA 的共享 value；与 key 一起构成 2C 宽度的缓存。",
          symbols: [
            { tex: R`v_t=x_tS_v`, shape: R`[B,\,L,\,C]`, label: "Shared value", value: "C = 256", note: "MFA-KR 只缓存 key，v 由 kM 派生" }
          ]
        },
        {
          id: "cache", kind: "state",
          x: 756, y: 658, w: 148, h: 46,
          label: "共享 KV cache",
          tex: R`k_{1:t},v_{1:t}`, texShape: R`2\times[B,\,L,\,C]`,
          fallback: "shared KV cache", shapeFallback: "2×[B,L,C]",
          relates: ["kf", "vf"],
          description: "缓存宽度 2C 与 head 数 m 无关；MFA-KR 变体减半到 C。",
          symbols: [
            { tex: R`k_{1:t},v_{1:t}`, shape: R`2\times[B,\,L,\,C]`, label: "MFA cache", value: "2C = 512 elements/token（BF16 为 1 KiB/层）", note: "24 层合计约 24 KiB/token" },
            { tex: R`k_{1:t}`, shape: R`[B,\,L,\,C]`, label: "MFA-KR cache", value: "C = 256 elements/token", note: "key reuse 以小幅质量折损换半数缓存" }
          ]
        },
        {
          id: "qc", kind: "weight", glyph: "down",
          x: 94, y: 588,
          label: "逐头 QK 矩阵",
          tex: R`Q_c`, texShape: R`[m,\,C,\,C]`,
          fallback: "Q_c", shapeFallback: "[m,C,C]",
          relates: ["qheads"],
          description: "每个 head 一个完整 C×C 矩阵：容量放在权重里，而不是缓存里。",
          symbols: [
            { tex: R`Q_c`, shape: R`[m,\,C,\,C]`, label: "Per-head QK circuits", value: "m = 18 heads × 256×256", note: "每头 factorization rank 可达 C" }
          ]
        },
        {
          id: "kr", kind: "weight", glyph: "pair", dashed: true, tone: "orange",
          x: 774, y: 588,
          label: "MFA-KR value 派生",
          tex: R`M=I{+}\mathrm{diag}(\alpha)N`, texShape: R`[C,\,C]`,
          fallback: "M = I + diag(α)N", shapeFallback: "[C,C]", titleSize: 8.2,
          relates: ["kf", "vf"],
          description: "可选 key-reuse 变体：v = kM，α 零初始化使训练从 v = k 出发；缓存从 2C 降到 C。",
          symbols: [
            { tex: R`M=I+\operatorname{diag}(\alpha)N`, shape: R`[C,\,C]`, label: "Key-reuse matrix", value: "α 零初始化 ⇒ 初始 v = k", note: "value 被约束到 key 派生族" }
          ]
        },
        {
          id: "qheads", kind: "activation",
          x: 86, y: 512,
          label: "逐头 query",
          tex: R`q_{t,c}`, texShape: R`[B,\,m,\,L,\,C]`,
          fallback: "q_tc", shapeFallback: "[B,m,L,C]",
          description: "18 个 C 维 query views；只存在于计算中，不进缓存。",
          symbols: [
            { tex: R`q_{t,c}=\bar q_tQ_c`, shape: R`[B,\,m,\,L,\,C]`, label: "Per-head queries", value: "m = 18, C = 256", note: "head 差异全部来自 Q_c" }
          ]
        },
        {
          id: "ropeq", kind: "operator", tone: "cyan",
          x: 133, y: 453,
          label: "RoPE（query 侧）",
          tex: R`R_t`, fallback: "R_t",
          description: "全部 heads 在同一 C 维空间用同一组 RoPE 频率旋转。",
          symbols: [
            { tex: R`R_t`, shape: R`[C,\,C]`, label: "Rotary operator", value: "base 500,000（论文 common settings）", note: "value 不旋转" }
          ]
        },
        {
          id: "ropek", kind: "operator", tone: "cyan",
          x: 373, y: 453,
          label: "RoPE（key 侧）",
          tex: R`R_s`, fallback: "R_s",
          description: "共享 key 每 token 只旋转一次即可服务全部 heads。",
          symbols: [
            { tex: R`R_s`, shape: R`[C,\,C]`, label: "Rotary operator", value: "每 token 一次", note: "相对位移由两侧配对消去" }
          ]
        },
        {
          id: "score", kind: "activation", titleSize: 8.4,
          x: 160, y: 380, w: 220, h: 52,
          label: "逐头注意力权重",
          tex: R`a^{(c)}=\operatorname{softmax}\!\big(\tfrac{q_{t,c}k_s^{\top}}{\sqrt C}+M\big)`,
          texShape: R`[B,\,m,\,L,\,L]`,
          fallback: "a = softmax(q_tc k_s^T/√C + M)", shapeFallback: "[B,m,L,L]",
          description: "共享底片被 18 套 QK circuit 读取；score 保留完整 head 轴。",
          symbols: [
            { tex: R`a^{(c)}`, shape: R`[B,\,m,\,L,\,L]`, label: "Attention weights", value: "TER 代理 = m·C = 18 × 256", note: "TER 是容量代理，不是精度定理" }
          ]
        },
        {
          id: "mul", kind: "operator",
          x: 613, y: 389,
          label: "共享 value 读取",
          tex: R`\otimes`, fallback: "⊗",
          description: "18 个 head 的权重共同读取同一份共享 value。",
          symbols: [
            { tex: R`m_{t,c}=\textstyle\sum_s a^{(c)}_{t,s}v_s`, shape: R`[B,\,m,\,L,\,C]`, label: "Weighted read", value: "共享 V 广播", note: "MFA-KR 下 v 由 kM 现场派生" }
          ]
        },
        {
          id: "m", kind: "activation",
          x: 566, y: 316,
          label: "逐头读取结果",
          tex: R`m_{t,c}`, texShape: R`[B,\,m,\,L,\,C]`,
          fallback: "m_tc", shapeFallback: "[B,m,L,C]",
          description: "每个 head 聚合出的 C 维内容，等待逐头 O_c 写回。",
          symbols: [
            { tex: R`m_{t,c}`, shape: R`[B,\,m,\,L,\,C]`, label: "Per-head reads", value: "m = 18, C = 256", note: "compute-only 中间激活" }
          ]
        },
        {
          id: "oc", kind: "weight", glyph: "up",
          x: 574, y: 246,
          label: "逐头 VO 矩阵",
          tex: R`O_c`, texShape: R`[m,\,C,\,C]`,
          fallback: "O_c", shapeFallback: "[m,C,C]",
          relates: ["o"],
          description: "每个 head 独立的 C×C 写回矩阵，与 Q_c 一起构成逐头 circuit。",
          symbols: [
            { tex: R`O_c`, shape: R`[m,\,C,\,C]`, label: "Per-head VO circuits", value: "18 × 256×256", note: "o_t = Σ_c m_tc O_c^⊤" }
          ]
        },
        {
          id: "o", kind: "activation",
          x: 566, y: 170,
          label: "聚合输出",
          tex: R`o_t`, texShape: R`[B,\,L,\,C]`,
          fallback: "o_t", shapeFallback: "[B,L,C]",
          description: "18 个 head 的写回求和；宽度回到共享 C。",
          symbols: [
            { tex: R`o_t=\textstyle\sum_c m_{t,c}O_c^{\top}`, shape: R`[B,\,L,\,C]`, label: "Aggregated output", value: "C = 256", note: "随后经 W^O 回到 d_model" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 574, y: 100,
          label: "输出投影",
          tex: R`W^{O}`, texShape: R`[C,\,d]`,
          fallback: "W^O", shapeFallback: "[C,d]",
          relates: ["u"],
          description: "把 C 维聚合结果映射回残差流宽度。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[C,\,d_{\mathrm{model}}]`, label: "Output projection", value: "256 × 2048", note: "u_t = W^O o_t" }
          ]
        },
        {
          id: "u", kind: "activation", tone: "gather",
          x: 566, y: 24,
          label: "层输出",
          tex: R`u_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "u_t", shapeFallback: "[B,L,d_model]",
          description: "MFA 层输出；缓存宽度与 head 数在结构上解耦。",
          symbols: [
            { tex: R`u_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 2048", note: "增加 m 只增加权重与算力" }
          ]
        }
      ],
      edges: [
        { from: "x", to: "sq", tone: "compute", bend: 788 },
        { from: "x", to: "sk", tone: "compute" },
        { from: "x", to: "sv", tone: "compute", bend: 788 },
        { from: "sq", to: "qbar", tone: "weight" },
        { from: "sk", to: "kf", tone: "weight" },
        { from: "sv", to: "vf", tone: "weight" },
        { from: "qbar", to: "qc", tone: "compute" },
        { from: "qc", to: "qheads", tone: "weight" },
        { from: "qheads", to: "ropeq", tone: "compute" },
        { from: "ropeq", to: "score", tone: "cyan", toAt: 0.3, bend: 444 },
        { from: "kf", to: "ropek", tone: "compute" },
        { from: "ropek", to: "score", tone: "cyan", toAt: 0.8, bend: 440 },
        { from: "kf", to: "kr", tone: "orange", dashed: true, fromSide: "top", fromAt: 0.85, bend: 642, label: "OPTIONAL · KR", lx: 650, ly: 632 },
        { from: "kr", to: "mul", tone: "orange", dashed: true, fromSide: "top", toSide: "right", label: R`v_s=k_sM`, lx: 760, ly: 500 },
        { from: "kf", to: "cache", tone: "state", dashed: true, fromSide: "bottom", fromAt: 0.8, toSide: "bottom", bend: 718 },
        { from: "vf", to: "cache", tone: "state", dashed: true },
        { from: "vf", to: "mul", tone: "compute" },
        { from: "score", to: "mul", tone: "compute" },
        { from: "mul", to: "m", tone: "gather" },
        { from: "m", to: "oc", tone: "compute" },
        { from: "oc", to: "o", tone: "weight" },
        { from: "o", to: "wo", tone: "gather" },
        { from: "wo", to: "u", tone: "gather" }
      ]
    },
    tpa: {
      modules: [
        {
          id: "x", kind: "activation", tone: "cyan",
          x: 402, y: 712, w: 136, h: 46,
          label: "输入残差流",
          tex: R`x_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "x_t", shapeFallback: "[B,L,d_model]",
          description: "每个 token 现场生成六路 contextual 因子：三个 head 轴 A 因子与三个 channel 轴 B 因子。",
          symbols: [
            { tex: R`x_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 1600", note: "T6-XL 官方研究配置（48 层）" }
          ]
        },
        {
          id: "waq", kind: "weight", glyph: "down", w: 104,
          x: 43, y: 620,
          label: "A_Q 因子生成器",
          tex: R`W^{A_Q}`, texShape: R`[d,\,R_Q h]`,
          fallback: "W^AQ", shapeFallback: "[d,RQ·h]",
          relates: ["aq"],
          description: "生成 query 的 token-dependent head-mixing 因子。",
          symbols: [
            { tex: R`W^{A_Q}`, shape: R`[d_{\mathrm{model}},\,R_Q h]`, label: "Query head-factor generator", value: "1600 × (6 × 78)", note: "R_Q = 6, h = 78" }
          ]
        },
        {
          id: "wbq", kind: "weight", glyph: "down", w: 104,
          x: 193, y: 620,
          label: "B_Q 因子生成器",
          tex: R`W^{B_Q}`, texShape: R`[d,\,R_Q d_h]`,
          fallback: "W^BQ", shapeFallback: "[d,RQ·dh]",
          relates: ["bq"],
          description: "生成 query 的 channel 模式因子；RoPE 稍后逐行旋转它。",
          symbols: [
            { tex: R`W^{B_Q}`, shape: R`[d_{\mathrm{model}},\,R_Q d_h]`, label: "Query channel-factor generator", value: "1600 × (6 × 64)", note: "d_h = 64" }
          ]
        },
        {
          id: "wak", kind: "weight", glyph: "down", w: 104,
          x: 343, y: 620,
          label: "A_K 因子生成器",
          tex: R`W^{A_K}`, texShape: R`[d,\,R_K h]`,
          fallback: "W^AK", shapeFallback: "[d,RK·h]",
          relates: ["ak"],
          description: "key 的 head 轴因子生成器；其输出进入因子缓存。",
          symbols: [
            { tex: R`W^{A_K}`, shape: R`[d_{\mathrm{model}},\,R_K h]`, label: "Key head-factor generator", value: "1600 × (2 × 78)", note: "R_K = 2 决定缓存的秩预算" }
          ]
        },
        {
          id: "wbk", kind: "weight", glyph: "down", w: 104,
          x: 493, y: 620,
          label: "B_K 因子生成器",
          tex: R`W^{B_K}`, texShape: R`[d,\,R_K d_h]`,
          fallback: "W^BK", shapeFallback: "[d,RK·dh]",
          relates: ["bk"],
          description: "key 的 channel 因子生成器；输出预旋转后写入缓存。",
          symbols: [
            { tex: R`W^{B_K}`, shape: R`[d_{\mathrm{model}},\,R_K d_h]`, label: "Key channel-factor generator", value: "1600 × (2 × 64)", note: "缓存里保存的是 R_t(B_K)" }
          ]
        },
        {
          id: "wav", kind: "weight", glyph: "down", w: 104,
          x: 643, y: 620,
          label: "A_V 因子生成器",
          tex: R`W^{A_V}`, texShape: R`[d,\,R_V h]`,
          fallback: "W^AV", shapeFallback: "[d,RV·h]",
          relates: ["av"],
          description: "value 的 head 轴因子生成器。",
          symbols: [
            { tex: R`W^{A_V}`, shape: R`[d_{\mathrm{model}},\,R_V h]`, label: "Value head-factor generator", value: "1600 × (2 × 78)", note: "R_V = 2" }
          ]
        },
        {
          id: "wbv", kind: "weight", glyph: "down", w: 104,
          x: 793, y: 620,
          label: "B_V 因子生成器",
          tex: R`W^{B_V}`, texShape: R`[d,\,R_V d_h]`,
          fallback: "W^BV", shapeFallback: "[d,RV·dh]",
          relates: ["bv"],
          description: "value 的 channel 因子生成器；B_V 不参与旋转。",
          symbols: [
            { tex: R`W^{B_V}`, shape: R`[d_{\mathrm{model}},\,R_V d_h]`, label: "Value channel-factor generator", value: "1600 × (2 × 64)", note: "对 value 施 RoPE 会注入绝对相位" }
          ]
        },
        {
          id: "aq", kind: "activation", w: 118,
          x: 36, y: 528,
          label: "A_Q 头轴因子",
          tex: R`A_Q(x_t)`, texShape: R`[B,\,L,\,R_Q,\,h]`,
          fallback: "A_Q", shapeFallback: "[B,L,RQ,h]",
          description: "token-dependent 的 head mixing：不同 token 把 channel 模式按不同强度写入不同 heads。",
          symbols: [
            { tex: R`A_Q(x_t)`, shape: R`[B,\,L,\,R_Q,\,h]`, label: "Query head factors", value: "R_Q = 6, h = 78", note: "区别于 MQA 的固定 head-sharing" }
          ]
        },
        {
          id: "bq", kind: "activation", w: 118,
          x: 186, y: 528,
          label: "B_Q 通道因子",
          tex: R`B_Q(x_t)`, texShape: R`[B,\,L,\,R_Q,\,d_h]`,
          fallback: "B_Q", shapeFallback: "[B,L,RQ,dh]",
          description: "64 维 channel 模式；每一 rank row 独立接受 RoPE 旋转。",
          symbols: [
            { tex: R`B_Q(x_t)`, shape: R`[B,\,L,\,R_Q,\,d_h]`, label: "Query channel factors", value: "R_Q = 6, d_h = 64", note: "旋转只作用于 channel 轴" }
          ]
        },
        {
          id: "ak", kind: "activation", w: 118,
          x: 336, y: 528,
          label: "A_K 头轴因子",
          tex: R`A_K(x_t)`, texShape: R`[B,\,L,\,R_K,\,h]`,
          fallback: "A_K", shapeFallback: "[B,L,RK,h]",
          description: "key 的 head 轴因子；与位置无关，直接进入因子缓存。",
          symbols: [
            { tex: R`A_K(x_t)`, shape: R`[B,\,L,\,R_K,\,h]`, label: "Key head factors", value: "R_K = 2", note: "不旋转，原样缓存" }
          ]
        },
        {
          id: "bk", kind: "activation", w: 118,
          x: 486, y: 528,
          label: "B_K 通道因子",
          tex: R`B_K(x_t)`, texShape: R`[B,\,L,\,R_K,\,d_h]`,
          fallback: "B_K", shapeFallback: "[B,L,RK,dh]",
          description: "key 的 channel 因子；写缓存前按位置 t 预旋转成 B̃_K。",
          symbols: [
            { tex: R`B_K(x_t)`, shape: R`[B,\,L,\,R_K,\,d_h]`, label: "Key channel factors", value: "R_K = 2, d_h = 64", note: "R_t(A^⊤B) = A^⊤R_t(B) 保证合法" }
          ]
        },
        {
          id: "av", kind: "activation", w: 118,
          x: 636, y: 528,
          label: "A_V 头轴因子",
          tex: R`A_V(x_t)`, texShape: R`[B,\,L,\,R_V,\,h]`,
          fallback: "A_V", shapeFallback: "[B,L,RV,h]",
          description: "value 的 head 轴因子，直接进入因子缓存。",
          symbols: [
            { tex: R`A_V(x_t)`, shape: R`[B,\,L,\,R_V,\,h]`, label: "Value head factors", value: "R_V = 2", note: "与 B_V 配对收缩出 value 聚合" }
          ]
        },
        {
          id: "bv", kind: "activation", w: 118,
          x: 786, y: 528,
          label: "B_V 通道因子",
          tex: R`B_V(x_t)`, texShape: R`[B,\,L,\,R_V,\,d_h]`,
          fallback: "B_V", shapeFallback: "[B,L,RV,dh]",
          description: "value 的 channel 因子；保持不旋转，原样缓存。",
          symbols: [
            { tex: R`B_V(x_t)`, shape: R`[B,\,L,\,R_V,\,d_h]`, label: "Value channel factors", value: "R_V = 2, d_h = 64", note: "value 无旋转配对可消去相位" }
          ]
        },
        {
          id: "ropebq", kind: "operator", tone: "cyan",
          x: 228, y: 461,
          label: "逐行 RoPE（B_Q）",
          tex: R`R_t`, fallback: "R_t",
          description: "对 B_Q 的每一 rank row 施位置旋转，等价于旋转重建后的每头 Q。",
          symbols: [
            { tex: R`R_t(B_Q)`, shape: R`[R_Q,\,d_h]`, label: "Row-wise rotation", value: "每行独立旋转", note: "A 因子与位置无关" }
          ]
        },
        {
          id: "ropebk", kind: "operator", tone: "cyan",
          x: 528, y: 461,
          label: "逐行 RoPE（B_K，预旋转）",
          tex: R`R_t`, fallback: "R_t",
          description: "写入缓存前完成旋转；解码时无需回头重旋历史因子。",
          symbols: [
            { tex: R`\widetilde B_K=R_t(B_K)`, shape: R`[R_K,\,d_h]`, label: "Pre-rotated key factors", value: "缓存已含位置", note: "相对位置由两侧配对保持" }
          ]
        },
        {
          id: "cache", kind: "state", w: 200, titleSize: 9.2,
          x: 670, y: 390,
          label: "因子化 KV cache",
          tex: R`A_K,\widetilde B_K,A_V,B_V`,
          texShape: R`[B,\,L,\,(R_K{+}R_V)(h{+}d_h)]`,
          fallback: "factor cache", shapeFallback: "[B,L,(RK+RV)(h+dh)]",
          relates: ["ak", "bk", "av", "bv"],
          description: "历史 token 只以低秩因子存在；完整 MHA-shaped 缓存从不物化。",
          symbols: [
            { tex: R`A_K,\widetilde B_K,A_V,B_V`, shape: R`[B,\,L,\,(R_K{+}R_V)(h{+}d_h)]`, label: "Persistent factors", value: "(2+2) × (78+64) = 568 elements/token", note: "完整 MHA-shaped 对照为 2 × 78 × 64 = 9984" }
          ]
        },
        {
          id: "score", kind: "activation", titleSize: 7.8, w: 230, h: 52,
          x: 130, y: 382,
          label: "因子域 score 收缩",
          tex: R`s=\tfrac{1}{R_QR_K}\sum_{p,r}A_QA_K\,(B_Q[p]{\cdot}B_K[r])`,
          texShape: R`[B,\,h,\,L,\,L]`,
          fallback: "s = (1/RQRK)Σ AQ·AK·(BQ·BK)", shapeFallback: "[B,h,L,L]",
          description: "先算 R_Q×R_K 个 channel 内积，再用 A 因子逐头加权；从不重建完整历史 K。",
          symbols: [
            { tex: R`s_{t,s,i}`, shape: R`[B,\,h,\,L,\,L]`, label: "Factor-domain scores", value: "h = 78 heads", note: "channel 内积形状 [B,L,S,R_Q,R_K] 与 head 数无关" }
          ]
        },
        {
          id: "smax", kind: "operator", glyph: "pill", w: 86, h: 28, titleSize: 8.2,
          x: 202, y: 326,
          label: "缩放 softmax",
          tex: R`\operatorname{softmax}`, fallback: "softmax",
          description: "对因子域收缩出的 score 做缩放与因果 mask 后的精确 softmax。",
          symbols: [
            { tex: R`a=\operatorname{softmax}(s/\sqrt{d_h}+M)`, shape: R`[B,\,h,\,L,\,L]`, label: "Attention weights", value: "scale = 1/√64", note: "精确 attention，无近似" }
          ]
        },
        {
          id: "o", kind: "activation", titleSize: 8.2, w: 230,
          x: 130, y: 232,
          label: "因子域 value 聚合",
          tex: R`o_{t,i}=\tfrac1{R_V}\sum_{s,r}a\,A_V[r,i]\,B_V[r]`,
          texShape: R`[B,\,h,\,L,\,d_h]`,
          fallback: "o = (1/RV)Σ a·AV·BV", shapeFallback: "[B,h,L,dh]",
          description: "value 聚合同样在因子域直接收缩，这是 FlashTPA decoding 的代数基础。",
          symbols: [
            { tex: R`o_{t,i}`, shape: R`[B,\,h,\,L,\,d_h]`, label: "Per-head outputs", value: "78 × 64 = 4992 inner width", note: "attention inner width 大于 d_model = 1600" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 189, y: 150,
          label: "输出投影",
          tex: R`W^{O}`, texShape: R`[h\,d_h,\,d]`,
          fallback: "W^O", shapeFallback: "[h·dh,d]",
          relates: ["u"],
          description: "把 4992 维 inner width 压回 1600 维残差流。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[h\,d_h,\,d_{\mathrm{model}}]`, label: "Output projection", value: "4992 × 1600", note: "参数量配平选择，不是维度错误" }
          ]
        },
        {
          id: "u", kind: "activation", tone: "gather",
          x: 181, y: 60,
          label: "层输出",
          tex: R`u_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "u_t", shapeFallback: "[B,L,d_model]",
          description: "TPA 层输出；缓存收益来自低秩因子化，不改变输出形状。",
          symbols: [
            { tex: R`u_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 1600", note: "T6-XL block size 1024 训练" }
          ]
        }
      ],
      edges: [
        { from: "x", to: "waq", tone: "compute", bend: 690 },
        { from: "x", to: "wbq", tone: "compute", bend: 684 },
        { from: "x", to: "wak", tone: "compute", bend: 678 },
        { from: "x", to: "wbk", tone: "compute", bend: 678 },
        { from: "x", to: "wav", tone: "compute", bend: 684 },
        { from: "x", to: "wbv", tone: "compute", bend: 690 },
        { from: "waq", to: "aq", tone: "weight" },
        { from: "wbq", to: "bq", tone: "weight" },
        { from: "wak", to: "ak", tone: "weight" },
        { from: "wbk", to: "bk", tone: "weight" },
        { from: "wav", to: "av", tone: "weight" },
        { from: "wbv", to: "bv", tone: "weight" },
        { from: "bq", to: "ropebq", tone: "compute" },
        { from: "ropebq", to: "score", tone: "cyan" },
        { from: "aq", to: "score", tone: "compute", toAt: 0.25, bend: 446 },
        { from: "bk", to: "ropebk", tone: "compute" },
        { from: "ropebk", to: "cache", tone: "state", fromSide: "top", toSide: "bottom", toAt: 0.25, bend: 446, label: "预旋转写入", lx: 636, ly: 470 },
        { from: "ak", to: "cache", tone: "state", fromSide: "top", toSide: "bottom", toAt: 0.4, bend: 452 },
        { from: "av", to: "cache", tone: "state", fromSide: "top", toSide: "bottom", toAt: 0.6, bend: 458 },
        { from: "bv", to: "cache", tone: "state", toAt: 0.875 },
        { from: "cache", to: "score", tone: "state", dashed: true, fromSide: "left", toSide: "right", toAt: 33 / 52, label: "读取历史因子", lx: 515, ly: 404 },
        { from: "score", to: "smax", tone: "compute" },
        { from: "smax", to: "o", tone: "compute" },
        { from: "cache", to: "o", tone: "state", dashed: true, fromSide: "top", toSide: "right", label: R`A_V,B_V`, lx: 700, ly: 246 },
        { from: "o", to: "wo", tone: "gather" },
        { from: "wo", to: "u", tone: "gather" }
      ]
    },
    dsa: {
      groups: [
        { id: "indexer", label: "LIGHTNING INDEXER · 低成本选址", tone: "control", members: ["wqi", "wki", "qi", "ki", "kicache", "logits", "topk", "selected"] },
        { id: "core", label: "SPARSE MLA CORE · LATENT 复用", tone: "gather", titlePos: "bottom", members: ["wmla", "qmla", "mlacache", "gather", "gathered"] }
      ],
      modules: [
        {
          id: "h", kind: "activation", tone: "cyan",
          x: 402, y: 880, w: 136, h: 46,
          label: "输入残差流",
          tex: R`h_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "h_t", shapeFallback: "[B,L,d_model]",
          description: "同一 hidden state 同时驱动低成本 Lightning Indexer 与昂贵的 MLA core。",
          symbols: [
            { tex: R`h_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 7168", note: "DeepSeek-V3.2-Exp 671B 代表配置" }
          ]
        },
        {
          id: "wqi", kind: "weight", glyph: "down",
          x: 94, y: 800,
          label: "Indexer query/权重投影",
          tex: R`W^{Iq},W^{Iw}`, texShape: R`[\,\cdot\,,H^{I}d_I{+}H^{I}]`,
          fallback: "W^Iq, W^Iw", shapeFallback: "[·,HI·dI+HI]", titleSize: 8.8,
          relates: ["qi"],
          description: "生成 64 个 128 维 indexer query 与逐头聚合权重 w^I；官方实现中 q^I 由共享 MLA query latent 投影。",
          symbols: [
            { tex: R`W^{Iq}`, shape: R`[d_c',\,H^{I}d_I]`, label: "Indexer query projection", value: "H^I = 64, d_I = 128", note: "自 MLA query latent（官方实现）" },
            { tex: R`W^{Iw}`, shape: R`[\,\cdot\,,H^{I}]`, label: "Head-weight generator", value: "每头一个标量 w^I", note: "ReLU 后跨头加权求和" }
          ]
        },
        {
          id: "wki", kind: "weight", glyph: "down",
          x: 274, y: 800,
          label: "Indexer key 投影",
          tex: R`W^{Ik}`, texShape: R`[d,\,d_I]`,
          fallback: "W^Ik", shapeFallback: "[d,dI]",
          relates: ["ki"],
          description: "共享 index key：每 token 只有一份 128 维，LayerNorm 后写入低精度缓存。",
          symbols: [
            { tex: R`W^{Ik}`, shape: R`[d_{\mathrm{model}},\,d_I]`, label: "Shared index-key projection", value: "7168 × 128", note: "所有 indexer heads 共享 key" }
          ]
        },
        {
          id: "wmla", kind: "weight", glyph: "down", titleSize: 8.4,
          x: 554, y: 800,
          label: "MLA core 投影组",
          tex: R`W^{DQ},W^{UQ},W^{QR}`, texShape: R`d\to H\,(d_h^{C}{+}d_h^{R})`,
          fallback: "MLA projections", shapeFallback: "d → H·(dhC+dhR)",
          relates: ["qmla", "mlacache"],
          description: "core 完整继承 MLA 的 query/latent 投影族；缓存条目也由 W^DKV/W^KR 产生。",
          symbols: [
            { tex: R`W^{DQ},W^{UQ},W^{QR}`, shape: R`d\to[B,\,H,\,L,\,d_h^{C}{+}d_h^{R}]`, label: "Core query projections", value: "H = 128, 192 = 128 + 64", note: "与 MLA 章完全同构" },
            { tex: R`W^{DKV},W^{KR}`, shape: R`d\to[B,\,L,\,d_c{+}d_h^{R}]`, label: "Latent/RoPE-key projections", value: "d_c = 512, d_h^R = 64", note: "写入 MLA latent cache" }
          ]
        },
        {
          id: "qi", kind: "activation",
          x: 86, y: 712,
          label: "Indexer queries",
          tex: R`q_{t,j}^{I},w_{t,j}^{I}`, texShape: R`[B,\,L,\,H^{I},\,d_I]`,
          fallback: "q^I, w^I", shapeFallback: "[B,L,HI,dI]",
          description: "64 个低维 query 加逐头聚合权重；两侧都经 pRoPE 与 Hadamard 配对变换。",
          symbols: [
            { tex: R`q_{t,j}^{I}`, shape: R`[B,\,L,\,H^{I},\,d_I]`, label: "Indexer queries", value: "H^I = 64, d_I = 128", note: "pRoPE → Hadamard → FP8（官方路径）" },
            { tex: R`w_{t,j}^{I}`, shape: R`[B,\,L,\,H^{I}]`, label: "Aggregation weights", value: "每头标量", note: "跨头加权求和 ReLU 项" }
          ]
        },
        {
          id: "ki", kind: "activation",
          x: 266, y: 712,
          label: "共享 index key",
          tex: R`k_s^{I}`, texShape: R`[B,\,L,\,d_I]`,
          fallback: "k^I", shapeFallback: "[B,L,dI]",
          description: "每 token 一份 128 维共享 key；写入缓存前完成 pRoPE 与 Hadamard。",
          symbols: [
            { tex: R`k_s^{I}`, shape: R`[B,\,L,\,d_I]`, label: "Shared index key", value: "d_I = 128", note: "Hadamard 服务 FP8 数值分布，不注入位置" }
          ]
        },
        {
          id: "qmla", kind: "activation",
          x: 546, y: 712,
          label: "MLA core query",
          tex: R`q_t^{C},q_t^{R}`, texShape: R`[B,\,H,\,L,\,d_h^{C}{+}d_h^{R}]`,
          fallback: "q^C, q^R", shapeFallback: "[B,H,L,192]",
          description: "高维 core query：128 个 heads、每头 128 内容 + 64 RoPE 维。",
          symbols: [
            { tex: R`q_t^{C},q_t^{R}`, shape: R`[B,\,H,\,L,\,d_h^{C}{+}d_h^{R}]`, label: "Core queries", value: "H = 128, 192-d/head", note: "只对选中的候选集合打分" }
          ]
        },
        {
          id: "kicache", kind: "state",
          x: 256, y: 616,
          label: "Indexer key cache",
          tex: R`k_{1:L}^{I}`, texShape: R`[B,\,L,\,d_I]`,
          fallback: "k^I cache", shapeFallback: "[B,L,dI]",
          relates: ["ki"],
          description: "低维低精度的第二缓存：Indexer 用它扫描全部历史位置。",
          symbols: [
            { tex: R`k_{1:L}^{I}`, shape: R`[B,\,L,\,d_I]`, label: "Index-key cache", value: "128 elements/token（FP8 可选）", note: "与 MLA cache 并存，合计 704 逻辑元素/token" }
          ]
        },
        {
          id: "mlacache", kind: "state",
          x: 736, y: 616,
          label: "MLA latent cache",
          tex: R`c_{1:L}^{KV},k_{1:L}^{R}`, texShape: R`[B,\,L,\,d_c{+}d_h^{R}]`,
          fallback: "MLA cache", shapeFallback: "[B,L,576]",
          relates: ["wmla", "gather"],
          description: "core 的持久缓存与 MLA 完全一致：512 维 latent + 64 维共享 RoPE key。",
          symbols: [
            { tex: R`c_{1:L}^{KV},k_{1:L}^{R}`, shape: R`[B,\,L,\,d_c{+}d_h^{R}]`, label: "Core persistent state", value: "512 + 64 = 576 elements/token", note: "稀疏选择不改变缓存口径" }
          ]
        },
        {
          id: "logits", kind: "activation", titleSize: 7.8, w: 200, h: 52,
          x: 140, y: 532,
          label: "全历史 Indexer logits",
          tex: R`I_{t,s}=\sum_j w_{t,j}^{I}\operatorname{ReLU}(q_{t,j}^{I\top}k_s^{I})`,
          texShape: R`[B,\,L_q,\,L_k]`,
          fallback: "I_ts = Σ w·ReLU(qI·kI)", shapeFallback: "[B,Lq,Lk]",
          description: "低成本扫描每个历史位置；这是 DSA 里唯一 O(L²) 但极低维的路径。",
          symbols: [
            { tex: R`I_{t,s}`, shape: R`[B,\,L_q,\,L_k]`, label: "Indexer logits", value: "64 heads × 128-d 点积", note: "训练时以 KL 对齐主 attention 分布" }
          ]
        },
        {
          id: "topk", kind: "operator", glyph: "diamond", tone: "orange", w: 64, h: 40, titleSize: 8.6,
          x: 208, y: 450,
          label: "token-level top-k 路由",
          tex: R`\mathrm{top}\text{-}k`, fallback: "top-k",
          description: "每个 query 只保留 2048 个候选位置；离散路由存在漏召回风险。",
          symbols: [
            { tex: R`\operatorname{TopK}_s(I_{t,s},k)`, shape: R`[B,\,L_q,\,L_k]\to[B,\,L_q,\,k]`, label: "Routing operator", value: "k = 2048", note: "Indexer 输入 detach，LM 损失不经此路由训练" }
          ]
        },
        {
          id: "selected", kind: "activation", tone: "orange",
          x: 176, y: 356,
          label: "选中位置集合",
          tex: R`\mathcal I_t`, texShape: R`[B,\,L_q,\,k]`,
          fallback: "I_t indices", shapeFallback: "[B,Lq,k]",
          description: "纯地址信息：core 按这些位置去 MLA cache 取回原始条目。",
          symbols: [
            { tex: R`\mathcal I_t`, shape: R`[B,\,L_q,\,k]`, label: "Selected positions", value: "top-k = 2048", note: "原型 DSA 没有额外 raw-token 滑窗" }
          ]
        },
        {
          id: "gather", kind: "operator", glyph: "pill", tone: "gather", w: 86, h: 28, titleSize: 8.2,
          x: 567, y: 365,
          label: "按地址取回条目",
          tex: R`\operatorname{Gather}`, fallback: "Gather",
          description: "把选中位置的原始 latent/RoPE-key 条目交给高维 core。",
          symbols: [
            { tex: R`\operatorname{Gather}(\{c^{KV},k^{R}\},\mathcal I_t)`, shape: R`[B,\,L_q,\,k,\,576]`, label: "Gather operator", value: "逻辑形状；实现可分页/分块", note: "地址来自 Indexer，内容来自 MLA cache" }
          ]
        },
        {
          id: "gathered", kind: "activation", w: 150, titleSize: 9,
          x: 535, y: 252,
          label: "候选核心条目",
          tex: R`\{c_s^{KV},k_s^{R}\}_{s\in\mathcal I_t}`,
          texShape: R`[B,\,L_q,\,k,\,576]`,
          fallback: "gathered entries", shapeFallback: "[B,Lq,k,576]",
          description: "选中的 2048 条 512+64 维原始条目；不做任何压缩或近似。",
          symbols: [
            { tex: R`\{c_s^{KV},k_s^{R}\}_{s\in\mathcal I_t}`, shape: R`[B,\,L_q,\,k,\,d_c{+}d_h^{R}]`, label: "Gathered core entries", value: "512 latent + 64 RoPE", note: "candidate-only 精确 MLA 的输入" }
          ]
        },
        {
          id: "core", kind: "activation", titleSize: 8.4, w: 200, h: 52,
          x: 510, y: 170,
          label: "候选内精确 MLA core",
          tex: R`o_t=\operatorname{softmax}_{s\in\mathcal I_t}\!(S_{t,s})\,V`,
          texShape: R`[B,\,H,\,L,\,d_h]`,
          fallback: "o = softmax_I(S)V", shapeFallback: "[B,H,L,dh]",
          description: "softmax 只在选中集合内重新归一化；MLA 的吸收执行形态原样保留。",
          symbols: [
            { tex: R`o_t`, shape: R`[B,\,H,\,L,\,d_h]`, label: "Sparse core output", value: "H = 128, d_h = 128（V 侧）", note: "选中后的 score 仍按 MLA 定义计算" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 554, y: 100,
          label: "输出投影",
          tex: R`W^{O}`, texShape: R`[H d_h,\,d]`,
          fallback: "W^O", shapeFallback: "[H·dh,d]",
          relates: ["u"],
          description: "core 输出写回残差流；与 MLA 的吸收式输出等价。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[H d_h,\,d_{\mathrm{model}}]`, label: "Output projection", value: "(128 × 128) × 7168", note: "可与 W^UV 预先合并" }
          ]
        },
        {
          id: "u", kind: "activation", tone: "gather",
          x: 546, y: 24,
          label: "层输出",
          tex: R`u_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "u_t", shapeFallback: "[B,L,d_model]",
          description: "DSA 层输出；稀疏化只减少 core 读取的位置数量。",
          symbols: [
            { tex: R`u_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 7168", note: "精确 attention 限于选中集合" }
          ]
        }
      ],
      edges: [
        { from: "h", to: "wqi", tone: "compute", bend: 860 },
        { from: "h", to: "wki", tone: "compute", bend: 854 },
        { from: "h", to: "wmla", tone: "compute", bend: 868 },
        { from: "wqi", to: "qi", tone: "weight" },
        { from: "wki", to: "ki", tone: "weight" },
        { from: "wmla", to: "qmla", tone: "weight" },
        { from: "wmla", to: "mlacache", tone: "state", dashed: true, fromSide: "top", fromAt: 0.8, toSide: "bottom", bend: 774, label: "latent 写入", lx: 760, ly: 786 },
        { from: "ki", to: "kicache", tone: "state" },
        { from: "qi", to: "logits", tone: "compute", toAt: 0.25, bend: 596 },
        { from: "kicache", to: "logits", tone: "state", toAt: 0.75, bend: 596 },
        { from: "logits", to: "topk", tone: "orange" },
        { from: "topk", to: "selected", tone: "orange" },
        { from: "selected", to: "gather", tone: "orange", label: "地址", lx: 436, ly: 368 },
        { from: "mlacache", to: "gather", tone: "state", fromSide: "top", toSide: "right", label: "内容", lx: 760, ly: 500 },
        { from: "gather", to: "gathered", tone: "gather" },
        { from: "gathered", to: "core", tone: "compute" },
        { from: "qmla", to: "core", tone: "compute", fromSide: "left", toSide: "left", bend: 480, label: R`q^{C},q^{R}`, lx: 480, ly: 460 },
        { from: "core", to: "wo", tone: "gather" },
        { from: "wo", to: "u", tone: "gather" }
      ]
    },
    csa: {
      groups: [
        { id: "compression", label: "OVERLAP 压缩 · m=4", tone: "compute", titlePos: "bottom", members: ["wc", "ccomp", "ccompcache"], pad: 12 },
        { id: "indexing", label: "COMPRESSED INDEXER · 选址", tone: "control", members: ["wic", "kic", "kiccache", "wiq", "qi", "logits", "topk", "selected"], pad: 12 }
      ],
      modules: [
        {
          id: "h", kind: "activation", tone: "cyan",
          x: 422, y: 920, w: 136, h: 46,
          label: "原始 token 流",
          tex: R`H`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "H", shapeFallback: "[B,L,d_model]",
          description: "CSA 同时构建重叠 compressed KV、独立 index keys、core query 与原始局部窗口。",
          symbols: [
            { tex: R`H`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 7168", note: "DeepSeek-V4-Pro 代表配置" }
          ]
        },
        {
          id: "wc", kind: "weight", glyph: "down", titleSize: 8.8,
          x: 74, y: 840,
          label: "KV compressor（重叠）",
          tex: R`\mathcal C_{\theta_C}`, texShape: R`2m\ \text{tokens}\to c`,
          fallback: "compressor θ_C", shapeFallback: "2m tokens → c",
          relates: ["ccomp"],
          description: "两路 a/b 投影对 2m=8 的重叠窗口做逐通道 softmax 加权求和，得到一个 512 维条目。",
          symbols: [
            { tex: R`\theta_C`, shape: R`[d_{\mathrm{model}},\,c]\ (\times\,a/b)`, label: "Core compressor params", value: "c = 512 · span 8 · stride 4", note: "重叠 2m receptive field" }
          ]
        },
        {
          id: "wic", kind: "weight", glyph: "down", titleSize: 8.8,
          x: 274, y: 840,
          label: "Index compressor（独立）",
          tex: R`\mathcal C_{\theta_I}`, texShape: R`\theta_I\ne\theta_C`,
          fallback: "compressor θ_I", shapeFallback: "θ_I ≠ θ_C",
          relates: ["kic"],
          description: "独立参数的第二个压缩器，只产生 128 维 index keys；与内容缓存职责分离。",
          symbols: [
            { tex: R`\theta_I`, shape: R`[d_{\mathrm{model}},\,d_I]`, label: "Index compressor params", value: "d_I = 128", note: "参数与 θ_C 不共享" }
          ]
        },
        {
          id: "wiq", kind: "weight", glyph: "down", titleSize: 8.8,
          x: 474, y: 840,
          label: "Indexer query 投影",
          tex: R`W^{Iq},W^{Iw}`, texShape: R`d\to H^{I}d_I`,
          fallback: "W^Iq, W^Iw", shapeFallback: "d → HI·dI",
          relates: ["qi"],
          description: "生成 64 个 128 维 indexer query 与逐头聚合权重。",
          symbols: [
            { tex: R`W^{Iq},W^{Iw}`, shape: R`[d,\,H^{I}d_I{+}H^{I}]`, label: "Indexer query projections", value: "H^I = 64, d_I = 128", note: "扫描的是压缩候选，成本再降 m 倍" }
          ]
        },
        {
          id: "wq", kind: "weight", glyph: "down",
          x: 674, y: 840,
          label: "Core query 投影",
          tex: R`W^{Q}`, texShape: R`[d,\,H_q c]`,
          fallback: "W^Q", shapeFallback: "[d,Hq·c]",
          relates: ["qcore"],
          description: "128 个 512 维 core query heads；同一 512 维条目兼作 K 与 V。",
          symbols: [
            { tex: R`W^{Q}`, shape: R`[d_{\mathrm{model}},\,H_q c]`, label: "Core query projection", value: "7168 → 128 × 512", note: "core head dim = c = 512" }
          ]
        },
        {
          id: "ccomp", kind: "activation", tone: "control",
          x: 66, y: 752,
          label: "重叠压缩条目",
          tex: R`C^{\mathrm{Comp}}`, texShape: R`[B,\,\lfloor L/m\rfloor,\,c]`,
          fallback: "C^Comp", shapeFallback: "[B,L/m,c]",
          description: "每个条目汇总 2m=8 个 raw tokens，序列长度缩短约 4 倍；同一条目兼作 key 与 value。",
          symbols: [
            { tex: R`C^{\mathrm{Comp}}`, shape: R`[B,\,\lfloor L/m\rfloor,\,c]`, label: "Compressed KV entries", value: "m = 4, c = 512", note: "末 64 维带 partial RoPE" }
          ]
        },
        {
          id: "kic", kind: "activation", tone: "control",
          x: 266, y: 752,
          label: "压缩 index keys",
          tex: R`K^{I\mathrm{Comp}}`, texShape: R`[B,\,\lfloor L/m\rfloor,\,d_I]`,
          fallback: "K^IComp", shapeFallback: "[B,L/m,dI]",
          description: "只负责打分与选址的低维 keys；与 C^Comp 是不同的缓存。",
          symbols: [
            { tex: R`K^{I\mathrm{Comp}}`, shape: R`[B,\,\lfloor L/m\rfloor,\,d_I]`, label: "Compressed index keys", value: "d_I = 128", note: "地址与内容分离" }
          ]
        },
        {
          id: "qi", kind: "activation",
          x: 466, y: 752,
          label: "Indexer queries",
          tex: R`q_t^{I},w_t^{I}`, texShape: R`[B,\,L,\,H^{I},\,d_I]`,
          fallback: "q^I, w^I", shapeFallback: "[B,L,HI,dI]",
          description: "64 个 indexer query 扫描约 L/m 个压缩候选。",
          symbols: [
            { tex: R`q_t^{I},w_t^{I}`, shape: R`[B,\,L,\,H^{I},\,d_I]`, label: "Indexer queries", value: "H^I = 64", note: "每 query 位置逐条目打分" }
          ]
        },
        {
          id: "qcore", kind: "activation",
          x: 666, y: 752,
          label: "Core queries",
          tex: R`q_t`, texShape: R`[B,\,H_q,\,L,\,c]`,
          fallback: "q_t", shapeFallback: "[B,Hq,L,c]",
          description: "core 前经 per-head RMSNorm；末 64 维施 partial RoPE。",
          symbols: [
            { tex: R`q_t`, shape: R`[B,\,H_q,\,L,\,c]`, label: "Core queries", value: "H_q = 128, c = 512", note: "RoPE slice = 末 64 维" }
          ]
        },
        {
          id: "ccompcache", kind: "state",
          x: 56, y: 656,
          label: "C^Comp 内容缓存",
          tex: R`C_{1:\lfloor t/m\rfloor}^{\mathrm{Comp}}`, texShape: R`[B,\,\lfloor L/m\rfloor,\,c]`,
          fallback: "C^Comp cache", shapeFallback: "[B,L/m,c]",
          relates: ["ccomp", "gather"],
          description: "core 内容条目的缓存；被 top-k 地址按需取回。",
          symbols: [
            { tex: R`C_{1:\lfloor t/m\rfloor}^{\mathrm{Comp}}`, shape: R`[B,\,\lfloor L/m\rfloor,\,c]`, label: "Content cache", value: "512 elements/entry", note: "长度轴缩短 m 倍" }
          ]
        },
        {
          id: "kiccache", kind: "state",
          x: 256, y: 656,
          label: "K^IComp 地址缓存",
          tex: R`K_{1:\lfloor t/m\rfloor}^{I\mathrm{Comp}}`, texShape: R`[B,\,\lfloor L/m\rfloor,\,d_I]`,
          fallback: "K^IComp cache", shapeFallback: "[B,L/m,dI]",
          relates: ["kic"],
          description: "低维选址缓存；Indexer 从约 L/4 个压缩候选中打分。",
          symbols: [
            { tex: R`K_{1:\lfloor t/m\rfloor}^{I\mathrm{Comp}}`, shape: R`[B,\,\lfloor L/m\rfloor,\,d_I]`, label: "Index-key cache", value: "128 elements/entry", note: "与内容缓存职责分离" }
          ]
        },
        {
          id: "swa", kind: "state",
          x: 826, y: 656,
          label: "原始局部窗口",
          tex: R`H_{t-w:t}`, texShape: R`[B,\,w,\,d]`,
          fallback: "raw window", shapeFallback: "[B,w,d]",
          description: "未压缩的最近 token 窗口，弥补 compressed core 的细节损失。",
          symbols: [
            { tex: R`H_{t-w:t}`, shape: R`[B,\,w,\,d_{\mathrm{model}}]`, label: "Raw local window", value: "w = 128 tokens", note: "与 compressed global branch 并行进入同一 softmax" }
          ]
        },
        {
          id: "logits", kind: "activation", titleSize: 7.6, w: 200, h: 52,
          x: 230, y: 560,
          label: "压缩候选 logits",
          tex: R`I_{t,s}=\sum_j w_{t,j}^{I}\operatorname{ReLU}(q_{t,j}^{I\top}K_s^{I\mathrm{Comp}})`,
          texShape: R`[B,\,L,\,\lfloor L/m\rfloor]`,
          fallback: "I = Σ w·ReLU(qI·K^IComp)", shapeFallback: "[B,L,L/m]",
          description: "候选单位是 compressed entry；扫描成本相对 raw 序列再降 m 倍。",
          symbols: [
            { tex: R`I_{t,s}`, shape: R`[B,\,L,\,\lfloor L/m\rfloor]`, label: "Indexer logits", value: "约 L/4 个候选", note: "低维低精度路径" }
          ]
        },
        {
          id: "topk", kind: "operator", glyph: "diamond", tone: "orange", w: 64, h: 40, titleSize: 8.6,
          x: 298, y: 490,
          label: "entry-level top-k",
          tex: R`\mathrm{top}\text{-}k`, fallback: "top-k",
          description: "从压缩候选中选出 1024 条；每条覆盖 2m=8 个 raw tokens。",
          symbols: [
            { tex: R`\operatorname{TopK}_s(I_{t,s},k)`, shape: R`[B,\,L,\,\lfloor L/m\rfloor]\to[B,\,L,\,k]`, label: "Routing operator", value: "k = 1024", note: "top-k 单位是 compressed entry" }
          ]
        },
        {
          id: "selected", kind: "activation", tone: "orange",
          x: 266, y: 396,
          label: "选中条目地址",
          tex: R`\mathcal J_t`, texShape: R`[B,\,L,\,k]`,
          fallback: "J_t indices", shapeFallback: "[B,L,k]",
          description: "top-k 地址下传给 Gather，从内容缓存取回对应条目。",
          symbols: [
            { tex: R`\mathcal J_t`, shape: R`[B,\,L,\,k]`, label: "Selected entries", value: "top-k = 1024", note: "约覆盖 8192 个 raw tokens" }
          ]
        },
        {
          id: "gather", kind: "operator", glyph: "pill", tone: "gather", w: 86, h: 28, titleSize: 8.2,
          x: 517, y: 326,
          label: "取回选中内容",
          tex: R`\operatorname{Gather}`, fallback: "Gather",
          description: "按地址从 C^Comp 缓存取回 512 维条目，交给 shared-KV core。",
          symbols: [
            { tex: R`\operatorname{Gather}(C^{\mathrm{Comp}},\mathcal J_t)`, shape: R`[B,\,L,\,k,\,c]`, label: "Gathered entries", value: "one shared KV head", note: "逻辑形状；实现可分页" }
          ]
        },
        {
          id: "rms", kind: "operator", glyph: "pill", tone: "cyan", w: 96, h: 28, titleSize: 8.2,
          x: 682, y: 316,
          label: "per-head RMSNorm",
          tex: R`\operatorname{RMSNorm}`, fallback: "RMSNorm",
          description: "core 前对 query heads 与唯一 compressed-KV head 归一化。",
          symbols: [
            { tex: R`\operatorname{RMSNorm}(q_t)`, shape: R`[B,\,H_q,\,L,\,c]`, label: "Pre-core normalization", value: "query 与 KV 两侧", note: "报告 §2.3.3 的必要步骤" }
          ]
        },
        {
          id: "core", kind: "activation", titleSize: 7.8, w: 220, h: 52,
          x: 620, y: 210,
          label: "shared-KV core（单一 softmax）",
          tex: R`o_t=\operatorname{MQA}_{K=V}(q_t;\mathrm{sink},C_{\mathcal J_t},\mathrm{SWA})`,
          texShape: R`[B,\,H_q,\,L,\,c]`,
          fallback: "o = MQA(q; sink, C_J, SWA)", shapeFallback: "[B,Hq,L,c]",
          description: "sink、选中全局摘要与 SWA 在同一个 softmax 中归一化；每头 sink logit 使真实权重和可小于 1。",
          symbols: [
            { tex: R`o_t`, shape: R`[B,\,H_q,\,L,\,c]`, label: "Core output", value: "128 heads 读同一份条目", note: "sink 为逐头可学习 logit" }
          ]
        },
        {
          id: "irope", kind: "operator", tone: "cyan",
          x: 713, y: 149,
          label: "输出 inverse RoPE",
          tex: R`R_{-t}`, fallback: "R_-t",
          description: "输出 RoPE slice 按原 query 位置 t 反旋转，使条目贡献只依赖 π_s − t。",
          symbols: [
            { tex: R`R_{-t}`, shape: R`[64,\,64]`, label: "Inverse rotary", value: "末 64 维 slice", note: "漏掉它会破坏相对位置恒等式" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up", titleSize: 8.8,
          x: 674, y: 88,
          label: "分组输出投影",
          tex: R`W^{OA},W^{OB}`, texShape: R`H_q c\to d`,
          fallback: "W^OA, W^OB", shapeFallback: "Hq·c → d",
          relates: ["u"],
          description: "按组 W^OA → Concat → W^OB 两级写回残差流。",
          symbols: [
            { tex: R`W^{OA},W^{OB}`, shape: R`[H_q c,\,d_{\mathrm{model}}]`, label: "Grouped output projection", value: "分组两级投影", note: "官方实现的写回顺序" }
          ]
        },
        {
          id: "u", kind: "activation", tone: "gather",
          x: 666, y: 16,
          label: "层输出",
          tex: R`u_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "u_t", shapeFallback: "[B,L,d_model]",
          description: "CSA 层输出；全局压缩、稀疏选择与局部窗口在此汇合。",
          symbols: [
            { tex: R`u_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 7168", note: "V4-Pro CSA layers（compress ratio 4）" }
          ]
        }
      ],
      edges: [
        { from: "h", to: "wc", tone: "compute", fromSide: "left", toSide: "right", bend: 240 },
        { from: "h", to: "wic", tone: "compute", bend: 894 },
        { from: "h", to: "wiq", tone: "compute", bend: 894 },
        { from: "h", to: "wq", tone: "compute", bend: 900 },
        { from: "h", to: "swa", tone: "state", dashed: true, fromSide: "right", toSide: "bottom", label: "raw 窗口 w=128", lx: 760, ly: 932 },
        { from: "wc", to: "ccomp", tone: "weight" },
        { from: "wic", to: "kic", tone: "weight" },
        { from: "wiq", to: "qi", tone: "weight" },
        { from: "wq", to: "qcore", tone: "weight" },
        { from: "ccomp", to: "ccompcache", tone: "state" },
        { from: "kic", to: "kiccache", tone: "state" },
        { from: "qi", to: "logits", tone: "compute", fromSide: "top", toSide: "bottom", toAt: 0.75, bend: 624 },
        { from: "kiccache", to: "logits", tone: "state" },
        { from: "logits", to: "topk", tone: "orange" },
        { from: "topk", to: "selected", tone: "orange" },
        { from: "selected", to: "gather", tone: "orange", fromSide: "right", toSide: "bottom", label: "地址", lx: 470, ly: 408 },
        { from: "ccompcache", to: "gather", tone: "state", fromSide: "top", toSide: "left", label: "内容条目", lx: 150, ly: 330 },
        { from: "gather", to: "core", tone: "gather", fromSide: "top", toSide: "left" },
        { from: "qcore", to: "rms", tone: "compute" },
        { from: "rms", to: "core", tone: "cyan" },
        { from: "swa", to: "core", tone: "state", dashed: true, fromSide: "top", toSide: "right", toAt: 0.33, label: "局部分支", lx: 906, ly: 440 },
        { from: "core", to: "irope", tone: "compute" },
        { from: "irope", to: "wo", tone: "compute" },
        { from: "wo", to: "u", tone: "gather" }
      ]
    },
    hca: {
      groups: [
        { id: "compression", label: "HEAVY 压缩 · m′=128", tone: "compute", titlePos: "bottom", members: ["wc", "ccomp", "gate", "ccompcache"], pad: 12 }
      ],
      modules: [
        {
          id: "h", kind: "activation", tone: "cyan",
          x: 402, y: 732, w: 136, h: 46,
          label: "原始 token 流",
          tex: R`H`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "H", shapeFallback: "[B,L,d_model]",
          description: "HCA 把已闭合的 128-token blocks 压成全局摘要，同时保留当前局部窗口。",
          symbols: [
            { tex: R`H`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 7168", note: "DeepSeek-V4-Pro 代表配置" }
          ]
        },
        {
          id: "wc", kind: "weight", glyph: "down", titleSize: 8.8,
          x: 114, y: 652,
          label: "非重叠块压缩器",
          tex: R`\mathcal C_{\theta_C}`, texShape: R`m'\ \text{tokens}\to c`,
          fallback: "compressor θ_C", shapeFallback: "m' tokens → c",
          relates: ["ccomp"],
          description: "块内逐通道 softmax(Z+B)⊙C 加权求和，把 128 个位置压成一条 512 维摘要。",
          symbols: [
            { tex: R`\theta_C`, shape: R`[d_{\mathrm{model}},\,c]\ (\times\,C/Z)`, label: "Block compressor params", value: "m′ = 128, c = 512", note: "每个输出通道是块内值的凸组合" }
          ]
        },
        {
          id: "wq", kind: "weight", glyph: "down",
          x: 414, y: 652,
          label: "Core query 投影",
          tex: R`W^{Q}`, texShape: R`[d,\,H_q c]`,
          fallback: "W^Q", shapeFallback: "[d,Hq·c]",
          relates: ["qcore"],
          description: "128 个 512 维 query heads；对全部已闭合摘要做 dense 读取。",
          symbols: [
            { tex: R`W^{Q}`, shape: R`[d_{\mathrm{model}},\,H_q c]`, label: "Core query projection", value: "7168 → 128 × 512", note: "core head dim = c = 512" }
          ]
        },
        {
          id: "ccomp", kind: "activation", tone: "control",
          x: 106, y: 564,
          label: "块摘要条目",
          tex: R`C_i^{\mathrm{Comp}}`, texShape: R`[B,\,\lfloor L/m'\rfloor,\,c]`,
          fallback: "C_i^Comp", shapeFallback: "[B,L/m',c]",
          description: "每个完整 block 产生一条 512 维条目；同一条目兼作 key 与 value。",
          symbols: [
            { tex: R`C_i^{\mathrm{Comp}}`, shape: R`[B,\,\lfloor L/m'\rfloor,\,c]`, label: "Block summaries", value: "m′ = 128, c = 512", note: "末 64 维按压缩位置 π_j 施 partial RoPE" }
          ]
        },
        {
          id: "qcore", kind: "activation",
          x: 406, y: 564,
          label: "Core queries",
          tex: R`q_t`, texShape: R`[B,\,H_q,\,L,\,c]`,
          fallback: "q_t", shapeFallback: "[B,Hq,L,c]",
          description: "core 前经 per-head RMSNorm；末 64 维施 partial RoPE。",
          symbols: [
            { tex: R`q_t`, shape: R`[B,\,H_q,\,L,\,c]`, label: "Core queries", value: "H_q = 128, c = 512", note: "RoPE slice = 末 64 维" }
          ]
        },
        {
          id: "gate", kind: "operator", glyph: "pill", tone: "control", w: 110, h: 28, titleSize: 8.4,
          x: 115, y: 494,
          label: "因果发布门",
          tex: R`m'(i{+}1)\le t`, fallback: "causal publish",
          description: "只有已关闭的 block 才发布到全局缓存；未闭合块由局部窗口覆盖。",
          symbols: [
            { tex: R`m'(i{+}1)\le t`, shape: R`\lfloor t/m'\rfloor\ \text{visible blocks}`, label: "Publish condition", value: "无 Indexer、无 top-k", note: "避免泄露未来信息" }
          ]
        },
        {
          id: "ccompcache", kind: "state",
          x: 96, y: 402,
          label: "已闭合块缓存",
          tex: R`\{C_i^{\mathrm{Comp}}\}_{m'(i+1)\le t}`, texShape: R`[B,\,\lfloor t/m'\rfloor,\,c]`,
          fallback: "completed cache", shapeFallback: "[B,t/m',c]", titleSize: 8.8,
          relates: ["ccomp"],
          description: "全局缓存随完成 block 数增长，长度轴缩短 128 倍。",
          symbols: [
            { tex: R`\{C_i^{\mathrm{Comp}}\}`, shape: R`[B,\,\lfloor t/m'\rfloor,\,c]`, label: "Persistent summaries", value: "one shared KV entry/block", note: "不为每个 raw token 保存完整 KV" }
          ]
        },
        {
          id: "swa", kind: "state",
          x: 686, y: 402,
          label: "原始局部窗口",
          tex: R`H_{t-w:t}`, texShape: R`[B,\,w,\,d]`,
          fallback: "raw window", shapeFallback: "[B,w,d]",
          description: "覆盖当前未闭合 block 与精细顺序的未压缩局部分支。",
          symbols: [
            { tex: R`H_{t-w:t}`, shape: R`[B,\,w,\,d_{\mathrm{model}}]`, label: "Raw local window", value: "w = 128 tokens", note: "与 compressed-dense global branch 并行" }
          ]
        },
        {
          id: "rms", kind: "operator", glyph: "pill", tone: "cyan", w: 96, h: 28, titleSize: 8.2,
          x: 422, y: 328,
          label: "per-head RMSNorm",
          tex: R`\operatorname{RMSNorm}`, fallback: "RMSNorm",
          description: "core 前对 query heads 与唯一 compressed-KV head 归一化。",
          symbols: [
            { tex: R`\operatorname{RMSNorm}(q_t)`, shape: R`[B,\,H_q,\,L,\,c]`, label: "Pre-core normalization", value: "query 与 KV 两侧", note: "报告 §2.3.3 的必要步骤" }
          ]
        },
        {
          id: "core", kind: "activation", titleSize: 7.8, w: 220, h: 52,
          x: 360, y: 222,
          label: "compressed-dense core（单一 softmax）",
          tex: R`o_t=\operatorname{MQA}_{K=V}(q_t;\mathrm{sink},C_{\mathrm{all}},\mathrm{SWA})`,
          texShape: R`[B,\,H_q,\,L,\,c]`,
          fallback: "o = MQA(q; sink, C_all, SWA)", shapeFallback: "[B,Hq,L,c]",
          description: "没有 Indexer 或 top-k：每个 query dense 读取全部 ⌊t/m′⌋ 条已闭合摘要与 SWA。",
          symbols: [
            { tex: R`A^{\mathrm{global}}`, shape: R`[B,\,H_q,\,L,\,\lfloor L/m'\rfloor]`, label: "Global weights", value: "dense · 因果排除未闭合块", note: "sink + 全部摘要 + SWA 一次归一化" }
          ]
        },
        {
          id: "irope", kind: "operator", tone: "cyan",
          x: 453, y: 161,
          label: "输出 inverse RoPE",
          tex: R`R_{-t}`, fallback: "R_-t",
          description: "输出 slice 按 raw query 位置 t 反旋转，把 R_{π_j} 变成相对旋转 R_{π_j−t}。",
          symbols: [
            { tex: R`R_{-t}`, shape: R`[64,\,64]`, label: "Inverse rotary", value: "末 64 维 slice", note: "π_j 由实现定义，不能擅自取块索引" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up", titleSize: 8.8,
          x: 414, y: 100,
          label: "分组输出投影",
          tex: R`W^{OA},W^{OB}`, texShape: R`H_q c\to d`,
          fallback: "W^OA, W^OB", shapeFallback: "Hq·c → d",
          relates: ["u"],
          description: "按组 W^OA → Concat → W^OB 两级写回残差流。",
          symbols: [
            { tex: R`W^{OA},W^{OB}`, shape: R`[H_q c,\,d_{\mathrm{model}}]`, label: "Grouped output projection", value: "分组两级投影", note: "与 CSA 共享写回配方" }
          ]
        },
        {
          id: "u", kind: "activation", tone: "gather",
          x: 406, y: 24,
          label: "层输出",
          tex: R`u_t`, texShape: R`[B,\,L,\,d_{\mathrm{model}}]`,
          fallback: "u_t", shapeFallback: "[B,L,d_model]",
          description: "HCA 层输出；全局重压缩 dense 读取与局部窗口在此汇合。",
          symbols: [
            { tex: R`u_t`, shape: R`[B,\,L,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 7168", note: "V4-Pro HCA layers（compress ratio 128）" }
          ]
        }
      ],
      edges: [
        { from: "h", to: "wc", tone: "compute", fromSide: "left", toSide: "right", bend: 280 },
        { from: "h", to: "wq", tone: "compute" },
        { from: "h", to: "swa", tone: "state", dashed: true, fromSide: "right", toSide: "bottom", label: "raw 窗口 w=128", lx: 650, ly: 744 },
        { from: "wc", to: "ccomp", tone: "weight" },
        { from: "wq", to: "qcore", tone: "weight" },
        { from: "ccomp", to: "gate", tone: "control" },
        { from: "gate", to: "ccompcache", tone: "state" },
        { from: "qcore", to: "rms", tone: "compute" },
        { from: "rms", to: "core", tone: "cyan" },
        { from: "ccompcache", to: "core", tone: "state", fromSide: "top", toSide: "left", label: "全部已闭合块", lx: 220, ly: 236 },
        { from: "swa", to: "core", tone: "state", dashed: true, fromSide: "top", toSide: "right", label: "局部分支", lx: 790, ly: 300 },
        { from: "core", to: "irope", tone: "compute" },
        { from: "irope", to: "wo", tone: "compute" },
        { from: "wo", to: "u", tone: "gather" }
      ]
    },
    kda: {
      modules: [
        {
          id: "x", kind: "activation", tone: "cyan",
          x: 417, y: 902, w: 136, h: 46,
          label: "输入残差流",
          tex: R`x_t`, texShape: R`[B,\,T,\,d_{\mathrm{model}}]`,
          fallback: "x_t", shapeFallback: "[B,T,d_model]",
          description: "KDA block 的输入：三条 conv 参数路径（q/k/v）与三条直接门控路径（α/β/g）并行出发。",
          symbols: [
            { tex: R`x_t`, shape: R`[B,\,T,\,d_{\mathrm{model}}]`, label: "Input activations", value: "d_model = 7168", note: "Kimi K3 · 2.8T total / 104B activated" }
          ]
        },
        {
          id: "wq", kind: "weight", glyph: "down", w: 104,
          x: 58, y: 822,
          label: "q 投影",
          tex: R`W^{Q}`, texShape: R`[d,\,H d_k]`,
          fallback: "W^Q", shapeFallback: "[d,H·dk]",
          relates: ["q"],
          description: "q 路径的线性投影；随后经 ShortConv、激活与 L2 归一化。",
          symbols: [
            { tex: R`W^{Q}`, shape: R`[d_{\mathrm{model}},\,H d_k]`, label: "Query projection", value: "7168 → 96 × 128", note: "H = 96, d_k = 128" }
          ]
        },
        {
          id: "wk", kind: "weight", glyph: "down", w: 104,
          x: 208, y: 822,
          label: "k 投影",
          tex: R`W^{K}`, texShape: R`[d,\,H d_k]`,
          fallback: "W^K", shapeFallback: "[d,H·dk]",
          relates: ["k"],
          description: "k 路径投影；L2 归一化让 delta 擦除具有清晰的方向投影几何。",
          symbols: [
            { tex: R`W^{K}`, shape: R`[d_{\mathrm{model}},\,H d_k]`, label: "Key projection", value: "7168 → 96 × 128", note: "‖k‖=1 时 I−βkk^⊤ 只收缩 key 方向" }
          ]
        },
        {
          id: "wv", kind: "weight", glyph: "down", w: 104,
          x: 358, y: 822,
          label: "v 投影",
          tex: R`W^{V}`, texShape: R`[d,\,H d_v]`,
          fallback: "W^V", shapeFallback: "[d,H·dv]",
          relates: ["v"],
          description: "v 路径投影；经 ShortConv 与激活，但不做 L2 归一化。",
          symbols: [
            { tex: R`W^{V}`, shape: R`[d_{\mathrm{model}},\,H d_v]`, label: "Value projection", value: "7168 → 96 × 128", note: "d_v = 128" }
          ]
        },
        {
          id: "walpha", kind: "weight", glyph: "pair", w: 104, titleSize: 8.6,
          x: 508, y: 822,
          label: "α 低秩门生成器",
          tex: R`W_\alpha^{\downarrow},W_\alpha^{\uparrow}`, texShape: R`d\to r\to H d_k`,
          fallback: "Wα↓, Wα↑", shapeFallback: "d → r → H·dk",
          relates: ["alpha"],
          description: "逐通道 decay 的低秩生成器：降维再升维，经 softplus/exp 得到每通道保留率。",
          symbols: [
            { tex: R`W_\alpha^{\downarrow}W_\alpha^{\uparrow}`, shape: R`[d,\,r]\times[r,\,H d_k]`, label: "Low-rank decay generator", value: "128 channels / head", note: "log α = −exp(A_h)·softplus(·)" }
          ]
        },
        {
          id: "wbeta", kind: "weight", glyph: "down", w: 104,
          x: 658, y: 822,
          label: "β 写入率投影",
          tex: R`W^{\beta}`, texShape: R`[d,\,H]`,
          fallback: "W^β", shapeFallback: "[d,H]",
          relates: ["beta"],
          description: "每 head 每 token 一个标量写入率；不经过 ShortConv。",
          symbols: [
            { tex: R`W^{\beta}`, shape: R`[d_{\mathrm{model}},\,H]`, label: "Delta-rate projection", value: "标量 / head / token", note: "β = σ(W^β x)" }
          ]
        },
        {
          id: "wg", kind: "weight", glyph: "down", w: 104,
          x: 808, y: 822,
          label: "输出门投影",
          tex: R`W^{G}`, texShape: R`[d,\,H d_v]`,
          fallback: "W^G", shapeFallback: "[d,H·dv]",
          relates: ["g"],
          description: "数据依赖的输出 gate；K3 配置为 full-rank（Kimi Linear 首发为低秩）。",
          symbols: [
            { tex: R`W^{G}`, shape: R`[d_{\mathrm{model}},\,H d_v]`, label: "Output-gate projection", value: "full-rank（K3 口径）", note: "读出处 RMSNorm(o) ⊙ σ(g)" }
          ]
        },
        {
          id: "convq", kind: "weight", glyph: "down", w: 76, h: 30, titleSize: 8.6,
          x: 72, y: 752,
          label: "q ShortConv",
          tex: R`\mathrm{Conv}_4`, texShape: R`[H d_k,\,4]`,
          fallback: "Conv4", shapeFallback: "[H·dk,4]", shapeSize: 7,
          relates: ["q"],
          description: "width-4 causal depthwise 卷积：为 q 提供局部顺序 shortcut。",
          symbols: [
            { tex: R`\operatorname{ShortConv}_4`, shape: R`[H d_k,\,4]`, label: "Depthwise causal conv", value: "width = 4", note: "只作用于 q/k/v 路径" }
          ]
        },
        {
          id: "convk", kind: "weight", glyph: "down", w: 76, h: 30, titleSize: 8.6,
          x: 222, y: 752,
          label: "k ShortConv",
          tex: R`\mathrm{Conv}_4`, texShape: R`[H d_k,\,4]`,
          fallback: "Conv4", shapeFallback: "[H·dk,4]", shapeSize: 7,
          relates: ["k"],
          description: "k 路径的 width-4 causal depthwise 卷积。",
          symbols: [
            { tex: R`\operatorname{ShortConv}_4`, shape: R`[H d_k,\,4]`, label: "Depthwise causal conv", value: "width = 4", note: "改善局部寻址与检索" }
          ]
        },
        {
          id: "convv", kind: "weight", glyph: "down", w: 76, h: 30, titleSize: 8.6,
          x: 372, y: 752,
          label: "v ShortConv",
          tex: R`\mathrm{Conv}_4`, texShape: R`[H d_v,\,4]`,
          fallback: "Conv4", shapeFallback: "[H·dv,4]", shapeSize: 7,
          relates: ["v"],
          description: "v 路径的 width-4 causal depthwise 卷积。",
          symbols: [
            { tex: R`\operatorname{ShortConv}_4`, shape: R`[H d_v,\,4]`, label: "Depthwise causal conv", value: "width = 4", note: "不是 KV cache，也不替代长程状态" }
          ]
        },
        {
          id: "actq", kind: "operator",
          x: 93, y: 689,
          label: "q 激活（Swish）",
          tex: R`\sigma`, fallback: "σ",
          description: "q 路径的 Swish/SiLU 激活。",
          symbols: [
            { tex: R`\operatorname{SiLU}`, shape: R`[B,\,T,\,H,\,d_k]`, label: "Activation", value: "逐元素", note: "允许有符号特征" }
          ]
        },
        {
          id: "actk", kind: "operator",
          x: 243, y: 689,
          label: "k 激活（Swish）",
          tex: R`\sigma`, fallback: "σ",
          description: "k 路径的 Swish/SiLU 激活。",
          symbols: [
            { tex: R`\operatorname{SiLU}`, shape: R`[B,\,T,\,H,\,d_k]`, label: "Activation", value: "逐元素", note: "随后 L2 归一化" }
          ]
        },
        {
          id: "actv", kind: "operator",
          x: 393, y: 689,
          label: "v 激活（Swish）",
          tex: R`\sigma`, fallback: "σ",
          description: "v 路径的 Swish/SiLU 激活；v 不做 L2 归一化。",
          symbols: [
            { tex: R`\operatorname{SiLU}`, shape: R`[B,\,T,\,H,\,d_v]`, label: "Activation", value: "逐元素", note: "v 保持无归一化" }
          ]
        },
        {
          id: "opalpha", kind: "operator", glyph: "pill", tone: "control", w: 96, h: 26, titleSize: 7.8,
          x: 512, y: 693,
          label: "softplus → exp",
          tex: R`\mathrm{softplus}\!\to\!\exp`, fallback: "softplus→exp",
          description: "把低秩输出变成 (0,1) 的逐通道保留率，带可学习下界。",
          symbols: [
            { tex: R`\alpha=\exp(-\exp(A_h^{\log})\operatorname{softplus}(\cdot))`, shape: R`(0,1)^{d_k}`, label: "Decay transform", value: "lower-bounded log decay", note: "每个 key channel 独立" }
          ]
        },
        {
          id: "opbeta", kind: "operator", tone: "control",
          x: 693, y: 689,
          label: "β sigmoid",
          tex: R`\sigma`, fallback: "σ",
          description: "sigmoid 把写入率压到 (0,1)。",
          symbols: [
            { tex: R`\beta_t=\sigma(W^{\beta}x_t)`, shape: R`(0,1)`, label: "Write-rate transform", value: "标量 / head", note: "控制 rank-1 delta 写入强度" }
          ]
        },
        {
          id: "opg", kind: "operator", tone: "control",
          x: 843, y: 689,
          label: "g sigmoid",
          tex: R`\sigma`, fallback: "σ",
          description: "输出 gate 的 sigmoid；在读出处与 RMSNorm(o) 逐元素相乘。",
          symbols: [
            { tex: R`\sigma(g_t)`, shape: R`(0,1)^{H d_v}`, label: "Output-gate transform", value: "数据依赖", note: "调节每通道读出强度" }
          ]
        },
        {
          id: "l2q", kind: "operator", glyph: "pill", tone: "cyan", w: 56, h: 24, titleSize: 8.2,
          x: 82, y: 638,
          label: "q L2 归一化",
          tex: R`\mathrm{L2}`, fallback: "L2",
          description: "q 的 L2 归一化（含 1/√d_k 缩放）。",
          symbols: [
            { tex: R`\operatorname{L2Norm}(q)`, shape: R`\|q\|_2=1`, label: "Normalization", value: "每 head", note: "稳定 delta 递推的谱" }
          ]
        },
        {
          id: "l2k", kind: "operator", glyph: "pill", tone: "cyan", w: 56, h: 24, titleSize: 8.2,
          x: 232, y: 638,
          label: "k L2 归一化",
          tex: R`\mathrm{L2}`, fallback: "L2",
          description: "单位 key 使 I−βkk^⊤ 的特征值全部落在 [0,1]。",
          symbols: [
            { tex: R`\operatorname{L2Norm}(k)`, shape: R`\|k\|_2=1`, label: "Normalization", value: "每 head", note: "β=1 时精确擦除 key 方向" }
          ]
        },
        {
          id: "q", kind: "activation", w: 118,
          x: 51, y: 560,
          label: "query 激活",
          tex: R`q_t`, texShape: R`[B,\,T,\,H,\,d_k]`,
          fallback: "q_t", shapeFallback: "[B,T,H,dk]",
          description: "读取固定状态的地址向量：o = S^⊤(q/√d_k)。",
          symbols: [
            { tex: R`q_t`, shape: R`[B,\,T,\,H,\,d_k]`, label: "Query", value: "H = 96, d_k = 128", note: "Linear→Conv→SiLU→L2" }
          ]
        },
        {
          id: "k", kind: "activation", w: 118,
          x: 201, y: 560,
          label: "key 激活",
          tex: R`k_t`, texShape: R`[B,\,T,\,H,\,d_k]`,
          fallback: "k_t", shapeFallback: "[B,T,H,dk]",
          description: "写入/擦除方向：decay 后沿 k 方向做 rank-1 delta 纠写。",
          symbols: [
            { tex: R`k_t`, shape: R`[B,\,T,\,H,\,d_k]`, label: "Key", value: "H = 96, d_k = 128", note: "单位化后进入 (I−βkk^⊤)" }
          ]
        },
        {
          id: "v", kind: "activation", w: 118,
          x: 351, y: 560,
          label: "value 激活",
          tex: R`v_t`, texShape: R`[B,\,T,\,H,\,d_v]`,
          fallback: "v_t", shapeFallback: "[B,T,H,dv]",
          description: "delta 写入的新内容：S ← A_t S + βkv^⊤。",
          symbols: [
            { tex: R`v_t`, shape: R`[B,\,T,\,H,\,d_v]`, label: "Value", value: "H = 96, d_v = 128", note: "Linear→Conv→SiLU（无 L2）" }
          ]
        },
        {
          id: "alpha", kind: "activation", tone: "control", w: 118,
          x: 501, y: 560,
          label: "逐通道 decay 门",
          tex: R`\alpha_t`, texShape: R`[B,\,T,\,H,\,d_k]`,
          fallback: "α_t", shapeFallback: "[B,T,H,dk]",
          description: "每个 head 的 128 个 key channels 独立衰减；乘积形成可学习距离衰减。",
          symbols: [
            { tex: R`\alpha_t`, shape: R`[B,\,T,\,H,\,d_k]`, label: "Channel decay", value: "128 channels / head / token", note: "位置感来自有序 decay，全程 NoPE" }
          ]
        },
        {
          id: "beta", kind: "activation", tone: "control", w: 118,
          x: 651, y: 560,
          label: "delta 写入率",
          tex: R`\beta_t`, texShape: R`[B,\,T,\,H]`,
          fallback: "β_t", shapeFallback: "[B,T,H]",
          description: "标量写入率：在旧关联与新值之间插值。",
          symbols: [
            { tex: R`\beta_t`, shape: R`[B,\,T,\,H]`, label: "Update rate", value: "scalar / head / token", note: "sigmoid delta-write strength" }
          ]
        },
        {
          id: "g", kind: "activation", tone: "control", w: 118,
          x: 801, y: 560,
          label: "输出门激活",
          tex: R`g_t`, texShape: R`[B,\,T,\,H,\,d_v]`,
          fallback: "g_t", shapeFallback: "[B,T,H,dv]",
          description: "输出 gate：绕过递推单元，在读出处调节内容。",
          symbols: [
            { tex: R`g_t`, shape: R`[B,\,T,\,H,\,d_v]`, label: "Output gate", value: "full-rank（K3）", note: "u = W^O[RMSNorm(o) ⊙ σ(g)]" }
          ]
        },
        {
          id: "core", kind: "activation", titleSize: 8, w: 260, h: 56,
          x: 205, y: 430,
          label: "Kimi Delta Attention 递推",
          tex: R`S_t=(I-\beta_tk_tk_t^{\top})\operatorname{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^{\top}`,
          texShape: R`o_t=S_t^{\top}(q_t/\sqrt{d_k})`,
          fallback: "S = (I−βkk^T)Diag(α)S + βkv^T", shapeFallback: "o = S^T q/√dk", shapeSize: 7.8,
          description: "受约束 DPLR：对角逐通道 decay 在前，绑定同一 key 的 rank-1 纠写在后；q 读出固定状态。",
          symbols: [
            { tex: R`A_t=(I-\beta_tk_tk_t^{\top})\operatorname{Diag}(\alpha_t)`, shape: R`[d_k,\,d_k]`, label: "Transition", value: "受约束 DPLR", note: "a ∥ k 且 b = k⊙α，换取稳定 chunk kernel" },
            { tex: R`o_t`, shape: R`[B,\,T,\,H,\,d_v]`, label: "Readout", value: "o = S^⊤(q/√d_k)", note: "每步 O(d_k d_v)" }
          ]
        },
        {
          id: "state", kind: "state",
          x: 576, y: 430,
          label: "固定递推状态",
          tex: R`S_t`, texShape: R`[B,\,H,\,d_k,\,d_v]`,
          fallback: "S_t", shapeFallback: "[B,H,dk,dv]",
          relates: ["core"],
          description: "每头固定 128×128 矩阵：历史被折叠进常数大小状态，不随 1M context 增长。",
          symbols: [
            { tex: R`S_t`, shape: R`[B,\,H,\,d_k,\,d_v]`, label: "Recurrent state", value: "96 × 128 × 128 = 1,572,864 elements/层/序列", note: "不含 dtype 与 ShortConv state" }
          ]
        },
        {
          id: "rms", kind: "operator", glyph: "pill", tone: "cyan", w: 96, h: 28, titleSize: 8.2,
          x: 287, y: 362,
          label: "读出 RMSNorm",
          tex: R`\operatorname{RMSNorm}`, fallback: "RMSNorm",
          description: "对每头读出做 RMSNorm；取代线性注意力的正分母归一化。",
          symbols: [
            { tex: R`\operatorname{RMSNorm}(o_t)`, shape: R`[B,\,T,\,H,\,d_v]`, label: "Readout normalization", value: "head-wise", note: "允许有符号特征的稳定读出" }
          ]
        },
        {
          id: "outgate", kind: "operator", tone: "control",
          x: 318, y: 293,
          label: "输出门逐元素乘",
          tex: R`\odot`, fallback: "⊙",
          description: "RMSNorm(o) 与 σ(g) 逐元素相乘后进入输出投影。",
          symbols: [
            { tex: R`\operatorname{RMSNorm}(o_t)\odot\sigma(g_t)`, shape: R`[B,\,T,\,H,\,d_v]`, label: "Gated readout", value: "逐元素", note: "数据依赖地调节写回内容" }
          ]
        },
        {
          id: "wo", kind: "weight", glyph: "up",
          x: 279, y: 210,
          label: "输出投影",
          tex: R`W^{O}`, texShape: R`[H d_v,\,d]`,
          fallback: "W^O", shapeFallback: "[H·dv,d]",
          relates: ["u"],
          description: "把 96 × 128 的门控读出写回残差流。",
          symbols: [
            { tex: R`W^{O}`, shape: R`[H d_v,\,d_{\mathrm{model}}]`, label: "Output projection", value: "(96 × 128) × 7168", note: "u = W^O[RMSNorm(o) ⊙ σ(g)]" }
          ]
        },
        {
          id: "u", kind: "activation", tone: "gather",
          x: 271, y: 130,
          label: "层输出",
          tex: R`u_t`, texShape: R`[B,\,T,\,d_{\mathrm{model}}]`,
          fallback: "u_t", shapeFallback: "[B,T,d_model]",
          description: "KDA 层输出；按 3:1 主节奏与 Gated MLA 层级交错。",
          symbols: [
            { tex: R`u_t`, shape: R`[B,\,T,\,d_{\mathrm{model}}]`, label: "Layer output", value: "d_model = 7168", note: "93 层骨干中的 69 个 KDA 层" }
          ]
        },
        {
          id: "gmla", kind: "block", titleSize: 9.4,
          x: 656, y: 130,
          label: "Gated MLA 层级调度",
          tex: R`\text{Gated MLA}\times24`, texShape: R`23\times(3\,\text{KDA}{+}1\,\text{GMLA})+\text{GMLA}`,
          fallback: "Gated MLA ×24", shapeFallback: "23×(3 KDA+1 GMLA)+GMLA", shapeSize: 7.4,
          relates: ["u"],
          description: "周期性全局回看层：NoPE + 数据依赖输出 gate，保留随 token 增长的 latent cache；层级交错而非混头。",
          symbols: [
            { tex: R`\text{GMLA}`, shape: R`[24\ \text{layers},\,B,\,T,\,d]`, label: "Global layers", value: "69 KDA + 24 Gated MLA = 93 层", note: "官方未公开细分 q/kv latent rank，不补写" },
            { tex: R`T_{\max}`, shape: R`[B,\,1{,}048{,}576]`, label: "Context length", value: "1M tokens", note: "GMLA cache 仍随 token 数增长；KDA 状态不变" }
          ]
        }
      ],
      edges: [
        { from: "x", to: "wq", tone: "compute", bend: 882 },
        { from: "x", to: "wk", tone: "compute", bend: 874 },
        { from: "x", to: "wv", tone: "compute", bend: 866 },
        { from: "x", to: "walpha", tone: "control", bend: 866 },
        { from: "x", to: "wbeta", tone: "control", bend: 874 },
        { from: "x", to: "wg", tone: "control", bend: 882 },
        { from: "wq", to: "convq", tone: "weight" },
        { from: "wk", to: "convk", tone: "weight" },
        { from: "wv", to: "convv", tone: "weight" },
        { from: "convq", to: "actq", tone: "compute" },
        { from: "convk", to: "actk", tone: "compute" },
        { from: "convv", to: "actv", tone: "compute" },
        { from: "actq", to: "l2q", tone: "compute" },
        { from: "actk", to: "l2k", tone: "compute" },
        { from: "actv", to: "v", tone: "compute" },
        { from: "l2q", to: "q", tone: "cyan" },
        { from: "l2k", to: "k", tone: "cyan" },
        { from: "walpha", to: "opalpha", tone: "control" },
        { from: "opalpha", to: "alpha", tone: "control" },
        { from: "wbeta", to: "opbeta", tone: "control" },
        { from: "opbeta", to: "beta", tone: "control" },
        { from: "wg", to: "opg", tone: "control" },
        { from: "opg", to: "g", tone: "control" },
        { from: "q", to: "core", tone: "compute", toAt: 0.1, bend: 518 },
        { from: "k", to: "core", tone: "compute", toAt: 0.3, bend: 526 },
        { from: "v", to: "core", tone: "compute", toAt: 0.5, bend: 518 },
        { from: "alpha", to: "core", tone: "control", toAt: 0.7, bend: 526 },
        { from: "beta", to: "core", tone: "control", toAt: 0.9, bend: 534 },
        { from: "core", to: "state", tone: "state", label: "写入", lx: 520, ly: 446 },
        { from: "state", to: "state", tone: "state", label: R`S_{t-1}\ \text{递推}`, lw: 96 },
        { from: "core", to: "rms", tone: "gather" },
        { from: "rms", to: "outgate", tone: "gather" },
        { from: "g", to: "outgate", tone: "control", fromSide: "top", toSide: "right", label: R`\odot\,\sigma(g_t)`, lx: 620, ly: 298 },
        { from: "outgate", to: "wo", tone: "gather" },
        { from: "wo", to: "u", tone: "gather" },
        { from: "u", to: "gmla", tone: "orange", dashed: true, toAt: 23 / 52, label: "3 KDA : 1 GMLA 层级交错", lx: 528, ly: 140 }
      ]
    }
  };

  window.ATTENTION_CHAPTERS.forEach(function (chapter) {
    var enhancement = chapterEnhancements[chapter.id];
    if (!enhancement) {
      throw new Error("Missing chapter enhancement for " + chapter.id);
    }

    chapter.positionEncoding = enhancement.positionEncoding;
    chapter.attentionConfig = enhancement.attentionConfig;
    if (attentionConfigMaps[chapter.id]) {
      chapter.attentionConfig.key = chapter.id;
      chapter.attentionConfig.title = chapter.attentionConfig.model + " 模型结构参数图";
      chapter.attentionConfig.description =
        "选择任一模块，查看对应数学符号、张量 shape 与代表模型参数。";
      chapter.attentionConfig.modules = attentionConfigMaps[chapter.id].modules;
      chapter.attentionConfig.edges = attentionConfigMaps[chapter.id].edges;
      if (attentionConfigMaps[chapter.id].groups) {
        chapter.attentionConfig.groups = attentionConfigMaps[chapter.id].groups;
      }
    }

    enhancement.derivations.forEach(function (derivation) {
      chapter.derivations.push(derivation);
    });
    chapter.derivations.forEach(function (derivation) {
      if (!derivation.source) {
        derivation.source = enhancement.derivationSourceFallback;
      }
    });

    chapter.exercises.forEach(function (exercise, index) {
      var defaults = enhancement.existingExerciseMeta[index] || {
        kind: "concept",
        level: "foundation"
      };
      if (!exercise.kind) {
        exercise.kind = defaults.kind;
      }
      if (!exercise.level) {
        exercise.level = defaults.level;
      }
    });
    enhancement.exercises.forEach(function (exercise) {
      chapter.exercises.push(exercise);
    });

    if (chapter.derivations.length < 4 || chapter.derivations.length > 6) {
      throw new Error("Chapter " + chapter.id + " must have 4–6 derivations");
    }
    if (chapter.exercises.length !== 6) {
      throw new Error("Chapter " + chapter.id + " must have exactly 6 exercises");
    }
  });
})();
