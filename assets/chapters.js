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
      deck: "先建立一把统一的“成本尺”。MHA 的表达力来自每个查询头拥有独立的 K/V 子空间，但自回归解码时也必须为每层、每个历史 token 保存全部 K/V。",
      takeaway: "MHA 的关键不是“有很多头”，而是每个头都拥有一套独立的读地址 K 与读内容 V；这同时给出高容量与最大的 KV cache。",
      motivation: [
        "循环网络必须按时间步串行传播状态。MHA 让每个 token 在一层内直接读取任意历史位置，训练时可对整个序列并行。",
        "不同头使用不同投影子空间：一个头可学习局部搭配，另一个头可学习指代或远程依赖；最终经拼接和输出投影写回残差流。",
        "它后来成为所有 KV 共享、低秩压缩、稀疏选择和线性状态方法的参照系。理解后续架构，本质上是在回答：MHA 的哪部分可以共享、压缩或省略？"
      ],
      constraints: [
        { label: "训练算力", title: "注意力图是二次的", body: "长度为 L 时要形成 H 个 L×L 分数图，核心算量约为 O(L²d)。" },
        { label: "推理存储", title: "KV 随上下文线性增长", body: "每层每 token 缓存 H 组 K 和 V；长上下文与大 batch 会迅速吃满显存。" },
        { label: "解码带宽", title: "常常不是算不动，而是搬不快", body: "每生成一个 token，都要从 HBM 读取历史 KV；小 batch 解码常受内存带宽限制。" }
      ],
      intuitions: [
        { label: "Query", title: "我正在找什么？", body: "每个头产生一个不同的问题。" },
        { label: "Key", title: "我能被怎样索引？", body: "历史 token 为每个头提供独立地址。" },
        { label: "Value", title: "找到后读出什么？", body: "匹配权重对内容向量做加权汇总。" }
      ],
      diagram: { type: "heads", mode: "mha", caption: "MHA：每个 Q 头都配有独立 K/V 头，表达容量高，缓存也最大。" },
      derivations: [
        {
          title: "从投影到加权读取",
          body: R`对第 \(h\) 个头，令
            \[
            Q_h=XW_h^Q,\quad K_h=XW_h^K,\quad V_h=XW_h^V,
            \]
            \[
            O_h=\operatorname{softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right)V_h,\qquad
            Y=\operatorname{Concat}(O_1,\ldots,O_H)W^O.
            \]
            缩放因子 \(\sqrt{d_h}\) 抑制点积方差；因果掩码 \(M\) 禁止读取未来。`
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
      warning: "“头数减半”不等于“KV cache 减半”，除非 KV 头数也减半。很多模型的 Q 头数与 KV 头数并不相同。",
      exercises: [
        {
          q: R`某模型有 \(N=32\) 层、\(H=32\)、\(d_h=128\)，以 BF16 缓存一条 \(L=4096\) 的序列。忽略 batch 维，MHA KV cache 多大？`,
          hint: "代入 2NLHd_hb，BF16 的 b=2。",
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
      deck: "MQA 没有改变查询头的数量，也没有近似 softmax；它只让全部 Q 头共享一套 K/V，从源头减少自回归解码必须读取的数据。",
      takeaway: "把“提出多少个问题”和“保存多少份历史索引”解耦：保留多 Q 头，令 \(H_{kv}=1\)。",
      motivation: [
        "增量解码一次只处理一个新 token，矩阵乘法很窄，GPU 计算单元难以吃满；反而从 HBM 反复加载历史 K/V 成为主要瓶颈。",
        "Shazeer 的核心观察是：查询头需要多样性，但历史 token 不一定要为每个查询头保存独立地址与内容。",
        "因此 MQA 保留 H 个 Q 投影与 H 个输出通道，只把 K/V 投影压成一个共享头。它是结构性共享，不是量化或低秩近似。"
      ],
      constraints: [
        { label: "目标", title: "优化解码带宽", body: "训练阶段仍需计算完整的多 Q 头注意力；主要收益发生在增量推理。" },
        { label: "存储", title: "KV 缩小约 H 倍", body: "相同头维下，Hkv 从 H 变为 1；实际收益还取决于布局、量化和 kernel。" },
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
          body: R`令 \(Q_h=XW_h^Q\)，但只计算
            \[
            K=XW^K,\qquad V=XW^V.
            \]
            第 \(h\) 个输出仍为
            \[
            O_h=\operatorname{softmax}\!\left(\frac{Q_hK^\top}{\sqrt{d_h}}+M\right)V.
            \]
            因此 softmax 仍精确，查询仍多头；变化只发生在 K/V 的头维。`
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
      warning: "MQA 的论文结论是“显著更快且只有轻微质量损失”的特定实验结果，不是所有模型、任务和训练配方上的普遍定理。",
      exercises: [
        {
          q: "沿用上一章 32 层、32 头、头维 128、4096 长度、BF16 的例子，MQA KV cache 多大？",
          hint: "把 Hkv 从 32 改为 1。",
          answer: "约 64 MiB；相对 MHA 理论缩小 32 倍。"
        },
        {
          q: "MQA 是否把注意力计算从二次复杂度变成线性复杂度？",
          hint: "考察长度 L 对 QKᵀ 的影响。",
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
      deck: "GQA 把 MHA 与 MQA 放在同一条连续轴上：每组 Q 头共享一个 K/V 头，用 Hkv 直接控制质量—缓存—带宽的折中。",
      takeaway: R`GQA 不是第三种完全不同的算子，而是 \(1\le H_{kv}\le H_q\) 的统一参数化：MQA 是 1，MHA 是 \(H_q\)。`,
      motivation: [
        "MQA 的缓存最小，但单一 K/V 头可能成为表达瓶颈；MHA 表达充足，却为每个 Q 头重复保存历史。",
        "GQA 把查询头分为 G 组，每组共享一套 K/V。模型设计者可以根据目标硬件和质量预算选择中间点。",
        "原论文还给出从 MHA checkpoint 升级到 GQA 的办法：组内 K/V 投影做均值池化，再用约原预训练算力 5% 的继续训练恢复能力。5% 是该论文配方，不是固定标准。"
      ],
      constraints: [
        { label: "结构约束", title: "通常要求整除", body: "常见实现要求 Hq 能被 Hkv 整除，每个 KV 头服务 r=Hq/Hkv 个 Q 头。" },
        { label: "硬件约束", title: "布局与 kernel 要匹配", body: "理论缓存下降不保证 kernel 自动高效；广播和并行切分会影响实际吞吐。" },
        { label: "迁移成本", title: "可从 MHA uptrain", body: "均值池化只是初始化，仍需继续训练适应共享后的表示。" }
      ],
      intuitions: [
        { label: "类比", title: "小组共用资料员", body: "每位成员有问题；每组有一位资料员维护索引。" },
        { label: "旋钮", title: "Hkv 控制折中", body: "越大越接近 MHA，越小越接近 MQA。" },
        { label: "分组", title: "共享发生在组内", body: "组间仍可学习不同 K/V 子空间。" }
      ],
      diagram: { type: "heads", mode: "gqa", caption: "GQA：示意 8 个 Q 头按 2 个一组，共享 4 个 K/V 头。" },
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
      warning: "“GQA 速度与 MQA 相当、质量接近 MHA”来自特定 uptraining 实验。服务框架、batch、序列长度和并行策略改变后，最优 G 不一定相同。",
      exercises: [
        {
          q: "Hq=48、Hkv=8 时，每个 KV 头服务多少个 Q 头？相对 MHA 的 KV 缩减倍数是多少？",
          hint: "都等于 Hq/Hkv。",
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
      zhTitle: "多头潜变量注意力：压缩每个 token 的宽度",
      year: "2024",
      category: "dense",
      difficulty: "高阶",
      report: "DeepSeek-V2 Technical Report",
      deck: "MLA 不再让若干 Q 头直接共享一套完整 K/V，而是先把每个 token 压到低维 latent，缓存 latent；需要注意力时再为各头上投影恢复 K/V。",
      takeaway: "GQA 在“头数”上共享，MLA 在“低秩潜空间”里共享。只缓存联合压缩 latent，保留多头上投影的表达力。",
      motivation: [
        "MQA/GQA 通过减少 KV 头数降缓存，但 K/V 表示容量也随之减少。DeepSeek-V2 希望保留多头差异，同时继续降低 KV cache。",
        "MLA 对 K 与 V 做联合低秩压缩：每个 token 只保存一个低维向量 cKV；每个头的 K/V 由不同上投影从 cKV 重建。",
        "RoPE 会阻碍把上投影吸收到查询侧，因此 MLA 把位置相关的 RoPE 子空间与可低秩吸收的内容子空间分开，这就是 decoupled RoPE。"
      ],
      constraints: [
        { label: "缓存", title: "宽度从 Hdh 变成 dc", body: "每 token 主要缓存 cKV 与较小的 RoPE key；收益取决于 dc 和 dR。" },
        { label: "算子", title: "需要吸收或重建", body: "朴素地显式恢复各头 K/V 会增加算力；高效推理通常把上投影吸收到 Q/输出侧。" },
        { label: "位置编码", title: "RoPE 必须解耦", body: "位置依赖旋转与低秩投影一般不可交换，需要单独保存旋转 key 分量。" }
      ],
      intuitions: [
        { label: "类比", title: "保存源文件，不存多份导出", body: "latent 是紧凑源文件，各头按需用不同模板展开。" },
        { label: "低秩", title: "共享生成基底", body: "K/V 头都来自同一个 dc 维潜空间。" },
        { label: "RoPE", title: "位置水印另存", body: "内容可压缩吸收，位置旋转通道单独处理。" }
      ],
      diagram: { type: "latent", caption: "MLA：每个 token 只缓存低维 cKV 与解耦的 RoPE key，再供多头读取。" },
      derivations: [
        {
          title: "KV 联合低秩压缩",
          body: R`对 token 隐状态 \(h_t\)，先下投影
            \[
            c_t^{KV}=W^{DKV}h_t,\qquad c_t^{KV}\in\mathbb{R}^{d_c}.
            \]
            各头内容 key/value 由上投影得到
            \[
            k_{t,i}^{C}=W_i^{UK}c_t^{KV},\qquad
            v_{t,i}=W_i^{UV}c_t^{KV}.
            \]
            推理时缓存的是 \(c_t^{KV}\)，而不是全部 \(k_{t,i}^C,v_{t,i}\)。`
        },
        {
          title: "解耦 RoPE 与缓存通式",
          body: R`把 key 分为内容与位置两部分：
            \[
            k_{t,i}=[k_{t,i}^{C};k_t^{R}],\qquad
            q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],
            \]
            其中 \(q^R,k^R\) 应用 RoPE。于是每层每 token 的主要缓存宽度约为
            \[
            d_c+d_R
            \]
            而不是 MHA 的 \(2H_qd_h\)。精确布局以具体模型实现为准。`
        },
        {
          title: "为什么能吸收到查询侧",
          body: R`内容分数中
            \[
            q_i^\top k_i^C=q_i^\top W_i^{UK}c^{KV}
            =\big((W_i^{UK})^\top q_i\big)^\top c^{KV}.
            \]
            因此可预先把 \(W_i^{UK}\) 合并进查询投影，让注意力直接读取 latent；RoPE 部分因位置旋转而单独保留。`
        }
      ],
      warning: "DeepSeek-V2 报告中的“KV cache 减少 93.3%”同时包含 MLA、与 DeepSeek-67B 不同的层配置及平均 6-bit cache 量化；“吞吐提升 5.76×”也是整套系统相对指定基线的自报结果，二者都不是 MLA 算子的固定倍率。",
      exercises: [
        {
          q: R`若 MHA 每 token 缓存 \(2H d_h=2\times32\times128\) 个元素，而 MLA 缓存 \(d_c+d_R=512+64\) 个元素，理论元素数比是多少？`,
          hint: "用 8192/576。",
          answer: "约 14.22 倍。该比值未计布局、量化和临时工作区。"
        },
        {
          q: "为什么不能简单把完整 RoPE key 也都吸收到查询投影？",
          hint: "位置旋转矩阵随 token 位置变化。",
          answer: R`因为 \(R_tW\) 一般不等于固定矩阵 \(WR_t\)，位置相关旋转破坏固定的低秩吸收。MLA 因而把 RoPE 通道解耦并单独缓存。`
        }
      ],
      sources: [
        { label: "DeepSeek-AI (2024), DeepSeek-V2 Technical Report", url: "https://arxiv.org/abs/2405.04434" },
        { label: "DeepSeek-V2 official repository", url: "https://github.com/deepseek-ai/DeepSeek-V2" },
        { label: "DeepSeek-AI (2024), DeepSeek-V3 Technical Report", url: "https://arxiv.org/abs/2412.19437" }
      ]
    },
    {
      id: "dsa",
      order: 4,
      title: "DSA",
      fullTitle: "DeepSeek Sparse Attention",
      zhTitle: "DeepSeek 稀疏注意力：先索引，再精读",
      year: "2025",
      category: "sparse",
      difficulty: "高阶",
      report: "DeepSeek-V3.2-Exp",
      deck: "DSA 把昂贵的 core attention 拆成两阶段：低成本 Lightning Indexer 为每个 query 找 top-k 历史 token，随后只在这些 token 上运行高维 MLA 注意力。",
      takeaway: "DSA 的核心不是预先固定窗口，而是内容驱动的 learned top-k：便宜地找候选，昂贵地精确读取。",
      motivation: [
        "MLA 解决了每个 token 缓存过宽的问题，但 dense attention 仍要让每个 query 与所有历史位置做高维交互；上下文极长时，算量仍随 L² 增长。",
        "DSA 引入 Lightning Indexer：用低维、低成本路径估计相关性，为每个 query 选择 k 个历史位置；core attention 只对选中 token 计算。",
        "DeepSeek-V3.2-Exp 以接近 V3.1-Terminus 的训练配置验证稀疏化，官方称输出质量基本持平，并开源训练/推理 kernel。"
      ],
      constraints: [
        { label: "选择预算", title: "k 决定精度与成本", body: "k 太小会漏掉关键 token；太大则接近 dense attention。" },
        { label: "索引成本", title: "索引器本身也要扫描", body: "若对所有历史位置打分，索引路径仍含 L² 项，只是维度和精度更低。" },
        { label: "训练系统", title: "离散 top-k 难优化", body: "需要索引损失、稳定训练与专用稀疏 kernel；纸面稀疏不自动等于硬件高效。" }
      ],
      intuitions: [
        { label: "阶段 1", title: "目录检索", body: "Indexer 只看短摘要，快速列出候选页码。" },
        { label: "阶段 2", title: "打开原文", body: "Core attention 在候选 token 上做完整高维读取。" },
        { label: "不同于窗口", title: "远处也能被选中", body: "选择由内容相关性决定，不只依赖距离。" }
      ],
      diagram: { type: "sparse", caption: "DSA：低成本索引器扫描历史，top-k 路由到高维 core attention。" },
      derivations: [
        {
          title: "Lightning Indexer 的正式评分",
          body: R`设 Indexer 有 \(H^I\) 个 query 头，共享一条低维 key。官方评分为
            \[
            I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}
            \operatorname{ReLU}\!\left((q^I_{t,j})^\top k^I_s\right),
            \qquad
            \mathcal I_t=\operatorname{TopK}_s(I_{t,s},k).
            \]
            高维核心注意力只在集合 \(\mathcal I_t\) 上计算
            \[
            o_t=\sum_{j\in\mathcal I_t}
            \operatorname{softmax}_{j\in\mathcal I_t}
            \left(\frac{q_t^\top k_j}{\sqrt d}\right)v_j.
            \]
            top-k 后的 softmax 对候选集合重新归一化。V3.2 的公开配置为
            \(H^I=64,d^I=128,k=2048\)，Indexer QK 路径使用 FP8。`
        },
        {
          title: "复杂度要分两条路径看",
          body: R`若 indexer 维度为 \(d_I\)，core 头维为 \(d\)，则粗略有
            \[
            C_{\mathrm{index}}\sim O(L^2d_I),\qquad
            C_{\mathrm{core}}\sim O(Lkd).
            \]
            DSA 的价值来自 \(d_I\ll d\)、低精度索引和 \(k\ll L\)。它不应被粗暴写成“所有部分都严格 O(Lk)”。`
        },
        {
          title: "Indexer 用独立对齐目标训练",
          body: R`Dense warm-up 阶段从主注意力聚合目标分布 \(p_{t,:}\)，训练
            \[
            \mathcal L^I=\sum_t D_{\mathrm{KL}}\!\left(
            p_{t,:}\,\|\,\operatorname{Softmax}(I_{t,:})\right).
            \]
            稀疏阶段改为在选中集合上对齐。官方设计将 Indexer 输入从主模型计算图 detach：Indexer 由
            \(\mathcal L^I\) 优化，语言模型损失不穿过离散 top-k 直接反传。`
        }
      ],
      warning: "内容稀疏注意力存在检索召回率风险：被 indexer 漏掉的 token 不会进入 core attention，后者再精确也无法补救。",
      exercises: [
        {
          q: R`当 \(L=131072,k=2048\) 时，core attention 的位置对数量相对 dense 减少多少倍？`,
          hint: "比较 L² 与 Lk。",
          answer: R`理想比值为 \(L/k=64\) 倍。这里只比较 core 位置对，不包括 indexer、局部路径和系统开销。`
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
      order: 5,
      title: "CSA",
      fullTitle: "Compressed Sparse Attention",
      zhTitle: "压缩稀疏注意力：先缩短历史，再检索摘要",
      year: "2026",
      category: "sparse",
      difficulty: "前沿",
      report: "DeepSeek-V4 Technical Report",
      deck: "CSA 先把相邻 token 学习式压成更短的 KV 序列，再让 Lightning Indexer 从压缩条目中选 top-k；局部滑窗保留近期未压缩细节。",
      takeaway: "MLA 压“每条记录有多宽”，CSA 压“历史有多少条”并稀疏读取：先把 L 变成 L/m，再从中选 k。",
      motivation: [
        "到百万 token，上下文条目数本身成为瓶颈：即使每条 KV 已被 MLA 压窄，逐 token 存储和检索仍很昂贵。",
        "CSA 每 m 个 token 压成一个条目，再用 DSA Lightning Indexer 选 top-k 压缩条目，同时攻击 cache 长度与 core attention 读取量。",
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
      diagram: { type: "compressed", mode: "csa", caption: "CSA：两路重叠学习式压缩把序列缩短 m 倍，再做 top-k；右侧滑窗保留原始近期 token。" },
      derivations: [
        {
          title: "沿序列维做重叠学习式压缩",
          body: R`CSA 产生两路 KV 流 \(C^a,C^b\) 与逐维门控 \(Z^a,Z^b\)。对第 \(i\) 个输出，把当前 a 块与前一 b 块的 \(2m\) 个位置逐维 softmax 加权：
            \[
            C_i^{Comp}=
            \sum_{j=mi}^{m(i+1)-1}S_j^a\odot C_j^a+
            \sum_{j=m(i-1)}^{mi-1}S_j^b\odot C_j^b.
            \]
            相邻输出有重叠，但步长仍为 \(m\)，所以历史长度从 \(L\) 变为约 \(L/m\)，不是 \(L/(2m)\)。`
        },
        {
          title: "Indexer、core 与 cache 要分别计算",
          body: R`CSA 对每个 query 扫描 \(L/m\) 个压缩候选，再选择 \(k\) 个压缩条目：
            \[
            C_{\mathrm{index}}\sim O(L^2d_I/m),\qquad
            C_{\mathrm{CSA,core}}\sim O(Lkd),\qquad
            \mathrm{Cache}_{CSA}\sim O((L/m)d_c).
            \]
            还需加窗口大小 \(w\) 的局部注意力 \(O(Lwd)\)。固定 \(m\) 时 Indexer 渐近仍是平方项，只是候选轴更短且使用低维低精度。`
        },
        {
          title: "报告中的正式配置",
          body: R`DeepSeek-V4 使用 \(m=4,w=128\)，Indexer 为 64 头、头维 128，QK 路径使用 FP4。V4-Flash 的 top-k 为 512，V4-Pro 为 1024。这些是模型配置，不是 CSA 定义中的固定常数。`
        }
      ],
      warning: "DeepSeek-V4 是 2026 年预览技术报告。其百万上下文效率数字、FP4 indexer 和召回率均应标注为官方自报，并等待更广泛的独立复现。",
      exercises: [
        {
          q: R`L=1,000,000，CSA 的 m=4、k=512。压缩池有多少条目？每个 query 的 core 只读其中多少比例？`,
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
      order: 6,
      title: "HCA",
      fullTitle: "Heavily Compressed Attention",
      zhTitle: "重压缩注意力：用粗粒度摘要换全局稠密视野",
      year: "2026",
      category: "sparse",
      difficulty: "前沿",
      report: "DeepSeek-V4 Technical Report §2.3.2",
      deck: "HCA 把每 m′ 个 token 压成一个条目，压缩率远高于 CSA；由于历史已非常短，它取消 Indexer 与 top-k，直接稠密读取全部压缩条目。",
      takeaway: "HCA 不做稀疏选择：把 key 轴从 L 重压到 L/m′，再运行 dense attention。它避免 top-k 漏召回，却承担更强的压缩损失。",
      motivation: [
        "CSA 保留较细的摘要并依赖 learned top-k，适合内容检索，但 Indexer 仍要扫描候选且可能漏掉相关块。",
        "HCA 选择相反的极端：把每个大块压成一个 KV 条目，使全局摘要短到无需检索器即可全部读取。",
        "DeepSeek-V4 交错使用 CSA 与 HCA：CSA 提供较高分辨率选择，HCA 提供便宜、稳定且无 top-k 漏召回的全局概览。"
      ],
      constraints: [
        { label: "压缩损失", title: "128 个 token → 1 条摘要", body: "极强的信息瓶颈难以保留符号、引用和 token 级精确细节。" },
        { label: "计算", title: "压缩后仍是 dense", body: "query 轴未压缩，prefill 仍含 O(L²/m′) 项，并非严格线性。" },
        { label: "局部性", title: "必须并联滑窗", body: "未闭合块与近期细节由 w=128 的原始 KV 窗口补足。" }
      ],
      intuitions: [
        { label: "Compress", title: "把整卷做成目录", body: "每 128 个 token 合成一个粗粒度全局条目。" },
        { label: "Read", title: "完整阅读短目录", body: "没有 Indexer，也没有 top-k 选择遗漏。" },
        { label: "Trade-off", title: "不漏目录，但目录会丢细节", body: "选择误差消失，压缩误差变成主要风险。" }
      ],
      diagram: { type: "compressed", mode: "hca", caption: "HCA：重压缩后的 key 轴足够短，可以 dense 读取全部摘要；局部滑窗保留近期原始 token。" },
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
            压缩历史长度为 \(\lfloor L/m'\rfloor\)。与 CSA 不同，HCA 没有两路重叠压缩。`
        },
        {
          title: "dense over compressed history",
          body: R`HCA 对全部压缩条目与局部窗口做核心注意力：
            \[
            C_{\mathrm{HCA}}\sim O\!\left(L(L/m'+w)d\right),\qquad
            \mathrm{Cache}_{HCA}\sim O((L/m')d_c+wd_c).
            \]
            DeepSeek-V4 采用 \(m'=128,w=128\)。常数约降 128 倍，但固定 \(m'\) 时 prefill 的渐近阶仍是平方。`
        }
      ],
      warning: "HCA 不是稀疏注意力，也不是 mHC 超连接。它对压缩序列做 dense attention；没有 top-k 漏召回不等于没有信息损失。",
      exercises: [
        {
          q: R`当 \(L=1{,}048{,}576,m'=128,w=128\) 时，每个 HCA query 最多读取多少个全局摘要与局部 token？`,
          hint: "先算 L/m′。",
          answer: "全局摘要 8192 条，再加 128 个局部 token，共 8320 个输入条目。"
        },
        {
          q: R`为什么 HCA 不能称为严格 \(O(L)\) attention？`,
          hint: "m′ 是固定常数，query 数仍为 L。",
          answer: R`因为 key 数为 \(L/m'\)，所有 L 个 query 都 dense 读取它们，prefill 为 \(O(L^2/m')\)；固定 \(m'\) 不改变平方渐近阶。`
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
      order: 7,
      title: "Linear Attention",
      fullTitle: "Kernelized Linear Attention",
      zhTitle: "线性注意力：把历史折叠进固定状态",
      year: "2020",
      category: "linear",
      difficulty: "高阶",
      report: "Transformers are RNNs",
      deck: "线性注意力通过核特征映射与矩阵乘法结合律，不再显式构造 L×L 注意力图；因果推理可写成固定大小状态的递推。",
      takeaway: "先算 QKᵀ 再乘 V 是二次的；若相似度可分解为 φ(q)ᵀφ(k)，就能先累计 KV，再让 q 读取。",
      motivation: [
        "softmax(QKᵀ)V 的计算顺序会显式形成长度平方的分数矩阵，极长序列训练和推理代价高。",
        "若相似度写成核内积 φ(q)ᵀφ(k)，利用结合律可先计算 Σφ(k)vᵀ，序列维被汇总到固定矩阵状态。",
        "因果场景中该汇总可递推更新，因此线性 Transformer 同时具有并行训练形式与 RNN 式常数状态推理形式。"
      ],
      constraints: [
        { label: "核约束", title: "不再是精确 softmax", body: "必须选择可分解特征映射 φ；核的归纳偏置决定模型可表达的相似性。" },
        { label: "状态容量", title: "固定状态会发生干扰", body: "所有历史写入同一个矩阵，精确复制和多键检索常弱于 full attention。" },
        { label: "硬件现实", title: "O(L) 不等于一定更快", body: "短序列上，成熟的 FlashAttention 可能因更高算术强度而更快。" }
      ],
      intuitions: [
        { label: "Dense", title: "每次翻全部档案", body: "query 与每条历史逐一比较。" },
        { label: "Linear", title: "维护统计台账", body: "历史到来时写入固定状态，query 直接查台账。" },
        { label: "代价", title: "位置可解释性下降", body: "通常无法还原一个显式 L×L 注意力图。" }
      ],
      diagram: { type: "linear", caption: "线性注意力：每个 (k,v) 写入状态 S,z；q 从固定状态读取，不保存完整历史表。" },
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
      warning: "“Linear Attention”是一个方法族，不是单一公式。不同论文可能用正特征核、随机特征、无归一化递推或门控状态；复杂度相同不代表性质相同。",
      exercises: [
        {
          q: R`取标量 \(\phi(q)=q,\phi(k)=k\)，依次写入 \((k_1,v_1)=(1,2)\)、\((k_2,v_2)=(3,4)\)。忽略归一化，q=2 时输出多少？`,
          hint: "S=Σkv。",
          answer: R`\(S=1\times2+3\times4=14\)，输出 \(qS=28\)。`
        },
        {
          q: R`特征维 r=64、value 维 128，单头状态 S 有多少元素？若上下文从 4K 变到 1M，S 是否变大？`,
          hint: "S 的形状是 r×dv。",
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
      order: 8,
      title: "DeltaNet",
      fullTitle: "DeltaNet & Gated DeltaNet",
      zhTitle: "Delta 更新：让固定状态学会定点改写",
      year: "2024–25",
      category: "linear",
      difficulty: "前沿",
      report: "Gated Delta Networks · ICLR 2025",
      deck: "普通线性注意力只会不断叠加写入，旧值与新值容易冲突。Delta rule 先读出当前预测，再只写入残差；Gated DeltaNet 再加入可学习遗忘。",
      takeaway: "加法记忆是“追加”；Delta rule 是“按 key 修正”；gate 是“先清场再修正”。三者逐步减少固定状态中的干扰。",
      motivation: [
        "普通线性状态 S←S+kvᵀ 对相同或相近 key 反复写入时会累积冲突，无法像字典一样覆盖旧值。",
        "Delta rule 计算当前状态对 key 的预测 Sᵀk，用目标 value 与预测之差作为写入量，因此更新集中在尚未记住的信息。",
        "Gated DeltaNet 在 delta 更新前对整个状态乘可学习衰减 αt，兼具快速遗忘与精确键值修改；论文还给出并行 chunk 算法。"
      ],
      constraints: [
        { label: "状态", title: "仍是有限容量矩阵", body: "delta 改善冲突，但不能让固定大小状态拥有无限无损记忆。" },
        { label: "并行", title: "递推需转成 chunk 算法", body: "逐 token 写法直观但训练低效，需要 WY/扫描类变换使用 Tensor Cores。" },
        { label: "混合", title: "精确检索仍可能不足", body: "实际前沿模型常周期性插入 full/sliding attention 层补充显式读取。" }
      ],
      intuitions: [
        { label: "累加", title: "每次把答案叠上去", body: "相同 key 的多个值会互相污染。" },
        { label: "Delta", title: "只写纠错量", body: "先问记忆当前答什么，再写入目标与答案之差。" },
        { label: "Gate", title: "可学习橡皮擦", body: "αt 控制全局旧记忆保留多少。" }
      ],
      diagram: { type: "delta", caption: "Delta rule：读取旧预测，计算误差，再沿 key 方向定点修正；gate 控制遗忘。" },
      derivations: [
        {
          title: "Delta rule 是一步在线梯度下降",
          body: R`令状态 \(S\in\mathbb R^{d_k\times d_v}\)，希望 \(S^\top k_t\approx v_t\)。瞬时损失
            \[
            \ell_t(S)=\frac12\|S^\top k_t-v_t\|^2.
            \]
            对 S 做步长 \(\beta_t\) 的梯度下降：
            \[
            S_t=S_{t-1}+\beta_t k_t\big(v_t-S_{t-1}^\top k_t\big)^\top.
            \]
            展开后得到 rank-1 状态转移。`
        },
        {
          title: "Gated DeltaNet 的标量遗忘",
          body: R`在单位范数 key 的常用约定下，Gated DeltaNet 写为
            \[
            S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}
            +\beta_tk_tv_t^\top,
            \qquad o_t=S_t^\top q_t.
            \]
            \(\alpha_t\in(0,1)\) 是每头标量 gate；\(\beta_t\) 控制定点改写强度。`
        }
      ],
      warning: "DeltaNet 的不同文献采用不同矩阵朝向、归一化与 update/read 顺序。比较公式时先核对 S 的形状以及 k、v 是列向量还是行向量。",
      exercises: [
        {
          q: R`若 \(\|k\|=1,\beta=1\)，证明 delta 更新后 \(S_t^\top k=v\)。`,
          hint: R`把 \(S_t=S+\;k(v-S^\top k)^\top\) 左乘到 k 上。`,
          answer: R`\(S_t^\top k=S^\top k+(v-S^\top k)k^\top k=v\)。因此对该 key 完成一次精确覆盖。`
        },
        {
          q: R`若连续 100 步没有 delta 写入且 \(\alpha=0.99\)，旧状态幅度剩多少？`,
          hint: R`计算 \(0.99^{100}\)。`,
          answer: "约 0.366。gate 相当于可学习的记忆半衰期控制。"
        }
      ],
      sources: [
        { label: "Schlag et al. (2021), Linear Transformers Are Secretly Fast Weight Programmers", url: "https://arxiv.org/abs/2102.11174" },
        { label: "Yang, Kautz & Hatamizadeh (2025), Gated Delta Networks", url: "https://arxiv.org/abs/2412.06464" },
        { label: "NVlabs official GatedDeltaNet repository", url: "https://github.com/NVlabs/GatedDeltaNet" },
        { label: "Qwen3-Next official architecture note: 3:1 Gated DeltaNet + attention", url: "https://qwenlm.github.io/blog/qwen3-next/" }
      ]
    },
    {
      id: "kda",
      order: 9,
      title: "KDA",
      fullTitle: "Kimi Delta Attention",
      zhTitle: "Kimi Delta Attention：逐通道控制记忆",
      year: "2025",
      category: "hybrid",
      difficulty: "前沿",
      report: "Kimi Linear Technical Report",
      deck: "KDA 把 Gated DeltaNet 每头一个标量遗忘率升级为逐 key 通道的对角 gate，并把转移限制为高效的 diagonal-plus-rank-1 形式；Kimi Linear 再以 3:1 比例混合 KDA 与 MLA。",
      takeaway: "不同记忆通道需要不同时间尺度。KDA 用 Diag(αt) 做细粒度衰减，用 rank-1 delta 做内容定点改写，再用周期性 MLA 恢复无压缩全局读取。",
      motivation: [
        "Gated DeltaNet 的 αt 是每头标量，头内所有 key 通道共享同一遗忘速度；这限制了同时追踪短期句法与长期主题的能力。",
        "KDA 使用对角 gate Diag(αt)，每个 key 通道拥有独立衰减；随后使用 Householder 风格 rank-1 delta 变换修正键值关联。",
        "一般 DPLR 转移表达力强但 chunk 并行昂贵。KDA 约束低秩项与 key 绑定，减少高精度二级 chunk 与额外 matmul，兼顾表达力和硬件效率。"
      ],
      constraints: [
        { label: "数值", title: "细粒度累积易不稳定", body: "逐通道衰减在长 chunk 的乘除中会产生精度问题，需要专门 UT/WY 形式。" },
        { label: "算子", title: "收益依赖定制 kernel", body: "简单 Python 递推无法体现 KDA 的吞吐优势；官方开源 FLA kernel 与 vLLM 实现。" },
        { label: "容量", title: "仍需 global attention", body: "Kimi Linear 用 3 KDA : 1 MLA，而不是完全移除 full attention。" }
      ],
      intuitions: [
        { label: "Channel gate", title: "一排不同速度的沙漏", body: "每个通道选择自己的记忆半衰期。" },
        { label: "Delta", title: "同地址定点覆盖", body: "rank-1 更新减少相似 key 之间的污染。" },
        { label: "Hybrid", title: "三次压缩记忆，一次全局翻档", body: "大部分层便宜递推，周期性 MLA 做精确全局纠偏。" }
      ],
      diagram: { type: "kda", caption: "KDA 的 diagonal gate + rank-1 delta 转移，以及 Kimi Linear 的 3:1 KDA/MLA 混合。" },
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
          body: R`忽略 delta 项时，第 r 个 key 通道经历
            \[
            S_{t,r}\approx \alpha_{t,r}S_{t-1,r}.
            \]
            若 \(\alpha_{t,r}=\alpha_r\) 近似恒定，其半衰期为
            \[
            \tau_{1/2,r}=\frac{\log 0.5}{\log \alpha_r}.
            \]
            因而一个头内部可以并存多种时间尺度。`
        },
        {
          title: "混合层的缓存直觉",
          body: R`若每 4 层中 3 层 KDA 只保留固定状态，1 层 MLA 保留随 L 增长的 cache，则长上下文下，线性增长部分约只来自四分之一层。官方报告据此给出“KV cache 最多降低 75%”的自报结果；实际还包括状态与实现开销。`
        }
      ],
      warning: "Kimi Linear 报告中的“最高 6×/6.3× 解码吞吐、最多 75% KV cache 降低”取决于 1M 上下文、硬件、batch、kernel 和比较基线。网站将其视为官方测量，不泛化为固定承诺。",
      exercises: [
        {
          q: R`某通道 \(\alpha=0.999\)，近似半衰期是多少步？`,
          hint: R`用 \(\log(0.5)/\log(0.999)\)。`,
          answer: "约 693 步。若另一通道 α=0.9，半衰期约 6.58 步，说明同一头可并存长期与短期记忆。"
        },
        {
          q: R`比较 \(A_t=(I-\beta kk^\top)\operatorname{Diag}(\alpha)\) 与一般 DPLR \(D-ab^\top\)。KDA 约束了什么？`,
          hint: "观察 rank-1 项左右向量与 k、α 的关系。",
          answer: "KDA 的低秩项不是自由 a、b，而与同一个 key k 及对角 gate 绑定；表达空间更受限，但 chunk 算法更稳定、高效。"
        },
        {
          q: "为什么 Kimi Linear 的 3:1 不能理解成每个 block 内并行做 75% KDA、25% MLA？",
          hint: "报告说 layerwise hybrid。",
          answer: "它是层级交错：连续 3 个 KDA 层后有 1 个 MLA 层；不是同一层内按权重线性混合两种输出。"
        }
      ],
      sources: [
        { label: "Kimi Team (2025), Kimi Linear: An Expressive, Efficient Attention Architecture", url: "https://arxiv.org/abs/2510.26692" },
        { label: "MoonshotAI official Kimi-Linear repository", url: "https://github.com/MoonshotAI/Kimi-Linear" },
        { label: "Flash Linear Attention official KDA kernels", url: "https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda" }
      ]
    }
  ];
})();
