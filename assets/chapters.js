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
        { label: "训练算力", title: "注意力图是二次的", body: R`长度为 \(L\) 时要形成 \(H\) 个 \(L\times L\) 分数图，核心算量约为 \(O(L^2d)\)。` },
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
      deck: "MQA 没有改变查询头的数量，也没有近似 softmax；它只让全部 Q 头共享一套 K/V，从源头减少自回归解码必须读取的数据。",
      takeaway: R`把“提出多少个问题”和“保存多少份历史索引”解耦：保留多 Q 头，令 \(H_{kv}=1\)。`,
      motivation: [
        "增量解码一次只处理一个新 token，矩阵乘法很窄，GPU 计算单元难以吃满；反而从 HBM 反复加载历史 K/V 成为主要瓶颈。",
        "Shazeer 的核心观察是：查询头需要多样性，但历史 token 不一定要为每个查询头保存独立地址与内容。",
        "因此 MQA 保留 H 个 Q 投影与 H 个输出通道，只把 K/V 投影压成一个共享头。它是结构性共享，不是量化或低秩近似。"
      ],
      constraints: [
        { label: "目标", title: "优化解码带宽", body: "训练阶段仍需计算完整的多 Q 头注意力；主要收益发生在增量推理。" },
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
          hint: R`把 \(H_{kv}\) 从 32 改为 1。`,
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
      deck: R`GQA 把 MHA 与 MQA 放在同一条连续轴上：每组 Q 头共享一个 K/V 头，用 \(H_{kv}\) 直接控制质量—缓存—带宽的折中。`,
      takeaway: R`GQA 不是第三种完全不同的算子，而是 \(1\le H_{kv}\le H_q\) 的统一参数化：MQA 是 1，MHA 是 \(H_q\)。`,
      motivation: [
        "MQA 的缓存最小，但单一 K/V 头可能成为表达瓶颈；MHA 表达充足，却为每个 Q 头重复保存历史。",
        "GQA 把查询头分为 G 组，每组共享一套 K/V。模型设计者可以根据目标硬件和质量预算选择中间点。",
        "原论文还给出从 MHA checkpoint 升级到 GQA 的办法：组内 K/V 投影做均值池化，再用约原预训练算力 5% 的继续训练恢复能力。5% 是该论文配方，不是固定标准。"
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
      zhTitle: "多头潜变量注意力：压缩每个 token 的宽度",
      year: "2024",
      category: "dense",
      difficulty: "高阶",
      report: "DeepSeek-V2 Technical Report",
      deck: "MLA 不再让若干 Q 头直接共享一套完整 K/V，而是先把每个 token 压到低维 latent，缓存 latent；需要注意力时再为各头上投影恢复 K/V。",
      takeaway: "GQA 在“头数”上共享，MLA 在“低秩潜空间”里共享。只缓存联合压缩 latent，保留多头上投影的表达力。",
      motivation: [
        "MQA/GQA 通过减少 KV 头数降缓存，但 K/V 表示容量也随之减少。DeepSeek-V2 希望保留多头差异，同时继续降低 KV cache。",
        R`MLA 对 K 与 V 做联合低秩压缩：每个 token 只保存一个低维向量 \(c^{KV}\)；每个头的 K/V 由不同上投影从 \(c^{KV}\) 重建。`,
        "RoPE 会阻碍把上投影吸收到查询侧，因此 MLA 把位置相关的 RoPE 子空间与可低秩吸收的内容子空间分开，这就是 decoupled RoPE。"
      ],
      constraints: [
        { label: "缓存", title: R`宽度从 \(Hd_h\) 变成 \(d_c\)`, body: R`每 token 主要缓存 \(c^{KV}\) 与较小的 RoPE key；收益取决于 \(d_c\) 和 \(d_R\)。` },
        { label: "算子", title: "需要吸收或重建", body: "朴素地显式恢复各头 K/V 会增加算力；高效推理通常把上投影吸收到 Q/输出侧。" },
        { label: "位置编码", title: "RoPE 必须解耦", body: "位置依赖旋转与低秩投影一般不可交换，需要单独保存旋转 key 分量。" }
      ],
      intuitions: [
        { label: "类比", title: "保存源文件，不存多份导出", body: "latent 是紧凑源文件，各头按需用不同模板展开。" },
        { label: "低秩", title: "共享生成基底", body: R`K/V 头都来自同一个 \(d_c\) 维潜空间。` },
        { label: "RoPE", title: "位置水印另存", body: "内容可压缩吸收，位置旋转通道单独处理。" }
      ],
      diagram: { type: "latent", caption: R`MLA：每个 token 只缓存低维 \(c^{KV}\) 与解耦的 RoPE key，再供多头读取。` },
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
        R`MLA 解决了每个 token 缓存过宽的问题，但 dense attention 仍要让每个 query 与所有历史位置做高维交互；上下文极长时，算量仍随 \(L^2\) 增长。`,
        "DSA 引入 Lightning Indexer：用低维、低成本路径估计相关性，为每个 query 选择 k 个历史位置；core attention 只对选中 token 计算。",
        "DeepSeek-V3.2-Exp 以接近 V3.1-Terminus 的训练配置验证稀疏化，官方称输出质量基本持平，并开源训练/推理 kernel。"
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
            DSA 的价值来自 \(d_I\ll d\)、低精度索引和 \(k\ll L\)。它不应被粗暴写成“所有部分都严格 \(O(Lk)\)”。`
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
          hint: R`比较 \(L^2\) 与 \(Lk\)。`,
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
        { label: "计算", title: "压缩后仍是 dense", body: R`query 轴未压缩，prefill 仍含 \(O(L^2/m')\) 项，并非严格线性。` },
        { label: "局部性", title: "必须并联滑窗", body: R`未闭合块与近期细节由 \(w=128\) 的原始 KV 窗口补足。` }
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
        { label: "硬件现实", title: R`\(O(L)\) 不等于一定更快`, body: "短序列上，成熟的 FlashAttention 可能因更高算术强度而更快。" }
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

  // Keep the base chapter records readable.  Detailed, cross-cutting teaching
  // material lives in this id-keyed pass so that its schema can evolve without
  // duplicating or rewriting the original chapter objects above.
  var chapterEnhancements = {
    mha: {
      positionEncoding: {
        title: "原始 MHA 使用加性正弦位置编码",
        summary: "《Attention Is All You Need》在词嵌入进入编码器/解码器栈之前加入固定正弦位置编码（也报告了效果相近的 learned embedding）。MHA 算子本身并不强制这种选择；现代 MHA 常改用 RoPE、相对位置偏置或 ALiBi。",
        equation: R`\[
          \operatorname{PE}(p,2i)=\sin\!\left(p/10000^{2i/d_{\text{model}}}\right),\qquad
          \operatorname{PE}(p,2i+1)=\cos\!\left(p/10000^{2i/d_{\text{model}}}\right).
        \]`,
        steps: [
          { label: "原论文", title: "先加位置，再投影 Q/K/V", body: R`输入为 \(x_p=e_p+\operatorname{PE}(p)\in\mathbb R^{d_{\text{model}}}\)，随后各头用自己的 \(W_h^Q,W_h^K,W_h^V\) 投影。` },
          { label: "作用点", title: "位置不属于 head-sharing 定义", body: "把正弦编码换成 learned absolute embedding 不会把 MHA 变成另一种头共享结构。" },
          { label: "现代实现", title: "RoPE/相对偏置是后来的常见替代", body: R`现代解码器常旋转 \(q_p,k_p\) 或给 score 加 \(b_{p-s}\)；这是实现选择，不应倒写成 2017 原论文的机制。` }
        ],
        caveat: "RoPE 长度扩展、频率缩放与插值属于具体模型配方；不能仅由“MHA”三个字推断。"
      },
      derivationSourceFallback: "Vaswani et al. (2017), §3.2（attention）与 §3.5（positional encoding）",
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
      positionEncoding: {
        title: "MQA 不定义新的位置编码",
        summary: "Shazeer 的改动是让全部 query 头共享一套 K/V；论文并未把某种位置编码写进 MQA 定义。原模型继承其 Transformer 基线的位置处理，现代 MQA 解码器则常与 RoPE 配套。",
        equation: R`\[
          O_h=\operatorname{softmax}\!\left(
          \frac{Q_hK^\top}{\sqrt{d_h}}+B_{\text{pos}}+M\right)V,
        \]`,
        steps: [
          { label: "不变量", title: "共享 K/V 不等于共享位置", body: R`无论 \(B_{\text{pos}}\) 来自加性 embedding、相对 bias 还是旋转后的 Q/K，MQA 的结构条件都只是 \(H_{kv}=1\)。` },
          { label: "原论文", title: "只替换注意力投影布局", body: "Fast Transformer Decoding 的贡献是 one write-head；它没有提出名为 MQA 的专属 PE。" },
          { label: "现代实现", title: "共享 key 只旋转一次", body: R`采用 RoPE 时，共享 \(k_s\) 可按位置 \(s\) 旋转一次，再被所有 \(Q_h\) 读取；query 头仍各自旋转。` }
        ],
        caveat: "不能从“MQA”推断 RoPE base、缩放方法或是否使用相对 bias；这些都由具体 checkpoint 决定。"
      },
      derivationSourceFallback: "Shazeer (2019), Multi-query attention 与 decoding-cost 相关章节",
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
      positionEncoding: {
        title: "GQA 与位置机制正交",
        summary: "GQA 论文从 T5 checkpoint uptrain，沿用 T5 的桶化相对位置偏置；GQA 本身只规定 query-head 到 KV-head 的分组。现代 Llama 类 GQA 更常在每个 Q/K 头上使用 RoPE。",
        equation: R`\[
          S_{h,t,s}=\frac{q_{h,t}^{\top}k_{g(h),s}}{\sqrt{d_h}}
          +b_{\operatorname{bucket}(t-s)}+M_{t,s}.
        \]`,
        steps: [
          { label: "原论文", title: "uptrain 继承 T5 相对偏置", body: R`位置项 \(b_{\operatorname{bucket}(t-s)}\) 与组映射 \(g(h)\) 是两条独立轴；池化 K/V 不要求池化相对偏置。` },
          { label: "分组", title: "每个 KV 组仍保留位置索引", body: R`共享的是内容投影 \(W_g^K,W_g^V\)，不是把多个历史位置合并。` },
          { label: "现代实现", title: "RoPE 常在共享前的头表示上应用", body: R`对组 \(g\)，旋转后的 \(R_s k_{g,s}\) 被组内所有 query 头读取；这不改变 \(H_q/H_{kv}\) 的 cache 比。` }
        ],
        caveat: "同为 GQA 的模型可能使用 T5 bias、RoPE、ALiBi 或其他方案；不能跨 checkpoint 搬用位置参数。"
      },
      derivationSourceFallback: "Ainslie et al. (2023), §2（GQA 定义）与 §3（uptraining）",
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
          source: "Ainslie et al. (2023), §3.1（mean pooling initialization）"
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
          source: "Ainslie et al. (2023), §2（grouped-query attention）"
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
      positionEncoding: {
        title: "MLA 原生使用 decoupled RoPE",
        summary: "DeepSeek-V2 把可吸收的内容子空间与较小的 RoPE 子空间拆开：内容 K/V 来自共享 latent，位置 key 单独缓存。现代 DeepSeek MLA 实现通常保留这一原则，只可能改变 RoPE scaling、维度和 kernel。",
        equation: R`\[
          s_{t,s}^{(i)}=
          (q_{t,i}^{C})^\top W_i^{UK}c_s^{KV}
          +(R_tq_{t,i}^{R})^\top(R_sk_s^{R})
          =(\widetilde q_{t,i}^{C})^\top c_s^{KV}
          +(q_{t,i}^{R})^\top R_{s-t}k_s^{R}.
        \]`,
        steps: [
          { label: "内容通道", title: "固定上投影可被吸收", body: R`\(\widetilde q_{t,i}^{C}=(W_i^{UK})^\top q_{t,i}^{C}\)，所以 score 可直接与缓存 latent \(c_s^{KV}\) 点积。` },
          { label: "位置通道", title: "旋转依赖 token 位置", body: R`正交旋转满足 \(R_t^\top R_s=R_{s-t}\)，给出相对位移；但 \(R_sW_i^{UK}\) 不是可预先吸收的固定矩阵。` },
          { label: "现代实现", title: "吸收式与重建式应数值等价", body: "训练可显式重建多头 K/V，decode 可把 K 上投影吸收到 Q 侧、V 上投影吸收到输出侧；两者只是计算图不同。" }
        ],
        caveat: "MLA cache 不是只有 latent：decoupled RoPE key 也必须保存；具体宽度和量化格式应以模型配置为准。"
      },
      derivationSourceFallback: "DeepSeek-V2 Technical Report (2024), §2.1.1（MLA）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "counterexample", level: "intermediate" }
      ],
      derivations: [
        {
          title: "K 与 V 的两侧吸收完整保留输出",
          body: R`**原式。** 对头 \(i\)，\(k_{s,i}^{C}=W_i^{UK}c_s\)、\(v_{s,i}=W_i^{UV}c_s\)，并令
            \(a_{t,s,i}\) 为含内容与 RoPE 分数的 softmax 权重。**补全代数。**
            \[
            (q_{t,i}^{C})^\top k_{s,i}^{C}
            =((W_i^{UK})^\top q_{t,i}^{C})^\top c_s,
            \]
            \[
            o_{t,i}=\sum_sa_{t,s,i}W_i^{UV}c_s
            =W_i^{UV}\bar c_{t,i},\quad
            \bar c_{t,i}=\sum_sa_{t,s,i}c_s.
            \]
            再把 \(W_i^{UV}\) 与该头对应的 \(W_i^O\) 合并，即
            \(W_i^Oo_{t,i}=(W_i^OW_i^{UV})\bar c_{t,i}\)。**张量形状。**
            \(c_s\in\mathbb R^{d_c}\)、\(W_i^{UK}\in\mathbb R^{d_h^C\times d_c}\)、
            \(W_i^{UV}\in\mathbb R^{d_v\times d_c}\)、\(\bar c_{t,i}\in\mathbb R^{d_c}\)。
            **直观。** K 的展开搬到 query 左侧，V 的展开搬到输出右侧，中间直接读紧凑 latent。**边界。**
            该恒等式依赖投影线性且 softmax 权重在 V 投影之前确定；RoPE 子空间因位置相关而不能并入同一个固定内容投影。`,
          source: "DeepSeek-V2 Technical Report (2024), §2.1.1（low-rank KV compression 与 decoupled RoPE）"
        }
      ],
      exercises: [
        {
          kind: "derivation",
          level: "advanced",
          q: R`从 \(R_t^\top R_s=R_{s-t}\) 推出 RoPE 点积只依赖相对位移，并说明为什么这不使 \(R_sW^{UK}\) 成为固定矩阵。`,
          hint: R`写成 \((R_tq)^\top(R_sk)=q^\top R_t^\top R_sk\)。`,
          answer: R`\((R_tq)^\top(R_sk)=q^\top R_{s-t}k\)，旋转关系只含 \(s-t\)。但对所有历史位置 \(s\)，\(R_sW^{UK}\) 随 \(s\) 改变；除非投影与所有旋转特殊地可交换，否则不存在一个与位置无关的吸收矩阵。`
        },
        {
          kind: "code-shape",
          level: "advanced",
          q: R`给定 \(c^{KV}\) 为 \([B,L,d_c]\)，每头 \(W_i^{UK}\) 为 \([d_h^C,d_c]\)。显式重建全部内容 key 后的形状是什么？吸收式 query 的形状又是什么？`,
          hint: "重建多一个 head 轴；吸收后 query 落到 latent 维。",
          answer: R`显式 key 为 \([B,H,L,d_h^C]\)。吸收式把每头 query 乘 \((W_i^{UK})^\top\)，得到 \([B,H,L_q,d_c]\)，它与缓存 \([B,1,L,d_c]\) 点积；这节省 cache，不保证 query-side 算量总更小。`
        },
        {
          kind: "design",
          level: "advanced",
          q: R`增大 \(d_c\) 与增大 \(d_R\) 分别主要改善什么，又分别增加什么成本？`,
          hint: "一个控制内容低秩容量，一个控制位置子空间。",
          answer: R`增大 \(d_c\) 提高 K/V 内容重建容量，但线性增加 latent cache 与吸收式点积宽度；增大 \(d_R\) 提高 RoPE 位置通道容量，却线性增加不可吸收的 position-key cache。两者应分别做质量—带宽消融。`
        },
        {
          kind: "complexity",
          level: "intermediate",
          q: R`若每 token 缓存 \(d_c+d_R=576\) 个 BF16 元素，40 层、长度 128K、batch 2 的 MLA cache 约多大？`,
          hint: R`计算 \(BNL(d_c+d_R)b\)，这里 K/V 已联合进 latent，不能再乘 2。`,
          answer: R`\(2\times40\times131072\times576\times2=12{,}079{,}595{,}520\) 字节，约 11.25 GiB。该估算不含对齐、量化元数据与运行时工作区。`
        }
      ]
    },

    dsa: {
      positionEncoding: {
        title: "DSA 继承 MLA RoPE，并给 Indexer 部分旋转",
        summary: "DeepSeek-V3.2 的 core attention 仍建立在 MLA 上；Lightning Indexer 的低维 Q/K 也只对指定 RoPE 子维应用旋转。Indexer 中用于低精度计算的正交 Hadamard 变换不是位置编码。",
        equation: R`\[
          q_{t,j}^{I}=[R_tq_{t,j}^{I,R};q_{t,j}^{I,N}],\qquad
          k_s^{I}=[R_sk_s^{I,R};k_s^{I,N}],
        \]`,
        steps: [
          { label: "核心路径", title: "稀疏选择不改 MLA 位置定义", body: "top-k 只缩小被 core MLA 读取的位置集合；选中后的内容/位置 score 仍按 MLA 计算。" },
          { label: "索引路径", title: "Indexer Q/K 使用 partial RoPE", body: R`只有 \(d_R\) 个子维旋转，其余 \(d_I-d_R\) 个子维保持内容表示；所有 head 共享 indexer key。` },
          { label: "实现细节", title: "Hadamard rotation 不是 PE", body: R`若对 q、k 同施正交 \(H\)，则 \((Hq)^\top(Hk)=q^\top k\)；它服务 FP8 数值分布，不注入 token 位置。` }
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
            \(p_t\) 由主 attention 各头重要性聚合后沿历史轴 L1 归一化。稀疏阶段把同一目标限制到选中集合
            \(\mathcal S_t\) 上再做式 (4) 的对齐。**张量形状。** \(I,p,\widehat p\in\mathbb R^{B\times L_q\times L_k}\)，
            top-k 索引为 \([B,L_q,k]\)。**直观。** Indexer 学的是“主 attention 会把概率放在哪里”，不是直接回归 value。**边界。**
            top-k 是离散路由，且报告将 Indexer 输入 detach；语言模型损失不经该选择直接训练 Indexer，漏召回只能由其独立 KL 信号与稀疏训练适配缓解。`,
          source: "DeepSeek-V3.2 Technical Report (2025), §2.1, Eqs. (3)–(4)"
        }
      ],
      exercises: [
        {
          kind: "derivation",
          level: "advanced",
          q: R`推导 \(D_{\mathrm{KL}}(p\|\operatorname{softmax}(I))\) 对 logit \(I_s\) 的梯度。`,
          hint: R`使用 \(\log\operatorname{softmax}(I)_u=I_u-\log\sum_ve^{I_v}\)。`,
          answer: R`去掉与 \(I\) 无关的 \(\sum_up_u\log p_u\) 后，损失为 \(-\sum_up_uI_u+\log\sum_ve^{I_v}\)。因 \(\sum_up_u=1\)，求导得 \(-p_s+e^{I_s}/\sum_ve^{I_v}=\widehat p_s-p_s\)。`
        },
        {
          kind: "counterexample",
          level: "advanced",
          q: "构造一种情形：Indexer 的 top-1 分数误差很小，却让 core 输出发生巨大变化。",
          hint: "让两个候选 score 很接近，但 value 完全相反。",
          answer: R`设正确候选 A 与错误候选 B 的 index score 只差 \(\epsilon\)，而 core value 分别为 \(v_A=u,v_B=-u\)。微小量化/估计误差可交换 top-1，使输出从 \(u\) 跳到 \(-u\)，变化范数为 \(2\|u\|\)。离散 top-k 在边界处不连续。`
        },
        {
          kind: "code-shape",
          level: "advanced",
          q: R`Indexer query 为 \([B,L,H^I,d_I]\)，共享 key cache 为 \([B,S,d_I]\)，权重为 \([B,L,H^I]\)。写出 score 的目标形状和 head 聚合。`,
          hint: R`先得到 \([B,L,H^I,S]\)，ReLU 后乘权重并沿 \(H^I\) 求和。`,
          answer: R`点积得到 \(D_{b,t,h,s}=\langle q_{b,t,h},k_{b,s}\rangle\)，形状 \([B,L,H^I,S]\)；计算 \(\sum_h w_{b,t,h}\operatorname{ReLU}(D_{b,t,h,s})\) 后为 \([B,L,S]\)，再沿 S 做 top-k 得 \([B,L,k]\)。`
        },
        {
          kind: "design",
          level: "advanced",
          q: R`若把 \(k\) 从 2048 减到 512，应至少监控哪些指标来判断是否值得？`,
          hint: "同时看召回、语言质量和系统收益。",
          answer: "监控 dense-teacher attention mass/关键 token 的 top-k recall、长上下文与检索任务质量、验证损失、Indexer KL、core FLOPs、端到端 prefill/TPOT；还要按距离和任务类型分桶，避免平均召回掩盖远程依赖退化。"
        }
      ]
    },

    csa: {
      positionEncoding: {
        title: "CSA 使用 partial RoPE，并对输出 inverse RoPE",
        summary: "DeepSeek-V4 的压缩条目同时充当 key 与 value。报告只旋转每个 query/KV entry 的末 64 维；由于 value 也携带旋转后的绝对位置，core 输出对应维还必须用 query 位置的逆旋转去除绝对相位。",
        equation: R`\[
          R_{-t}\sum_{j\in\mathcal S_t}a_{t,j}R_jv_j^R
          =\sum_{j\in\mathcal S_t}a_{t,j}R_{j-t}v_j^R.
        \]`,
        steps: [
          { label: "Partial", title: "只旋转末 64 维", body: R`写成 \(c_j=[c_j^N;c_j^R]\)，仅对 \(c_j^R\in\mathbb R^{64}\) 应用 \(R_j\)，其余通道保持非旋转内容。` },
          { label: "Inverse", title: "输出按 query 位置反旋转", body: R`对输出 RoPE slice 左乘 \(R_{-t}=R_t^\top\)，使条目 \(j\) 的贡献依赖相对位移 \(j-t\)。` },
          { label: "现代实现", title: "不能套普通 QK-only RoPE", body: "因为同一压缩 entry 兼作 value，若只旋转 Q/K 而忘记 output inverse rotation，计算图就不再等价于报告定义。" }
        ],
        caveat: "压缩条目的索引与原 token 位置不是一一对应；必须使用实现规定的 compressed-position 序列与 RoPE scaling，不能擅自改成块中心。"
      },
      derivationSourceFallback: "DeepSeek-V4 Technical Report (2026), §2.3.1（CSA）",
      existingExerciseMeta: [
        { kind: "complexity", level: "foundation" },
        { kind: "derivation", level: "intermediate" }
      ],
      derivations: [
        {
          title: "inverse RoPE 把 value 的绝对相位变成相对相位",
          body: R`**原式。** CSA 的旋转 value slice 产生
            \[
            o_t^R=\sum_{j\in\mathcal S_t}a_{t,j}R_jv_j^R.
            \]
            **补全代数。** 报告在 query 位置 \(t\) 应用逆旋转：
            \[
            \widetilde o_t^R=R_{-t}o_t^R
            =\sum_ja_{t,j}R_t^\top R_jv_j^R
            =\sum_ja_{t,j}R_{j-t}v_j^R.
            \]
            **张量形状。** 每个压缩 entry \(c_j\in\mathbb R^{d_c}\)，RoPE slice
            \(v_j^R,o_t^R\in\mathbb R^{64}\)，权重 \(a_t\in\mathbb R^k\)。**直观。**
            先随资料条目的位置旋转，汇总后再站到 query 的坐标系观察，于是只剩相对距离。**边界。**
            该恒等式要求同频率旋转且 \(R_{-t}=R_t^\top\)；只对 Q/K 旋转、漏掉输出逆旋转，或给压缩条目使用不一致的位置表都会破坏它。`,
          source: "DeepSeek-V4 Technical Report (2026), §2.3.3, Eq. (26)"
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
          q: R`输入 H 为 \([B,L,d]\)，压缩率 \(m=4\)，压缩宽度 \(d_c\)。写出两路 \(C^a,C^b,Z^a,Z^b\) 与最终压缩池的典型形状。`,
          hint: "两路投影仍保留 token 轴；沿每个 2m 覆盖窗口逐通道归一化后，每 m 步产出一个 entry。",
          answer: R`投影流通常都是 \([B,L,d_c]\)。窗口权重在覆盖轴上逐通道 softmax；步长为 4，因此完整块部分的输出约为 \([B,\lfloor L/4\rfloor,d_c]\)。边界 remainder、cache carry 与 overlap 的具体条目数需按实现处理。`
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
      positionEncoding: {
        title: "HCA 与 CSA 共享 partial/inverse RoPE",
        summary: "HCA 虽取消 Indexer 和 top-k，位置处理没有退化为普通 dense RoPE：压缩 KV entry 兼作 value，末 64 维做 partial RoPE，attention 输出同一 slice 再按 query 位置做 inverse RoPE。",
        equation: R`\[
          \widetilde o_t^R
          =R_{-t}\sum_{j=1}^{\lfloor L/m'\rfloor}a_{t,j}R_jc_j^R
          =\sum_ja_{t,j}R_{j-t}c_j^R.
        \]`,
        steps: [
          { label: "压缩位置", title: "每个重压缩条目仍有序号", body: R`HCA 把 \(m'\) 个 token 合成一个 entry，但压缩序列仍按因果顺序编号并进入 RoPE。` },
          { label: "全局读取", title: "dense 只表示不做 top-k", body: "全部压缩条目参与 softmax，并不意味着忽略相对位置。" },
          { label: "输出坐标", title: "逆旋转保留相对贡献", body: R`用 \(R_{-t}\) 把混合后的 RoPE value slice 拉回 query 坐标系；这是 V4 报告的专门步骤。` }
        ],
        caveat: "HCA 还并联未压缩滑窗；全局压缩分支与局部分支的位置索引/频率配置必须按官方实现对齐。"
      },
      derivationSourceFallback: "DeepSeek-V4 Technical Report (2026), §2.3.2（HCA）",
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
            **张量形状。** 块输入 \(C_i,Z_i\in\mathbb R^{m'\times d_c}\)，权重
            \(S_i\in\mathbb R^{m'\times d_c}\)，输出 \(c_i^{Comp}\in\mathbb R^{d_c}\)。
            **直观。** 不同通道可从块内不同 token 摘要信息，而不是全向量共用一个标量权重。**边界。**
            凸组合性质只针对投影后的 C 通道；前后线性层仍可产生块外数值范围，且 128→1 仍不可逆。`,
          source: "DeepSeek-V4 Technical Report (2026), §2.3.2, Eq. (22)"
        },
        {
          title: "重压缩 dense attention 的精确位置对计数",
          body: R`**原式。** 完整闭合块数 \(n_c=\lfloor L/m'\rfloor\)，每个 query 还看至多 \(w\) 个局部 token。**补全代数。**
            \[
            N_{\text{pairs}}\le L(n_c+w)
            =L\left(\left\lfloor\frac L{m'}\right\rfloor+w\right).
            \]
            因 \(\lfloor L/m'\rfloor\le L/m'\)，得到上界 \(L^2/m'+Lw\)。**张量形状。**
            全局 score 为 \([B,H,L,n_c]\)，局部 score 逻辑上为 \([B,H,L,w]\)。**直观。**
            HCA 完整读一份短目录，再查最近原文。**边界。** 当前未闭合块不能提前进入全局摘要；实际 causal 有效 pair 少于矩形上界，固定
            \(m'\) 时主项仍为 \(\Theta(L^2)\)。`,
          source: "DeepSeek-V4 Technical Report (2026), §2.3.2 与 §2.3.4"
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
          q: R`证明对正交 RoPE 矩阵 \(R_t\)，输出逆旋转把 \(R_j\) 变成相对旋转 \(R_{j-t}\)。`,
          hint: R`使用 \(R_{-t}=R_t^\top\) 和旋转群 \(R_aR_b=R_{a+b}\)。`,
          answer: R`\(R_{-t}R_j=R_{-t+j}=R_{j-t}\)，故 \(R_{-t}\sum_ja_jR_jv_j=\sum_ja_jR_{j-t}v_j\)。若不同位置使用不一致频率，群关系不再成立。`
        }
      ]
    },

    linear: {
      positionEncoding: {
        title: "核线性化不自动提供位置编码",
        summary: "《Transformers are RNNs》的核心贡献是核分解与因果状态递推，不是新的显式 PE。因果前缀使状态随时间更新，但无衰减的加法状态对同一前缀内写入顺序可交换；现代线性模型常另加 ShortConv、decay/gate 或专门的相对核。",
        equation: R`\[
          S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
          y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t+\varepsilon}.
        \]`,
        steps: [
          { label: "原论文", title: "位置不是核技巧的一部分", body: "该算子可以接收已含位置特征的输入，但不能从 kernel linear attention 名称推断 RoPE 或某种固定 embedding。" },
          { label: "因果顺序", title: "prefix 边界提供弱顺序信号", body: R`时刻 \(t\) 只能读 \(j\le t\)，但 \(\sum_{j\le t}\phi(k_j)v_j^\top\) 对这些项的排列可交换。` },
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
            用固定宽统计量换掉位置轴。**边界。** “关于 L 线性”假设 \(r,d_v\) 不随 L 增长；若为逼近 softmax 而让
            \(r=\Theta(L)\)，则成本重新变成平方级。`,
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
      positionEncoding: {
        title: "DeltaNet/Gated DeltaNet 依赖 ShortConv 与有序状态",
        summary: "原始 DeltaNet/Gated DeltaNet token mixer 不给线性层 Q/K 套 RoPE；Q/K/V 投影后经过短因果卷积，递推更新和 gate 又使早晚写入产生不同状态。现代混合模型可在另外的 attention 层使用 RoPE，但这不等于 linear 层也使用 RoPE。",
        equation: R`\[
          q_t,k_t=\operatorname{L2Norm}\!\left(
          \operatorname{SiLU}(\operatorname{ShortConv}(Wx)_{t})\right),\qquad
          S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
        \]`,
        steps: [
          { label: "局部顺序", title: "ShortConv 看最近若干 token", body: "因果 depthwise 卷积让同一 token 在不同局部排列下生成不同 q/k/v，是显式 RoPE 之外的局部次序通道。" },
          { label: "全局顺序", title: "状态转移不可交换", body: R`一般 \(A_tA_s\ne A_sA_t\)，其中 \(A_t=\alpha_t(I-\beta_tk_tk_t^\top)\)，所以写入顺序影响最终记忆。` },
          { label: "现代混合", title: "分层区分 PE", body: "例如混合架构中的 full/sliding-attention 层可使用其自己的 RoPE；Gated DeltaNet 层仍按 ShortConv+state 工作。" }
        ],
        caveat: "不要为解释位置感而虚构一项 additive PE：论文消融明确把 ShortConv、gate 和状态动力学作为组成部分。"
      },
      derivationSourceFallback: "Schlag et al. (2021), delta-rule章节；Yang et al. (2025), §3（Gated DeltaNet）",
      existingExerciseMeta: [
        { kind: "derivation", level: "intermediate" },
        { kind: "complexity", level: "foundation" }
      ],
      derivations: [
        {
          title: "从逐元素微分补全在线梯度下降",
          body: R`**原式。** 取 \(S\in\mathbb R^{d_k\times d_v}\) 和
            \[
            \ell_t(S)=\frac12\|S^\top k_t-v_t\|_2^2.
            \]
            **补全代数。** 令误差 \(e=S^\top k_t-v_t\in\mathbb R^{d_v}\)。微分
            \[
            d\ell=e^\top d(S^\top k_t)
            =e^\top(dS)^\top k_t
            =\operatorname{tr}\!\left((k_te^\top)^\top dS\right),
            \]
            故 \(\nabla_S\ell=k_te^\top\)。一步 SGD 给出
            \[
            S_t=S_{t-1}-\beta_tk_t(S_{t-1}^\top k_t-v_t)^\top
            =(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
            \]
            **张量形状。** \(k_t\in\mathbb R^{d_k}\)、\(v_t,e_t\in\mathbb R^{d_v}\)，两个外积均为
            \(\mathbb R^{d_k\times d_v}\)。**直观。** 先用快权重回答，再只沿当前 key 写入预测误差。**边界。**
            这是单样本、一步显式 SGD；\(\|k_t\|=1\) 与 \(0\le\beta_t\le1\) 有助于稳定和精确覆盖，但不保证不同 key 正交或长期无干扰。`,
          source: "Yang, Kautz & Hatamizadeh (2025), §3.1（online regression view）"
        },
        {
          title: "gated delta 等于先衰减再纠错",
          body: R`**原式。**
            \[
            S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
            \]
            **补全代数。** 令预衰减状态 \(\widetilde S_{t-1}=\alpha_tS_{t-1}\)，则
            \[
            S_t=\widetilde S_{t-1}
            +\beta_tk_t\left(v_t-\widetilde S_{t-1}^{\top}k_t\right)^\top.
            \]
            因此若 \(\|k_t\|=1,\beta_t=1\)，有 \(S_t^\top k_t=v_t\)。**张量形状。**
            标量 \(\alpha_t,\beta_t\) 广播到 \([d_k,d_v]\) 状态。**直观。** 先让所有旧记忆褪色，再基于褪色后的答案做定点修正。**边界。**
            若把误差错误地写成 \(v_t-S_{t-1}^\top k_t\)，展开式会少一个 \(\alpha_t\) 交叉项；矩阵朝向不同的论文还需整体转置后比较。`,
          source: "Yang, Kautz & Hatamizadeh (2025), Eq. (10) and §3.1"
        }
      ],
      exercises: [
        {
          kind: "counterexample",
          level: "advanced",
          q: R`构造 \(\|k_1\|=\|k_2\|=1\) 但写入第二个 key 会破坏第一个 key 的读取。`,
          hint: "选两个不正交 key。",
          answer: R`取 \(k_1=(1,0)^\top\)，\(k_2=(1,1)^\top/\sqrt2\)，先令 \(S^\top k_1=v_1\)。第二次 delta 写入含 \(k_2e_2^\top\)，对 \(k_1\) 的输出变化为 \(e_2(k_2^\top k_1)=e_2/\sqrt2\)，一般非零。精确覆盖当前 key 不代表不干扰相似 key。`
        },
        {
          kind: "code-shape",
          level: "intermediate",
          q: R`批量多头实现中，Q/K 为 \([B,H,L,d_k]\)、V 为 \([B,H,L,d_v]\)。递推 state 与单步误差应是什么形状？`,
          hint: "state 不含 L 轴。",
          answer: R`state 为 \([B,H,d_k,d_v]\)。时刻 t 的预测 \(S^\top k_t\) 与误差都是 \([B,H,d_v]\)；\(k_t e_t^\top\) 广播外积回到 \([B,H,d_k,d_v]\)。`
        },
        {
          kind: "design",
          level: "advanced",
          q: "何时应在 Gated DeltaNet 模型中周期性插入 full/SWA 层？",
          hint: "固定状态擅长压缩记忆，但不擅长任意精确回看。",
          answer: "当任务需要多针检索、逐字复制、长距离精确引用或状态碰撞明显时插入显式 attention。应消融插入频率，联合测 retrieval、局部质量、状态/cache、训练吞吐与 decode TPOT。"
        },
        {
          kind: "derivation",
          level: "advanced",
          q: R`令 \(\widetilde S=\alpha S\)。证明 \(\beta=1,\|k\|=1\) 时 gated delta 更新后对 k 的读取精确等于 v。`,
          hint: R`使用 \(S^+=\widetilde S+k(v-\widetilde S^\top k)^\top\)。`,
          answer: R`\((S^+)^\top k=\widetilde S^\top k+(v-\widetilde S^\top k)k^\top k=v\)。若 \(\|k\|\ne1\)，则会多出 \(\|k\|^2\) 因子，除非相应调整步长。`
        }
      ]
    },

    kda: {
      positionEncoding: {
        title: "Kimi Linear 用 NoPE；KDA 自己承担位置感",
        summary: "Kimi Linear 报告明确让全局 MLA 层也使用 NoPE。位置与 recency 主要来自 KDA 的数据依赖逐通道 decay、有序 DPLR 转移和 Q/K/V 前 ShortConv，而不是显式 RoPE。",
        equation: R`\[
          S_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
          +\beta_tk_tv_t^\top,\qquad
          \alpha_t\in[0,1]^{d_k}.
        \]`,
        steps: [
          { label: "局部", title: "ShortConv 编码邻域次序", body: "Q/K/V 的短因果卷积对局部排列敏感，不需要额外虚构绝对位置向量。" },
          { label: "长程", title: "逐通道乘积形成可学习距离衰减", body: R`从位置 \(s\) 到 \(t\) 的某通道保留量包含 \(\prod_{u=s+1}^{t}\alpha_{u,r}\)，天然依赖经过的有序步数与内容。` },
          { label: "原报告", title: "周期性 MLA 也采用 NoPE", body: "Kimi Linear 把位置责任交给 KDA，NoPE MLA 提供全局无位置内容读取；这与 DeepSeek MLA 的 decoupled RoPE 配方不同。" }
        ],
        caveat: "后续采用 KDA 的模型可以选择不同混合层位置方案；但不能把后续实现的 RoPE 反推为 Kimi Linear 原报告的 KDA 机制。"
      },
      derivationSourceFallback: "Kimi Linear Technical Report (2025), §2.2–§3（KDA）",
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

  window.ATTENTION_CHAPTERS.forEach(function (chapter) {
    var enhancement = chapterEnhancements[chapter.id];
    if (!enhancement) {
      throw new Error("Missing chapter enhancement for " + chapter.id);
    }

    chapter.positionEncoding = enhancement.positionEncoding;

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
