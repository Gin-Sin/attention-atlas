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
      zhTitle: "多头潜变量注意力：压缩每个 token 的宽度",
      year: "2024",
      category: "dense",
      difficulty: "高阶",
      report: "DeepSeek-V2 Technical Report",
      deck: R`MLA 先把每个 token 压到低维 \(c_t^{KV}\)，并另存共享的 decoupled-RoPE key \(k_t^R\)。decode 主路径吸收 K/V 上投影，直接读取缓存，而不是逐步显式重建全部多头 K/V。`,
      takeaway: R`GQA 在头轴共享，MLA 在低秩潜空间共享；DeepSeek-V2 每层每 token 的完整注意力缓存是 \(c_t^{KV}+k_t^R\)，宽度 \(d_c+d_h^R\)。`,
      motivation: [
        "MQA/GQA 通过减少 KV 头数降缓存，但 K/V 表示容量也随之减少。DeepSeek-V2 希望保留多头差异，同时继续降低 KV cache。",
        R`MLA 对 K 与 V 做联合低秩压缩：内容 K/V 都由低维 \(c^{KV}\) 上投影；位置相关的共享 \(k^R\) 另行生成和缓存。显式重建便于解释，吸收式计算才是高效 decode 的主路径。`,
        "RoPE 会阻碍把上投影吸收到查询侧，因此 MLA 把位置相关的 RoPE 子空间与可低秩吸收的内容子空间分开，这就是 decoupled RoPE。"
      ],
      constraints: [
        { label: "缓存", title: R`完整宽度是 \(d_c+d_h^R\)`, body: R`每 token 缓存 latent 与共享 RoPE key，不是“只有 \(d_c\)”；收益取决于两者宽度及存储精度。` },
        { label: "算子", title: "decode 采用两侧吸收", body: "概念图可显式恢复各头 K/V；高效 decode 把 K 上投影吸收到 Q 侧、V 上投影吸收到输出侧。" },
        { label: "位置编码", title: "RoPE 必须解耦", body: "位置依赖旋转与低秩投影一般不可交换，需要单独保存旋转 key 分量。" }
      ],
      intuitions: [
        { label: "类比", title: "保存源文件，不存多份导出", body: "latent 是紧凑源文件，各头按需用不同模板展开。" },
        { label: "低秩", title: "共享生成基底", body: R`K/V 头都来自同一个 \(d_c\) 维潜空间。` },
        { label: "RoPE", title: "位置水印另存", body: "内容可压缩吸收，位置旋转通道单独处理。" }
      ],
      diagram: { type: "latent", caption: R`MLA：缓存 \(c_t^{KV}\) 与共享 \(k_t^R\)；decode 通过吸收后的多头 query/输出投影直接读取这份紧凑缓存。` },
      derivations: [
        {
          title: "KV 联合低秩压缩",
          body: R`论文 §2.1.2 先定义原始低秩向量
            \[
            \bar c_t^{KV}=W^{DKV}h_t,\qquad \bar c_t^{KV}\in\mathbb{R}^{d_c}.
            \]
            论文主公式可把它直接记为 \(c_t^{KV}\)；训练细节与发布 checkpoint 在压缩 latent 后施加额外 RMSNorm，可写为
            \(c_t^{KV}=\operatorname{RMSNorm}(\bar c_t^{KV})\)。各头内容 key/value 由上投影得到
            \[
            k_{t,i}^{C}=W_i^{UK}c_t^{KV},\qquad
            v_{t,i}=W_i^{UV}c_t^{KV}.
            \]
            checkpoint 缓存的是归一化后的 \(c_t^{KV}\)，而不是全部 \(k_{t,i}^C,v_{t,i}\)；“原始线性 latent”与“实际缓存 latent”不可混称。`
        },
        {
          title: "解耦 RoPE 与缓存通式",
          body: R`把 key 分为内容与位置两部分：
            \[
            k_{t,i}=[k_{t,i}^{C};k_t^{R}],\qquad
            q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],
            \]
            其中 \(q^R,k^R\) 应用 RoPE，且
            \(q_{t,i}^C,k_{s,i}^C\in\mathbb R^{d_h}\)、
            \(q_{t,i}^R,k_s^R\in\mathbb R^{d_h^R}\)。注意力 logit 缩放为
            \[
            \ell_{t,s,i}=\frac{q_{t,i}^{\top}k_{s,i}}{\sqrt{d_h+d_h^R}}.
            \]
            §2.1.4 因而给出每层每 token 的总缓存宽度
            \[
            d_c+d_h^R
            \]
            而不是 MHA 的 \(2H_qd_h\)。`
        },
        {
          title: "decode 主路径为何能两侧吸收",
          body: R`内容分数中
            \[
            q_i^\top k_i^C=q_i^\top W_i^{UK}c^{KV}
            =\big((W_i^{UK})^\top q_i\big)^\top c^{KV}.
            \]
            因此可把 \(W_i^{UK}\) 合并进查询侧；同理 \(W_i^{UV}\) 可与输出投影合并。decode 直接对 latent 做 core attention，RoPE key 因位置旋转而单独保留；显式恢复 K/V 只是等价说明或训练实现选择。`
        }
      ],
      warning: R`“KV cache 减少 93.3%”和“最大生成吞吐 5.76×”是 DeepSeek-V2 官方对指定 checkpoint、DeepSeek-67B 基线、层配置、量化与系统实现的自报结果，不是 MLA 算子的固定倍率；结构性缓存公式应单独写成 \(d_c+d_h^R\)。`,
      exercises: [
        {
          q: R`若 MHA 每 token 缓存 \(2H d_h=2\times32\times128\) 个元素，而 MLA 缓存 \(d_c+d_h^R=512+64\) 个元素，理论元素数比是多少？`,
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
          q: R`当 \(L=131072,k=2048\) 时，core attention 的位置对数量相对 dense 减少多少倍？`,
          hint: R`比较 \(L^2\) 与 \(Lk\)。`,
          answer: R`理想比值为 \(L/k=64\) 倍。这里只比较 core 位置对，不包括带 \(H^Id_I\) 因子的全历史 Indexer 和系统开销；DSA 本身没有另加局部窗口。`
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
      order: 6,
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
      order: 7,
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
      order: 8,
      title: "DeltaNet",
      fullTitle: "DeltaNet & Gated DeltaNet",
      zhTitle: "Delta 更新：让固定状态学会定点改写",
      year: "2024–25",
      category: "linear",
      difficulty: "前沿",
      report: "Gated Delta Networks · ICLR 2025",
      deck: R`普通线性注意力不断叠加写入；Delta rule 先读再写残差，Gated DeltaNet 再加入遗忘。论文用 \(F_t\in\mathbb R^{d_v\times d_k}\)，本站统一转置为 \(S_t=F_t^\top\in\mathbb R^{d_k\times d_v}\)。`,
      takeaway: "加法记忆是“追加”；Delta rule 是“按 key 修正”；gate 是“先清场再修正”。三者逐步减少固定状态中的干扰。",
      motivation: [
        R`普通线性状态 \(S\leftarrow S+kv^{\mathsf T}\) 对相同或相近 key 反复写入时会累积冲突，无法像字典一样覆盖旧值。`,
        R`Delta rule 计算当前状态对 key 的预测 \(S^{\mathsf T}k\)，用目标 value 与预测之差作为写入量，因此更新集中在尚未记住的信息。`,
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
      warning: R`Gated DeltaNet 论文写 \(F_t\in\mathbb R^{d_v\times d_k},\,o_t=F_tq_t\)；本站公式使用转置约定 \(S_t=F_t^\top,\,o_t=S_t^\top q_t\)。官方 block 只有 q/k/v 经过 ShortConv；\(\alpha,\beta\) 走直接线性投影，\(\beta=\sigma(W^\beta x)\)，实现还会把归一化后的 q 乘 \(d_k^{-1/2}\)。`,
      exercises: [
        {
          q: R`若 \(\|k\|=1,\beta=1\)，证明 delta 更新后 \(S_t^\top k=v\)。`,
          hint: R`把 \(S_t=S+\;k(v-S^\top k)^\top\) 左乘到 k 上。`,
          answer: R`\(S_t^\top k=S^\top k+(v-S^\top k)k^\top k=v\)。因此对该 key 完成一次精确覆盖。`
        },
        {
          q: R`若连续 100 步没有 delta 写入且 \(\alpha=0.99\)，旧状态幅度剩多少？`,
          hint: R`计算 \(0.99^{100}\)。`,
          answer: "在没有新写入且 α 恒定的 decay-only 情形下约 0.366；数据依赖 gate 与后续 delta 写入会改变实际可检索寿命。"
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
      positionEncoding: {
        title: "MLA 原生使用 decoupled RoPE",
        summary: R`DeepSeek-V2 把 \(d_h\) 维内容子空间与 \(d_h^R\) 维 RoPE 子空间拆开：缓存 \(c_s^{KV}\) 与共享 \(k_s^R\)，attention logit 除以 \(\sqrt{d_h+d_h^R}\)。\(c_t^Q\in\mathbb R^{d_c'}\) 是 query bottleneck，展开后的每头 \(q_{t,i}^C\in\mathbb R^{d_h}\)，两者不能混淆。`,
        equation: R`\[
          s_{t,s}^{(i)}=
          \frac{(q_{t,i}^{C})^\top W_i^{UK}c_s^{KV}
          +(R_tq_{t,i}^{R})^\top(R_sk_s^{R})}
          {\sqrt{d_h+d_h^R}}
        \]
        \[
          =(\widetilde q_{t,i}^{C})^\top c_s^{KV}
          /\sqrt{d_h+d_h^R}
          +(q_{t,i}^{R})^\top R_{s-t}k_s^{R}/\sqrt{d_h+d_h^R}.
        \]`,
        steps: [
          { label: "内容通道", title: "decode 直接读 latent", body: R`\(\widetilde q_{t,i}^{C}=(W_i^{UK})^\top q_{t,i}^{C}\)，V 上投影也吸收到输出侧；这是 decode 主路径，不必显式重建多头 K/V。` },
          { label: "位置通道", title: "旋转依赖 token 位置", body: R`正交旋转满足 \(R_t^\top R_s=R_{s-t}\)，给出相对位移；但 \(R_sW_i^{UK}\) 不是可预先吸收的固定矩阵。` },
          { label: "缓存与归一化", title: R`总宽度 \(d_c+d_h^R\)`, body: R`论文代数先定义原始 \(\bar c=W^{DKV}h\)；训练细节/发布 checkpoint 对 compressed latent 做 RMSNorm 并缓存归一化结果，再另存 \(k^R\)。` }
        ],
        caveat: "MLA cache 不是只有 latent：decoupled RoPE key 也必须保存；具体宽度和量化格式应以模型配置为准。"
      },
      derivationSourceFallback: "DeepSeek-V2 Technical Report (2024), §2.1.2–§2.1.4 与 Appendix C（完整 MLA 公式）",
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
            \(c_s\in\mathbb R^{d_c}\)、\(W_i^{UK}\in\mathbb R^{d_h\times d_c}\)、
            \(W_i^{UV}\in\mathbb R^{d_v\times d_c}\)、\(\bar c_{t,i}\in\mathbb R^{d_c}\)。
            **直观。** K 的展开搬到 query 左侧，V 的展开搬到输出右侧，中间直接读紧凑 latent。**边界。**
            该恒等式依赖投影线性且 softmax 权重在 V 投影之前确定；RoPE 子空间因位置相关而不能并入同一个固定内容投影。`,
          source: "DeepSeek-V2 Technical Report (2024), §2.1.2（KV/query compression）、§2.1.3（decoupled RoPE）与 Appendix C"
        },
        {
          title: R`为什么 \(q^R\) 逐头、\(k^R\) 却全头共享`,
          body: R`**设定。** DeepSeek-V2 定义逐头 RoPE query
            \(q_{t,i}^R=R_tW_i^{QR}c_t^Q\)，却只保留一个全头共享的 RoPE key
            \(k_t^R=R_tW^{KR}h_t\)。这个不对称不是数学必然，而是一次精确的成本—收益权衡：昂贵的一侧（key 进缓存）做 MQA 式共享，免费的一侧（query 现算即弃）保留逐头自由度。
            **补全代数。** key 侧：\(k^R\) 在 latent 之外裸缓存，宽度直接线性计入每 token 缓存。共享时总宽为
            \[
            d_c+d_h^R=512+64=576;
            \]
            若改为逐头，则变成
            \[
            d_c+Hd_h^R=512+128\times64=8704,
            \]
            位置通道一项就把 MLA 相对 MHA 的缓存压缩全部吃回。query 侧：\(q_{t,i}^R\) 只为当前 token 计算一次、不进缓存，逐头化的代价仅是可忽略的 \(O(Hd_h^R)\) 投影算量。而每头分数中的位置项
            \[
            (R_tq_{t,i}^R)^\top(R_sk_s^R)=(q_{t,i}^R)^\top R_{s-t}k_s^R
            \]
            对每个头独立成立相对位置性质：不同头以不同的 \(W_i^{QR}\) 对同一 \(k^R\) 产生不同的幅度与相位响应，位置敏感度的头间多样性完全由 query 侧承担。
            **张量形状。** 逐头 \(q^R\) 为 \([B,H,L_q,d_h^R]\)；共享 \(k^R\) 缓存为 \([B,1,L,d_h^R]\)，以 stride-0 广播给全部 \(H\) 个头——正是 MQA 的共享技巧在位置通道上的翻版。
            **直观。** 在随序列长度增长的维度（key cache）上做最激进的共享，在一次性计算的维度（query）上保留全部自由度。
            **边界。** 共享意味着所有头读取同一个 key 侧位置基底；头间的 key 侧位置差异被放弃，只能靠增大 \(d_h^R\) 换容量并按线性代价计入缓存。DSA/CSA 等后续架构原样继承了这一共享布局。`,
          source: "DeepSeek-V2 Technical Report (2024), §2.1.3（decoupled RoPE：逐头 RoPE query 与共享 RoPE key 的定义）与 §2.1.4（KV cache 宽度对比）"
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
          q: R`给定 \(c^Q\in\mathbb R^{d_c'}\)、每头内容 query \(q_i^C\in\mathbb R^{d_h}\)、\(c^{KV}\) 为 \([B,L,d_c]\)。写出展开 query、显式内容 key 与吸收式 query 的形状。`,
          hint: "先区分 query bottleneck、每头内容维和 KV latent 维。",
          answer: R`\(W^{UQ}c^Q\) 展开为 \([B,H,L_q,d_h]\)；显式内容 key 为 \([B,H,L,d_h]\)。吸收式再乘 \((W_i^{UK})^\top\) 得 \([B,H,L_q,d_c]\)，与缓存 \([B,1,L,d_c]\) 点积。\(d_c'\)、\(d_h\)、\(d_c\) 是三种不同宽度。`
        },
        {
          kind: "design",
          level: "advanced",
          q: R`增大 \(d_c\) 与增大 \(d_h^R\) 分别主要改善什么，又分别增加什么成本？`,
          hint: "一个控制内容低秩容量，一个控制位置子空间。",
          answer: R`增大 \(d_c\) 提高 K/V 内容重建容量，但线性增加 latent cache 与吸收式点积宽度；增大 \(d_h^R\) 提高 RoPE 位置通道容量，却线性增加不可吸收的 position-key cache。两者应分别做质量—带宽消融。`
        },
        {
          kind: "complexity",
          level: "intermediate",
          q: R`若每 token 缓存 \(d_c+d_h^R=576\) 个 BF16 元素，40 层、长度 128K、batch 2 的 MLA cache 约多大？`,
          hint: R`计算 \(BNL(d_c+d_h^R)b\)，这里 K/V 已联合进 latent，不能再乘 2。`,
          answer: R`\(2\times40\times131072\times576\times2=12{,}079{,}595{,}520\) 字节，约 11.25 GiB。该估算不含对齐、量化元数据与运行时工作区。`
        }
      ]
    },

    dsa: {
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
      positionEncoding: {
        title: "DeltaNet/Gated DeltaNet 依赖 ShortConv 与有序状态",
        summary: R`Gated DeltaNet 不给 q/k 套 RoPE。只有 q/k/v 走 ShortConv+SiLU；\(\alpha,\beta\) 都由 hidden state 直接线性投影，\(\beta=\sigma(W^\beta x)\)。q/k L2Norm 后，官方/主流 kernel 还把 q 乘 \(d_k^{-1/2}\)。`,
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
          S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
        \]`,
        steps: [
          { label: "路径边界", title: "ShortConv 只在 q/k/v", body: R`\(\alpha,\beta\) 使用 linear-only gate path；给 \(\beta\) 加 ShortConv 会变成另一种参数化。` },
          { label: "全局顺序", title: "状态转移不可交换", body: R`一般 \(A_tA_s\ne A_sA_t\)，其中 \(A_t=\alpha_t(I-\beta_tk_tk_t^\top)\)，所以写入顺序影响最终记忆。` },
          { label: "矩阵约定", title: R`本站 \(S=F^\top\)`, body: R`论文常写 \(F_t\in\mathbb R^{d_v\times d_k},o_t=F_tq_t\)；本站转置为 \(S_t\in\mathbb R^{d_k\times d_v},o_t=S_t^\top q_t\)。` }
        ],
        caveat: "不要为解释位置感而虚构一项 additive PE：论文消融明确把 ShortConv、gate 和状态动力学作为组成部分。"
      },
      derivationSourceFallback: "Schlag et al. (2021), delta-rule；Yang et al. (2025), Eq. (10)、token-mixer block 与 official implementation",
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
            **约定。** Gated DeltaNet 论文状态为 \(F\in\mathbb R^{d_v\times d_k}\)；这里令 \(S=F^\top\)。**补全代数。** 令误差 \(e=S^\top k_t-v_t\in\mathbb R^{d_v}\)。微分
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
