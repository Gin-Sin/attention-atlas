# Attention Atlas

面向大模型算法工程师的中文注意力架构课程。课程沿两条相互汇合的演化路径组织：

- `MHA → MQA → GQA → MLA → DSA → CSA / HCA`
- `Linear Attention → DeltaNet / Gated DeltaNet → KDA → Hybrid Attention`

CSA 与 HCA 是两章独立内容。全站共 10 章、40 个推导和 60 道练习，学习进度只保存在浏览器 `localStorage`，不会上传。

## 章节内容契约

`assets/chapters.js` 必须按下列顺序提供且只提供 10 个 id：

`mha, mqa, gqa, mla, dsa, csa, hca, linear, gated-delta, kda`

每章遵守同一份可验证契约：

- `order` 从 0 连续递增，`category` 只能是 `dense`、`sparse`、`linear` 或 `hybrid`。
- 动机、约束、数学直觉、架构图、警告和权威来源组成基础教学内容；来源使用可解析的 HTTP(S) URL。
- `positionEncoding` 必须包含标题、摘要、公式、实现边界，以及恰好 3 个带 `label/title/body` 的步骤。
- 数学推导为 4–6 个；每个推导都必须有 `title/body/source`。
- 每章恰好 6 道练习；每题都必须有 `kind/level/q/hint/answer`。
- `diagram` 配置必须能由 `assets/diagrams.js` 构建出非空 SVG、编号导读和一句话记忆；图中每个结构块都要映射到有效的实现块。
- 每章 id 必须在 `assets/implementations.js` 中有同名 PyTorch 实现记录。

### 位置编码与时序注入

位置部分不是通用 RoPE 模板，而是逐章说明原论文或报告实际采用的机制：原始 MHA 的加性正弦编码、MQA/GQA 与位置机制的正交关系、MLA/DSA 的 decoupled 或 partial RoPE、CSA/HCA 的 partial + inverse RoPE，以及 Linear/Delta/KDA 中由前缀、ShortConv、衰减和有序状态转移承担的时序信息。具体模型配方与架构定义会明确分开。

## PyTorch 教学实现

`pytorch/` 是实现源码的唯一真源。每个文件用 `# [Block NN] 标题` 与 `# [/Block NN]` 标出连续编号的教学块；生成器把完整源码、块标题、代码和真实行号同步到浏览器资源。

章节页把架构图和代码放在同一个交互工作台：点击或键盘激活任一 SVG 结构块，会在 JetBrains Mono IDE 面板中跳转到对应 PyTorch 实现。面板支持 Prism Python 语法高亮、块级跳转、完整源码模式、真实行号高亮、复制与下载；CDN 不可用时会退化为安全的纯文本代码。

架构与实现块映射如下：

- `mha → pytorch/mha.py`：7 块，正弦位置、独立 Q/K/V、完整 KV cache 与 causal attention。
- `mqa → pytorch/mqa.py`：7 块，RoPE、单 KV 头、广播读取与缓存。
- `gqa → pytorch/gqa.py`：8 块，分组 KV 投影、无复制的 head-group 映射与 causal attention。
- `mla → pytorch/mla.py`：11 块，低秩 Q/KV、decoupled RoPE、latent cache、吸收式/重建式路径。
- `dsa → pytorch/dsa.py`：10 块，Indexer pRoPE/Hadamard/FP8 模拟、对齐损失、Indexer 打分、dense MLA 教师分布、top-k 选址、latent/RoPE/Indexer 三缓存追加与候选 gather、吸收式候选 MLA、输出写回与增量解码等价 smoke test。
- `csa → pytorch/csa.py`：10 块，重叠压缩、索引打分与 top-k、原始 Q/K/V 通道、摘要 gather、局部窗口、共享 softmax、输出投影与 smoke test。
- `hca → pytorch/hca.py`：10 块，非重叠重压缩、逆 RoPE 规范化、完结摘要缓存、局部窗口、dense 摘要读取、输出投影与 smoke test。
- `linear → pytorch/linear_attention.py`：5 块，ELU+1、prefix/recurrent 等价与固定状态。
- `gated-delta → pytorch/gated_delta.py`：5 块，ShortConv、标量门控 delta 递推与教学边界。
- `kda → pytorch/kda.py`：7 块，逐通道 DPLR、NoPE 全局层与 3:1 层级混合。

修改 `pytorch/` 后重新生成浏览器 bundle：

```bash
python3 tools/sync_pytorch_examples.py
```

只检查同步状态而不写文件：

```bash
python3 tools/sync_pytorch_examples.py --check
```

不要直接编辑生成文件 `assets/implementations.js`。

## 校验与测试

内容校验器仅使用 Node 内置模块。它在隔离的 `vm` context 中执行浏览器 IIFE，检查章节契约、全部架构—代码交互节点、首页链接与统计、源码块行号、富文本渲染入口，以及生成 bundle 的同步状态：

```bash
node tools/validate_content.mjs
```

运行 Python 单元测试和十套 CPU 参考实现的微型前向测试：

```bash
python3 -m unittest discover -s tests -v
```

静态站点、bundle 生成和内容校验不依赖 PyTorch。单元测试中的张量前向用例需要 `torch`；未安装或当前平台无法加载时，这些用例会被跳过，marker/bundle 测试仍会运行。需要完整验证时，请按 PyTorch 官方针对当前系统与硬件的说明安装 `torch`。

## 本地浏览

可以直接打开 `index.html`。为避免浏览器对本地脚本或资源施加额外限制，推荐在项目根目录启动静态服务器：

```bash
python3 -m http.server 8000
```

然后访问 `http://localhost:8000/`。

## 文件结构

- `index.html`：发展脉络、课程入口、当前内容统计与 KV cache 计算器
- `chapter.html?id=<id>`：统一章节渲染页
- `assets/chapters.js`：十章教学内容、位置机制、推导、练习与来源
- `assets/diagrams.js`：十套技术报告级 Attention Block SVG 与编号导读
- `assets/implementations.js`：由 `pytorch/` 确定性生成的浏览器代码 bundle
- `assets/course.js`：章节渲染、图解—代码交互工作台、进度与计算器
- `assets/styles.css`：响应式 editorial-academic 设计系统
- `pytorch/`：可导入、可测试的 PyTorch 教学实现
- `tools/`：实现同步器与内容校验器
- `tests/`：同步、marker 和 PyTorch 前向测试
