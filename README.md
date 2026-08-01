# Attention Atlas

面向大模型算法工程师的中文注意力架构课程。按架构演化讲解：

`MHA → MQA → GQA → MLA → DSA → CSA/HCA`

其中 CSA 与 HCA 分为独立章节。并行分支为：

`Linear Attention → DeltaNet / Gated DeltaNet → KDA → Hybrid Attention`

## 使用

直接打开 `index.html`，或在本目录启动任意静态文件服务器：

```bash
python3 -m http.server 8000
```

然后访问 `http://localhost:8000/`。

## 文件

- `index.html`：发展脉络、课程入口、KV cache 计算器
- `chapter.html?id=<id>`：统一章节渲染页
- `assets/chapters.js`：十章的教学内容、公式、练习与来源
- `assets/diagrams.js`：十套技术报告级 Attention Block SVG 与编号导读
- `assets/course.js`：章节渲染、SVG 图解、进度与计算器
- `assets/styles.css`：响应式 editorial-academic 设计系统

学习进度只保存在浏览器 `localStorage`，不会上传。
