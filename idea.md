# Agent 负载中的 MoE Expert 预测式预取优化

## 背景介绍

### 场景描述

MoE 模型把 Transformer 中的部分 FFN 层替换为多个专家，每个 token 经过 router 选择少量专家计算。例如 DeepSeek-V4-Flash 是 284B 总参数、每 token 激活 13B（256 routed experts，top-6）；dots3-note Preview 是 280B 总参数、每 token 激活 16B（256 routed experts + 1 shared，top-8）。

MoE 的优势是计算稀疏，但系统问题也来自这里：**需要存储的专家很多，实际激活的专家很少，且激活集合由运行时 router 决定**。

这带来两个典型场景：

- **端侧或显存不足场景**：GPU 内只能保存一部分 experts。未命中的 expert 需要从 CPU、UFS、NVMe 或远端内存加载。router 在模型内部才给出专家选择，加载信息出现得太晚，导致 expert miss stall。
- **服务端显存充足场景**：experts 分布在多个 GPU 上（Expert Parallel）。router 会造成 token dispatch、all-to-all 通信、per-expert 负载不均和小 batch GEMM。问题不是能不能放下 experts，而是如何放置、复制和调度 experts。

因此核心问题是：**系统需要在 router 结果真正出现之前，提前准备即将被访问的 expert 状态**——端侧需要提前加载，服务端需要提前复制热点 expert。

### Agent 负载的特殊性

普通 LLM Serving 看到的是一串独立请求，系统不知道下一个请求是什么。

Agent Serving 不同。一个 Agent Application 由 Orchestrator 调度多个 LLM 请求，形成一个有向无环图（DAG）。每个节点是一次 LLM 调用，节点之间存在依赖关系和条件分支。一个 Agent 节点还可以 spawn 子 Agent，子 Agent 完成后 join 回父 Agent。系统在请求进入模型之前就已经知道：

- 这是哪个 Agent Template（planner、coder、searcher、reviewer 等）；
- 当前在 DAG 中的位置；
- 哪些节点已完成、正在执行或尚未执行；
- 图中哪些后续节点可能接下来运行。

这些信息可以在 router 执行之前就被利用。

### 现有方案的局限

已有 MoE 优化工作（MoE-Infinity、ProMoE、ExpertFlow 等）主要利用：

- token 或 sequence 级别的历史 router trace；
- 当前请求的中间 hidden states（前几层 router 结果预测后续层）；
- 最近窗口内的 expert 访问统计。

这些信息有一个共同问题：**它们都发生在请求内部，或者发生在 expert 已经开始变热之后**。预测信号出现得太晚，对端侧 offload 来说预取 lead time 不足，对服务端来说无法提前做 replica placement。

没有工作从 Agent 层面进行设计——利用"同一个 Agent Template 在不同任务中反复使用"这一事实来做 expert 维度的预测。

## 核心想法

**同一种 Agent 在不同任务中，可能具有稳定的 expert 使用规律。**

一个 coder Agent 无论处理什么具体任务，它的 system prompt、tool schema 和输出风格是固定的，因此它在 MoE 模型中激活的 expert 分布可能呈现跨任务的稳定性。如果这个规律成立，系统就可以在 Agent 节点还没有生成任何 prompt 内容之前，提前预测它会使用哪些 expert，并将这些 expert 从 CPU 搬运到 GPU（端侧），或提前在多 GPU 上复制即将变热的 expert（服务端），把 H2D 时间和计算时间遮蔽掉。

具体地，结合以下三类信息进行预测：

1. **Agent Template 的跨图历史**：同一个 Agent Template 在过去不同 Application 中被调用时积累的 router 轨迹统计；
2. **当前 Application 的执行图结构**：DAG 中节点的依赖关系、已完成节点的信息、后续 ready 节点的组合；
3. **任务类型**：当前 Application 属于哪类任务（coding、search、discussion 等）。

预测对象是：未来时间窗口内，每一层的每个 expert 会被多少 token 使用、大致什么时候开始使用。

主要用途对应两个场景：

- **Prefetch（端侧/内存受限场景）**：预测即将使用的 expert，提前从 CPU 搬运到 GPU，减少 expert miss stall；
- **Proactive Replica Placement（服务端/EP 场景）**：预测即将变热的 expert，提前在多 GPU 上复制，降低 Expert Parallel 中的负载不均和 all-to-all tail latency。

预测器使用轻量 ML 模型（GNN + MLP），利用 Agent 图结构和历史 embedding 预测同一请求内后续 layer 会使用哪些 expert、使用量以及大致使用时间。

## 创新点

### 创新点 1：从 Agent 层面做 expert 预测

原有工作都是在请求粒度和 token 粒度进行预测和缓存优化。本工作首次利用"同一个 Agent Template 在不同 Application 之间是共享的"这一事实，把 expert 预测从"请求内部/router 之后"提前到"request admission 阶段"。这提供了更长的 prefetch lead time。

### 创新点 2：同时利用 Agent 历史和图结构

不仅利用 Agent 自身的历史 router 统计，还利用当前 Application DAG 的结构信息。同一个 coder Agent 放在 planner 后面和放在 reviewer 后面可能有不同的 expert 需求——图结构编码了这种上下文差异。

### 创新点 3：面向 expert cache 的预测式预取与冗余放置

区别于调度层面的优化（Teola、Autellix 等只优化"先执行谁"），本工作优化的是"GPU 里提前放好什么"。在 MoE 服务中，瓶颈已经从"调度谁"变成"GPU 里放不下所有 expert"（端侧）和"哪些 expert 即将过热需要提前复制"（服务端），因此预测未来 demand、提前做 CPU→GPU 搬运或跨 GPU 复制，比单纯优化 batch 更有收益空间。

## 与已有工作的区别

| 工作             | 优化对象                 | 使用信号                       | 与本工作的区别                                     |
| ---------------- | ------------------------ | ------------------------------ | -------------------------------------------------- |
| MoE-Infinity     | Expert offloading        | request-level activation trace | 本工作使用 Agent Template 历史作为更早信号         |
| ProMoE           | Expert proactive caching | 中间 hidden states             | 本工作在 request admission 阶段预测，不等中间结果  |
| ExpertFlow       | Predictive expert cache  | routing path predictor         | 本工作把 Agent 图结构用于 expert demand 预测       |
| Teola / Autellix | Agent 请求调度           | Agent 图和概率                 | 它们优化调度顺序，本工作优化 expert 驻留和复制策略 |
| vLLM EPLB        | Expert replica balance   | 最近负载统计（reactive）       | 本工作用未来 Agent mix 做 proactive placement      |

## 方法概述

### Agent Template Embedding

每个稳定的 Agent Template 维护一个 embedding，由固定描述信息（system prompt、tool schema）的编码加上历史 router 轨迹的修正组成。新 Agent 没有历史时依赖描述编码；随着运行次数增加逐渐使用自己的历史。同一 Agent 在不同任务类型下保存独立的 expert 使用统计。

### 图感知预测

将 Application DAG 和 Agent 节点特征输入轻量 GNN，传播前驱/后继信息后，对每个未来节点预测：

- 是否会执行（分支概率）；
- 大致执行时间区间；
- 每层会使用哪些 expert 以及 token 数量。

### 端侧：Prefetch 决策

根据预测的 expert demand 和当前 GPU cache 状态，计算每个 expert 的未来收益（执行概率 × 使用量 × miss cost），提前将高收益 expert 从 CPU/NVMe 搬运到 GPU。真实 router 仍然执行，预测错误只影响性能不影响正确性——未命中时退化为按需加载。

### 服务端：Proactive Replica Placement

根据预测的未来 Agent mix 和 expert demand，识别即将变热的 expert，在 router 结果出现之前就提前在多 GPU 上复制这些 expert。当真实 router 结果到来时，token 被分派到负载较轻的 replica，降低 all-to-all tail latency 和 per-GPU 负载不均。区别于现有 EPLB 的 reactive 策略（先观察到热再复制），本方法是 proactive 的（根据未来 Agent 图预测热点）。

## 泛化验证设计

| 任务                  | 测试时的新东西                         | 核心问题                      |
| --------------------- | -------------------------------------- | ----------------------------- |
| G1：新任务实例        | 具体输入内容没见过                     | Agent 历史能否跨任务复用      |
| G2：新 Application 图 | 图结构没见过，但 Agent Template 都见过 | 已学的 Agent 能否组合到新图中 |
| G3：新 Agent Template | 某个 Agent 的全部历史都没有            | 冷启动能否在线适应            |

## 实验模型与数据

使用两个 MoE 模型运行 Agent 任务并进行优化验证：

- **DeepSeek-V4-Flash**：284B / 13B active，256 experts，top-6，1M context；
- **dots3-note Preview**：280B / 16B active，256 experts + 1 shared，top-8，512K context。

Agent 负载来源于 AgentX 开源的真实 agentic coding traces（`semianalysisai/cc-traces-weka-062126`），包含 393 个 Claude Code sessions。该数据集提供：

- 完整的 Agent DAG 结构：主 agent 的线性请求链、sub-agent 的 spawn/join 拓扑、并行分支和依赖关系；
- 每个请求的元信息：agent_id（区分主 agent 和各 sub-agent）、请求时间戳、input/output token 数、inter-turn delay；
- KV-cache block hash（用于还原 prefix 复用结构）。

将这些 traces replay 到上述两个 MoE 模型上，采集每层的 router logits 和 expert 选择结果，构建带有 Agent 元数据标注的 router trace 数据集。在此基础上验证 Agent-conditioned expert 预测和预取/冗余放置的效果。

## 预期收益

- **端侧**：expert miss stall 下降，预测准确时 expert 已在 GPU 中等待，H2D 搬运时间被计算时间遮蔽；
- **服务端**：hot expert replica 提前到位，all-to-all tail latency 下降，per-GPU 负载更均衡；
- **正确性保证**：真实 router 没有被替代，模型行为不变，输出完全一致。