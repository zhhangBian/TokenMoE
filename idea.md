# 基于 Agent Application 图的 MoE Expert 预测与预取

主要的创新点为：

- **将 Agent 作为可跨任务复用的逻辑模板。** 对每个 Agent Template 保存跨任务、跨 Application 的历史 expert 访问规律，并使用 Agent 描述生成的 embedding 支持冷启动。
- **利用整个 Application 图预测尚未执行的 Agent 调用。** 不只根据当前 token、当前 prompt 或最近请求预测 expert，而是在未来节点的 prompt 尚未完全生成时，根据 Agent Template、任务类型、图结构、条件分支和已经完成的前驱节点，预测未来每层会使用哪些 expert、使用多少、何时首次使用。
- **把预测结果用于 CPU 到单 GPU 的 expert cache 和 prefetch。** 预测器不修改 MoE router，只决定哪些 expert 应提前传入 GPU、保留或淘汰；预测错误只影响性能，不影响模型输出。

本文档是整体研究指引，不受到当前代码实现的限制。第一阶段只研究**静态可见、带条件分支的 Agent Application 图**，并把系统动作锁定为**单 GPU 场景下的 expert cache 和 prefetch**。动态图、跨模型直接迁移和强化学习控制器均作为后续扩展。

真正需要证明的是：**在任务类型、当前可见 prompt 和历史 routing 信息都已经给预测器之后，Agent Template 和真实图结构是否仍然提供额外信息；这些信息是否比现有方法更早出现，并最终减少真实的 expert 加载等待。**

# MoE Expert Cache 和 Agent Application

## MoE expert offload

Mixture-of-Experts 模型在每个 MoE 层中包含多个 expert。对每个 token，模型自身的 router 只选择少量 expert 计算，因此单次计算是稀疏的，但所有 expert 权重的总量仍然很大。

当 router 选择的 expert 不在 GPU 中时，当前 token 必须等待权重传输完成。这个等待称为 `expert miss stall`。普通 LRU 只能在 expert 已经被使用之后更新缓存，无法利用 Agent 程序中预先暴露的未来执行结构。

prefetch 的作用是：在未来调用真正到达 MoE 层之前，把可能使用的 expert 传入 GPU。如果传输能够与前驱 Agent、工具调用、其他 Application 或当前模型计算重叠，那么原本处于关键路径上的传输可以被隐藏。

这里有两个必须同时满足的条件：

- 未来 expert 需要能够被预测；
- 预测必须足够早，早到可以覆盖 CPU 到 GPU 的传输时间。

## Agent Application

为了避免“Agent”“任务”“节点”混用，全文统一使用下面的术语。

| 术语 | 定义 | 例子 |
| --- | --- | --- |
| Application Template | 预先注册的静态条件有向无环图，描述一类 Agent 应用的节点、边、并行关系和条件分支 | 代码修复流程、研究报告流程 |
| Application Instance | 某个 Application Template 在一个具体任务上的一次完整运行，也称 Episode | 修复某一个代码问题 |
| Agent Template | 跨任务稳定的逻辑 Agent，具有固定 role、system prompt、tool schema 和版本 | planner、coder、tester |
| Agent Invocation | 一个 Agent Template 在某个 Application Instance 中的一次实际执行 | coder 在第 17 个任务中的调用 |
| Task Type | 运行前直接提供的离散任务标签 | code repair、research、data analysis |
| Template Memory | 从过去已完成的 Agent Invocation 中积累的 expert routing 统计和学习表示 | coder 在不同任务中的逐层 expert 分布 |

本文还会反复使用以下词：

| 词 | 直接含义 |
| --- | --- |
| frontier | 当前依赖已经满足、可以进入执行队列的节点集合 |
| residual | 在通用描述或全局统计之上，针对某个 Agent/Application 学到的修正量 |
| prediction head | 共享模型后面专门输出某一种目标的小型输出层 |
| oracle | 允许读取真实未来的不可部署上界，用来判断系统最多能获益多少 |
| Belady | 知道完整未来访问序列的最优淘汰策略；这里只优化 eviction，不允许提前加载 |
| makespan | 一个完整 Application Instance 从开始到全部结束的时间 |
| cumulative regret | 在线前若干次运行累计比参考策略多付出的 stall 或 makespan |
| JSD | 两个 expert 概率分布的差异，0 表示完全相同，越大表示差异越明显 |
| NLL | 模型给真实结果分配的负对数概率，越低越好 |
| Brier score | 预测概率与真实 0/1 结果的均方差，越低越好 |
| ECE | 预测置信度与真实命中率之间的校准误差，越低越好 |
| CDF | “不超过某个横轴值”的样本比例曲线 |

一个 Application Template 记为：

$$
G=(V,E)
$$

其中：

- $V^{agent}$ 是 Agent Invocation 节点，通过 $a(v)$ 关联稳定的 Agent Template，并产生 MoE expert demand；
- $V^{tool}$ 是工具、CPU 函数或外部服务的 timing node，不产生 expert demand，但其运行时间会改变后继 Agent 的 ready time；
- $V=V^{agent}\cup V^{tool}$；
- 每条边 $u\rightarrow v$ 表示 $v$ 依赖 $u$ 的结果。

如果某个 Agent framework 不把工具调用暴露为独立节点，至少需要把工具类型、开始/结束时间和结果可用时间记录为边 delay。工具等待通常是最有价值的预取窗口，不能在图模型中消失。

图在运行前可见，但不代表所有未来输入都可见。未来节点的固定 system prompt、tool schema、role 和位置已经知道；前驱 Agent 的输出、工具结果和由它们拼接出的动态 prompt 尚未产生。

图允许包含：

- 条件分支：某条边或某个节点只在条件满足时执行；
- 并行节点：多个节点可以同时 ready；
- 多个并发 Application Instance：不同任务的图同时运行；
- 相同 Agent Template 的复用：同一个 planner、reviewer 或 coder 可以出现在多个 Application Template 中。

第一阶段不支持运行中新增未知节点或未知边。分支的具体结果可以在运行时确定，但所有候选节点和候选边需要在运行前注册。

每个条件分支还需要在 Application Template 中注册明确语义：

- 一个 split 是互斥单选、多选，还是多个独立条件；
- 一个 join 等待所有已选择前驱，还是任一前驱；
- 每个节点由什么 branch guard 决定是否运行。

这些语义由真实 Application runtime 执行，预测器只估计尚未确定的分支结果，不能自行改变图的控制流。

## Agent 为什么可能有用

当前请求的 prompt 确实会直接影响 router，因此不能把 Agent metadata 当成 prompt 的替代品。Agent 的价值主要来自三个更早、更稳定的信号。

第一，**Agent Template 跨任务重复出现**。例如 coder 的具体代码输入每次不同，但固定 system prompt、工具集合、输出格式和工作目标相似。历史 Invocation 可能形成可复用的 expert 访问先验。

第二，**Application 图在未来请求可独立调度之前已经可见**。即使后继节点的动态 prompt 尚未生成，也已经知道它可能由哪个 Agent Template 执行、位于哪个分支、依赖哪些前驱以及距离当前执行还有多少步。

第三，**图结构决定需求到达的顺序和概率**。相同的 Agent 集合以不同依赖边组合时，未来 expert 的总集合可能接近，但首次使用时间、紧迫程度和并发冲突会不同。这些差异正是 cache 和 prefetch 需要的信息。

因此，这项工作不应该只预测“某个 Agent 常用哪些 expert”，而应预测：

```text
未来哪个 Agent 节点会运行
  -> 大约何时运行
  -> 每一层会使用哪些 expert
  -> 每个 expert 会接收多少 token
  -> 该 expert 最晚应在何时进入 GPU
```

# 之前工作已经做到什么

现有工作已经覆盖了 expert 预测、任务相关 expert 热度和 Agent 图资源管理的多个部分，因此不能把贡献写成“第一次预测 expert”或“第一次利用 Agent 图做 serving”。

## 当前请求内部的 expert 预测

[SiDA-MoE](https://proceedings.mlsys.org/paper_files/paper/2024/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) 和 [ExpertFlow](https://arxiv.org/abs/2410.17954) 已经说明，可以根据当前输入在实际 MoE 计算之前预测 routing path，并用于 expert 预取或 CPU-GPU 管理。[FineMoE](https://arxiv.org/abs/2502.05370) 也使用 prompt semantic hints 和历史 expert map。因此，不能声称现有方法都只能在 router 执行后反应。

[Patterns behind Chaos](https://arxiv.org/abs/2510.05497) 系统分析了 MoE routing 的时间和空间规律：

- 相邻层之间存在 expert 条件相关性；
- 相邻 token 之间存在 expert 复用；
- prefill 的 expert 分布可以提示同一请求后续 decode 的分布；
- 不同任务和语言具有不同的热门 expert；
- expert 之间存在稳定的共同激活关系。

这篇工作给本项目的重要启发是：现有预测的时间尺度主要位于**同一个请求内部**，包括层、token 和 prefill 到 decode。本工作要增加的是更高一层的时间尺度：

```text
layer -> token -> prefill/decode -> Agent Invocation -> Application graph
```

它同时构成一个强 baseline：任务类型热门 expert、相邻层、跨 token 和 prefill 到 decode 的预测都必须加入比较。**任务标签指导 expert prefetch 本身已经不是创新点。**

这篇工作可以被本文具体复用为：

1. 用 layer、token、prefill/decode 三个请求内时间尺度作为 baseline，再增加 Agent Invocation 和 Application graph 两个跨请求尺度；
2. 实现基于条件频率表的轻量 predictor，而不是只比较 LRU；
3. 使用 expert 共同激活关系初始化或正则化 layer-expert embedding；
4. 使用 prefill 预测同一请求 decode，和本文预测后继 Agent Invocation 形成时间边界清楚的对比；
5. 按 Task Type、语言、层和 prefill/decode 分层报告结果，避免总体热门 expert 掩盖差异。

它公开的普通请求 routing trace 可以用于复现请求内 baseline 或预训练 layer-expert 表示，但其中没有 Agent Template、Application 图、分支、节点时间和多 Application 并发信息，不能用于证明本文的 Agent 图结论。

可直接参考的 artifact 包括：

- [expert selection trace](https://huggingface.co/datasets/core12345/MoE_expert_selection_trace)：用于复现 layer/token/prefill-decode baseline；
- [trace analysis](https://huggingface.co/datasets/core12345/MoE_trace_analysis)：用于核对 cross-layer、cross-token 和 co-activation 统计；
- [wafer-scale simulator](https://github.com/zhongkaiyu/waferscale_gpu_moe_sim)：可借用 trace parser、conditional-frequency predictor 和 oracle/learned predictor 的实验组织，不能照搬硬件模型；
- [real-GPU artifact](https://github.com/zhongkaiyu/moe_exp_placement)：可参考 router instrumentation 和 placement 实现。

其中基于当前 expert 查询 cross-token 条件表、再选 $\lceil0.2E\rceil$ 个候选 expert 的轻量 predictor，应作为明确 baseline。该论文的真实 GPU case 面向 8×H100 上的 prefill-aware placement，不是单 GPU CPU expert offload，其加速数字不能直接外推到本文场景。

## Agent 图上的资源预测

[KVFlow](https://arxiv.org/abs/2507.07400) 利用 Agent Step Graph 预测步骤距离，用于 KV cache 淘汰和预取；[PBKV](https://arxiv.org/abs/2605.06472) 预测未来 Agent 调用，用于动态 workflow 的 KV 管理；[Pythia](https://arxiv.org/abs/2604.25899) 从 workflow 和 role 预测调用路径、输出长度和资源需求。

这些工作说明 Agent 图确实可以提供未来资源信息，但它们没有回答本文的具体问题：**能否用 Agent Template 和图上下文预测冻结 MoE router 在未来调用中的逐层 expert 需求。**

因此，Agent ID、embedding、概率图遍历、GNN、路径预测和主动 warmup 都不能单独作为 novelty。本文可能成立的交叉点只能是：**原生逐层 MoE expert first-use demand、G2/G3 泛化，以及 CPU 到单 GPU 的 expert cache/prefetch。**

[Sem-MoE](https://arxiv.org/abs/2503.04398) 已经使用 token-expert affinity 进行请求聚合和 expert placement，因此 expert-aware scheduling 也不是本文贡献。[PA-MoE](https://arxiv.org/abs/2602.17038) 使用 Agent trajectory phase 影响模型内 expert assignment，但它训练或改变路由行为；本文只预测冻结模型原生 router 的未来需求。

## 本工作的准确边界

本文研究的是：给定一个已经注册、正在执行的静态条件 Agent 图，使用跨任务积累的 Agent Template 记忆和当前可见的图状态，在未来 Agent 调用的动态 prompt 尚未完全生成时，预测固定 MoE 模型原生 router 的逐层 expert 需求，并据此执行 CPU 到单 GPU 的 expert cache 和 prefetch。

固定 MoE checkpoint、router 实现和 token IDs 后，routing 由模型计算决定。完整 prompt 已经生成时，Agent metadata 通常只是 prompt 的子集或低成本代理，不能被描述成信息论上“超越完整 prompt”的新信息。本文的强结论严格限定在**未来 prompt 尚未物化**的阶段；prompt-ready 之后只比较不同信号在预测成本、准确率和可用提前量上的权衡。

本文不声称：

- 首次进行 expert prediction；
- 首次根据任务类型预测热门 expert；
- 首次使用 Agent 图管理系统资源；
- 首次进行 expert-aware request scheduling；
- Agent embedding 可以直接迁移到另一个 MoE 模型；
- 预测器可以替代或修改模型 router。

如果实验最终只表明 Agent Template 与 prompt 内容相关，而真实图边没有额外价值，那么“graph-conditioned”不能作为贡献。如果预测提升不能转化为真实加载等待或端到端延迟下降，那么这只能是一项 routing characterization，而不是完整的系统工作。

# 整体思路梳理

系统的完整流程为：

```text
运行前注册：
  Application Template
  + Agent Template 描述
  + 离散 Task Type

运行中可见：
  已完成节点及其真实 routing trace
  + 已确定的分支
  + ready/running/pending 状态
  + 当前 expert cache 和传输状态

预测器：
  Agent Template Memory
  + Agent embedding
  + 有向图编码
  -> 未来节点运行概率
  -> 未来节点到达时间
  -> 每层 expert 出现概率和 token 数
  -> 首次使用时间与不确定性

控制器：
  汇总所有并发 Application 的未来需求
  -> prefetch / keep / evict
  -> CPU 到 GPU 异步传输

执行与更新：
  真实 MoE router 正常执行
  -> 记录真实 expert 使用
  -> 先评估本次预测，再更新 Template Memory
  -> 对剩余图重新预测
```

这个设计中，预测器和控制器需要分开。

- 预测器只回答“未来需求是什么”；
- 控制器根据显存、带宽和 deadline 决定“现在做什么”；
- 真实 router 始终回答“当前 token 实际去哪个 expert”。

这样可以分别判断：

1. Agent 图是否真的提高预测；
2. 即使预测正确，当前硬件是否存在可利用的 prefetch 空间；
3. 系统收益来自更好的信息，还是来自不同的 cache 策略。

# G1、G2、G3 三项研究任务

G1、G2、G3 不是三个附加实验，而是这项工作的三种基本泛化能力。方法设计、数据切分和最终结论都围绕它们展开。

| 任务 | 测试时什么是新的 | 测试时什么已经见过 | 主要问题 |
| --- | --- | --- | --- |
| G1 | 新的具体任务和新的 Application Instance | Application Template、Agent Templates、Task Type | 同一应用中的历史能否帮助新任务 |
| G2 | 完整的新 Application Template，即新的图组合 | 图中所有 Agent Templates 都在其他 Application 中见过 | Agent 记忆能否在新图中组合使用 |
| G3 | 至少一个新的 Agent Template | MoE 模型、任务标签、其他 Agent、图结构描述 | 新 Agent 能否靠描述冷启动并在线学习 |

## G1：已见 Application 的新任务

训练数据来自某个已知 Application Template 的历史任务实例，测试数据来自同一个 Application Template 的新任务。

例如：

```text
训练：
  code-repair graph 上的 issue 1 ... issue N

测试：
  同一个 code-repair graph 上未见过的 issue N+1
```

测试时允许使用：

- 完整静态图；
- Agent Template 身份和描述；
- 训练阶段积累的 Template Memory；
- 当前任务的离散 Task Type；
- 当前时刻已经真实产生的信息。

测试时禁止使用：

- 当前测试 Episode 后续节点的真实 router trace；
- 尚未产生的工具输出和 Agent 输出；
- 从同一测试 Episode 未来调用中统计出的 expert 分布。

G1 主要证明跨任务稳定性。如果 G1 失败，说明同一 Agent Template 在不同任务上的 routing 差异过大，Agent 历史很难作为可靠先验。

## G2：已见 Agent 的新图组合

测试时留出一个完整的 Application Template，但这个新图中的每个 Agent Template 都在其他 Application 中出现过。

例如：

```text
训练图：
  planner -> searcher -> writer
  planner -> coder -> reviewer

测试图：
  planner -> {searcher, coder} -> reviewer -> writer
```

G2 的关键不是简单地把一个新图加入训练集，而是测试**组合泛化**：

- planner 的历史来自多个旧 Application；
- reviewer 的历史来自多个旧 Application；
- 新图注册后，预测器需要根据新的边、并行关系和分支重新估计到达顺序；
- 新图本身的专属历史在第一次运行时为零。

为避免 GNN 只记住 Agent 集合，G2 必须包含：

- 相同或高度相似的节点集合；
- 不同的真实依赖边；
- 不同的并行和 join 关系；
- 至少一组不同的条件分支。

G2 是最能体现 Agent 图意义的任务。G2 成功意味着跨 Application 保存的 Agent Template Memory 可以被新的图结构重新组合，而不是只能记住某个固定 workflow。

## G3：新 Agent Template 的冷启动

测试时出现训练期间完全未见的 Agent Template。该 Agent 没有可查表的历史 residual，只能依赖它的稳定描述：

- role；
- system prompt；
- tool schema；
- 输入输出格式；
- 模板版本；
- Task Type；
- 图中的前驱、后继和位置。

G3 应在已见或严格配对的 Application 图上进行，保持图拓扑、Task Type 和其他 Agent 尽量不变，只替换或留出目标 Agent Template。不能同时使用一个全新的图和一个全新的 Agent，否则无法判断失败来自 G2 还是 G3。

G3 再分成：

- G3-a：图中的结构位置和能力类别已见，只把该位置替换为 descriptor 不同的新 Agent Template；这是冷启动主结果；
- G3-b：Agent 的 role 或能力类别也未见；这是更难的开放集结果；
- new Agent + new graph：只作为 compound stress test，不用于单独支持 G2 或 G3。

G3 需要报告两部分：

1. 第一次调用前的 `K=0` 冷启动能力；
2. 完成 `K=1,2,4,8,16` 次真实调用后的在线学习曲线。

G3 的目标不是要求一个未见 Agent 立即达到已见 Agent 的准确率，而是回答：

- descriptor embedding 是否优于全局或任务级 fallback；
- 需要多少次调用才能接近已见 Agent；
- 冷启动期间的错误会造成多少额外 stall 和带宽浪费；
- 在线记忆的累计收益何时超过学习成本。

## G4：更换 MoE 模型

更换 MoE 模型后，expert 的数量、功能和 router 都可能变化。本文**不承诺 expert embedding 或 Template Memory 直接跨模型迁移**。

G4 只作为外部有效性检查：对每个新 MoE 模型重新采集 routing trace、重新训练 expert 相关参数，再检查 G1、G2、G3 的现象是否仍然存在。

# 问题定义

## 当前时刻允许看到什么

在时间 $\tau$，系统可见状态拆成需求预测状态 $S_\tau^{forecast}$ 和控制状态 $S_\tau^{control}$。它们都只包含因果上已经可用的信息：

$$
S_\tau^{forecast} =
\{
G,\ c,\ \text{initial task input},\
\text{Agent descriptors},\
\text{visible prompt parts}_{\leq\tau},\
\text{observed predecessor/tool outputs}_{\leq\tau},\
\text{node status}_{\leq\tau},\
\text{observed branch outcomes}_{\leq\tau},\
\text{completed routing traces}_{<\tau},\
\text{queue and timing history}_{\leq\tau}
\}
$$

控制器另外可见：

$$
S_\tau^{control}
=
\{
\text{cache state}_{\tau},\
\text{H2D state}_{\tau},\
\text{GPU memory budget}_{\tau}
\}
$$

后续公式中的 $S_\tau$ 表示二者组成的完整可见状态；expert demand predictor 只读取 $S_\tau^{forecast}$，controller 再读取预测结果和 $S_\tau^{control}$。

其中 $c$ 是运行前给出的 Task Type。

如果某个未来节点的完整 prompt 在当前时刻已经能够合法生成，那么它必须提供给 prompt baseline，也可以提供给完整模型。不能为了突出 Agent 图而人为隐藏已经可见的 prompt。

如果 prompt 仍依赖未完成的前驱输出，则只能使用其固定模板、已知静态部分和图上下文，不能使用未来真实内容。

每一个输入字段都需要记录 `available_time`。训练、离线 replay 和在线推理都按同一时间戳遮蔽未来信息。

## 未来节点的随机变量

对一个尚未完成的节点 $v$、MoE 层 $l$ 和 expert $e$，定义：

- $C_j$：第 $j$ 个条件 split 的离散分支结果；
- $R_v$：节点最终是否执行；
- $Q_v$：节点依赖满足、进入 frontier 的 ready time；
- $E_v$：节点被固定请求调度器放入模型队列的 enqueue time；
- $T_v$：节点真正开始模型推理的 start time；
- $Z_{v,l,e}$：该节点是否至少使用一次 expert $e$；
- $N_{v,l,e}$：在 $Z_{v,l,e}=1$ 时路由到 expert $e$ 的 token 数；
- $F_{v,l,e}$：从节点开始到第一次使用 expert $e$ 的时间。

expert 的绝对首次使用时间为：

$$
U_{v,l,e}=T_v+F_{v,l,e}
$$

每个节点的运行事件不是独立变量。Application Template 为节点注册 guard $g_v$，因此：

$$
R_v=g_v(C_1,C_2,\ldots)
$$

互斥分支使用一个 categorical outcome，而不是对每条边独立做 Bernoulli 预测。AND/OR join 按 Application Template 的真实语义计算。对于简单 DAG，可以通过动态规划得到节点 reach probability；存在相关分支时，对分支联合分布采样或显式枚举，不能用独立概率乘积破坏控制流约束。

prefill 和 decode 的需求规律可能不同，因此实际记录中为这些变量增加阶段 $s\in\{\text{prefill},\text{decode}\}$。正文中的公式省略 $s$，实现与实验中分开预测和报告。

系统真正需要的不是单个节点的 top-k expert 列表，而是所有并发 Application 中，未来哪些逐层 expert 会在什么 deadline 前首次被需要。

将 expert 的未来绝对首次使用时间划分为若干 bucket $b$，定义“按首次使用 deadline 分组的需求”：

$$
D^{deadline,(i)}_{\tau,b,l,e}
=
\sum_{v\in V_i^{agent}}
R_v\,
\mathbf{1}(U_{v,l,e}-\tau\in b)\,
Z_{v,l,e}\,
N_{v,l,e}
$$

这里 $i$ 表示当前某一个 Application Instance。这个量表示：从当前时刻看，该实例中在未来时间窗口 $b$ 内第一次需要第 $l$ 层 expert $e$ 的所有节点，最终会向它路由多少 token。多个并发实例在 controller 一节统一聚合。

预测器输出其条件期望。一个可解释的近似分解为：

$$
\widehat D^{deadline,(i)}_{\tau,b,l,e}
=
\sum_{v\in V_i^{agent}}
\widehat p^{run}_v
\cdot
\widehat p^{use}_{v,l,e}
\cdot
\widehat p^{first}_{v,l,e,b}
\cdot
\widehat \mu^{tokens\mid use}_{v,l,e}
$$

其中：

- $\widehat p^{run}_v=P(R_v=1\mid S_\tau^{forecast})$，由分支分布和图 guard 推导；
- $\widehat p^{use}_{v,l,e}=P(Z_{v,l,e}=1\mid R_v=1,S_\tau^{forecast})$；
- $\widehat p^{first}_{v,l,e,b}=P(U_{v,l,e}-\tau\in b\mid R_v=1,Z_{v,l,e}=1,S_\tau^{forecast})$；
- $\widehat \mu^{tokens\mid use}_{v,l,e}=E[N_{v,l,e}\mid R_v=1,Z_{v,l,e}=1,S_\tau^{forecast}]$。

这个量把一次 Invocation 的总 token 数归到该 expert 的首次使用 bucket，用于衡量“是否值得在第一次使用前搬入”，**不是严格的逐 token 时间窗访问量**。第一阶段中：

- ML 预测只决定 pre-start prefetch，以及 Invocation 之间是否保留某个 expert；
- 一个 Invocation 开始使用某 expert 后，运行中 residency 使用所有方法共享的 refcount/LRU 策略；
- 不使用这个公式声称能够预测 Invocation 内部每个 decode 时刻的 eviction；
- 若未来要让 ML 决定 Invocation 内 keep/evict，必须另外预测逐时间窗访问量或 last-use time。

实际模型同时预测节点开始时间 $T_v$ 和 expert 相对首次使用时间 $F_{v,l,e}$，再通过二者分布的卷积得到 $U_{v,l,e}$；也可以直接预测绝对 first-use bucket。无论采用哪种实现，都必须保留 branch/reach、node start、expert use、first use 和 count 五个可解释输出，方便检查错误来自哪里。

## 预测目标

对每个候选未来节点，预测器至少输出：

```text
1. 每个尚未确定的条件 split 会选择什么分支
2. 由图 guard 推导出的节点运行概率
3. 节点在哪个时间窗口开始
4. 每层每个 expert 是否至少使用一次
5. 使用时的期望 token 数
6. 第一次使用该 expert 的绝对时间或 deadline
7. 每个预测的置信度
```

只预测热门 expert 身份不够。对 prefetch 而言，一个概率稍低但即将使用、加载代价很高的 expert，可能比一个概率高但很晚才使用的 expert 更值得优先传输。

# Agent 层面的 ML 预测

## Agent Template Memory

每个 Agent Template $a$ 都有稳定描述 $d_a$，包括 role、固定 system prompt、tool schema、输入输出格式和模板版本。

Agent 表示由两部分组成：

$$
z_a
=
f_\theta(d_a)
+
\frac{n_a}{n_a+\kappa}r_a
$$

其中：

- $f_\theta(d_a)$ 是从描述生成的通用 embedding；
- $r_a$ 是根据训练集历史 routing 学到的慢速专属 residual；
- $n_a$ 是训练阶段该 Agent 的历史调用数；
- $\kappa$ 控制需要多少训练样本才信任慢速 residual。

运行期间还维护一个不需要 SGD 的快速 memory $m_a^\tau$：

$$
z_a^\tau
=
f_\theta(d_a)
+
\frac{n_a^{train}}{n_a^{train}+\kappa_s}r_a^{slow}
+
\frac{n_a^{online}}{n_a^{online}+\kappa_f}g_\phi(m_a^\tau)
$$

$m_a^\tau$ 包含逐层 expert 计数、EMA token 数、运行时间、分支统计和最近相似 Invocation。$g_\phi$ 在训练阶段学会如何编码这些统计，测试时冻结；每次真实调用后只更新统计量。因此 G2/G3 的 `K=1,2,...` 提升有明确来源，不依赖测试时偷偷重新训练整个网络。

这个设计直接对应 G1、G2、G3：

- G1 中，描述、慢速 residual 和快速 memory 都可用；
- G2 中，同一个 Agent 的表示可以跨 Application 复用；
- G3 的第一次调用满足 $n_a^{train}=n_a^{online}=0$，只能使用描述 embedding；
- 随着在线样本增加，模型逐步信任新 Agent 的快速 memory；
- 周期性离线训练可以把快速 memory 吸收到新的慢速 residual，但这不计入同一轮 K-shot 测试。

不能直接为每个运行实例建立独立 ID embedding。那样只能记住训练任务，无法处理新任务、新图或新 Agent。

## Task Type 条件统计

由于任务类型在运行前已知，并且已有工作已经证明任务会影响 expert 热度，最简单的预测器应先维护 Task Type 条件分布。

对 Agent $a$、任务类型 $c$、层 $l$、expert $e$，分别维护“是否使用”和“使用多少”两套统计，不能把 token 计数与 Invocation 计数混为一个量。

expert 使用概率采用层次平滑：

$$
\pi^{use}_{a,c,l,e}
=
\frac{C^{use}_{a,c,l,e}+\alpha\pi^{use}_{c,l,e}}
{C^{inv}_{a,c,l}+\alpha}
$$

条件 token 数采用：

$$
\mu^{tokens\mid use}_{a,c,l,e}
=
\frac{S^{tokens}_{a,c,l,e}+\beta\mu^{tokens\mid use}_{c,l,e}}
{C^{use}_{a,c,l,e}+\beta}
$$

其中：

- $C^{inv}_{a,c,l}$ 是该 Agent 在该任务类型和层上的历史 Invocation 数；
- $C^{use}_{a,c,l,e}$ 是其中至少使用一次 expert $e$ 的 Invocation 数；
- $S^{tokens}_{a,c,l,e}$ 是这些 Invocation 路由给 expert $e$ 的总 token 数；
- $\alpha,\beta$ 决定小样本时向任务级先验回退的强度。

这个统计模型不是最终方法，而是重要的第一层 baseline。复杂 ML 模型至少要稳定优于它，否则没有必要引入 GNN。

所有 Template Memory 都按下面的 scope 隔离：

```text
MoE checkpoint / model ID
router 实现与版本
权重精度或量化方式
tokenizer 与 chat template 版本
Agent Template 版本
Task Type
```

任一项改变都不能无条件复用旧的 expert 统计。更换 MoE checkpoint 时默认新建 memory 并重新训练。

## 跨 Application 的两级记忆

同一个 Agent Template 在不同 Application 中通常相似，但它的输入来源和图位置会改变具体需求。因此使用两级记忆：

$$
M(a,A,c)
=
M_{\text{global}}(a,c)
+
\frac{n_{a,A}^{train}}{n_{a,A}^{train}+\kappa_A}
\Delta M_{\text{app}}(a,A,c)
+
\frac{n_{a,A}^{online}}{n_{a,A}^{online}+\kappa_{fast}}
g_A(m_{a,A}^\tau)
$$

其中：

- $M_{\text{global}}$ 聚合该 Agent 在所有历史 Application 中的信息；
- $\Delta M_{\text{app}}$ 表示它在某个 Application 中的偏差；
- $m_{a,A}^\tau$ 是无需梯度更新的快速 Application 统计；
- 新 Application 第一次运行时两个计数都为 0，只使用 global memory；
- 新图运行若干次后，先通过快速统计适配；周期性训练时才更新慢速 application residual。

这个结构是 G2 的核心。如果只为每个 Application 单独建表，新图永远冷启动；如果只用全局 Agent 统计，又会忽略新图带来的上下文差异。

## 节点输入

对当前图中的节点 $v$，初始特征 $x_v^\tau$ 包括：

```text
Agent 信息：
  descriptor embedding
  template residual
  跨 Application 的历史 routing memory

任务信息：
  离散 Task Type embedding
  运行前已经可见的初始任务输入 embedding

图信息：
  入度、出度、拓扑深度
  距离当前 frontier 的步数
  是否在并行分支、join 或条件分支
  前驱和后继 Agent Template

运行状态：
  pending / ready / running / completed / skipped
  已确认分支
  已完成前驱输出或工具结果的 causal embedding
  已完成前驱的真实 expert 摘要和运行时间

内容信息：
  当前时刻已经可见的 prompt 静态部分
  若完整 prompt 已合法生成，则加入完整 prompt embedding

时间信息：
  当前时间、队列负载
  ready / enqueue / start 的历史
  已完成节点的实际运行时间
```

预测需求与系统状态需要解耦。节点的 expert 身份和 token 数 head 只使用 Agent、任务、内容和图；节点到达时间 head 可以使用队列负载和已完成节点时间；expert resident/loading 状态只进入 controller，不进入 expert demand predictor，避免缓存策略反过来污染“真实会使用什么 expert”的预测。

Tool timing node 没有 Agent embedding。它的特征只包括 tool/service 类型、固定 schema、历史时长分布、当前状态和已经可见的结果摘要；其 expert use/count/first-use loss 全部 mask，只参与图的时间传播和条件分支预测。

## 有向图编码

第一阶段采用简单的两层有向 GraphSAGE 或 GAT，不直接使用大型异构图模型。目标是先证明真实边提供信息，而不是依靠模型复杂度获得小幅提升。

一层消息传递可以写为：

$$
h_v^{(k+1)}
=
\sigma\left(
W_0h_v^{(k)}
+
W_{in}\operatorname{AGG}_{u\rightarrow v}h_u^{(k)}
+
W_{out}\operatorname{AGG}_{v\rightarrow w}h_w^{(k)}
+
W_g g_\tau
\right)
$$

其中：

- $h_v^{(0)}=x_v^\tau$；
- 入边聚合已经或可能向当前节点提供内容的前驱；
- 出边聚合当前节点在未来图中的作用；
- $g_\tau$ 是当前 Application 和全局运行状态；
- 所有运行时特征都经过时间遮蔽。

需要同时实现一个忽略边方向的 `node-set model`。它看到完全相同的节点和节点特征，但看不到真实边，用来判断 GNN 的收益究竟来自图结构还是仅仅来自 Agent 列表。

## Layer-Expert embedding 和预测头

对固定 MoE 模型中的每个 $(l,e)$ 学习一个 layer-expert embedding $q_{l,e}$。它只表示该模型中的 expert，不承诺跨模型迁移。

对节点最终表示 $h_v$，使用以下输出头：

条件 split $j$ 的分支结果：

$$
\widehat p_j^{branch}
=
\operatorname{softmax}(W_j h_{\operatorname{split}(j)})
$$

上式用于互斥单选 split；多选或独立条件使用 sigmoid multi-label head。分支组类型由 Application Template 给出。

节点运行概率 $\widehat p_v^{run}$ 由所有相关 $\widehat p_j^{branch}$ 和图 guard $g_v$ 推导，不使用互相矛盾的独立 node classifier。

节点开始时间窗口：

$$
\widehat p_{v}^{time}
=
\operatorname{softmax}(W_t h_v)
$$

expert 至少使用一次的概率：

$$
\widehat p_{v,l,e}^{use}
=
\operatorname{sigmoid}(h_v^\top q_{l,e}+b_{l,e})
$$

expert 在确定使用时的 token 数：

$$
\widehat\mu_{v,l,e}^{tokens\mid use}
=
\operatorname{softplus}
\left(
g_\mu(h_v,q_{l,e})
\right)
$$

首次使用时间可以预测为离散 bucket，或者使用 survival head 预测在 deadline 前尚未使用的概率。第一阶段优先使用离散 bucket，因为它容易校准，也能直接对应 prefetch 时间窗。

first-use head 预测 $F_{v,l,e}$ 或 $U_{v,l,e}$：

$$
\widehat p_{v,l,e}^{first}
=
\operatorname{softmax}
\left(
g_f(h_v,q_{l,e})
\right)
$$

factorized 的 node embedding 与 layer-expert embedding 点积，比直接输出一个固定巨型向量更容易共享 Agent 表示，也更容易分析哪些 Agent 与哪些 expert 相似。

## 训练目标

整体使用监督学习，而不是让强化学习同时承担 expert 预测。

$$
\mathcal{L}
=
\lambda_b\mathcal{L}_{branch}
+
\lambda_t\mathcal{L}_{time}
+
\lambda_z\mathcal{L}_{use}
+
\lambda_n\mathcal{L}_{count}
+
\lambda_f\mathcal{L}_{first}
+
\lambda_c\mathcal{L}_{calibration}
$$

具体为：

- $\mathcal{L}_{branch}$：每个条件 split 真实结果的 categorical 或 multi-label loss；
- $\mathcal{L}_{time}$：时间 bucket 的交叉熵或生存损失；
- $\mathcal{L}_{use}$：逐层逐 expert 的多标签二元交叉熵；
- $\mathcal{L}_{count}$：Poisson 或 Negative Binomial 计数损失；
- $\mathcal{L}_{first}$：expert first-use bucket 的交叉熵或生存损失；
- $\mathcal{L}_{calibration}$：Brier score 或校准正则。

没有执行的分支节点只通过真实 branch outcome 和 guard 监督 reach，不应该把所有 expert count 简单当作普通负样本，否则分支预测错误和 expert 预测错误会混在一起。expert use/count/first-use loss 只在节点真实运行时计算，count/first-use loss 再只在对应 expert 真实使用时计算。

由于 expert 标签稀疏，需要按层报告结果，并使用 class weighting 或负样本采样。但最终比较必须回到完整 expert 集合，不能只在采样后的容易子集上报告准确率。

## 简单模型到完整模型

实验不应一开始只给出一个 GNN。需要按下面的顺序逐步增加信息：

| 模型 | 使用的信息 | 作用 |
| --- | --- | --- |
| M0 Task prior | Task Type | 对应已有 task-aware expert 热度 |
| M1 Template frequency | Task Type + Agent 历史统计 | 检查简单统计是否已经足够 |
| M2 Probabilistic DAG | Template frequency + branch frequency + duration，通过真实 guard 遍历图 | 检查简单概率图是否已经足够 |
| M3 Agent MLP | Agent embedding + 当前可见内容，不看图边 | 检查 ML embedding 的作用 |
| M4 Node-set model | 图中所有节点，但忽略真实边 | 控制“只看 Agent 集合” |
| M5 Graph predictor | Agent embedding + 有向真实图 | 检查 learned message passing 的增量 |
| M6 Graph + online memory | M5 + 快速在线统计和校准 | 完整方法 |

如果 M1 或 M2 已经与 M6 相同，应选择简单方法，而不是为了形式保留 GNN。如果 M5 只改善到达时间、不改善 expert 身份，也要如实拆开报告；这种结果仍可能支持“图负责 deadline，Agent Template 负责 expert identity”的系统设计。

## 在线学习

在线学习分为快、慢两层。

快速更新不做在线 SGD，只更新：

- 分支的 Beta/Dirichlet 计数；
- Agent 每层 expert 使用的滑动计数；
- token 数和节点运行时间的指数移动平均；
- 当前 Task Type 下的快速条件统计；
- 预测校准参数；
- 最近相似 Invocation 的小型 memory。

G1/G2/G3 的在线曲线默认冻结所有神经网络参数，只允许上述快速统计发生变化。另设“周期性再训练”实验时必须单独标注训练成本和使用的数据，不能与快速在线结果混报。

慢速更新周期性训练：

- descriptor encoder；
- GNN；
- layer-expert embeddings；
- Agent 和 Application residual；
- 多任务预测头。

在线评估严格使用 `test-then-update`：

```text
1. 在节点执行前保存预测
2. 节点由真实 MoE router 正常执行
3. 用真实 trace 评估刚才的预测
4. 评估完成后才更新该 Agent 的 memory
5. 使用更新后的状态重新预测剩余图
```

不能先用当前节点的真实 trace 更新 memory，再把它计入同一节点的预测成绩。

## 对 RL-Schedule 的借鉴

[RL-Schedule](https://github.com/pku-lemonade/RL-Schedule) 最值得借鉴的不是具体网络或动作，而是**预测器与控制器解耦**：

```text
历史和结构数据
  -> predictor 形成结构化未来状态
  -> controller 根据状态选择系统动作
```

对应到本工作：

- 监督学习模型负责预测 Agent 图上的未来 expert demand；
- cache/prefetch controller 接收 demand、置信度、deadline、显存和带宽；
- controller 的 reward 或目标是实际 stall、应用 makespan 和传输浪费，不是预测准确率。

RL-Schedule 的 predictor 将计算节点、通信链路和时间历史编码为图状态，再把预测结果作为控制器状态的一部分。本文可对应为：

| RL-Schedule 中的概念 | 本文中的对应物 |
| --- | --- |
| 计算/链路节点特征 | Agent 节点、图边、Task Type、routing memory |
| 图编码与时间历史 | 有向 GNN + Agent Template 的在线统计 |
| predictor 输出 | branch/reach/start/use/count/first-use/uncertainty |
| RL state 的预测通道 | prefetch controller 的未来 demand 输入 |
| 调度动作 | prefetch/keep/evict |
| 真实运行代价 | application makespan、stall、wasted bytes |

不能直接照搬其具体网络和动作空间。RL-Schedule 面对的是相对固定的物理拓扑与低维状态；本文面对的是条件分支、多 Application 并发和逐层逐 expert 的高维 demand，候选动作也随 cache 内容变化。

第一阶段不直接使用 RL controller，原因是：

1. prefetch、keep 和 evict 已经有清晰的容量、带宽和 deadline 约束；
2. 贪心、背包或短窗口优化器更容易解释预测为何带来收益；
3. RL 会把 predictor 错误和 controller 错误混在一起；
4. 条件 Agent 图上的动作空间会随 expert 数和候选节点变化，训练稳定性本身会成为额外问题。

正确顺序是：

```text
perfect oracle + 简单 controller
  -> learned predictor + 同一个 controller
  -> 证明预测带来的系统增量
  -> 最后再考虑 RL 是否进一步改善复杂资源竞争
```

# Expert Cache 和 Prefetch

## Cache 状态

每个 $(l,e)$ expert 在任意时刻处于：

```text
resident：
  权重已经完整位于 GPU，可立即执行

prefetching：
  CPU 到 GPU 的异步传输正在进行

absent：
  权重只在 CPU，需要按需加载或预取
```

对每个 expert 记录：

- 权重字节数；
- CPU pinned memory 地址；
- GPU cache slot；
- 当前状态和传输完成事件；
- 最近使用时间；
- 预测下次使用概率和 deadline；
- 淘汰后重新加载的代价。

缓存容量必须以真实可用显存计算：

```text
GPU expert cache
= GPU 总显存
- 非 expert 权重
- KV cache 预算
- activation 和 workspace
- runtime safety margin
```

不能通过挤占 KV cache 或制造不现实的小缓存来夸大收益。主实验优先选择在目标 GPU 上自然产生 expert 内存压力的 MoE 模型。

## 从预测到动作

对一个当前不在 GPU 的 expert，控制器需要估计提前加载的净收益：

$$
B_{\tau,l,e}
=
p^{deadline}_{\tau,l,e}
\cdot C^{stall}_{l,e}
-
(1-p^{deadline}_{\tau,l,e})C^{waste}_{l,e}
-
C^{evict}_{l,e}
-
C^{bandwidth}_{l,e}
$$

其中：

- $p^{deadline}_{\tau,l,e}$ 是在加载 deadline 前使用该 expert 的校准概率；
- $C^{stall}$ 是不预取时可能暴露在关键路径上的加载等待；
- $C^{waste}$ 是预测错误、传入后未使用的代价；
- $C^{evict}$ 是为了腾出空间淘汰其他 expert 的未来代价；
- $C^{bandwidth}$ 是本次传输占用有限 H2D 带宽造成的机会成本。

不同 expert 的收益不是严格可加的：多个并行传输的暴露 stall 由最晚完成者和 copy-engine contention 共同决定，同一 expert 也不能在多个 bucket 重复加载。因此第一阶段不声称求解一个可加的全局最优问题，而采用 **receding-horizon marginal-benefit greedy**：每发生一个系统事件就基于最新状态重算短窗口内的边际收益。

具体流程为：

1. 汇总所有 Application 的候选 expert；
2. 每个 absent expert 只生成一个 load candidate，deadline 取预测的最早有效 first-use bucket；
3. 用当前 resident、in-flight copy 和固定 scheduler 模拟 candidate 的完成时间；
4. 无法在 deadline 前完成、会超过逐时刻 cache 容量或 H2D in-flight budget 的 candidate 直接排除；
5. 若需要 eviction，把被淘汰 expert 的未来机会成本计入边际 benefit；
6. 每次选择单位字节正边际收益最高的一个动作，更新模拟状态后重新计算剩余 candidate；
7. 多个 transfer 的 stall 使用事件时间线模拟，不把它们的收益简单相加；
8. 置信度过低时退回 LRU/LFU 或 task-hot 策略。

需要保存每个 bucket 的 resident set、loading set、copy start/end 和 eviction 事件，以验证任意时刻都不超过 GPU cache 容量与 H2D 并发预算。这个 greedy 是可解释的实用策略，不宣称最优；deadline-aware full-future controller 作为上界。

## 多 Application、并行和分支

多个 Application Instance 并发时，控制器不能只预测“下一个请求”。它需要聚合所有图的未来需求：

$$
\widehat D^{deadline,all}_{\tau,b,l,e}
=
\sum_i \widehat D^{deadline,(i)}_{\tau,b,l,e}
$$

如果多个图即将使用同一 expert，应提高其优先级；如果两个并行分支需要不同 expert，则应根据运行概率、关键路径和传输 deadline 竞争带宽。

并行节点的 enqueue/start 顺序仍由固定 scheduler 决定。完整方法可以预测该顺序造成的 $E_v,T_v$，但不能暗中重排它；否则系统收益会混入 scheduling contribution。

条件分支在未确定前使用历史学习的概率。分支一旦由真实前驱结果确定：

- 被排除分支的需求立即清零；
- 被选择分支的概率变为 1；
- 剩余图重新运行预测；
- 已经发起但不再需要的传输可以取消，若无法取消则计入 wasted bytes。

图结构的主要系统价值可能不是改变“最终会用哪些 expert”，而是改变“哪些 expert 应该先传”。因此必须分别评估 expert identity 和 first-use/deadline。

## 决策和重新预测的时机

系统在以下事件发生时重新计算剩余图需求：

1. 新 Application Instance 注册；
2. 任意 Agent Invocation 完成；
3. 条件分支被确定；
4. 某个未来节点的完整 prompt 变为可见；
5. 新节点进入 ready/running 状态；
6. expert 传输完成或按需 miss 改变 cache；
7. 周期性控制 tick 到达。

每次预测都保存决策时间和当时的可见特征快照。这样才能分别评价 graph-registration、frontier-update、prompt-ready 和 prefill 阶段的准确率与提前量。

## 正确性和退化路径

预测器永远不修改真实 router 的输出。

```text
预测命中：
  router 选择的 expert 已经 resident，直接执行

预测遗漏：
  按真实 router 结果同步或异步加载，等待完成后执行

错误预取：
  expert 占用缓存和带宽，但不参与错误 token 的计算
```

prefetch 不改变 router 决策和模型语义，但不笼统承诺跨 kernel、batch 或 runtime 的逐比特一致。在相同随机种子和确定性执行设置下，需要验证启用与关闭预测得到相同 token 输出和相同真实 router 选择。系统必须提供以下退化路径：

- 预测服务超时：使用 task-hot 或 LRU；
- 新 Agent 无历史：descriptor + Task Type fallback；
- 图信息缺失：退化为 Agent MLP 或 prompt/history predictor；
- 置信度低：不做高代价预取；
- H2D 饱和：停止低收益 prefetch，只保留按需加载。

# 实施路线

## 阶段 0：数据和系统机会

完成：

- 统一 Application/Agent/Invocation/Task 记录格式；
- router trace、branch、prompt availability 和 first-use 时间戳；
- CPU-GPU expert 传输 microbenchmark；
- exact trace replay；
- deadline-aware perfect future oracle。

进入下一阶段的条件：

> 目标模型和硬件上存在可观的 oracle stall reduction，并且数据覆盖矩阵可以区分 Task、Agent 和 Graph。

## 阶段 1：简单统计和 Insight

完成：

- Task-Hot、history、prompt 和 request-internal baselines；
- hierarchical Template frequency；
- Agent 稳定性；
- lead time；
- node-set/edge 负对照；
- G1 数据切分。

进入下一阶段的条件：

> Agent Template 在 prompt+Task Type 之外有增量，或者真实图至少对 first-use/deadline 有稳定增量。

## 阶段 2：ML Graph Predictor

完成：

- descriptor encoder；
- Agent residual；
- two-level Application memory；
- directed GraphSAGE/GAT；
- branch/reach/start/use/count/first-use heads；
- calibration；
- test-then-update；
- G1、G2、G3 全部离线实验。

进入下一阶段的条件：

> Full model 相对 prompt+history、Probabilistic DAG、Agent MLP 和 node-set baseline 在 held-out 数据上稳定提升，并且预测开销远小于可隐藏的传输时间。

## 阶段 3：真实 Expert Cache/Prefetch

完成：

- CPU pinned expert store；
- GPU expert cache；
- async prefetch 和完成事件；
- cost-aware controller；
- branch update；
- 多 Application demand aggregation；
- 错误退化路径；
- 输出与 router 正确性验证。

## 阶段 4：最终实验

完成：

- 多模型独立训练；
- G1/G2/G3；
- 多 Application；
- cache、带宽、并发和负载 sweep；
- 所有 baseline 和消融；
- Episode-level 统计；
- 最坏情况和负结果。

# 后续扩展

以下方向有价值，但不属于第一阶段主贡献：

## 动态 Application 图

运行时允许新增节点和边。已有 Template Memory 可以继续使用，但图预测器需要支持增量编码和未知后继分布。

## RL Controller

当多种传输、eviction、优先级和负载控制相互作用，简单 greedy 明显低于 oracle controller 时，再冻结 predictor，把 forecast、uncertainty、cache、带宽和队列作为 RL state。

RL 的 reward 应使用：

```text
- application critical-path latency
- expert stall
- wasted transfer
- harmful eviction
- fairness violation
```

不能使用 prediction accuracy 代替系统 reward。

## 跨 MoE 模型

Agent descriptor encoder 可以尝试复用，但 layer-expert embedding、routing memory 和预测 head 默认全部重新训练。只有经过显式 expert 对齐实验后，才能讨论跨模型迁移。

## 其他系统动作

未来可以将同一个 future demand forecast 用于：

- request scheduling；
- 多 GPU expert placement；
- expert replication；
- 分层 CPU/CXL/NVMe 管理。

这些动作不能在第一阶段同时加入，否则无法判断系统收益来自 expert prefetch、请求重排还是 placement。

# 最终应形成的贡献

如果核心实验成立，论文贡献可以概括为：

1. **问题与观察。** 定义 Agent Application 图上的跨调用、pre-materialization MoE expert demand forecast，并证明 Agent Template 和真实图结构在强 prompt/history baseline 之外具有可利用信息。
2. **预测方法。** 提出支持 G1 新任务、G2 新图组合和 G3 新 Agent 冷启动的 Agent Template Memory 与有向图 ML predictor，联合预测节点运行、时间、逐层 expert 使用、token 数和不确定性。
3. **系统方法。** 将所有并发 Application 的未来需求转化为受容量、H2D 带宽和 deadline 约束的 CPU 到单 GPU expert cache/prefetch，不修改原生 router 的决策逻辑，并在确定性配置下验证 token 与 route 一致。
4. **完整验证。** 通过可识别的数据设计、强 request-internal baseline、oracle opportunity、真实 runtime、G1/G2/G3、负对照和资源 sweep，说明收益来自更早的 Agent 图信息，而不是任务标签、prompt 泄漏或更宽松的系统资源。

这四项中，第一项和第四项最关键。没有对增量信息、提前量和 oracle 系统机会的严格证明，再复杂的 Agent embedding、GNN 或 cache controller 都不足以形成有意义的科研工作。
