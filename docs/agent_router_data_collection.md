# Agent Application Router 数据采集设计

本文档定义 TokenMoE 后续数据采集的最小运行时协议。目标是让同一份原始数据能够支持：

- 分析同一 Agent Template 在不同任务中的 expert 使用规律；
- 训练基于 Agent 和运行图的 ML 预测器；
- 验证条件分支和动态生成的图结构是否提供提前量；
- 回放 CPU 到单 GPU 的 expert cache/prefetch 策略。

本文档只规定采集脚本必须记录的运行时数据。模型配置、完整 Agent 定义、完整 Application 定义等静态信息由实验配置保存，采集脚本只记录能够关联这些定义的 ID，不在每条 trace 中重复静态内容。

## 1. 核心设计选择

第一阶段采用以下约束：

1. Router 保持不变。我们预测 Router 将产生的 expert demand，不训练或替换 Router。
2. `node_type` 的集合固定，例如 `Planner`、`Coder`、`Reviewer`、`Tool`。
3. 每次运行中实际出现的节点实例和边可以根据条件分支动态变化，但第一阶段的合法变化必须来自运行前注册的图生成规则。
4. 运行时图使用只追加事件记录，不反复保存整张图。
5. 同一节点可以发起多次 LLM 请求。
6. prefill 和 decode 的 Router 结果都必须采集。
7. 所有事件使用同一可比较的单调时钟。

动态图的含义是：节点类型集合固定，但节点实例数量、实际依赖边和执行路径在运行过程中逐步确定。

例如，静态类型集合始终是：

```text
Planner, Coder, Reviewer, Finalizer
```

一次运行可能产生：

```text
Planner-1 -> Coder-1 -> Reviewer-1 -> Finalizer-1
```

另一次运行可能产生：

```text
Planner-1 -> Coder-1 -> Reviewer-1
                         -> Coder-2 -> Reviewer-2
                         -> Finalizer-1
```

`Coder-1` 和 `Coder-2` 使用相同的 `node_type`，但它们是不同的运行时节点。

## 2. 统一术语

### 2.1 Application Definition

`Application Definition` 是运行前可获得的静态定义，包括：

- 允许使用的 `node_type` 集合；
- Agent Template 与 node type 的关系；
- 初始节点；
- 已知的条件分支和图生成规则；
- 分支、依赖和汇合语义。

Application Definition 中还应为初始节点、初始边、分支规则和图生成规则提供稳定 ID。运行时事件只引用这些 ID。

这些内容不由运行时采集脚本重复保存。每次运行只记录 `application_definition_id` 作为外部引用。

### 2.2 Application Run

`Application Run` 表示：

> 一个 Application Definition 在一个具体 benchmark 任务上的一次完整执行，从 orchestrator 接受任务开始，到系统确认不会再创建节点或边，且所有已创建节点进入终态为止。

以下情况产生新的 `application_run_id`：

- 同一任务重新独立执行；
- 同一任务更换 Application Definition；
- 同一任务更换随机种子后重新执行；
- 整个 Application 失败后从头重启。

以下情况仍属于原来的 Application Run：

- 条件分支选择不同路径；
- 运行时创建新的 Coder 或 Reviewer 节点；
- 某个节点重试；
- 一个节点发出多次 LLM 请求；
- 工具调用失败后重试。

Application Run 的开始时间不是第一次 LLM 请求的时间。结束时间也不是最后一个 token 返回的时间，而是整个 Application 进入 `succeeded`、`failed`、`cancelled` 或 `timeout` 终态的时间。

### 2.3 Node Run

`Node Run` 是某次 Application Run 中实际创建的一个节点实例，使用 `node_run_id` 标识。

`node_type_id` 表示图中的逻辑类型，例如 `Coder`。`agent_template_id` 表示实际使用的 Agent 实现，包括其 prompt 和工具配置。多个 Agent Template 可以属于同一种 node type；同一个 Agent Template 也可以被多个 Node Run 复用。

`node_template_id` 表示 Application Definition 中预先声明的节点位置。初始节点通常带有该字段；由图生成规则动态创建且没有固定位置的节点可以为空，并改为引用 `generation_rule_id`。

同一种 node type 可以出现多次：

```text
Coder-1: node_type=Coder, iteration_index=1
Coder-2: node_type=Coder, iteration_index=2
```

### 2.4 LLM Request

`LLM Request` 是一个 Node Run 发起的一次真实模型调用，使用 `llm_request_id` 标识。一个 Node Run 可以对应零个、一个或多个 LLM Request。

工具节点可能不产生 LLM Request。Agent 节点则可能因为多轮交互、工具调用或重试产生多个 LLM Request。

`Tool Call` 表示某个 Node Run 内的一次真实工具调用，使用 `tool_call_id` 标识。它属于一个确定的 `node_run_id` 和 `attempt_id`，工具结果通过 `TOOL_RESULT_AVAILABLE` 事件变为因果可见。

### 2.5 Router Dispatch

`Router Dispatch` 表示某个 MoE 层的一次实际 Router 分发事件，使用 `dispatch_id` 标识。一个 batch 中可以同时包含来自多个 Application Run 的 token。

Router Dispatch 不属于单个 LLM Request。一个 batch dispatch 可以包含多个请求，二者通过 Route Assignment 建立多对多关联：

```text
application_run_id
  -> node_run_id
    -> llm_request_id
      -> request token
           |
           +-> route_assignment <- dispatch_id
```

### 2.6 ID 作用域

以下实体 ID 在整个采集数据集中全局唯一：

```text
application_run_id
node_run_id
llm_request_id
dispatch_id
batch_id
event_id
edge_run_id
branch_decision_id
tool_call_id
segment_id
```

`agent_template_id` 在外部 Agent registry 中全局稳定。`node_type_id`、`node_template_id`、`generation_rule_id`、`branch_rule_id` 和静态 transition ID 的作用域是一个 `application_definition_id`，离线 join 时必须同时携带 Application Definition 引用。

以下字段是局部序号：

```text
attempt_id
  在一个 node_run_id 内从 0 开始递增。

request_index_within_node
  在一个 node_run_id 内单调递增，跨 attempt 不重置。

sequence_id
  在一个 llm_request_id 内唯一，表示 n>1、beam 或其他多输出产生的模型序列；
  普通单输出请求固定为 0，它不是 batch slot。

token_position
  在 (llm_request_id, sequence_id, phase) 内唯一；prefill 和 decode 分别从 0 开始。
```

## 3. 静态信息和运行时信息的边界

以下内容由实验启动器或外部 manifest 保存，不进入每条运行时事件：

- 模型、checkpoint、tokenizer 和 Router 配置；
- 完整 Application Definition；
- Agent system prompt、tool schema 和输入输出 schema；
- benchmark 的完整静态内容；
- Agent embedding、prompt embedding 和 graph embedding；
- 由原始 trace 计算得到的 expert 频率、RouteSig 和预测标签。

运行时数据仍需要保存少量引用字段：

```text
experiment_config_id
application_definition_id
agent_template_id
benchmark_item_id
```

这些字段只用于把运行时数据关联到外部静态配置，不重复保存静态配置本身。外部配置必须不可变并带版本或 hash，否则无法判断两次采集是否真的使用了同一套定义。

## 4. 运行时采集记录

建议把运行时数据拆成六类记录，而不是写入一个大 JSON。

### 4.1 Application Run Record

每次完整运行一条记录：

```text
application_run_id           必需，全局唯一
application_definition_id    必需，关联外部静态定义
experiment_config_id         必需，关联本次实验配置
benchmark_item_id            必需，原始任务标识
task_type                    必需，运行前可见的离散任务标签
retry_of_application_run_id  可空，整个 Application 重启时填写
run_started_at               必需
run_finished_at              结束后填写
run_status                   running/succeeded/failed/cancelled/timeout
```

并发运行多个 Application 时，每个 Application 使用不同的 `application_run_id`，但共享同一时钟域。

启动时立即写入基础 Run Record。结束状态由 `APPLICATION_RUN_FINISHED` 事件确定，`run_finished_at` 和最终 `run_status` 可以在离线整理时物化，避免进程异常退出时整条 Run 记录丢失。

### 4.2 Graph Event

所有图事件使用统一头部：

```text
event_id             全局唯一，用于去重和建立因果关系
schema_version       事件格式版本
application_run_id   所属 Application Run
event_sequence       同一 Run 内严格递增，表示完整运行状态版本
event_type           事件类型
event_time           事件在系统中真正可见的时间
recorded_time        collector 实际写入日志的时间，推荐保留
clock_domain_id      时间所属的单调时钟域
producer_id          orchestrator 或 tool runtime
graph_revision       事件发生后的运行图版本
caused_by_event_id   可空，触发本事件的前序事件
```

`event_time` 用于判断一个特征在预测时是否已经可见。`recorded_time` 只用于诊断日志延迟，不能作为因果时间。

同一 Application Run 的 `event_sequence` 由 orchestrator 中的中央 sequencer 唯一分配。Tool runtime 将事件发送给该 sequencer 后再落盘；模型服务的 Router Dispatch 不进入 Graph Event 序列，而是使用独立的 `dispatch_id` 和时间戳。CPU 到单 GPU 的第一阶段部署要求所有 producer 位于同一主机，并使用同一个系统单调时钟。跨主机时钟同步不属于第一阶段范围。

第一阶段至少支持以下事件。

#### APPLICATION_RUN_STARTED

表示 orchestrator 已接受任务，Application Run 正式开始。它必须早于任何节点创建事件。

#### NODE_CREATED

```text
node_run_id
node_type_id
node_template_id             可空，动态节点可能没有固定图位置
agent_template_id            Agent 在运行时绑定时必需
iteration_index
creation_index
created_by_branch_decision_id
generation_rule_id           可空
```

`node_type_id` 来自固定的类型集合。`node_run_id` 表示本次运行中实际出现的节点。

初始节点的 `created_by_branch_decision_id` 和 `generation_rule_id` 可以为空。其他动态节点必须同时引用触发它的 branch decision 和已注册的 generation rule。

#### EDGE_CREATED

```text
edge_run_id
source_node_run_id
target_node_run_id
dependency_type
join_group_id                可空
created_by_branch_decision_id
generation_rule_id           可空
```

即使某条边可以由静态分支规则推导，也建议记录 `EDGE_CREATED`。它体量很小，却能验证 orchestrator 的实际行为是否符合静态定义。

初始边的 `created_by_branch_decision_id` 和 `generation_rule_id` 可以为空。其他动态边必须能够追溯到已注册的 generation rule。

#### EDGE_DEACTIVATED

```text
edge_run_id
deactivated_by_branch_decision_id   可空
reason
```

该事件只改变后续可达关系，不删除此前已经发生的依赖事实。

#### BRANCH_DECIDED

```text
branch_decision_id
decision_node_run_id
branch_rule_id
selected_outcome_id
selected_transition_ids
```

`selected_transition_ids` 引用 Application Definition 中注册的静态 transition ID，不是运行时 `edge_run_id`。

候选 outcome 和生成规则已经静态定义时，不需要在每次事件中重复保存。若后续阶段允许候选集合本身在运行时产生，应先记录：

```text
BRANCH_OPENED
branch_decision_id
candidate_outcome_ids
```

只有 `BRANCH_DECIDED.event_time` 之后，真实分支结果才能成为预测特征。

#### Node Lifecycle Events

```text
NODE_READY
NODE_ENQUEUED
NODE_STARTED
NODE_FINISHED
NODE_FAILED
NODE_CANCELLED
NODE_SKIPPED
ATTEMPT_FAILED
```

共同字段为：

```text
node_run_id
attempt_id
status_reason       可空
```

推荐状态转换为：

```text
NODE_CREATED
  -> NODE_READY
  -> NODE_ENQUEUED
  -> NODE_STARTED
  -> NODE_FINISHED | NODE_FAILED | NODE_CANCELLED

NODE_CREATED
  -> NODE_SKIPPED | NODE_CANCELLED
```

`NODE_FAILED` 表示该 Node Run 已经终止，不会再次执行。可以恢复的单次失败记录为 `ATTEMPT_FAILED`，随后使用新的 `attempt_id` 重新进入 `NODE_READY` 或 `NODE_ENQUEUED`。这样重试不会把已经发生的请求和 Router trace 覆盖掉。

不要从日志中删除已经出现过的节点或边。运行时不再使用时，记录 `NODE_CANCELLED` 或 `EDGE_DEACTIVATED`。这样才能重建过去任意时刻预测器实际看到的图。

#### Tool Events

若工具被建模为独立的 `Tool` Node Run，可以直接使用节点生命周期事件。若工具调用嵌套在 Agent Node 内，至少记录：

```text
TOOL_STARTED
TOOL_RESULT_AVAILABLE
TOOL_FINISHED
```

共同字段为：

```text
tool_call_id
node_run_id
attempt_id
tool_name
tool_status
result_ref          在 TOOL_RESULT_AVAILABLE 中填写
```

工具结果只有在 `TOOL_RESULT_AVAILABLE.event_time` 之后才能成为后继节点的输入，不能用 `TOOL_FINISHED` 代替。

#### APPLICATION_RUN_FINISHED

只有当 orchestrator 明确声明不会继续创建节点或边，并且所有已经创建的 Node Run 都进入终态后，才能记录该事件。

```text
final_status        succeeded/failed/cancelled/timeout
failure_reason      可空
```

### 4.3 LLM Request Record

每次真实模型调用记录：

```text
llm_request_id
application_run_id
node_run_id
attempt_id
request_index_within_node
request_created_at
event_sequence_at_create
graph_revision_at_create
prompt_ready_at
request_enqueued_at
inference_started_at
first_token_at
inference_finished_at
request_status
```

必须区分：

- `NODE_STARTED`：orchestrator 开始执行节点；
- `prompt_ready_at`：完整 prompt 已经生成；
- `request_enqueued_at`：请求进入模型服务队列；
- `inference_started_at`：模型真正开始执行。

`event_sequence_at_create` 用于重建请求产生时完整可见的运行状态，包括节点状态和已经确定的分支。`graph_revision_at_create` 用于快速定位当时的图结构。离线 ML 代码不能读取更高 event sequence 或 graph revision 中才出现的信息。

### 4.4 Prompt Segment 和 Token Record

每个 prompt 片段记录：

```text
segment_id
llm_request_id
segment_type              system/task/predecessor_output/tool_output/other
source_node_run_id        可空
source_tool_call_id       可空
produced_at
available_at
token_start
token_end                 右开区间
content_ref               原文或可无损恢复原文的外部引用
```

每个进入模型的 token 记录：

```text
llm_request_id
sequence_id
phase                     prefill/decode
token_position
token_id
segment_id                可空，decode token 通常没有 prompt segment
```

`available_at` 是防止未来信息泄漏的核心字段。预测时只能使用 `available_at` 不晚于预测时刻的内容。

Token ID 不能视为匿名数据。它通常可以还原原始文本，因此公开数据时需要单独处理隐私和数据许可。

### 4.5 Router Dispatch Record

每个 MoE 层的实际 Router 分发事件记录：

```text
dispatch_id
batch_id
layer_id
phase
router_decision_time
expert_demand_time
clock_domain_id
```

`expert_demand_time` 表示系统真正需要对应 expert 的时间。不能使用 cache miss 之后的 compute start 作为 demand time，否则会把 miss stall 错误地计入可用提前量。

### 4.6 Route Assignment Record

每个 token 在每层的最终 top-k 路由结果记录：

```text
dispatch_id
application_run_id
node_run_id
llm_request_id
sequence_id
phase
token_position
layer_id
topk_slot
expert_id
router_score             可选
```

最小唯一键是：

```text
(llm_request_id, sequence_id, phase, token_position, layer_id, topk_slot)
```

`expert_id` 保存真正执行的最终 expert。如果模型存在 capacity drop 或 reroute，可额外保存原始 Router 候选，但不能用候选 ID 替代最终 dispatch ID。

## 5. 动态图的记录规则

### 5.1 固定类型，动态实例

固定的是 `node_type_id` 集合，不是 `node_run_id` 集合。每次创建节点都分配新的 `node_run_id`。

如果工作流中存在逻辑循环，应展开为新的 Node Run：

```text
Coder-1 -> Reviewer-1 -> Coder-2 -> Reviewer-2
```

不要反复复用同一个 Coder 或 Reviewer 节点实例。这样每个运行时图都可以表示为随时间增长的执行 DAG。

### 5.2 分支和图变化

推荐的事件顺序是：

```text
BRANCH_DECIDED
  -> NODE_CREATED
  -> EDGE_CREATED
  -> NODE_READY
```

新节点和边使用 `created_by_branch_decision_id` 或 `caused_by_event_id` 指向分支事件。

第一阶段要求所有合法图变化都由 Application Definition 中注册的规则定义。因此预测器在运行前知道候选 node type、transition 和 generation rule，只是不知道哪条分支会发生、哪些节点和边最终会被物化。

未来可以扩展到 Agent 创建静态规则中没有出现过的新连接。现有事件协议仍能记录这种结构，但它属于另一个“未知图生成”问题：创建事件发生前，预测器既不知道真实结构，也不能把最终图作为输入。该场景不计入第一阶段 G1/G2 结论。

### 5.3 图版本

以下操作增加 `graph_revision`：

```text
BRANCH_DECIDED
NODE_CREATED
EDGE_CREATED
EDGE_DEACTIVATED
NODE_CANCELLED（当它改变未来可达图时）
```

节点普通状态变化不一定增加 graph revision，但必须增加 `event_sequence`。

离线重放时，从 Application Run 开始顺序应用图事件，即可恢复任意 revision 的运行图。可以周期性保存图快照用于加速恢复，但快照不是原始真值，不能替代事件日志。

### 5.4 未选择分支的监督

只保存最终物化图是不够的，因为这会丢失未选择分支的候选空间。正确做法是：

- 候选分支来自外部 Application Definition，或由 `BRANCH_OPENED` 记录；
- `BRANCH_DECIDED` 保存真实 outcome；
- `NODE_CREATED` 和 `EDGE_CREATED` 保存 outcome 实际产生的结构；
- 未选择路径不伪造 Node Run，只通过 branch outcome 表示其未发生。

## 6. Expert Cache/Prefetch 的附加事件

Router 预测数据可以不包含 cache 状态。但要进行真实 CPU 到单 GPU 的系统实验，需要额外记录：

```text
cache_event_id
transfer_id             load_started/load_finished 配对时填写
event_time
layer_id
expert_id
event_type             demand/hit/miss/load_started/load_finished/evicted
bytes
application_run_id      可空
node_run_id             可空
llm_request_id          可空
dispatch_id             可空
policy_reason
```

一次 load 或 eviction 可能由多个 Application Run 的聚合预测触发，因此上述来源 ID 都允许为空，不能强制把系统动作归给单个请求。需要表达多个需求来源时，另建关联表：

```text
cache_event_source:
  cache_event_id
  application_run_id
  node_run_id          可空
  llm_request_id       可空
  dispatch_id          可空
```

这部分应作为独立的系统事件流保存。不要在每次 Router route 中复制完整 cache snapshot；resident set 可以通过 cache event 顺序重放得到。

## 7. 推荐存储结构

低频、结构不固定的事件使用 JSONL；高频 token 和 route 使用 Parquet 或 Arrow：

```text
runtime/
  application_runs.jsonl
  graph_events.jsonl
  llm_requests.jsonl
  prompt_segments.jsonl
  request_tokens.parquet
  router_dispatches.parquet
  route_assignments.parquet
  cache_events.parquet          # 系统实验阶段可选
```

Agent framework 负责记录 Application Run、动态图和 Node Run。模型服务负责记录 Router Dispatch 和 Route Assignment。两侧通过以下 ID 对齐：

```text
application_run_id
node_run_id
llm_request_id
```

这些 ID 应通过 OpenAI-compatible 请求的 metadata 或等价的 side channel 一起传给本地 MoE 服务。

## 8. 最小采集集合

本节是采集实现的必需字段索引。各事件 payload 的具体语义以第 4 节为准。

```text
Application Run：
  application_run_id
  application_definition_id
  experiment_config_id
  benchmark_item_id
  task_type
  retry_of_application_run_id（可空）
  run_started_at

Graph Event 公共头：
  event_id
  schema_version
  application_run_id
  event_sequence
  event_type
  event_time
  clock_domain_id
  producer_id
  graph_revision
  caused_by_event_id（可空）
  recorded_time（推荐）

Graph Event payload：
  NODE_CREATED：node_run_id, node_type_id, agent_template_id,
                node_template_id/generation_rule_id,
                iteration_index, creation_index,
                created_by_branch_decision_id（初始节点可空）
  EDGE_CREATED：edge_run_id, source_node_run_id, target_node_run_id,
                dependency_type, join_group_id（可空）, generation_rule_id,
                created_by_branch_decision_id（初始边可空）
  EDGE_DEACTIVATED：edge_run_id,
                    deactivated_by_branch_decision_id（可空）, reason
  BRANCH_DECIDED：branch_decision_id, decision_node_run_id,
                  branch_rule_id, selected_outcome_id,
                  selected_transition_ids
  Node lifecycle：node_run_id, attempt_id, status_reason（可空）
  Tool lifecycle：tool_call_id, node_run_id, attempt_id,
                  tool_name, tool_status, result_ref（结果事件填写）
  APPLICATION_RUN_FINISHED：final_status, failure_reason（可空）

LLM Request：
  llm_request_id
  application_run_id
  node_run_id
  attempt_id
  request_index_within_node
  request_created_at
  event_sequence_at_create
  graph_revision_at_create
  prompt_ready_at
  request_enqueued_at
  inference_started_at
  first_token_at
  inference_finished_at
  request_status

Prompt Segment：
  segment_id, llm_request_id, segment_type,
  source_node_run_id/source_tool_call_id（可空）, produced_at,
  available_at, token_start, token_end, content_ref

Request Token：
  llm_request_id, sequence_id, phase, token_position,
  token_id, segment_id（可空）

Router Dispatch：
  dispatch_id, batch_id, layer_id, phase,
  router_decision_time, expert_demand_time, clock_domain_id

Route Assignment：
  dispatch_id, application_run_id, node_run_id, llm_request_id,
  sequence_id, phase, token_position, layer_id,
  topk_slot, expert_id, router_score（可空）
```

第一阶段不默认采集：

```text
完整模型静态配置
完整 Application 静态图
完整 Agent 静态定义
hidden state
完整 Router logits
Agent/prompt/graph embedding
派生 expert 频率和预测标签
每个时刻的完整图 snapshot
每个时刻的完整 cache snapshot
```

`router_score` 体量相对较小，建议保留，但它不是最小必需字段。

## 9. 数据校验规则

采集完成后必须验证：

1. `application_run_id`、`node_run_id`、`llm_request_id` 和 `dispatch_id` 不可混用。
2. 同一 Application Run 内的 `event_sequence` 严格递增且可去重。
3. `event_time` 和 Router 时间属于可比较的单调时钟域。
4. Node Run 必须先 `NODE_CREATED`，之后才能进入其他状态或发起 LLM Request。
5. `EDGE_CREATED` 的源节点和目标节点必须已经创建。
6. 分支产生的节点和边必须能够追溯到对应的 `BRANCH_DECIDED`。
7. 一个节点只有在真实依赖满足后才能进入 `NODE_READY`。
8. 重试创建新的 `attempt_id`，不能覆盖旧请求或旧事件。
9. `event_sequence_at_create` 和 `graph_revision_at_create` 不能包含请求产生之后才出现的状态、节点或边。
10. Route Assignment 引用的 token 必须存在，expert ID 必须合法。
11. prefill 和 decode 必须分别标记，不能混合 token position。
12. 所有 Node Run 必须在 Application Run 结束前进入终态。
13. capture on/off 必须通过输出等价性测试，确保采集逻辑没有改变模型行为。

## 10. Benchmark 选择

现成 benchmark 通常提供任务和 Agent workflow，但不会包含目标 MoE 模型的 Router trace。因此所有主实验都必须通过目标 MoE 重新执行并采集，其他模型产生的 expert ID 不能直接迁移。

| Benchmark | 主要用途 | 限制 |
| --- | --- | --- |
| [TeamBench](https://github.com/ybkim95/TeamBench) | Planner、Executor、Verifier 角色固定，任务数量多且有确定性评分；适合 G1 | 默认 Application 结构较固定，单独使用不足以证明 G2 |
| [MAS-PromptBench](https://github.com/juyangbai/MAS-PromptBench) | 包含多类 reasoning、coding、tool-use 任务和多种 Agent topology；适合受控 G2 | 必须冻结各 topology 的 Agent prompt，避免把 Agent Template 变化误认为图变化 |
| [MacNet](https://arxiv.org/abs/2406.07155) | 显式 DAG，支持 chain、tree、mesh、layer、random 等结构；适合图结构实验 | 原始实现可能让边引入额外 Critic 调用，需构造相同节点集合和调用预算的配对图 |
| [MultiAgentBench/MARBLE](https://arxiv.org/abs/2503.01935) | 原生多 Agent，覆盖 research、coding、database 等真实协作场景；适合外部验证 | 工具和动态交互较多，本地 MoE 需要可靠的 function calling |
| [ChatDev 2.0](https://github.com/OpenBMB/ChatDev/blob/main/docs/user_guide/en/workflow_authoring.md) | 可以定义节点、条件边、工具和动态执行；适合作为可控 Application runner | 它是 workflow 框架，不是独立任务数据集 |
| [Patterns behind Chaos trace](https://huggingface.co/datasets/core12345/MoE_expert_selection_trace) | 用于复现 token、layer、prefill/decode 的 Router baseline 和参考 trace 格式 | 没有 Agent、动态图、分支和运行时间，不能作为主训练数据 |

### 10.1 推荐首轮组合

第一阶段采用三层数据：

1. 使用 TeamBench 收集固定 Agent Application 在新任务上的数据，支持 G1。
2. 使用 MAS-PromptBench 的任务和 MacNet 风格配对图，固定 node type、Agent Template 和节点数量，只改变图生成规则，支持 G2。
3. 使用 MARBLE 的 coding、database 和 research 场景做真实多 Agent 外部验证。

ChatDev 2.0 可用于构造具有明确分支规则的可控 Application。Patterns behind Chaos trace 只用于无 Agent 的 Router baseline。

### 10.2 采集矩阵

为了分离任务、Agent 和图结构的影响，数据必须形成交叉组合：

```text
benchmark item
  x Application Definition
  x 固定 Agent Template 集合
  x random seed
```

同一个 Agent Template 必须出现在多个任务和多个 Application Definition 中。同一任务也应由多张配对图执行。

### 10.3 G1、G2、G3 切分

#### G1：已见 Application、已见 Agent、新任务

- 训练和测试使用相同 Application Definition 和 Agent Templates；
- 测试集包含新的完整 benchmark item；
- 同一 benchmark item 的所有节点和请求必须属于同一 split。

#### G2：已见 Agent、新图组合

- 测试 Application Definition 整体不进入训练集；
- 测试图使用的 node types 和 Agent Templates 都在训练数据中出现过；
- 不按运行结束后的最终物化图随机切分，避免利用 branch outcome 选择测试样本；
- 优先使用相同 node type 集合、不同图生成规则的配对 Application。

#### G3：未见 Agent Template 冷启动

- 某个 Agent Template 的全部历史从训练集移除；
- 测试时只使用其外部静态描述、当前可见图和任务输入；
- 第一次运行结束后，才允许使用该 Agent 新收集到的 Router 历史；
- 报告第 0、1、2、4、8、16 次运行后的学习曲线。

## 11. 当前 trace 格式的关系

当前 [`tokenmoe.trace.v3`](trace_format.md) 已经保存 request、prompt token、逐层 expert ID 和 Router score，可以继续作为 Router 子记录使用。

但当前格式仍以一次 prompt admission 为单位，逐层 Router 数组只与 prompt token 对齐。新采集协议需要在其外层增加：

- Application Run；
- Node Run；
- 动态图事件；
- 请求和图 revision 的关联；
- decode Router trace；
- 可比较的 Router demand 时间。

新协议不要求立即替换现有分析代码。可以先让 Application 侧和模型服务侧分别输出事件，再通过共同 ID 生成离线统一数据集。
