# Agent Application Router 数据采集设计

本数据采集用于：

- 分析同一 Agent Template 在不同任务中的 expert 使用规律；
- 训练基于 Agent 和运行图的 ML 预测器；
- 验证条件分支和动态生成的图结构是否提供提前量；
- 回放 CPU 到单 GPU 的 expert cache/prefetch 策略。

## 运行时采集记录

### Application Run Record

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

### LLM Request Record

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

### Prompt Segment 和 Token Record

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

### Router Dispatch Record

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

`expert_demand_time` 表示系统真正需要对应 expert 的时间。

### Route Assignment Record

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
router_score
```

## 动态图的记录规则

新节点和边使用 `created_by_branch_decision_id` 或 `caused_by_event_id` 指向分支事件。

如果所有合法图变化都由静态规则定义，那么预测器在运行前知道候选结构，只是不知道哪条分支会发生。

如果 Agent 可以在运行时创建静态规则中没有出现过的新连接，采集协议仍能通过 `NODE_CREATED` 和 `EDGE_CREATED` 记录它们，但这些结构在创建事件发生前不可见。实验中不能把最终图提前提供给预测器。

## Expert Cache/Prefetch 的附加事件

Router 预测数据可以不包含 cache 状态。但要进行真实 CPU 到单 GPU 的系统实验，需要额外记录：

```text
cache_event_id
event_time
layer_id
expert_id
event_type             demand/hit/miss/load_started/load_finished/evicted
bytes
application_run_id
node_run_id
llm_request_id
dispatch_id
policy_reason
```

这部分应作为独立的系统事件流保存。不要在每次 Router route 中复制完整 cache snapshot；resident set 可以通过 cache event 顺序重放得到。

## Benchmark 选择

现成 benchmark 通常提供任务和 Agent workflow，但不会包含目标 MoE 模型的 Router trace。因此所有主实验都必须通过目标 MoE 重新执行并采集，其他模型产生的 expert ID 不能直接迁移。

| Benchmark                                                    | 主要用途                                                     | 限制                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [TeamBench](https://github.com/ybkim95/TeamBench)            | Planner、Executor、Verifier 角色固定，任务数量多且有确定性评分；适合 G1 | 默认 Application 结构较固定，单独使用不足以证明 G2           |
| [MAS-PromptBench](https://github.com/juyangbai/MAS-PromptBench) | 包含多类 reasoning、coding、tool-use 任务和多种 Agent topology；适合受控 G2 | 必须冻结各 topology 的 Agent prompt，避免把 Agent Template 变化误认为图变化 |
| [MacNet](https://arxiv.org/abs/2406.07155)                   | 显式 DAG，支持 chain、tree、mesh、layer、random 等结构；适合图结构实验 | 原始实现可能让边引入额外 Critic 调用，需构造相同节点集合和调用预算的配对图 |
| [MultiAgentBench/MARBLE](https://arxiv.org/abs/2503.01935)   | 原生多 Agent，覆盖 research、coding、database 等真实协作场景；适合外部验证 | 工具和动态交互较多，本地 MoE 需要可靠的 function calling     |
| [ChatDev 2.0](https://github.com/OpenBMB/ChatDev/blob/main/docs/user_guide/en/workflow_authoring.md) | 可以定义节点、条件边、工具和动态执行；适合作为可控 Application runner | 它是 workflow 框架，不是独立任务数据集                       |
| [Patterns behind Chaos trace](https://huggingface.co/datasets/core12345/MoE_expert_selection_trace) | 用于复现 token、layer、prefill/decode 的 Router baseline 和参考 trace 格式 | 没有 Agent、动态图、分支和运行时间，不能作为主训练数据       |