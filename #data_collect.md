# Agent 负载下 MoE Router 数据采集规范

本规范定义 TokenMoE 自采数据的内容、产生方、格式与校验规则。目标是一次采集同时支撑 `idea.md` 中的四类用途：

| 用途               | 要回答的问题                                                 | 依赖的记录                                                |
| ------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| 内容预测           | 给定角色、刚调用的工具、工具结果类别，下一次请求新增 prefill 与 decode 在每层的 expert 使用分布 | 请求记录、逐 token 逐层路由、工具调用记录、角色模板注册表 |
| 时间预测           | 正在运行的工具还剩多久、结果是什么                           | 工具调用记录、带时间戳的工具输出流、主机负载              |
| 驻留模拟与系统实验 | 每层每个 expert 何时被哪个 session 需要；多 session 下的批组成与排队 | 引擎步记录、请求生命周期时间戳、模型静态信息、cache 事件  |
| 泛化切分 G1/G2/G3  | 哪些历史记录可以迁移到新任务、新组合、新角色                 | 角色模板版本、角色类型、应用定义、任务标识、种子          |

数据由三个产生方写出，通过共同 ID 离线对齐：agent harness（session、请求、prompt 组成）、工具执行器（工具调用与输出流）、模型服务即 TokenMoE vLLM fork（引擎步、路由）。

## 1. 设计决策

1. Router 不变。只记录 router 的真实选择，不训练或替换 router。
2. prefix caching 开启。命中的前缀不重算，其路由由 fork 从更早的计算中重新暴露；每个请求必须记录 `num_cached_tokens`，用来区分 cached prefix、new prefill 与 decode 三个阶段。
3. decode 路由必采。采集使用真实生成，不使用 `max_tokens=1` 的 prompt-only 模式。最后一个采样 token 没有路由，属正常现象。
4. 工具输出在执行器旁路捕获。记录工具的原始 stdout 与 stderr 流，harness 对 agent 的截断与过滤照常进行，agent 看到的内容不变。
5. 标签离线派生。采集时只记原始命令、退出码、输出与时间；`tool_type`、`outcome_class`、转移点标记等由版本化的离线规则生成，便于修改分类法后重算。
6. 时序按引擎步记录。每次前向记一条步记录，expert 的需求时刻由步开始时间加该层的延迟偏移得到，不为每层每步单独打时间戳。
7. 路由按请求存为数组。每个请求一份 `[T, L, K]` 的 expert ID 数组及可选的同形状 score 数组，不按 (token, layer, slot) 展开成行。
8. 子 agent 的创建建模为一次工具调用。`tool_type = spawn_agent` 的调用创建一个新 session，它的返回就是 join。单循环 agent、子 agent、多角色流水线共用一套记录。
9. 并发是记录下来的实验参数。路由只取决于 token 自身的上下文，不随并发变化；工具时长、排队与批组成随并发和负载变化，所以每条记录都能追溯到它所属的并发配置。高并发场景的离线叠加分析依赖各 session 自带的时间线、工具时长与主机负载。
10. 采集不改变模型行为。每个新模型或新并行配置在采集前通过 capture 开关两侧的逐 token 输出等价性测试。

## 2. 术语与 ID

| 术语                   | 含义                                                         | ID                                   |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------ |
| Experiment Config      | 一次采集的全部静态配置：模型、引擎参数、harness 版本、并发数、种子 | `experiment_config_id`               |
| Role Template          | 一个 agent 角色的静态定义：system prompt、工具列表、采样参数 | `agent_template_id` + `version_hash` |
| Application Definition | 多角色应用的静态结构：角色集合、依赖、分支规则；单循环 agent 为空 | `application_definition_id`          |
| Application Run        | 一个 benchmark 任务的一次完整执行，含其中所有 session        | `application_run_id`                 |
| Session                | 一个 agent 循环的一次运行实例；子 agent 是独立的 session     | `session_id`                         |
| LLM Request            | 一个 session 发出的一次模型调用                              | `llm_request_id`                     |
| Tool Call              | 模型输出中的一次工具调用及其执行                             | `tool_call_id`                       |
| Engine Step            | 模型服务的一次前向，一个 batch                               | `step_id`                            |
| Cache Event            | 系统实验阶段 expert 在存储层级间的搬运与替换事件             | `cache_event_id`                     |

全局唯一的 ID：`application_run_id`、`session_id`、`llm_request_id`、`tool_call_id`、`step_id`、`cache_event_id`。静态 ID：`experiment_config_id`、`agent_template_id`、`application_definition_id`、`benchmark_item_id`、`model_profile_id`、`host_id`、`engine_instance_id`。局部序号：`step_index`（session 内从 0 递增）、`attempt_id`（同一步重试时递增）、`chunk_index`（同一 tool call 同一 stream 内递增）。

### 时钟

所有时间戳使用同一可比较的单调时钟，记录 `clock_domain_id`。第一阶段要求 harness、工具执行器和模型服务在同一主机。跨主机时每个产生方记录时钟校准：

```text
clock_sync:
  producer_id
  measured_at
  offset_estimate_ms
  round_trip_ms
```

预测时能否使用某个特征，只看它的 `available_at` 或事件时间是否不晚于预测时刻；写入日志的时间只用于诊断。

## 3. 静态信息

静态信息由实验启动器保存，运行时记录只引用其 ID。所有静态对象不可变、带版本或 hash。

### 3.1 Experiment Config

```text
experiment_config_id
model_profile_id
engine: vllm_commit, tensor_parallel, expert_parallel, dtype, quantization,
        max_model_len, gpu_memory_utilization, max_num_seqs,
        enable_prefix_caching (必须为 true), scheduling_policy,
        capture_routed_experts, capture_router_scores
harness: name, version, sandbox_type (docker/local), cpu_quota, memory_limit
concurrency: num_concurrent_sessions, arrival_pattern
hosts: [host_id, gpu_model, gpu_count, cpu_model, dram_gb, link_type]
seed
```

### 3.2 Model Profile

```text
model_profile_id
model_id, revision_hash, architecture
num_layers, moe_layer_ids
num_routed_experts, top_k, num_shared_experts
expert_bytes_per_layer          按实际存储精度计算，cache 模拟必需
router_score_semantics          top-k 后、归一化方式
tokenizer_hash
mtp_disabled = true
multimodal_disabled = true
layer_latency_profile_ref       每 (engine config, batch 大小区间) 一份逐层前向延迟
```

### 3.3 Role Template Registry

```text
agent_template_id
version_hash
role_type                       planner/coder/reviewer/searcher/tester/generalist 等
harness, harness_version
system_prompt_ref
tool_schema_ref
sampling_params
description_ref                 G3 描述编码器的输入：system prompt 与工具列表的可读描述
```

`role_type` 用于 G3 区分"已见角色类型的新模板"与"全新角色类型"。同一 harness 内改动 system prompt 或工具列表必须产生新的 `version_hash`。

### 3.4 Application Definition

仅多角色应用需要：角色集合、初始节点与边、分支规则、图生成规则，每项带稳定 ID。单循环 agent 与运行时按需 spawn 子 agent 的 harness 不需要应用定义，`application_definition_id` 为空。

### 3.5 Benchmark Item

```text
benchmark_id, benchmark_version
benchmark_item_id
task_type                       运行前可见的离散标签
difficulty                      可空
```

## 4. 运行时记录

### 4.1 Application Run

```text
application_run_id
experiment_config_id
application_definition_id       可空
benchmark_id, benchmark_item_id, task_type
seed
retry_of_application_run_id     可空
concurrency_group_id            同时运行的一组 application run 共享
run_started_at
run_finished_at
run_status                      running/succeeded/failed/cancelled/timeout
benchmark_score                 可空，任务完成后回填
```

### 4.2 Session

```text
session_id
application_run_id
agent_template_id, version_hash
role_type
parent_session_id               子 agent 填父 session，否则为空
spawned_by_tool_call_id         创建该 session 的工具调用，否则为空
predecessor_session_ids         多角色流水线中的前驱，否则为空
created_at
started_at                      第一次组装 prompt
finished_at
final_status                    succeeded/failed/cancelled/timeout
```

单循环 agent 的一次运行是一个 application run 加一个 session。子 agent 的 session 在 `spawn_agent` 工具调用发出时创建，其 `created_at` 就是该调用的 `started_at`。

### 4.3 LLM Request

```text
llm_request_id
application_run_id
session_id
step_index
attempt_id
prev_tool_call_ids              本请求新进入 prompt 的工具结果对应的调用，可为多个
request_created_at              harness 组装完 prompt
request_sent_at                 发往模型服务
engine_received_at
first_scheduled_at              首次进入某个 engine step
first_token_at
inference_finished_at
response_received_at            harness 收到完整响应
num_cached_tokens               prefix cache 命中的 token 数
num_prompt_tokens
num_output_tokens
finish_reason                   stop/length/tool_call/abort/error
model_profile_id
engine_instance_id
prompt_ref                      渲染后的完整 prompt 文本
output_ref                      模型输出文本
request_status
```

三个阶段的 token 区间由上述字段直接给出：cached prefix 为 `[0, num_cached_tokens)`，new prefill 为 `[num_cached_tokens, num_prompt_tokens)`，decode 为 `[num_prompt_tokens, num_prompt_tokens + num_output_tokens)`。

`request_created_at` 与 `prev_tool_call_ids` 对应工具的 `finished_at` 之差是 harness 的组装开销；`request_sent_at` 之前的任何信息才允许作为该请求的预测特征。

### 4.4 Prompt Segment

每个请求的 prompt 由若干片段拼成，按渲染后的文本记录字符区间，并对齐到 token 区间：

```text
segment_id
llm_request_id
segment_type                    system/task/tool_output/harness_scaffold/agent_prior_output/predecessor_output/other
source_tool_call_id             tool_output 必填
source_session_id               predecessor_output 必填
produced_at
available_at
char_start, char_end
token_start, token_end          右开区间；对齐失败时为空并填 alignment_error
```

`harness_scaffold` 指 harness 在工具结果外包裹的固定文字；`agent_prior_output` 指本 session 之前轮次的模型输出。new prefill 区间内的片段是内容预测的直接对象。

### 4.5 Tool Call

```text
tool_call_id
llm_request_id_issuing          模型输出里包含这次调用的请求
llm_request_id_consuming        工具结果进入的请求，结束后回填；被丢弃时为空
session_id
call_index_within_request       同一请求内多个并行调用时递增
tool_name                       harness 内的工具名：bash、str_replace_editor、read_file、spawn_agent 等
tool_args_ref                   原始参数或命令行全文
issued_at                       模型生成完该调用的时刻，等于发出请求的 inference_finished_at
started_at                      工具进程或处理函数开始
finished_at                     工具进程或处理函数结束
exit_status                     进程退出码；非进程工具为空
timed_out                       bool
output_bytes_raw                原始 stdout 与 stderr 总字节数
output_bytes_seen               harness 截断后交给 agent 的字节数
spawned_session_id              spawn_agent 时填写
```

`issued_at` 是内容预测发生的时刻，`finished_at` 是下一次请求的 prompt 才可能存在的时刻，两者之差就是可用的 lead time。`tool_type` 与 `outcome_class` 不在此处记录，见第 5 节。

### 4.6 Tool Output Chunk

工具执行器以 pty 或管道 tee 的方式旁路记录原始输出流：

```text
tool_call_id
stream                          stdout/stderr
chunk_index
chunk_time
byte_offset, byte_length        指向该 tool call 的原始输出文件
```

每个 tool call 保存两个原始文件 `stdout` 与 `stderr`，chunk 只记时间与偏移。chunk 的划分以执行器读到数据的时刻为准，不按行切分。这份记录是进度读取器的全部训练数据：每个 chunk 时刻的剩余时间标签由 `finished_at - chunk_time` 派生。工作目录的文件系统事件不在第一阶段范围内。

### 4.7 Engine Step

```text
step_id
engine_instance_id
step_index
step_started_at
step_finished_at
num_tokens_total
entries: [(llm_request_id, phase, token_start, token_end)]
```

`entries` 列出本步为每个请求计算了哪些 token，`phase` 为 new_prefill 或 decode。批组成直接由 entries 给出。某个 token 在某层的 expert 需求时刻定义为该 token 所在步的 `step_started_at` 加模型 profile 中该层的延迟偏移；不得使用 cache miss 之后的实际计算开始时间，否则会把 miss 造成的停顿计入可用提前量。

### 4.8 Routing

每个请求一条，数组按 token 位置对齐：

```text
llm_request_id
model_profile_id
num_cached_tokens, num_prompt_tokens, num_output_tokens
token_ids            [T]          T = num_prompt_tokens + num_output_tokens
step_ids             [T]          计算该 token 的 step；cached prefix 为空
experts              [T_r, L, K]  T_r = T - 1，最后一个采样 token 无路由；uint16
scores               [T_r, L, K]  可选，fp16；缺失时记录 scores_unavailable_reason
unavailable_layer_ids
```

`L` 为 `moe_layer_ids` 的长度，层顺序与之一致。cached prefix 区间的路由由 fork 从更早的计算中重新暴露，保留用于分析，但不代表本请求的需求。expert ID 必须是 EPLB 映射之前的逻辑 ID。

### 4.9 Host Load

每个主机 1 Hz：

```text
time, host_id
load_avg_1m, cpu_util, mem_util
concurrent_tool_processes
gpu_util[], gpu_mem_used[]
net_rx_bytes, net_tx_bytes
```

用于解释同一命令在不同负载下的时长差异，以及检验进度读取器对负载变化的稳健性。

### 4.10 Cache 与搬运事件（系统实验阶段）

```text
cache_event_id
transfer_id                     load_started 与 load_finished 配对
event_time
layer_id, expert_id
event_type                      demand/hit/miss/load_started/load_finished/evicted
src_tier, dst_tier              gpu/dram/nvme
bytes
trigger                         on_demand/prefetch/eviction
policy_reason
llm_request_id, step_id         可空；由聚合预测触发的搬运不归属单个请求
```

作为独立事件流保存，不在路由记录中复制 cache 快照；驻留集通过按序重放事件得到。

### 4.11 多角色应用的图事件（仅 G2）

多角色应用另记只追加的图事件，用于恢复任意时刻预测器可见的图：`NODE_CREATED`（等价于 session 创建，附 `node_template_id` 或 `generation_rule_id`）、`EDGE_CREATED`、`EDGE_DEACTIVATED`、`BRANCH_DECIDED`（引用应用定义中的 `branch_rule_id` 与 `selected_outcome_id`）、节点生命周期事件。公共头部为 `event_id`、`application_run_id`、`event_sequence`、`event_time`、`graph_revision`、`caused_by_event_id`。每个 LLM Request 记录 `event_sequence_at_create`，离线代码不得读取更高序号的事件。未选择的分支只通过 branch outcome 表示，不伪造节点。

## 5. 离线派生标签

派生标签存放在 `derived/labels_v<N>/`，每个版本附带生成规则。

- **`tool_type`**：由 `tool_name` 与 `tool_args_ref` 归一化。首版分类：`test`（pytest、go test、cargo test、npm test 等）、`build`、`install`、`vcs`、`fs_read`、`fs_write`、`search`、`run_script`、`network`、`spawn_agent`、`other`。bash 类工具按命令行首个可执行程序与参数分类，管道取最重的一段。
- **`outcome_class`**：由 `exit_status`、`timed_out`、`tool_type` 与输出模式派生。首版取值：`pass`、`fail`、`error`、`timeout`、`empty`、`na`。`na` 用于不执行命令的工具，例如文件读取。
- **转移点标记**：请求级布尔量，满足任一条件为真：session 的第一步；子 agent 的第一步；`prev_tool_call_ids` 的 `tool_type` 与上一步不同；前驱角色与本 session 角色不同。
- **剩余时间标签**：每个 chunk 的 `finished_at - chunk_time`，以及在固定相对位置（如 50%、90%）处的快照，用于与时间预测基线比较。
- **需求直方图**：按 `(agent_template_id, tool_type, outcome_class, phase, layer)` 聚合的 expert 使用计数，是内容预测器的物化视图，可随时从路由记录重算。

## 6. 存储布局

```text
static/
  experiment_configs.jsonl
  model_profiles.jsonl
  layer_latency_profiles/<model_profile_id>.json
  role_templates.jsonl            + role_templates/<agent_template_id>/<version_hash>/{system_prompt.txt,tools.json}
  application_definitions.jsonl
  benchmark_items.jsonl
runtime/
  application_runs.jsonl
  sessions.jsonl
  llm_requests.jsonl
  prompts/<llm_request_id>.{prompt,output}.txt
  prompt_segments.parquet
  tool_calls.jsonl
  tool_output_chunks.parquet      + tool_outputs/<tool_call_id>.{stdout,stderr}
  engine_steps.parquet
  routing/<model_profile_id>/<llm_request_id>.npz
  host_load.parquet
  clock_sync.jsonl
  graph_events.jsonl              仅 G2
  cache_events.parquet            仅系统实验阶段
derived/
  labels_v<N>/
```

体积估算：256 expert 模型每 token 每层 8 个槽位，ID 2 字节加 score 2 字节，43 层约 1.4 KB；一个请求新增约 600 个 token 时约 0.8 MB。500 个任务、每任务 50 步约 20 GB 每模型每 benchmark，压缩前。四个模型、三个 benchmark 约 250 GB。若存储紧张，score 只对采样子集保留，ID 全量保留。

## 7. 校验规则

采集完成后必须通过：

1. 各类 ID 不可混用；`session_id`、`llm_request_id`、`tool_call_id`、`step_id` 全局唯一。
2. 同一 session 内 `step_index` 连续；重试产生新的 `attempt_id`，不覆盖旧请求。
3. 每个 `llm_request_id_consuming` 非空的 tool call，其 `finished_at` 不晚于对应请求的 `request_created_at`；`prev_tool_call_ids` 与之互相一致。
4. `issued_at` 等于发出请求的 `inference_finished_at`，误差在时钟精度内。
5. 所有 chunk 的 `chunk_time` 落在 `[started_at, finished_at]` 内，偏移与原始文件长度一致。
6. `num_cached_tokens <= num_prompt_tokens`；路由数组形状为 `[T-1, L, K]`，expert ID 落在 `[0, num_routed_experts)`，score 有限且非负。
7. 每个 new prefill 与 decode 的 token 恰好出现在一个 engine step 的 entries 中，且 `step_ids` 与之一致；cached prefix 的 token 不出现在任何 step 中。
8. `step_started_at` 单调递增；同一 step 内各 entry 的请求互不相同。
9. prompt segment 的字符区间不重叠且覆盖整个 prompt；对齐失败的片段有非空 `alignment_error`。
10. 子 agent session 的 `created_at` 不早于 `spawned_by_tool_call_id` 的 `started_at`，其 `finished_at` 不晚于该调用的 `finished_at`。
11. 所有 session 在 application run 结束前进入终态。
12. 每个 `experiment_config_id` 下存在通过的 capture 开关两侧逐 token 输出等价性测试记录。
13. 图事件（G2）：节点先创建后使用；边的两端已存在；分支产生的结构可追溯到 `BRANCH_DECIDED`；`event_sequence_at_create` 不引用其后的事件。
14. 跨主机采集时存在覆盖整个运行时间的 clock_sync 记录。

## 8. Benchmark 与 harness

所有主实验都必须通过目标 MoE 模型重新执行并采集，其他模型产生的 expert ID 不可迁移。

| harness                                    | benchmark                  | 图形态                     | 工具                 | 用途                                          |
| ------------------------------------------ | -------------------------- | -------------------------- | -------------------- | --------------------------------------------- |
| mini-SWE-agent                             | SWE-bench Verified         | 单循环                     | 单一 bash            | 最易插桩，与 Ask the Tool 直接可比，G1 主数据 |
| OpenHands                                  | SWE-bench Verified         | 单循环加 delegation        | bash、编辑器、浏览器 | 多工具类型；delegation 提供子 agent           |
| Terminal-Bench 2.0 自带 harness            | Terminal-Bench 2.0         | 单循环                     | 任意 shell 命令      | 工具长尾，进度读取器覆盖"所有工具"的主战场    |
| MetaGPT 或 ChatDev 2.0                     | 各自任务或 MAS-PromptBench | 多角色流水线，可构造配对图 | 代码执行             | G2 配对图；多角色下的 G3                      |
| Claude Code 形态，如 OpenCode 或 Codex CLI | 自选 coding 任务           | 主 agent 加并行子 agent    | 完整工具集           | 与 cc-traces 的时序和扇出对齐                 |

首批顺序：mini-SWE-agent、Terminal-Bench 2.0、OpenHands、多角色框架。

### 8.1 插桩点

- **请求元信息**：harness 通过 OpenAI-compatible 接口的 `extra_body.tokenmoe` 传入 `session_id`、`agent_template_id`、`version_hash`、`step_index`、`prev_tool_call_ids`；fork 在引擎侧记录请求与这些 ID 的映射，并把路由数组随响应返回或写入服务端文件。
- **工具执行器**：替换 harness 的命令执行函数，为每次执行分配 `tool_call_id`，以 pty 或管道 tee 原始输出并记录 chunk 时间，执行结束后按 harness 原有逻辑截断再交给 agent。docker 沙箱下包裹 `docker exec`。
- **子 agent**：spawn 类工具的执行函数在创建子 session 时写入 `spawned_session_id` 与 `parent_session_id`。
- **引擎步**：fork 的调度器在每步结束时写出 step 记录。

### 8.2 采集矩阵

```text
benchmark item x role template 集合 x model profile x seed x 并发配置
```

同一角色模板必须出现在多个任务上；G2 需要同一任务由相同角色集合、不同依赖关系的配对应用执行；每个模型至少覆盖单 session 与一个多 session 并发配置。工具时长记录必须来自真实并发运行，同一任务在不同并发配置下的重复运行用于量化负载对工具时长的影响。

### 8.3 cc-traces 的用途

`semianalysisai/cc-traces-weka-062126` 只包含 64-token 块的哈希与时间戳，没有文本、token 与角色标签，不能用于产生路由。它只用于两件事：校准多 session 叠加时的到达与工具间隔分布；检验自采数据的间隔分布与子 agent 扇出是否与真实 Claude Code 会话一致。

## 9. 模型

| 模型               | 专家配置                          | 备注                                                         |
| ------------------ | --------------------------------- | ------------------------------------------------------------ |
| Qwen3-30B-A3B      | 48 个 MoE 层，128 选 8            | fork 已支持，迭代与首轮特征化用                              |
| GPT-OSS-120B       | 36 层，128 选 4，无 shared expert | 模型类已在 fork 中，需补 score 语义与等价性验证              |
| DeepSeek-V4-Flash  | 43 层，256 选 6 加 1 shared       | 已在 fork 中注册，需验证捕获；FP8/FP4 权重约 168 GB          |
| dots3-note Preview | 256 选 8 加 1 shared，含 MTP      | fork 中不存在，需 rebase 到 2026 年 8 月后的 vLLM main；采集时关闭 MTP，只用文本 |

每个模型采集前完成：模型 profile 填写，逐层延迟 profile 测量，capture 开关两侧的输出等价性测试。

## 10. G1、G2、G3 切分规则

- **G1**：训练与测试使用相同的角色模板与应用形态，测试集为新的 benchmark item；同一 item 的全部 session 与请求属于同一 split。
- **G2**：测试应用的定义整体不进入训练集，但其角色模板均在训练集中出现过；不按运行结束后的最终图随机切分；优先使用相同角色集合、不同依赖关系的配对应用。
- **G3**：某个角色模板的全部历史从训练集移除，测试时只能使用其 `description_ref`、当前图与任务输入；第一次运行结束后才允许使用该角色新产生的记录；报告第 0、1、2、4、8、16 次运行后的学习曲线；区分 `role_type` 已见与未见两种情况。

三种切分都以 `request_sent_at` 为界：一个请求的预测只能使用在此之前 `available_at` 或事件时间已到达的记录。