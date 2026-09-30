# Agent 负载下 MoE Router 数据采集规范

本规范定义 TokenMoE 自采数据的内容、产生方、格式与校验规则。目标是一次采集同时支撑 仓库根目录 `#idea.md` 中的四类用途：

| 用途               | 要回答的问题                                                 | 依赖的记录                                                |
| ------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| 内容预测           | 给定角色、刚调用的工具、工具结果类别，下一次请求新增 prefill 与 decode 在每层的 expert 使用分布 | 请求记录、逐 token 逐层路由、工具调用记录、角色模板注册表 |
| 时间预测           | 正在运行的工具还剩多久、结果是什么                           | 工具调用记录、带时间戳的工具输出流、主机负载              |
| 驻留模拟与系统实验 | 每层每个 expert 何时被哪个 session 需要；多 session 下的批组成与排队 | 引擎步记录、请求生命周期时间戳、模型静态信息、cache 事件  |
| 泛化切分 G1/G2/G3  | 哪些历史记录可以迁移到新任务、新组合、新角色                 | 角色模板版本、角色类型、应用定义、任务标识、种子          |

数据由三个产生方写出，通过共同 ID 离线对齐：agent harness（session、请求、prompt 组成）、工具执行器（工具调用与输出流）、模型服务即 TokenMoE vLLM fork（引擎步、路由）。

## 1. 设计决策

1. Router 不变。只记录 router 的真实选择，不训练或替换 router。
2. prefix caching 开启。只保存本请求实际计算的路由，不重新暴露命中前缀的历史路由。`num_cached_tokens` 固定为首次调度的起始 token 位置；后续更长缓存命中造成的非连续区间用 token_positions 表示。抢占后重复计算标为 `recompute`，路由保留首次计算结果。
3. decode 路由必采。采集使用真实生成，不使用 `max_tokens=1` 的 prompt-only 模式。最后一个采样 token 没有路由，属正常现象。
4. 工具输出在执行器旁路捕获。将工具 stdout 与 stderr 合并成一个管道流，记录原始字节，harness 对 agent 的截断与过滤照常进行，agent 看到的内容不变。
5. 标签离线派生。采集时只记原始命令、退出码、输出与时间；`tool_type`、`outcome_class`、转移点标记等由版本化的离线规则生成，便于修改分类法后重算。
6. 时序按引擎步记录。每次前向记一条步记录，expert 的需求时刻由步开始时间加该层的延迟偏移得到，不为每层每步单独打时间戳。
7. 路由按请求存为 expert ID 数组，不按 (token, layer, slot) 展开成行；不采集 router scores。完整请求的行覆盖 `[num_cached_tokens, T-1)`；异常请求保存实际计算的 `[row_start, row_end)`，以 `routing_complete` 标记完整性。E ≤ 256 时用 uint8，否则用 uint16。
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

全局唯一的 ID：`application_run_id`、`session_id`、`llm_request_id`、`tool_call_id`、`engine_instance_id`、`step_id`、`cache_event_id`。静态 ID：`experiment_config_id`、`engine_config_id`、`agent_template_id`、`application_definition_id`、`benchmark_item_id`、`model_profile_id`、`host_id`。局部序号：`step_index`（session 内从 0 递增）、`attempt_id`（同一步重试时递增）、`chunk_index`（同一 tool call 内递增）。

运行时 ID 使用 `run_`、`ses_`、`req_`、`tc_`、`eng_` 前缀与按时间排序的随机后缀；`step_id = <engine_instance_id>:<step_index>`。引擎 step_index 在每个 engine lifetime 内从 0 递增。静态配置 ID 和模板版本使用规范化 JSON 的 SHA-256 前 16 位。

### 时钟

所有时间戳使用宿主机 `time.monotonic_ns()`，单位纳秒；`clock_domain_id` 为 `/proc/sys/kernel/random/boot_id`。每个产生方启动时记录单调时钟与 wall-clock 锚点对；工具时间戳在宿主机采集。第一阶段要求 harness、工具执行器和模型服务在同一主机。跨主机时每个产生方记录时钟校准：

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
engine_config_id                关联有效引擎配置和等价性记录
model_profile_id
engine: vllm_commit, tensor_parallel, expert_parallel, data_parallel, dtype, quantization,
        max_model_len, gpu_memory_utilization, max_num_seqs,
        enable_prefix_caching (必须为 true), scheduling_policy,
        max_num_batched_tokens, block_size, model_runner_version,
        async_scheduling (必须为 false), capture_routed_experts
harness: name, version, sandbox_type (docker/podman/local), cpu_quota, memory_limit
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
tokenizer_hash
mtp_disabled = true
multimodal_disabled = true
layer_latency_profile_ref       第一阶段为 null，系统实验阶段再测逐层前向延迟
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
vllm_request_id                 引擎内部 ID，含 vLLM 随机后缀
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
num_cached_tokens               首次调度起始 token 位置，抢占后不改写
num_preemptions
sampling_seed                   由 base_seed、benchmark_item_id、step_index、attempt_id 稳定派生
num_prompt_tokens
num_output_tokens
finish_reason                   stop/length/tool_call/abort/error
model_profile_id
engine_instance_id
prompt_ref                      渲染后的完整 prompt 文本
output_ref                      模型输出文本
request_status
assistant_response              完整的 content、reasoning、tool_calls
```

完整请求的 cached prefix 为 `[0, num_cached_tokens)`，new prefill 为 `[num_cached_tokens, num_prompt_tokens)`，实际计算的 decode 为 `[num_prompt_tokens, T-1)`，其中 `T = num_prompt_tokens + num_output_tokens`；最后一个采样 token 无路由。异常请求以实际的 row_end 截断；重算单独标为 recompute。

`first_scheduled_at` 取首次调度步的 dispatch 时间；`first_token_at` 取生成首 token 的步的 output-ready 时间；正常完成的 `inference_finished_at` 取完成步的 output-ready 时间，取消请求取处理取消时间。缺少引擎记录时 finalize 保留 harness 记录并标记 `engine_record_missing`。

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
token_start, token_end          右开区间；对齐失败时字符/token 区间都为空
alignment_error                成功时为 null，定位失败时必填
```

`harness_scaffold` 指 harness 在工具结果外包裹的固定文字；`agent_prior_output` 指本 session 之前轮次的模型输出。new prefill 区间内的片段是内容预测的直接对象。

离线解码引擎 prompt token IDs（保留特殊 token），按消息顺序精确定位，再尝试去空白定位；按 harness provenance 中的多个字符区间拆出工具 payload。同一调用的长输出头尾分别记为 tool_output，中间固定模板记为 harness_scaffold；输出文本不作修改。角色头、特殊 token、工具调用语法等未归属文字记为 other。字符区间索引到 prompt_ref 文件；增量解码得到 token 偏移，跨边界 token 归其首字符所在片段。无法定位的消息保留空区间和 alignment_error，未认领文字仍由 other 覆盖。pilot 消息定位率低于 99% 时按 Q1 回退到客户端渲染并通过 /v1/completions 发送 token IDs。

### 4.5 Tool Call

```text
tool_call_id
llm_request_id_issuing          模型输出里包含这次调用的请求
llm_request_ids_consuming        所有携带该结果的请求 ID 列表（包括重试）；未消费时为空列表
session_id
call_index_within_request       同一请求内多个并行调用时递增
tool_name                       harness 内的工具名：bash、str_replace_editor、read_file、spawn_agent 等
tool_args_ref                   原始参数或命令行全文
issued_at                       模型生成完该调用的时刻，等于发出请求的 inference_finished_at
parsed_at                       harness 解析命令的时间
started_at                      工具进程或处理函数开始
finished_at                     工具进程或处理函数结束
exit_status                     进程退出码；非进程工具为空
timed_out                       bool
output_bytes_raw                合并输出文件字节数
output_bytes_seen               交给 agent 的 observation 全文 UTF-8 字节数
spawned_session_id              spawn_agent 时填写
```

`issued_at` 是内容预测发生的时刻，`finished_at` 是下一次请求的 prompt 才可能存在的时刻，两者之差就是可用的 lead time。`tool_type` 与 `outcome_class` 不在此处记录，见第 5 节。

### 4.6 Tool Output Chunk

工具执行器以 stdout=PIPE、stderr=STDOUT（SWE-bench 使用 bash -c 和 BASH_ENV=/root/.bashrc） 合并管道 tee 原始输出流：

```text
tool_call_id
chunk_index
chunk_time
byte_offset, byte_length        指向该 tool call 的原始输出文件
```

每个 tool call 保存一个 `tool_outputs/<tool_call_id>.out` 文件；chunk 只记时间与偏移，offset 连续铺满文件。超时杀死 exec 进程、保留已收到的输出，按 mini-swe-agent 2.4.6 返回包含 partial output、returncode=-1、exception_info 的错误结果，再渲染原 observation。chunk 的划分以执行器读到数据的时刻为准，不按行切分。这份记录是进度读取器的全部训练数据：每个 chunk 时刻的剩余时间标签由 `finished_at - chunk_time` 派生。工作目录的文件系统事件不在第一阶段范围内。

### 4.7 Engine Step

```text
step_id
engine_instance_id
step_index
t_sched                        调度开始
step_started_at                 t_dispatch，派发模型执行前
step_finished_at                t_output，模型输出可用后
t_end                          处理模型输出后
num_tokens_total
num_running, num_waiting, kv_cache_usage
entries: [(llm_request_id|null, vllm_request_id, phase, token_start, token_end)]
```

`entries` 覆盖本步实际调度的全部 token，无 TokenMoE ID 的请求也记录（llm_request_id 为 null）。只记录至少计算一个 token 的步。每请求维护实际已计算区间（初始命中前缀作为已覆盖范围）：已覆盖位置为 recompute，首次计算且低于 num_prompt_tokens 为 new_prefill，其余首次计算为 decode；跨边界时拆分 entries，同请求在同一步可有多条但不重叠。批组成直接由 entries 给出。某个 token 在某层的 expert 需求时刻定义为该 token 所在步的 `step_started_at` 加模型 profile 中该层的延迟偏移；不得使用 cache miss 之后的实际计算开始时间，否则会把 miss 造成的停顿计入可用提前量。

### 4.8 Routing

每个带 TokenMoE ID 的完成请求一份未压缩 npz：

```text
token_ids            [T] int32    T = num_prompt_tokens + num_output_tokens
row_start            标量        = num_cached_tokens（从未调度时为 0）
row_end              标量        实际已捕获的右开结束位置
routing_complete     bool        完整生成且路由覆盖到 T-1
token_positions      [R] int32    每行绝对 token 位置，严格递增且唯一
experts              [R, L, K] uint8（E <= 256），否则 uint16
step_index           [R] int32，每行首次计算时的引擎步序号
layer_ids            [L]          与 layer_map.json 一致，按层号升序
```

完整请求要求 row_end = T-1。取消或报错请求保留实际计算的 `[row_start, row_end)`，routing_complete=false；未调度时保存零行，不伪造前缀命中。正在执行的请求取消时，等待当前步的路由归集后再写终态；已计算但未采样下一个 token 时 row_end 可以等于 T。

第 i 行对应绝对位置 token_positions[i]。连续情况等于 row_start+i，R=row_end-row_start。抢占后更长的缓存命中可能跳过本请求未计算的位置，R 可以更小；不补零、不挪用其他请求路由。后续若真正计算这些缺口，按 new_prefill/decode 记录，并在写出时按绝对位置排序。routing_complete 表示完整生成的实际计算均已捕获，不要求缓存缺口产生路由。L 只含实际绑定捕获的 MoE 层，dense 层不进入数组；layer_map.json 标记 router、monolithic 或 capture_source 路径，无绑定层则启动失败。expert ID 为 EPLB 映射前的逻辑 ID，取自实际执行步的请求 slice，不从 slot buffer 读取；重算保留首次结果，cached prefix 不存。

llm_request_id 由文件名给出；model_profile_id 在请求记录与 routing 父目录，token 计数和 engine_instance_id 在请求记录。逻辑 step_id 由 engine_instance_id 与每行 step_index 拼接，不在 npz 重复存字符串。

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
  equivalence/<engine_config_id>.json
raw/
  harness/<host>-<pid>/           harness 独立 jsonl shards 与工具原始输出
  engine/<engine_instance_id>/    engine_meta.json、layer_map.json、steps.jsonl、requests.jsonl、routing/
runtime/
  application_runs.jsonl
  sessions.jsonl
  llm_requests.jsonl
  prompts/<llm_request_id>.{prompt,output}.txt
  prompt_segments.parquet
  tool_calls.jsonl
  tool_output_chunks.parquet      + tool_outputs/<tool_call_id>.out
  engine_steps.parquet
  routing/<model_profile_id>/<llm_request_id>.npz
  host_load.parquet
  clock_sync.jsonl
  graph_events.jsonl              仅 G2
  cache_events.parquet            仅系统实验阶段
derived/
  labels_v<N>/
```

finalize 保留 raw；路由和工具输出硬链接到 runtime，跨文件系统时才复制。prompt/output 文本由引擎 token IDs 解码后保存（含特殊 token）；assistant 的解析结果留在请求记录。第一阶段不产生 graph/cache/clock_sync 事件及逐层延迟 profile。

估算假设每 session 60 步，每步实际计算约 1.3K token（总计 78K），上下文从 3K 增至 63K、累计约 2.0M token，N=1，未压缩并保留 raw。expert ID 每计算 token 占 L·K 字节（E > 256 时翻倍）。step_index 和 token_positions 各约 0.31 MB/session；其余项约 24 MB/session：token IDs 8 MB，prompt/output 文本 7 MB，引擎步 raw jsonl 6 MB 加 parquet 1 MB，工具输出 1 MB，harness 增量记录不足 1 MB。token IDs 和 prompt 文本随 session 长度呈平方增长。

| 模型 | L × K | expert IDs/session | 总计/session | 500 sessions |
|---|---|---|---|---|
| Qwen3-30B-A3B（调试） | 48 × 8 | 30 MB | 54 MB | 27 GB |
| GPT-OSS-120B | 36 × 4 | 11 MB | 35 MB | 18 GB |
| DeepSeek-V4-Flash | 43 × 6 | 20 MB | 44 MB | 22 GB |
| dots3-note | L × 8（L 从 config 读取） | 0.62·L MB | 0.62·L + 24 MB | 0.31·L + 12 GB |

每模型 N=1 加至少一个并发配置，总量约翻倍；pilot 按实际 token 数、文件占用（硬链接去重）和 session 数更新估算。

## 7. 校验规则

采集完成后必须通过：

1. 各类 ID 不可混用；`session_id`、`llm_request_id`、`tool_call_id`、`step_id` 全局唯一。
2. 同一 session 内 `step_index` 连续；重试产生新的 `attempt_id`，不覆盖旧请求。
3. 每个 `llm_request_ids_consuming` 非空的 tool call，其 `finished_at` 不晚于列表内每个请求的 `request_created_at`；`prev_tool_call_ids` 与之互相一致。
4. `issued_at` 等于发出请求的 `inference_finished_at`，误差在时钟精度内。
5. 所有 chunk 的 `chunk_time` 落在 `[started_at, finished_at]` 内，偏移与原始文件长度一致。
6. `0 <= num_cached_tokens <= num_prompt_tokens`；row_start 等于首次缓存量（未调度时为 0），`0 <= row_start <= row_end <= T`，数组形状为 `[R, L, K]`，token_positions 形状为 `[R]`、严格递增且位于 `[row_start,row_end)`；连续时 R=row_end-row_start，非连续时按 positions 校验。完整请求要求 routing_complete=true 且 row_end=T-1；异常请求按实际区间校验。layer_ids 与 profile 一致，expert ID 落在 `[0, num_routed_experts)`，dtype 遵循 §4.8；step_index 与路由行数一致。
7. 每个实际计算的 new_prefill/decode token 恰好出现一次非 recompute entry，step_index 与首次计算步一致；额外计算只能标为 recompute。首次缓存前缀不得产生 new_prefill/decode entry，抢占后可产生 recompute entry。异常或非连续请求以 token_positions 为准；不要求后来缓存跳过的缺口有本请求路由。
8. 同一 engine lifetime 内 step_started_at 单调递增；同一步内同一请求的 entries 不重叠，各 entry 长度之和等于 num_tokens_total。
9. prompt segment 的字符区间不重叠且覆盖整个 prompt；对齐失败的片段有非空 `alignment_error`。
10. 子 agent session 的 `created_at` 不早于 `spawned_by_tool_call_id` 的 `started_at`，其 `finished_at` 不晚于该调用的 `finished_at`。
11. 所有 session 在 application run 结束前进入终态。
12. 每个 experiment config 引用的 engine_config_id 必须有通过的 `static/equivalence/<engine_config_id>.json`。固定 50 个 SWE-bench 首步 prompt，batch=1、greedy、max_tokens=256，按 off/off/on/on 四次运行。on/off token 序列不一致的 prompt 数不得超过 off/off 基线加 1；每捕获行含 K 个不同合法 ID；两次 on 在 token 一致的位置路由也一致（含 TP > 1）。
13. 图事件（G2）：节点先创建后使用；边的两端已存在；分支产生的结构可追溯到 `BRANCH_DECIDED`；`event_sequence_at_create` 不引用其后的事件。
14. 跨主机采集时存在覆盖整个运行时间的 clock_sync 记录。

第一阶段实现规则 1–9、11、12；10、13、14 报告 not_applicable。validation.json 对每条规则给出状态、违规计数和最多 20 个例子 ID；任一适用规则失败时返回非零退出码。

## 8. Benchmark 与 harness

所有主实验都必须通过目标 MoE 模型重新执行并采集，其他模型产生的 expert ID 不可迁移。

| harness                                    | benchmark                  | 图形态                     | 工具                 | 用途                                          |
| ------------------------------------------ | -------------------------- | -------------------------- | -------------------- | --------------------------------------------- |
| mini-SWE-agent                             | SWE-bench Verified         | 单循环                     | 单一 bash            | 最易插桩，与 Ask the Tool 直接可比，G1 主数据 |
| OpenHands                                  | SWE-bench Verified         | 单循环加 delegation        | bash、编辑器、浏览器 | 多工具类型；delegation 提供子 agent           |
| Terminal-Bench 2.0 自带 harness            | Terminal-Bench 2.0         | 单循环                     | 任意 shell 命令      | 工具长尾，进度读取器覆盖"所有工具"的主战场    |
| MetaGPT 或 ChatDev 2.0                     | 各自任务或 MAS-PromptBench | 多角色流水线，可构造配对图 | 代码执行             | G2 配对图；多角色下的 G3                      |
| Claude Code 形态，如 OpenCode 或 Codex CLI | 自选 coding 任务           | 主 agent 加并行子 agent    | 完整工具集           | 与 cc-traces 的时序和扇出对齐                 |

首批只实现 mini-swe-agent 2.4.6 × SWE-bench Verified，单循环、单主机；其他 harness 留待后续。benchmark_score 和 layer_latency_profile_ref 均为 null；task_type 取 repo 名，difficulty 取数据集标签。

### 8.1 插桩点

- **请求元信息**：POST /v1/chat/completions 的 body 通过 `vllm_xargs.tokenmoe_llm_request_id` 传入本次请求 ID。session/template/step/工具 provenance 留在 harness 侧，按 ID 离线 join。重试沿用 step_index、递增 attempt_id、分配新请求 ID；seed 由 `(base_seed, benchmark_item_id, step_index, attempt_id)` 稳定 hash 派生。
- **消息与 reasoning**：保存 messages_delta 和逐消息 provenance（segment_type、source_tool_call_id、payload 字符偏移），tools 和采样默认值存 role template。消息列表不是前一请求的追加扩展时保存全量；assistant 的 content、reasoning、tool_calls 完整保存，后续请求回传 reasoning 字段。
- **工具执行器**：包裹 docker/podman exec，合并管道 tee 原始输出，每次 os.read 记录一个 chunk，返回值、超时异常和 observation 模板保持 mini-swe-agent 行为。每 session 独立容器，任何退出路径均清理。
- **引擎记录**：仅设置 TOKENMOE_TRACE_DIR 时启用，每 engine lifetime 独立目录；启动 worker 前导出 TOKENMOE_TRACE_ENGINE_DIR，TP rank 0 写 layer_map.json。engine_meta.json 记录 raw_format_version=1、有效配置、vLLM 版本、fork commit、时钟锚点和 PID。Scheduler 归集请求生命周期、计算区间与每步 routing slice；writer 线程写 steps.jsonl、requests.jsonl 和 routing/<id>.npz，shutdown 时 flush。启用 recorder 时跳过 API 路由组装，允许 upstream 保留 routed_experts: null 字段；关闭时保持 upstream 行为。
- **启动约束**：权重加载前拒绝关闭 prefix caching、未开启 routed-experts capture、启用 speculative decoding、async scheduling 或 DP > 1 的配置；PP、DCP/PCP、KV connector 按 upstream capture 约束拒绝。serve 显式使用 --no-async-scheduling。
- **离线关联**：按 llm_request_id 合并 producer 文件；issued_at 取 issuing request 的 inference_finished_at，llm_request_ids_consuming 从 prev_tool_call_ids 回填；解码 prompt/output、对齐片段、输出 §6 布局并运行 validator。
- **运行与环境**：每个 run 使用新目录并关联一个 engine lifetime；harness 连接已启动的 server，脚本负责 server 生命周期。独立 harness/fork venv，tokenmoe_collect 不 import vllm。旧 torch 2.11 venv 使用 `PYTHONPATH=/home/youwei/bzh/project/TokenMoE-vllm-0722` 加载保留的 90025dce2 代码和编译产物；v0.30.0 使用新 venv。

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
| Qwen3-30B-A3B | 48 个 MoE 层，128 选 8 | 仅本地 A100 调试；TP=2、YaRN ×4 到 128K、qwen3 reasoning/hermes tool parser，router 捕获 |
| GPT-OSS-120B | 36 层，128 选 4，无 shared expert | Hopper 主实验；补齐 monolithic Triton MXFP4 捕获，保持原有路由运算 |
| DeepSeek-V4-Flash | 43 层，256 选 6 加 1 shared | Hopper 主实验；默认非 MegaMoE 预期 router 路径，MegaMoE 为 capture_source；smoke 确认 backend 和 prefix cache |
| dots3-note Preview | 256 选 8 加 1 shared，含 MTP | v0.30.0 已注册；Hopper 主实验，预期 router 路径，L 从绑定层获得；关闭 MTP，仅文本 |

fork 基于 upstream v0.30.0（ced6857afa0e），分支 tokenmoe-v0.30.0-trace。Hopper parser 与 TP 按该版本 recipe 配置、smoke 确认。每模型先核对 layer_map、一次 routed request 与引擎步，再通过规则 12 等价性测试；本次不测逐层延迟。

## 10. G1、G2、G3 切分规则

- **G1**：训练与测试使用相同的角色模板与应用形态，测试集为新的 benchmark item；同一 item 的全部 session 与请求属于同一 split。
- **G2**：测试应用的定义整体不进入训练集，但其角色模板均在训练集中出现过；不按运行结束后的最终图随机切分；优先使用相同角色集合、不同依赖关系的配对应用。
- **G3**：某个角色模板的全部历史从训练集移除，测试时只能使用其 `description_ref`、当前图与任务输入；第一次运行结束后才允许使用该角色新产生的记录；报告第 0、1、2、4、8、16 次运行后的学习曲线；区分 `role_type` 已见与未见两种情况。

三种切分都以 `request_sent_at` 为界：一个请求的预测只能使用在此之前 `available_at` 或事件时间已到达的记录。