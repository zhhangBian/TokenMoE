# 采集状态与续采

驻留研究进度见 [screening-evidence.md](../optimization/docs/screening-evidence.md)：历史 Qwen 筛查已通过独立包复现，新增分析不改变原采集验收。缓存实测、工作负载扩展与较大模型资格须分别核验，不能以分析结果代替。

2026-10-23 的本地缓存与控制器诊断记录见 [backend-evidence.md](../optimization/docs/backend-evidence.md)，包括 TP2 全层张量对照、延迟分片、取消、故障退出和工具期间预取。正式 agent 性能对照尚未完成。

整理日期：2026-10-19。完成情况依据[采集对话截至 2026-10-17 的汇总](https://chatgpt.com/s/cx_6ad31ad8951481919643d0318553120c)。本次没有连接远端 DSW，也没有重新校验原始数据；以下完成标记沿用该汇总，待执行项目均未在本次启动。

正式全量目标是三个模型各 1,000 个会话，合计 3,000 个。记录中已完成并验收 1,000 个，DeepSeek、Dots 的 2,000 个正式会话待采集。历史 pilot 和诊断数据单独保存。

## 执行目标与 agent 职责

本轮以尽量完成三模型的实际采集、补齐任务覆盖和原始轨迹为目标。SWE-bench 解题正确率不作为停止或放行条件，也不要求用评分证明采集完成。正常提交、解题失败、工具错误、超时和步数耗尽都如实记录；已有有效轨迹和终态的失败会话属于采集结果。

正常提交率只用于诊断，不再要求达到 90% 才能继续全量。汇总命令显式使用 `--min-success-rate 0`。任务未成功提交时，执行 agent 分析原因并继续采集；不能仅因低提交率停在 pilot，或反复重跑同一批任务来提高比例。

执行采集的 agent 自行读取日志、定位运行问题、修改代码或配置、做针对性验证，并自动接续剩余任务。恢复私有运行时、修复容器启动、依赖、解析、调度、推理服务和落盘错误，都属于该 agent 的工作。模型生成了错误题解时保留实际输出，外部采集 agent 不向任务注入题解或伪造正常提交。

运行故障按以下方式处理：保留原始日志、输入、配置和失败状态，定位原因后修改，再用原失败输入验证；通过后以新会话或 attempt 接续。重试与原记录建立对应关系，不覆盖已完成数据，不重新采集 GPT 主集。同一条件下反复发生同一故障时，改变修复方法，不无限原样重试。只有访问入口、凭据或必要平台能力确实缺失、无法继续独立处理时，报告具体缺项。

数据结构、真实路由、工具原始输出、版本来源和任务资源限额仍须可信。没有发出真实模型请求的启动失败、丢失记录或损坏数据，标为待修复、待补采。collector 引入的数值差异、路由错位和数据缺失由 agent 修复后重采受影响部分；完整的业务失败轨迹继续保留。

## 已有数据

| 数据 | 已采集内容 | 完成标记 | 后续处理 |
|---|---|---|---|
| GPT-OSS 正式全量 | 500 个唯一任务，N1、N4 各 500 会话；正常提交 498/500、500/500 | 采集、终验、原始数据交付已完成 | 保留主集，不重复采集；恢复访问后核对交付回执与文件哈希 |
| DeepSeek eager pilot | N1、N4 各 20 会话；正常提交 8/20、10/20，22 个超时；该配置等价性 200 请求通过 | 历史 pilot 数据已产生，按旧提交率门槛未放行；正式全量未采集 | 分析超时；优先用有完整数据和有效等价性记录的配置继续全量，不等待提交率达到旧门槛 |
| Dots 历史 mem95/eager pilot | N1、N4 各 20 会话；正常提交 15/20、14/20，11 个超时 | 历史 pilot 数据已产生，按旧提交率门槛未放行；正式全量未采集 | 核对该配置的数据完整性、长输入和等价性记录，具备采集条件即继续；保留失败样本，与新候选分开 |
| Dots 新候选诊断 | 正式等价性仍有一个固定 prompt 的输出、路由差异；第 6 层 MLA 输出首次捕获数值差异 | 诊断未完成 | 定位差异来源，修复后重做该配置的正式等价性 |

GPT 的两个限额终止会话已有完整采集记录，属于主集中的失败结果。额外重试须另存，不覆盖原始会话。正常提交率与 SWE-bench 解题正确率分别记录，当前尚未计算 SWE-bench 得分。

GPT 主集还记录了：

- 12 个仓库、30,019 次 LLM 请求，对应路由文件齐全。
- 约 5.64 亿输入 token、482.31 万生成 token；输入量包含重复历史和缓存前缀。
- 30,017 次工具调用，原始输出约 4.53 GB。
- 约 396.75 万条引擎 step、682.19 万条输出 chunk、475.07 万条 prompt segment。
- 消息定位率 N1 99.87%、N4 99.80%；原始工具输出与 observation 的对应关系已通过记录中的验收。

这些数量沿用原汇总的精度，原始路径、精确计数及 SHA 待恢复远端或交付目录后核对。

## 待续采任务

- [x] GPT-OSS：正式全量 N1/N4 各 500 会话完成采集验收和原始数据交付。
- [ ] 恢复任务私有 namespace 运行时：取得启动器和包装入口源码，核验 keeper 身份、UID/GID 映射、VFS 存储及每任务 2 核、4 GiB 内核限额。
- [ ] 核对当前配置和状态：从实际 state/launch、进程身份及数据文件判断是否仍有任务运行，记录模型、权重、数据集、fork 和运行参数的版本；保留历史输出。
- [ ] DeepSeek：对历史 22 个超时会话按 `exit_reason`、请求结果和引擎进展重新分类，完成下文的时间分析。
- [ ] DeepSeek：优先恢复已有有效等价性记录、能稳定保存真实轨迹的配置，核对长输入、原始输出和开销。CUDA Graph 候选由 agent 继续修复验证，不把候选完成当成稳定配置继续采集的必要条件。
- [ ] DeepSeek：用 N1/N4 pilot 确认数据与运行流程可信后，继续正式全量 N1/N4 各 500 个任务，共 1,000 个会话；低提交率单独记录，超时按下文由 agent 自主分析和调整。
- [ ] Dots：比较失败 prompt 各 rank 的输入、权重、执行路径与第 6 层 MLA 输入/输出，找出首次实际计算差异；保留原失败张量和日志。
- [ ] Dots：核对历史已通过等价性的配置能否稳定运行长输入并保存完整数据，具备条件时用该配置继续；存在数值差异的新候选由 agent 独立修复，改动后重做其正式等价性和 pilot。
- [ ] Dots：数据与运行流程可信后，继续正式全量 N1/N4 各 500 个任务，共 1,000 个会话；历史 11 个超时和后续任务失败都保留，运行错误自动修复后接续。
- [ ] DeepSeek、Dots：完成各自全量终验、版本来源记录、原始数据交付及文件哈希核对。

SWE-bench 评分尚未执行，单独记录，不列入此次续采任务。每个模型的 pilot 仍检查消息定位率至少 99%、适用数据校验全部通过，以及 recorder 开销低于 5%；正常提交率作为统计保留。模型服务参数改变后，旧等价性记录不能替代新配置验证。改变时间预算的数据须与原 3,600 秒协议的数据分开保存。若远端控制器仍因提交率低于 90% 停止，执行 agent 同步修改该控制器的本轮策略并记录代码版本。

## 超时怎么处理

现有 DeepSeek 记录更支持“累计推理耗时用完会话预算”，还没有证据确认推理死锁。两个案例分别记录约 3,635 秒、3,581 秒推理耗时，工具执行约 15 秒、53 秒，且请求成功、路由记录完整。原始分段计时缺失，仍需区分排队、prefill、decode 和重试，不能把全部等待都算作 GPU 计算。

当前仓库的限制分别作用于不同层：

| 限制 | 当前配置 | 作用与判断方式 |
|---|---|---|
| 会话时间 | `time_limit_seconds: 3600` | 传给 mini-swe-agent 的会话墙钟预算；在每轮模型请求前检查，进行中的请求和工具动作可能使总耗时超出预算 |
| HTTP 请求等待 | `request_timeout: 600` | HTTPX 的连接、读取、写入和连接池等待设置；读取超时按网络数据块等待计算，不是整个会话的总截止时间 |
| 请求重试 | `request_retries: 2` | 同一轮最多三个 attempt；传输错误、429 和 5xx 可重试，各 attempt 有独立 ID 与 seed |
| 工具执行 | `tool_timeout: 60` | 作用于工具命令；应独立分析命令、输出、退出码和任务 cgroup 状态 |
| agent 步数 | `step_limit: 250` | 用完步数也会停止；仓库将 `LimitsExceeded` 与 `TimeExceeded` 都汇总为 `timeout`，须读取 `exit_reason` 区分 |
| 单次生成 | 模型配置的 `max_tokens: 8192` | 多次 `finish_reason=length` 可能表示长推理或循环生成，需检查实际输出和工具调用 |

上述行为来自当前仓库的 [launcher.py](tokenmoe_collect/launcher.py)、[client.py](tokenmoe_collect/client.py)，以及安装的 mini-swe-agent 2.4.6 `DefaultAgent.query`。HTTP 等待的语义见 [HTTPX 文档](https://www.python-httpx.org/advanced/timeouts/)。远端历史脚本曾做过修复，续采时须核对实际运行代码，不能直接假定与当前仓库一致。

对每个异常会话，记录 `session_id`、`exit_reason`、请求/attempt 数、生成 token 数、`finish_reason`、工具耗时和重试情况。再通过同一 `llm_request_id` 关联引擎记录，比较：

- `engine_received_at` 到 `first_scheduled_at`：排队等待。
- `first_scheduled_at` 到 `first_token_at`：首 token 之前的执行耗时，包含 prefill 等阶段。
- `first_token_at` 到 `inference_finished_at`：后续生成耗时，结合实际生成 token 数计算速度。
- `num_prompt_tokens`、`num_cached_tokens`、`num_preemptions`、引擎 step 的进展：长输入、缓存、抢占和调度影响。
- 原始响应与工具轨迹：重复推理、重复命令、连续生成到长度上限，以及正常提交是否在推进。

判定无进展时，至少比较两个相隔一定时间的引擎快照，窗口须覆盖同长度、同并发正常请求的合理 prefill 或生成耗时。单次 GPU 利用率低、日志暂时未更新，都不足以确认死锁。

| 观察结果 | 处理 |
|---|---|
| `TimeExceeded`，各请求正常完成，token 和任务操作持续推进，速度与同配置正常请求相符 | agent 自主对同一小批任务做 3,600 秒与 7,200 秒的有界对照；保持 GPU、实际请求 seed、采样和 token 上限不变，观察额外时间是否带来实际任务进展和更多有效轨迹，再决定续采预算 |
| `ReadTimeout`，引擎仍正常推进，但单个请求的合理耗时超过 600 秒 | 单独评估 HTTP 等待，例如 600 秒与 1,200 秒对照；检查旧请求是否已取消，防止重试增加排队负担。仅延长会话时间无法消除单请求读取超时 |
| `LimitsExceeded`，达到 250 步或反复执行相同动作 | 检查任务策略、提示词、工具协议和循环原因；延长等待时间不会增加步数预算 |
| 大量生成达到 8,192 token，内容重复、工具调用无进展 | 检查模型输出、停止条件、推理/工具解析和任务循环；生成仍在前进也可能是模型行为问题，不能直接判为引擎故障 |
| 引擎 step 与输出 token 长时间无进展，明显超过同输入正常耗时；出现 CUDA/NCCL 错误、worker 退出或服务不健康 | 保留 rank 日志、进程栈和原始输入，定位计算、通信、调度或服务故障；不能仅靠不断延长时间重试 |
| 主要时间在排队或频繁抢占 | 检查并发、KV 容量、批次和遗留请求；完成同配置 N1/N4 对照后再判断是否调整调度 |
| 工具超时或容器 OOM | 分析该命令、CPU 限流、内存事件和进程状态；与模型请求等待分开处理 |

3,600/7,200 秒和 600/1,200 秒对照均为待执行的诊断设计，本次未运行，也未修改正式配置。后续由执行 agent 依据进展自主选择和调整，每次只改变正在验证的时间参数。若延长后仍无进展，停止该对照并诊断、修复；若有实际进展，再依据吞吐和完成时间分布设置有上限的新预算并继续采集。预算变化写入新协议、配置和输出目录，完成对应的数据检查，原预算下的超时结果继续保留，不要求以题目解对或正常提交率提高作为调整依据。

Dots 当前候选还有数值一致性阻塞。延长等待不能使不一致的 token 或路由通过等价性，须先完成相应修复验证。

## Local Qwen collection completed — 2026-10-19

Experiment: `/home/youwei/bzh/dataset/tokenmoe_artifacts/qwen3-runtime-20261019-063626`.

All 40 primary planned units passed data validation: 20 Verified tasks at N1 and N4, with one model profile and one role version. The primary runs contain 685 attempted model calls (684 admitted and one evidenced context rejection), 1,021 tool calls, and 691,847 first-computed routing positions across 48 MoE layers. Rendered-field alignment is 100% in both runs; original rates are 97.5174% and 93.8178%, with all 249/524 proven template omissions retained. N4 session/request overlap peaks are both 4. Benchmark scores remain null and failed/limited trajectories remain in the dataset.

The 200-request off/off/on/on equivalence protocol passed with zero output or routing disagreements. The four-round fixed greedy/max_tokens=64 overhead comparison measured 3.689% overhead (7.220440 seconds off versus 7.486773 seconds on; two repetitions per mode, ten measured requests each). All 70 harness tests pass. The hardware, installed environments, selected image digests and actual 2 CPU/4 GiB container limits were verified.

CRLF validation, dropped optional Parquet evidence fields, and missing HTTP error-body capture were repaired. The one historical context rejection has separately disclosed exact-input replay evidence; its original response body remains marked unavailable. No original model outputs, raw tool bytes or routing arrays were changed or regenerated. Earlier derived reports and recovery provenance are retained.

Read `reports/runtime_summary.md`, `reports/data_notes.md`, `reports/coverage.json`, `reports/storage.json`, and `reports/artifact_checksums.json` under the experiment root. Primary data directories are `collection/N1/attempt-01/N1/` and `collection/N4/attempt-01/N4/`. G1 groups are in `derived/labels_v1/g1_groups.json`; pilot and other diagnostics are separate. Health snapshots were recorded during execution, and all experiment-owned model services have completed.
