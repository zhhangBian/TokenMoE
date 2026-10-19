# 三模型采集交接

采集 GPT-OSS-120B、DeepSeek-V4-Flash 和 dots3-note 执行 SWE-bench Verified 时的请求、工具输出和专家路由。每个模型运行 500 个任务，分别使用 N1、N4 两档并发，共 1,000 个会话；三个模型合计 3,000 个会话。

主流程采用“DSW 容器内的私有 namespace＋本地 Podman”：常驻 keeper 保留私有 user、mount、cgroup namespace，所有任务容器操作通过同一个包装入口进入其中执行。模型服务和采集脚本留在外层 DSW 容器内。整个流程使用同一 Linux 内核，无需宿主机 Docker 权限。字段定义和验收要求见 [data_collect.md](data_collect.md)，脚本参数见 [README.md](README.md)。

已完成数据、待续采任务和超时处理见 [COLLECTION_STATUS.md](COLLECTION_STATUS.md)。GPT 正式全量已有完成记录；DeepSeek、Dots 按该文件的待执行项目继续采集，避免重复已验收主集。

本轮以尽量完成采集为目标，SWE-bench 解题正确率不作为放行条件。正常提交率只作统计，不再用 90% 门槛停止全量。保留真实的失败、超时和限额轨迹；执行 agent 遇到运行错误后自行分析、修复、验证并接续任务。数据丢失、路由失真和采集器引入的数值差异仍须修复。

## 代码与运行环境

平台容器需要能访问 GPU，以及 Git、Python 3.12、uv 和私有运行时所使用的 Podman。CUDA 13 的驱动支持由平台提供。GPU 数量按下文的 TP 配置准备，模型兼容性通过 smoke 验证。下文命令均在分配到的平台容器内执行。

```bash
git clone --recurse-submodules https://github.com/zhhangBian/TokenMoE.git
cd TokenMoE
git -C vllm switch tokenmoe-v0.30.0-trace
```

主仓库使用 `main`，子模块 `vllm/` 使用 `tokenmoe-v0.30.0-trace`。已有仓库缺少子模块时，执行 `git submodule update --init --recursive`，再切换分支。

### 采集脚本位于 Docker 容器内时

运行时准备按以下顺序进行：

1. 创建本任务的私有 user、mount、cgroup namespace，由 keeper 常驻保留。记录 keeper 的 PID、启动时刻、完整命令、工作目录和 namespace ID，包装入口每次进入前核验身份，防止 PID 复用后进入其他进程。
2. 建立覆盖任务镜像所需 UID/GID 的映射。历史早期记录使用 `0 0 65536`，全量阶段又修过映射范围；新环境须按实际镜像核验，不能把早期范围当成所有镜像的固定值。
3. 在私有挂载视图中配置任务 cgroup，保留每任务 2 核、4 GiB 限制。PID、网络及已有受限 `/proc`、`/sys` 视图沿用外层 DSW 环境，`/proc`、`/sys` 按历史方案只读 bind。资源管理范围须来自平台已允许的权限。
4. 在这组 namespace 内配置任务专用的 VFS 镜像存储、runroot 和 tmpdir。镜像状态和临时路径由运行时统一管理；VFS 占用须纳入磁盘估算。
5. 通过包装入口调用本地 Podman，统一执行 pull、run、exec、rm 和 `check-runtime`。入口接受 Podman 参数，并固定本地执行，采集脚本不连接宿主机 Docker socket。

历史记录中的组件是 `private_runtime.py worker`、`private_podman_entry.py` 和 `private_podman.py`，早期使用 Podman 4.9.3。运行时启动器、包装入口、存储配置和 keeper 身份记录须一并交接，恢复运行时后用真实任务镜像检查内核限额。

本仓库已支持 `--runtime-executable` 入口适配和任务容器检查，尚未包含上述 namespace 启动器源码。以下命令从已部署、已核验的私有运行时入口开始；在新 DSW 上执行前，须恢复或补齐这段运行时准备。

创建私有 namespace 受平台的 seccomp、UID/GID 映射和 cgroup 权限约束。私有视图不会自动产生上层未授予的控制权。若启动或限额检查失败，保留具体错误；禁止关闭 cgroup 或去掉 2 核、4 GiB 限制放行正式采集。

## 安装和配置

在同一个 Bash 会话、仓库根目录执行下文命令。路径换成平台容器内可写的存储位置，`WORK_ROOT` 为本轮实验使用一个新目录。

```bash
set -euo pipefail
MODEL_ROOT=/data/tokenmoe/models
DATASET_DIR=/data/tokenmoe/datasets/SWE-bench_Verified
WORK_ROOT=/data/tokenmoe/runs/collection-01
HARNESS_VENV=/data/tokenmoe/venvs/harness
FORK_VENV=/data/tokenmoe/venvs/vllm030
HARNESS_PY="$HARNESS_VENV/bin/python"
CONFIG_ROOT="$WORK_ROOT/configs"
RUN_ROOT="$WORK_ROOT/results"
RUNTIME_ENTRY=/实际路径/private-runtime/bin/podman
test -x "$RUNTIME_ENTRY"
RUNTIME_ARGS=(--runtime podman --runtime-executable "$RUNTIME_ENTRY")
```

`RUNTIME_ENTRY` 填入本任务私有 Podman 包装入口的绝对路径。`test -x` 只检查文件可执行；keeper 身份、namespace 和存储状态由入口核验。仓库调用入口时会加 `--remote=false`。沿用私有运行时的 VFS、runroot 和 tmpdir 配置，不在采集过程中临时切换运行位置或镜像存储。

采集脚本和模型服务各用一个 venv：

```bash
uv venv --python 3.12 "$HARNESS_VENV"
uv pip install --python "$HARNESS_PY" -e 'collection[harness,dev]'
uv venv --python 3.12 "$FORK_VENV"
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install --python "$FORK_VENV/bin/python" --torch-backend=cu130 -e ./vllm
```

下载权重和数据集：

```bash
"$HARNESS_PY" collection/scripts/download_assets.py \
  --models gpt-oss-120b deepseek-v4-flash dots3-note \
  --model-root "$MODEL_ROOT" --dataset-dir "$DATASET_DIR"
```

已有全部权重时加 `--skip-models`，已有数据集时加 `--skip-dataset`。认证使用 Hugging Face 登录或 `HF_TOKEN`；指定下载源用 `--endpoint`。复现实验时，通过 `--model-revision 模型名=提交号`、`--dataset-revision 提交号` 固定版本，并记录实际下载的 revision。

默认权重目录为 `$MODEL_ROOT/openai/gpt-oss-120b`、`$MODEL_ROOT/deepseek-ai/DeepSeek-V4-Flash` 和 `$MODEL_ROOT/dots-studio/dots3-note-prev`。数据集目录中应有 Parquet 文件。权重、任务镜像和采集结果都占用磁盘，容量估算见[存储规范](data_collect.md#6-存储布局)。

生成本机配置：

```bash
"$HARNESS_PY" collection/scripts/prepare_configs.py \
  --model-root "$MODEL_ROOT" --dataset-dir "$DATASET_DIR" \
  --output-dir "$CONFIG_ROOT" --run-root "$RUN_ROOT" \
  "${RUNTIME_ARGS[@]}" --concurrency 1 4 --base-seed 42 \
  --step-limit 250 --time-limit-seconds 3600 \
  --tool-timeout 60 --request-timeout 600 --cpu-quota 2 --memory-limit 4g
mkdir -p "$RUN_ROOT/logs"
```

生成结果包含三份模型配置和六份运行配置。每份 `*-pilot.yaml` 含 20 个任务，每份 `*-full.yaml` 含 500 个任务，两者均运行 N1、N4。每个任务容器限制为 2 核、4 GiB。

权重目录不同时，生成配置的命令可加 `--model-path dots3-note=/实际路径`；调整 TP 用 `--tp gpt-oss-120b=4`。配置输出目录须为空或尚未创建。后面的采集命令使用生成的 YAML。

## 选模型，检查任务容器

每轮选择一个模型，GPU 数量须与模型配置中的 TP 一致。默认配置如下，设备编号换成机器上的空闲 GPU：

| `MODEL` | TP | `GPUS` 示例 |
|---|---|---|
| `gpt-oss-120b` | 2 | `0,1` |
| `deepseek-v4-flash` | 4 | `0,1,2,3` |
| `dots3-note` | 8 | `0,1,2,3,4,5,6,7` |

```bash
MODEL=gpt-oss-120b
GPUS=0,1
PORT=8000
MODEL_CONFIG="$CONFIG_ROOT/models/$MODEL.yaml"
PILOT_CONFIG="$CONFIG_ROOT/runs/$MODEL-pilot.yaml"
FULL_CONFIG="$CONFIG_ROOT/runs/$MODEL-full.yaml"

"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage pull-images --config "$PILOT_CONFIG" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-pull-pilot.log"

"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage check-runtime --config "$PILOT_CONFIG" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-runtime-check.log"
```

`check-runtime` 用首个已缓存的任务镜像实际启动容器，核对 Linux boot ID，并读取任务自身 cgroup 的 CPU 配额、周期和内存上限。配置为 2 核、4 GiB 时，读数须与之相符；忽略限额或读不到限额都不能通过。结果为 `passed: true` 后再运行 GPU 阶段。

镜像缺失、keeper 身份异常、namespace、UID/GID、cgroup 或挂载失败时，保留具体错误并修复，再重做检查。pilot/full 会在模型启动前通过同一包装入口执行任务检查，并保存包含内核限额读数的 `runtime_preflight.json`。keeper 的身份核验由包装入口负责。

## 验证和小规模采集

smoke 检查一条真实请求的层数、路由和引擎步记录：

```bash
"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage smoke --model-config "$MODEL_CONFIG" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --output-dir "$RUN_ROOT/$MODEL-smoke" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-smoke.log"
```

确认 `$RUN_ROOT/$MODEL-smoke/engine/smoke.json` 中为 `passed: true`。GPU 阶段由脚本启停模型服务，端口须空闲；服务启动超出默认 1,200 秒时，加 `--startup-timeout 秒数`。

等价性验证对固定 50 个首步 prompt 按 recorder 关闭、关闭、开启、开启运行，共 200 个请求：

```bash
"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage equiv --model-config "$MODEL_CONFIG" --dataset-dir "$DATASET_DIR" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --output-dir "$RUN_ROOT/$MODEL-equiv" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-equiv.log"
```

确认 `$RUN_ROOT/$MODEL-equiv/static/equivalence/*.json` 中为 `passed: true`，判断标准见[校验规则 12](data_collect.md#7-校验规则)。该测试每次最多生成 256 token，长会话的 OOM 和超时还要通过 pilot 检查。

等价性通过后，运行 N1、N4 各 20 个任务：

```bash
"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage pilot --config "$PILOT_CONFIG" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --equivalence-record "$RUN_ROOT/$MODEL-equiv" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-pilot.log"

"$HARNESS_PY" collection/scripts/summarize_runs.py \
  --run-dir "$RUN_ROOT/$MODEL-pilot" \
  --min-success-rate 0 \
  --output "$RUN_ROOT/logs/$MODEL-pilot-checks.json"
```

每个并发档都要通过数据校验，消息定位率至少 99%。命令中的 `--min-success-rate 0` 使正常提交率仅作为诊断统计，不因模型解题失败或低提交率停止采集。汇总中的 `automated_checks_passed: true` 只涵盖自动数据检查；仍须核对工具原始输出与 observation、测量 recorder 开销低于 5%，并按实际 token 数核对存储占用，见[实验验收](data_collect.md#71-实验验收)。

`Submitted` 表示 agent 正常提交。模型提交了错误题解或因限额结束时，完整轨迹仍作为采集结果保存。没有真实模型请求的启动失败、丢失或损坏的数据标为待补采。SWE-bench 解题得分不在本轮采集范围，`benchmark_score` 保持 `null`。

## 全量采集

当前模型的 pilot 确认运行流程、等价性和数据记录可信后，继续全量；低正常提交率单独记录，由执行 agent 分析处理：

```bash
"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage pull-images --config "$FULL_CONFIG" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-pull-full.log"

"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage full --config "$FULL_CONFIG" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --equivalence-record "$RUN_ROOT/$MODEL-equiv" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-full.log"

"$HARNESS_PY" collection/scripts/summarize_runs.py \
  --run-dir "$RUN_ROOT/$MODEL-full" \
  --min-success-rate 0 \
  --output "$RUN_ROOT/logs/$MODEL-full-checks.json"
```

`--stage full` 按传入配置的任务清单执行。本例使用生成的 `*-full.yaml`，包含 500 个任务。结束后核对 N1、N4 各有 500 个会话，并检查各档的 `validation.json`、`alignment.json` 和 `run_summary.json`。

换模型时，重新设置 `MODEL`、`GPUS` 和三个配置路径，再执行容器检查、smoke、等价性、pilot 和全量。修改采集代码、权重或服务参数后，用新实验目录重新验证；等价性记录须对应当前 `engine_config_id`。

长任务可放在 `tmux` 中。脚本没有自动断点续跑；中断后保留原目录，核对未完成任务，用新配置和新目录补跑。pilot/full 可用 `--output-dir` 指定新目录。

执行 agent 负责错误后的自动接续：保存日志、失败输入、配置和任务状态，定位运行问题并修复，用原失败输入做针对性验证，再以新会话或 attempt 重试受影响任务。已完成数据保留，失败重试记录来源关系。环境、容器、解析、推理服务或落盘错误都应主动处理；同一故障不能无限原样重试。私有启动器缺失时由执行 agent 恢复或实现并实测，只有缺少访问、凭据或必要平台能力而无法独立推进时才报告具体缺项。

## 历史采集结果（截至 2026-10-17）

完整数量、完成标记和续采清单维护在 [COLLECTION_STATUS.md](COLLECTION_STATUS.md)，依据当时的采集对话，尚未重新核对远端状态。记录中 GPT 正式全量已完成；DeepSeek pilot 未过正常提交率门槛，Dots 新候选未过等价性。

私有 namespace 运行时解决了早期容器启动阻塞，镜像回收和 UID/GID 映射问题也有后续修复记录。Dots 早期注意力临时显存 OOM 导致服务退出，连接失败和 trace drain 报错随后出现；降低 KV 预留后，原失败长输入重放通过。该历史问题与当前候选的数值差异分别追踪。

## 交付和故障定位

每个模型交付全量结果、等价性记录、实际配置和阶段日志：

| 路径 | 内容 |
|---|---|
| `$RUN_ROOT/<模型>-full/` | N1/N4 结果、引擎原始文件、校验报告 |
| `$RUN_ROOT/<模型>-equiv/` | 四轮请求、服务日志、等价性结果 |
| `$CONFIG_ROOT/` | 实际使用的模型和运行配置 |
| `$RUN_ROOT/logs/` | 阶段日志与 pilot/full 检查汇总 |

附上主仓库和 vLLM 的提交号、未提交的源码改动、权重及数据集 revision、GPU 型号和驱动版本，以及私有运行时源码、入口路径、keeper 身份记录、UID/GID 映射和限额验收报告。保留 `raw/`；搬运整个实验目录时使用 `rsync -aH` 等保留硬链接的方式，核对文件数量与哈希。数据布局见[存储规范](data_collect.md#6-存储布局)。

阶段失败返回非零时，保留现场，从对应日志定位：

| 现象 | 检查位置或处理方式 |
|---|---|
| `cannot clone`、`unshare: Operation not permitted` | `*-runtime-check.log` 和平台允许的 user/mount namespace 能力 |
| 包装入口失效或 keeper 身份不符 | 运行时启动记录、PID 启动时刻、完整命令、工作目录及 namespace ID；停止任务启动并恢复入口 |
| cgroup 只读或内核限额不符 | cgroup 委派、私有运行时入口及 `runtime_preflight.json` 中的限额读数 |
| proc 挂载或 UID/GID 错误 | 本地 Podman 的 namespace、UID/GID 映射；已有私有入口的实际检查结果 |
| 权重路径错误、OOM、服务退出 | 生成的模型 YAML、smoke 的 `server.log`、equiv 的 `round-*/server.log` |
| 镜像拉取失败 | `*-pull-pilot.log` 或 `*-pull-full.log`，检查平台容器的网络、存储驱动和磁盘 |
| `engine_config_id` 不匹配 | 使用当前模型配置重跑等价性验证 |
| pilot/full 校验失败或超时 | N1/N4 的结果 JSON，以及 `engine-N*/server.log` |
| 输出目录已存在 | 保留旧目录，另建实验目录 |

## Local Qwen execution ownership

For `local-qwen3-runtime-v1`, the executing agent performs authorized environment repairs, missing-asset downloads, image pulls, GPU runs, collection-fault diagnosis, and recovery. This local authorization supersedes historical instructions requiring the user to execute every command. Existing formal remote workflows retain their arrangements.

Use the [local Qwen commands](README.md#local-qwen-runtime-collection) and a new experiment directory under `/home/youwei/bzh/dataset/tokenmoe_artifacts/`. Keep main/fork commits, uncommitted patch and source hashes, model/tokenizer/dataset provenance, actual GPU UUIDs, image IDs/digests, and measured container limits. Native rootless Podman with `--network=none` has been verified locally for 2 CPU/4 GiB limits and a shared host boot clock; recheck the actual benchmark image before collecting.

The supervisor owns only its launched processes and uniquely named task containers. A collection error preserves logs and marks the experiment `needs_repair`; diagnose the original input, repair the issue, and resume affected units in new attempt directories. Model task failures remain valid observations when collection integrity holds. Do not replace tool output, inject solutions, relax capture equivalence, or discard failures to improve task metrics.

`health_checks.jsonl` contains 30-minute process/progress snapshots. `run_local.py --check-only` reads immediate health. Hand off the experiment path, `collection_manifest.json`, active process identity, latest health check, recovery events, report/checksum paths, and outstanding units. The local monitor is not an app automation or a promise of chat notifications after the active agent session ends.

The completed local reference delivery is `/home/youwei/bzh/dataset/tokenmoe_artifacts/qwen3-runtime-20261019-063626/`. Its 40 primary units are valid; read `reports/runtime_summary.md`, `reports/data_notes.md`, `reports/coverage.json` and `reports/artifact_checksums.json` before downstream use. N1/N4 have the same model profile and role version, while pilot and diagnostic stages remain separate. Prompt readers preserve CRLF, and Parquet export retains optional evidence fields appearing after the first row. The historical rejection replay remains explicitly distinguished from a captured original error body.
