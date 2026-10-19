# 三模型采集交接

采集 GPT-OSS-120B、DeepSeek-V4-Flash 和 dots3-note 执行 SWE-bench Verified 时的请求、工具输出和专家路由。每个模型运行 500 个任务，分别使用 N1、N4 两档并发，共 1,000 个会话；三个模型合计 3,000 个会话。

下文使用同一台 Linux 机器运行模型服务、采集脚本和任务容器。字段定义和验收要求见 [data_collect.md](data_collect.md)，脚本参数见 [README.md](README.md)。

## 代码与运行环境

机器需要支持 CUDA 13 的 NVIDIA 驱动，以及 Git、Python 3.12、uv 和容器运行时。GPU 数量按下文的 TP 配置准备，模型兼容性通过 smoke 验证。采集环境中使用 Docker CLI 连接宿主机 Docker；SWE-bench 镜像由宿主机拉取和启动。

```bash
git clone --recurse-submodules https://github.com/zhhangBian/TokenMoE.git
cd TokenMoE
git -C vllm switch tokenmoe-v0.30.0-trace
```

主仓库使用 `main`，子模块 `vllm/` 使用 `tokenmoe-v0.30.0-trace`。已有仓库缺少子模块时，执行 `git submodule update --init --recursive`，再切换分支。

### 采集脚本位于 Docker 容器内时

将同一宿主机的 Docker socket 挂到采集容器。SWE-bench 任务容器由宿主机创建，与采集容器并列运行；cgroup、挂载和 UID/GID 设置交给宿主机处理。

下面这段在宿主机的仓库目录执行。`COLLECTOR_IMAGE` 换成已有的采集镜像，镜像中应有 Bash、Git、uv、Python 和 Docker CLI。

```bash
COLLECTOR_IMAGE='你的采集镜像:标签'
mkdir -p /data/tokenmoe
docker run --rm -it --gpus all --ipc=host \
  --group-add "$(stat -c '%g' /var/run/docker.sock)" \
  --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  --mount type=bind,src="$PWD",dst=/workspace/TokenMoE \
  --mount type=bind,src=/data/tokenmoe,dst=/data/tokenmoe \
  --workdir /workspace/TokenMoE --entrypoint bash \
  "$COLLECTOR_IMAGE"
```

`/data/tokenmoe` 保存权重、数据集、venv 和采集结果。使用其他存储目录时，修改挂载和下文路径。Docker socket 具有宿主机管理权限，只挂到可信的采集容器；任务容器通过 `docker exec` 返回工具输出，不挂载该 socket 或采集目录。

已有 DSW 容器缺少宿主机 socket 时，需要平台挂载该入口，或将采集进程移到能启动任务容器的宿主机。在容器内安装 Docker CLI 无法补齐这些权限。

## 安装和配置

进入采集环境后，在同一个 Bash 会话、仓库根目录执行下文命令。路径按机器情况修改，`WORK_ROOT` 为本轮实验使用一个新目录。

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
RUNTIME_ARGS=(--runtime docker --runtime-socket /var/run/docker.sock)
```

直接在 Linux 上使用 Podman 时，将上面的数组改成 `RUNTIME_ARGS=(--runtime podman)`。Docker socket 位于其他路径时，修改 `--runtime-socket` 的值。

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

`check-runtime` 用首个已缓存的任务镜像实际启动容器，带上配置中的 CPU、内存限制，并核对它与采集进程的 Linux boot ID。结果为 `passed: true` 后再运行 GPU 阶段。它检查容器启动和同机条件，资源限制的内核读数仍需单独核验。

socket 不可访问、镜像缺失、cgroup 或挂载失败时，日志会保留具体错误。处理后重做检查。pilot/full 也会在模型启动前执行此检查，并保存 `runtime_preflight.json`。

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
  --output "$RUN_ROOT/logs/$MODEL-pilot-checks.json"
```

每个并发档都要通过数据校验，消息定位率至少 99%，正常提交率至少 90%。汇总中的 `automated_checks_passed: true` 只涵盖这些自动检查。启动全量前，还须核对工具原始输出与 observation、测量 recorder 开销低于 5%，并按实际 token 数核对存储占用，见[实验验收](data_collect.md#71-实验验收)。

`Submitted` 表示 agent 正常提交。SWE-bench 解题得分需另行测试，采集结果中的 `benchmark_score` 保持 `null`。

## 全量采集

当前模型的 pilot 和其余验收项目全部通过后，运行全量：

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
  --output "$RUN_ROOT/logs/$MODEL-full-checks.json"
```

`--stage full` 按传入配置的任务清单执行。本例使用生成的 `*-full.yaml`，包含 500 个任务。结束后核对 N1、N4 各有 500 个会话，并检查各档的 `validation.json`、`alignment.json` 和 `run_summary.json`。

换模型时，重新设置 `MODEL`、`GPUS` 和三个配置路径，再执行容器检查、smoke、等价性、pilot 和全量。修改采集代码、权重或服务参数后，用新实验目录重新验证；等价性记录须对应当前 `engine_config_id`。

长任务可放在 `tmux` 中。脚本没有自动断点续跑；中断后保留原目录，核对未完成任务，用新配置和新目录补跑。pilot/full 可用 `--output-dir` 指定新目录。

## 交付和故障定位

每个模型交付全量结果、等价性记录、实际配置和阶段日志：

| 路径 | 内容 |
|---|---|
| `$RUN_ROOT/<模型>-full/` | N1/N4 结果、引擎原始文件、校验报告 |
| `$RUN_ROOT/<模型>-equiv/` | 四轮请求、服务日志、等价性结果 |
| `$CONFIG_ROOT/` | 实际使用的模型和运行配置 |
| `$RUN_ROOT/logs/` | 阶段日志与 pilot/full 检查汇总 |

附上主仓库和 vLLM 的提交号、未提交的源码改动、权重及数据集 revision、GPU 型号和驱动版本。保留 `raw/`；搬运整个实验目录时使用 `rsync -aH` 等保留硬链接的方式，核对文件数量与哈希。数据布局见[存储规范](data_collect.md#6-存储布局)。

阶段失败返回非零时，保留现场，从对应日志定位：

| 现象 | 检查位置或处理方式 |
|---|---|
| Docker socket 权限不足 | 采集容器的 socket 挂载、用户所属组；在采集环境中检查 `docker --host unix:///var/run/docker.sock info` |
| cgroup、proc 挂载或 UID/GID 错误 | `*-runtime-check.log`，确认命令连接宿主机 daemon |
| 权重路径错误、OOM、服务退出 | 生成的模型 YAML、smoke 的 `server.log`、equiv 的 `round-*/server.log` |
| 镜像拉取失败 | `*-pull-pilot.log` 或 `*-pull-full.log`，检查宿主机 daemon 的网络和磁盘 |
| `engine_config_id` 不匹配 | 使用当前模型配置重跑等价性验证 |
| pilot/full 校验失败或超时 | N1/N4 的结果 JSON，以及 `engine-N*/server.log` |
| 输出目录已存在 | 保留旧目录，另建实验目录 |
