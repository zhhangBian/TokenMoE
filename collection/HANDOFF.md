# 三模型数据采集操作指引

使用本指引运行 GPT-OSS-120B、DeepSeek-V4-Flash 和 dots3-note。先执行第 1–4 步准备代码和资源，再为每个模型执行第 5–8 步，最后按第 9 步交付数据。

需要一台 Linux 主机，模型、harness 和任务容器均在这台机器运行。准备 Git、Python 3.12、uv、支持 CUDA 13 的 NVIDIA 驱动及可正常联网拉镜像的 Podman 或 Docker。三个 Hopper 模型的实机兼容性仍须在第 5–7 步确认。

采集字段、校验与验收要求见 [data_collect.md](data_collect.md)，脚本功能和参数说明见 [README.md](README.md)。

## 1. Clone 仓库

```bash
git clone --recurse-submodules https://github.com/zhhangBian/TokenMoE.git
cd TokenMoE
git -C vllm switch tokenmoe-v0.30.0-trace
```

`--recurse-submodules` 会同时拉取 [TokenMoE-vLLM](https://github.com/zhhangBian/TokenMoE-vLLM)，放在 `vllm/` 中。主仓库使用 `main`，vLLM 使用 `tokenmoe-v0.30.0-trace` 分支。

下面的目录换成机器上的实际路径，后续命令都在 `TokenMoE/` 下执行：

```bash
MODEL_ROOT=/data/models
DATASET_DIR=/data/datasets/SWE-bench_Verified
WORK_ROOT=/data/tokenmoe/collection-20260930-01
HARNESS_VENV=/data/venvs/tokenmoe-harness
FORK_VENV=/data/venvs/tokenmoe-vllm030
RUNTIME=podman                       # 使用 Docker 时改为 docker
set -euo pipefail
```

## 2. 安装环境

在仓库根目录执行：

```bash
uv venv --python 3.12 "$HARNESS_VENV"
uv pip install --python "$HARNESS_VENV/bin/python" -e 'collection[harness,dev]'
uv venv --python 3.12 "$FORK_VENV"
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install --python "$FORK_VENV/bin/python" --torch-backend=cu130 -e ./vllm
HARNESS_PY="$HARNESS_VENV/bin/python"
```

`HARNESS_VENV` 运行采集脚本，`FORK_VENV` 运行模型服务。

## 3. 下载权重和数据集

```bash
"$HARNESS_PY" collection/scripts/download_assets.py \
  --models gpt-oss-120b deepseek-v4-flash dots3-note \
  --model-root "$MODEL_ROOT" --dataset-dir "$DATASET_DIR"
```

下载完成后应有：

```text
MODEL_ROOT/
  openai/gpt-oss-120b/
  deepseek-ai/DeepSeek-V4-Flash/
  dots-studio/dots3-note-prev/
DATASET_DIR/
  data/*.parquet
```

已有全部权重时加 `--skip-models`；已有数据集时加 `--skip-dataset`。自定义权重目录在下一步指定。认证使用运行者的 Hugging Face 登录或 `HF_TOKEN`，镜像地址使用 `--endpoint`。下载和磁盘估算参数见脚本帮助及[规范 §6](data_collect.md#6-存储布局)；权重和容器镜像的空间另计。

## 4. 生成本机配置

```bash
CONFIG_ROOT="$WORK_ROOT/configs"
RUN_ROOT="$WORK_ROOT/results"
"$HARNESS_PY" collection/scripts/prepare_configs.py \
  --model-root "$MODEL_ROOT" --dataset-dir "$DATASET_DIR" \
  --output-dir "$CONFIG_ROOT" --run-root "$RUN_ROOT" \
  --runtime "$RUNTIME" --concurrency 1 4 --base-seed 42 \
  --step-limit 250 --time-limit-seconds 3600 \
  --tool-timeout 60 --request-timeout 600 --cpu-quota 2 --memory-limit 4g
mkdir -p "$RUN_ROOT/logs"
```

`$CONFIG_ROOT/models/` 下应有三份模型 YAML，`$CONFIG_ROOT/runs/` 下应有六份运行 YAML。每个模型的 `*-pilot.yaml` 包含 20 个任务，`*-full.yaml` 包含 500 个任务；两者都运行 N=1、N=4。

如果权重不在默认目录，生成时增加如 `--model-path dots3-note=/实际权重目录`。调整 GPU 数使用如 `--tp gpt-oss-120b=4`。生成的配置目录必须是新目录或空目录。

后续步骤均使用这些生成的 YAML。检查其中的路径和参数。修改代码、权重或服务参数后，换用新的实验目录并重新执行第 5–7 步。

## 5. 选择模型，执行 smoke

每次选择一个模型。下表是未修改 TP 时的参数；GPU 编号可替换为同样数量的空闲设备。

| `MODEL` | `GPUS` 示例 |
|---|---|
| `gpt-oss-120b` | `0,1` |
| `deepseek-v4-flash` | `0,1,2,3` |
| `dots3-note` | `0,1,2,3,4,5,6,7` |

```bash
MODEL=gpt-oss-120b
GPUS=0,1
PORT=8000                           # 选择空闲端口
MODEL_CONFIG="$CONFIG_ROOT/models/$MODEL.yaml"
PILOT_CONFIG="$CONFIG_ROOT/runs/$MODEL-pilot.yaml"
FULL_CONFIG="$CONFIG_ROOT/runs/$MODEL-full.yaml"

"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage smoke --model-config "$MODEL_CONFIG" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --output-dir "$RUN_ROOT/$MODEL-smoke" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-smoke.log"
```

终端结果和 `$RUN_ROOT/$MODEL-smoke/engine/smoke.json` 中应为 `passed: true`。脚本会自动启动服务、检查一条请求、停止服务，其他 GPU 阶段也一样；运行这些阶段时无需另外启动 `serve.sh`。

服务启动超过默认 1,200 秒时，可用 `--startup-timeout` 调整等待时间。

## 6. 执行等价性验证

```bash
"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage equiv --model-config "$MODEL_CONFIG" --dataset-dir "$DATASET_DIR" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --output-dir "$RUN_ROOT/$MODEL-equiv" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-equiv.log"
```

该步骤会多次启动模型服务。等待结束，确认 `$RUN_ROOT/$MODEL-equiv/static/equivalence/*.json` 的 `passed` 为 `true`，再继续。实验协议和判断标准见[规范规则 12](data_collect.md#7-校验规则)。

## 7. 运行并检查 pilot

```bash
"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage pull-images --config "$PILOT_CONFIG" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-pull-pilot.log"

"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage pilot --config "$PILOT_CONFIG" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --equivalence-record "$RUN_ROOT/$MODEL-equiv" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-pilot.log"

"$HARNESS_PY" collection/scripts/summarize_runs.py \
  --run-dir "$RUN_ROOT/$MODEL-pilot" \
  --output "$RUN_ROOT/logs/$MODEL-pilot-checks.json"
```

结束后应有 `$RUN_ROOT/$MODEL-pilot/N1/` 和 `N4/`，每个目录都包含 `validation.json`、`alignment.json`、`run_summary.json`。汇总输出中的 `automated_checks_passed` 应为 `true`。

按[规范 §7.1](data_collect.md#71-实验验收)完成验收，再执行全量。汇总脚本只检查自动项目，工具输出核对、开销测量和存储核对需另行完成。任何阶段返回非零时，先保留日志并处理失败原因。

## 8. 运行并检查全量

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

必须传入 `*-full.yaml`。`--stage full` 使用配置内的任务清单，不会自动补成 500 个任务。默认每个模型执行 1,000 个 session。

完成一个模型后，回到第 5 步修改 `MODEL`、`GPUS`，重新设置三个配置路径，再执行第 5–8 步。三个模型全部完成后，共得到六份并发档数据。

长任务可在 `tmux` 中运行。程序没有自动断点续跑；中断后保留原目录，确认剩余任务清单，用新配置和新目录补跑。`--output-dir` 可以为 pilot/full 指定新的输出目录。

## 9. 交付数据

先发送六个并发档的结果报告及 `$RUN_ROOT/logs/*-full-checks.json`。每个模型需要交付以下目录：

| 路径 | 内容 |
|---|---|
| `$RUN_ROOT/<模型>-full/` | N1/N4 的采集产物、引擎原始文件和日志 |
| `$RUN_ROOT/<模型>-equiv/` | 等价性结果和各轮记录 |
| `$CONFIG_ROOT/` | 实际使用的模型及运行配置 |
| `$RUN_ROOT/logs/` | 阶段日志和检查汇总 |

附上 GPU 型号和驱动信息。保留 `raw/`，搬运整个实验目录时使用 `rsync -aH` 等能保留硬链接的方式。数据内部布局见[规范 §6](data_collect.md#6-存储布局)。

## 失败时查看哪里

| 问题 | 先查看 / 处理 |
|---|---|
| 漏拉子模块 | 执行 `git submodule update --init --recursive`，然后切换 vLLM 分支 |
| 权重路径错误 / OOM / 模型启动失败 | 生成的模型 YAML，以及 smoke 的 `server.log` 或 equiv 的 `round-*/server.log` |
| 镜像拉取失败 / 容器网络权限错误 | pull-images 日志和 Podman/Docker 运行环境 |
| `engine_config_id` 不匹配 | 用当前模型服务配置重做第 6 步 |
| pilot/full 校验失败 | 对应 N1/N4 的结果 JSON，以及 `engine-N*/server.log` |
| 输出目录已存在 | 保留旧目录，使用新的实验目录 |
