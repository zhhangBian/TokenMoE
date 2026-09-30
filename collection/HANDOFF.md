# 从 clone 到三模型完整采集

本指南供实际运行者使用。依次对 **GPT-OSS-120B、DeepSeek-V4-Flash、dots3-note** 执行：环境准备 → 下载 → 生成配置 → smoke → 等价性 → pilot → 全量 → 交付。

默认全量规模为每模型 **500 个 SWE-bench Verified 任务 × 并发 N=1、N=4**，合计 3,000 个 session。每 session 最多 250 步、1 小时，种子为 42；这些参数均可修改。`succeeded` 表示 agent 正常提交，不表示通过 SWE-bench 测试；本项目暂不计算 benchmark 分数。

> **发布前提（维护者处理）：** 本地提交完成后，先推送 vLLM fork，再推送包含对应 submodule 指针与采集代码的主仓库版本，向运行者提供固定 commit 或 tag（下文 `RELEASE_REF`）。本地 commit 不代表远端已经可用；发布完成后再按本指南 clone。三个 Hopper 模型尚未完成实机验收，必须按顺序过验证，不能直接启动全量。

## 1. 准备机器

模型、harness 和任务容器必须运行在**同一台 Linux 主机**。需要 Git、Python 3.12、uv、支持 CUDA 13 的 NVIDIA 驱动，以及可启动容器并拉取镜像的 Podman 或 Docker。

```bash
git --version
python3 --version
uv --version
nvidia-smi
podman --version                 # 使用 Docker 时换成 docker --version
```

没有 uv 时可先运行 `python3 -m pip install --user uv`，并确保 `uv` 位于命令搜索路径。驱动和容器运行时由机器管理员准备。

| 模型参数名 | 权重仓库 | 当前 TP / GPU 数 | smoke 预期 MoE 层数 |
|---|---|---:|---:|
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 2 | 36 |
| `deepseek-v4-flash` | `deepseek-ai/DeepSeek-V4-Flash` | 4 | 43 |
| `dots3-note` | `dots-studio/dots3-note-prev`，BF16 | 8 | 45，以实际 layer map 核实 |

使用 Hopper GPU；上表为当前配置起点，显存是否足够须由 smoke 确认。按平均每 session 60 步估算，三模型两个并发档的采集数据约 **135 GB**；权重和 500 个容器镜像的空间另计，最终按 pilot 实测规划。

## 2. 设置路径并 clone

以下均在 **Bash** 中执行。示例路径需要换成机器上有写权限的目录；每批实验使用新的 `WORK_ROOT`。

```bash
export TOKENMOE_REPO=/work/TokenMoE
export MODEL_ROOT=/data/models
export DATASET_DIR=/data/datasets/SWE-bench_Verified
export WORK_ROOT=/data/tokenmoe/collection-20260930-01
export HARNESS_VENV=/data/venvs/tokenmoe-harness
export FORK_VENV=/data/venvs/tokenmoe-vllm030
export RELEASE_REF='<维护者提供的主仓库 commit 或 tag>'
set -euo pipefail

git clone https://github.com/zhhangBian/TokenMoE.git "$TOKENMOE_REPO"
cd "$TOKENMOE_REPO"
git checkout "$RELEASE_REF"
git submodule init
# .gitmodules 原地址使用 SSH；这里仅对本机改用 HTTPS。
git config submodule.vllm.url https://github.com/zhhangBian/TokenMoE-vLLM.git
git submodule update --init --recursive

# 确认取得的是带采集功能的版本。
test -f collection/scripts/run_collection.py
test -f vllm/vllm/tokenmoe_trace.py
git rev-parse HEAD
git -C vllm rev-parse HEAD
```

私有仓库须先取得 GitHub 访问权限。保留上述两个 commit 作为交付信息。**不要执行 `git submodule update --remote`**，它会偏离主仓库固定的 fork 版本；若文件检查失败，先向维护者确认发布版本。

## 3. 安装两个独立环境

以下命令继续在仓库根目录执行。使用新 venv，不覆盖已有实验环境。

```bash
uv venv --python 3.12 "$HARNESS_VENV"
uv pip install --python "$HARNESS_VENV/bin/python" -e 'collection[harness,dev]'
uv venv --python 3.12 "$FORK_VENV"
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install --python "$FORK_VENV/bin/python" --torch-backend=cu130 -e ./vllm
HARNESS_PY="$HARNESS_VENV/bin/python"

"$HARNESS_PY" collection/scripts/download_assets.py --help
"$HARNESS_PY" collection/scripts/prepare_configs.py --help
"$HARNESS_PY" collection/scripts/run_collection.py --help
```

fork 安装使用 v0.30.0 的预编译产物；不能替换为原版 `pip install vllm`。下面都用 harness Python 执行操作脚本，脚本通过 `--fork-venv` 调用模型服务环境。

## 4. 下载模型和数据集

[download_assets.py](scripts/download_assets.py) 是实际下载入口，无需复制或修改 Python 代码：

```bash
"$HARNESS_PY" collection/scripts/download_assets.py \
  --models gpt-oss-120b deepseek-v4-flash dots3-note \
  --model-root "$MODEL_ROOT" --dataset-dir "$DATASET_DIR" \
  --max-workers 8
```

权重会保存为 `$MODEL_ROOT/<权重仓库名>/`，例如 `$MODEL_ROOT/openai/gpt-oss-120b/`；数据集下载到 `$DATASET_DIR`。

常用参数：

- 已有全部权重：增加 `--skip-models`；已有数据集：增加 `--skip-dataset`。
- 仅下载一个模型：`--models gpt-oss-120b`。
- 固定权重版本：`--model-revision gpt-oss-120b=<HF commit>`，可重复；数据集使用 `--dataset-revision <HF commit>`。
- 指定可访问的 Hugging Face 端点：`--endpoint <地址>`；认证使用运行者自己的 Hugging Face 登录或 `HF_TOKEN`。
- 先看计划而不下载：增加 `--dry-run`。

## 5. 生成本机配置

[prepare_configs.py](scripts/prepare_configs.py) 生成三份模型配置及六份 pilot/full 配置，将所有路径写为绝对路径。输出配置目录必须为新目录或空目录，防止覆盖已确认的实验配置。

```bash
CONFIG_ROOT="$WORK_ROOT/configs"
RUN_ROOT="$WORK_ROOT/results"
"$HARNESS_PY" collection/scripts/prepare_configs.py \
  --models gpt-oss-120b deepseek-v4-flash dots3-note \
  --model-root "$MODEL_ROOT" --dataset-dir "$DATASET_DIR" \
  --output-dir "$CONFIG_ROOT" --run-root "$RUN_ROOT" \
  --runtime podman --concurrency 1 4 --base-seed 42 \
  --step-limit 250 --time-limit-seconds 3600 \
  --tool-timeout 60 --request-timeout 600 --cpu-quota 2 --memory-limit 4g
mkdir -p "$RUN_ROOT/logs"
```

生成布局：

```text
configs/models/<模型>.yaml
configs/runs/<模型>-pilot.yaml    20 个 repo 分层实例
configs/runs/<模型>-full.yaml     500 个实例
```

可选参数：`--runtime docker`；`--model-path dots3-note=/已有权重目录`；`--tp gpt-oss-120b=4`（可重复）；`--max-model-len 131072`；`--stages pilot`；`--dry-run`。未覆盖的模型参数沿用仓库配置。

确认生成的 YAML 后冻结配置。**代码、权重、TP、服务参数发生变化时，重做对应模型的等价性验证。** 不要自行关闭 prefix caching、打开 async scheduling、EP/DP 或 speculative decoding 来绕过启动错误。

## 6. 选择一个模型并预览

下面第 6–10 步，**三个模型各执行一遍**。先运行一个模型，完成后再切换到下一个，避免抢占同一组 GPU。

```bash
MODEL=gpt-oss-120b
GPUS=0,1                        # DeepSeek 当前需 4 张；dots3 当前需 8 张
PORT=8000                       # 选择空闲端口
MODEL_CONFIG="$CONFIG_ROOT/models/$MODEL.yaml"
PILOT_CONFIG="$CONFIG_ROOT/runs/$MODEL-pilot.yaml"
FULL_CONFIG="$CONFIG_ROOT/runs/$MODEL-full.yaml"

"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage smoke --model-config "$MODEL_CONFIG" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --output-dir "$RUN_ROOT/$MODEL-smoke" --dry-run
```

检查打印的权重路径、GPU、端口和输出目录。GPU 数量必须与生成模型配置中的 TP 一致；选择当前空闲的 GPU。所有 GPU 阶段都由脚本自动启停服务，**不要同时手动运行 `serve.sh`**。

## 7. 执行 smoke 和正式等价性验证

```bash
"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage smoke --model-config "$MODEL_CONFIG" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --output-dir "$RUN_ROOT/$MODEL-smoke" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-smoke.log"

"$HARNESS_PY" collection/scripts/run_collection.py \
  --stage equiv --model-config "$MODEL_CONFIG" --dataset-dir "$DATASET_DIR" \
  --fork-venv "$FORK_VENV" --gpus "$GPUS" --port "$PORT" \
  --output-dir "$RUN_ROOT/$MODEL-equiv" \
  2>&1 | tee "$RUN_ROOT/logs/$MODEL-equiv.log"
```

smoke 发一条请求，检查绑定层、合法路由和引擎步。结果在 `<模型>-smoke/engine/smoke.json`。等价性使用 50 个固定首步 prompt，依次启动 off/off/on/on 四次服务；结果在 `<模型>-equiv/static/equivalence/*.json`，必须为 `passed: true`。

默认每次服务启动等待 1,200 秒；大模型确需更久时可增加 `--startup-timeout 2400`。阶段失败后先看对应目录中的 `server.log` / `round-*/server.log`，保留日志并回传，不要跳过等价性门槛。

## 8. 拉 pilot 镜像并采集 20 个任务

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

脚本按配置依次运行 N=1、N=4，各自启动一个 engine lifetime，结束后自动停止服务、finalize 和 validate。`--equivalence-record` 可传具体 JSON，也可传只含一个等价性记录的实验目录；不需要手抄 engine ID。

**继续全量前须满足：** 两个并发档结构校验均通过；消息定位率 ≥99%；正常提交率 ≥90%。汇总脚本会检查这三项，未达标返回非零。还需人工核对原始工具输出与 observation，并单独测量 recorder 开销 <5%；汇总脚本不会把这两项宣称为已通过。不要通过调低阈值掩盖失败。

## 9. 运行完整 500 任务清单

pilot 验收通过后：

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

`--stage full` 使用所传配置中的任务清单，不会自动替换清单；务必传生成的 `*-full.yaml`。默认每模型得到 500×2=1,000 个 session。可在 `tmux` 中运行；pilot/full 也可用 `--output-dir <新目录>` 覆盖配置中的输出路径。

**没有自动断点续跑。** 中断后保留原目录，确认剩余实例清单，再用新配置、新输出目录补跑；重新执行同一命令不会自动接着跑。

## 10. 交付结果

每个模型的全量目录为 `$RUN_ROOT/<模型>-full/`：

```text
N1/、N4/                 static/、raw/、runtime/ 与三份结果 JSON
engine-N1/、engine-N4/   服务日志和引擎原始文件
pilot_summary.json       两个并发档的汇总（full 阶段也沿用此文件名）
```

先交付三个模型 N1/N4 的 `validation.json`、`alignment.json`、`run_summary.json`，以及汇总脚本输出。然后交付完整的模型实验目录、等价性目录、`$CONFIG_ROOT`、日志和两个 Git commit、GPU 型号、驱动信息。

保留 `raw/`。搬运整个实验根目录时保留硬链接，例如 `rsync -aH`，避免 raw 与 runtime 中的大文件重复占用空间。更详细的数据定义见 [采集规范](data_collect.md)。

## 常见问题

| 现象 | 处理 |
|---|---|
| clone 后没有操作脚本或 `vllm/tokenmoe_trace.py` | 发布版本不完整，联系维护者；不要安装旧 fork 继续运行 |
| CUDA / 扩展导入失败 | 检查 fork venv、CUDA 13 驱动和预编译 wheel；不要覆盖原有环境 |
| OOM / 不支持模型 backend | 回传模型配置、GPU 信息及 server.log，确认调整；调整后重做等价性 |
| Docker Hub 超时 / `/dev/net/tun` 无权限 | 修复镜像访问或容器网络；正式 SWE-bench 不自动替换为本地 smoke 任务 |
| `engine_config_id` 不匹配 | 当前引擎参数与等价性记录不同，按当前配置重做验证 |
| 输出目录已存在 | 换新实验目录；不要删除已有原始数据或假定会自动续跑 |
| 原始数据已产生但校验失败 | 保留整个目录与错误报告，先定位失败规则，再决定补跑范围 |
