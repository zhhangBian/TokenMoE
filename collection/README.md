# TokenMoE collection

给运行者的完整步骤见 [从 clone 到三模型完整采集](HANDOFF.md)。下载、配置生成、阶段运行与结果汇总分别使用 `scripts/download_assets.py`、`prepare_configs.py`、`run_collection.py`、`summarize_runs.py`；各脚本支持 `--help`，无需复制内嵌 Python 代码。

本目录采集 mini-swe-agent 2.4.6 的请求、工具输出和 vLLM 实际计算的逐 token 路由，并离线关联成 [数据规范](data_collect.md) 的布局。harness 不导入 vLLM。正式采集使用 SWE-bench Verified；`smoke-local.yaml` 是独立标记的本地小任务验证，不是 SWE-bench 成绩。

## 环境

旧环境已保存在 `/home/youwei/bzh/project/TokenMoE-vllm-0722`（90025dce2，包含原编译产物）。使用旧 venv 时必须设置：

```bash
PYTHONPATH=/home/youwei/bzh/project/TokenMoE-vllm-0722 \
  /home/youwei/bzh/venvs/tokenmoe-vllm/bin/python your_old_script.py
```

新环境彼此独立。以下命令在仓库根目录执行；安装和下载需要网络。

```bash
python3 -m venv /home/youwei/bzh/venvs/tokenmoe-harness
/home/youwei/bzh/venvs/tokenmoe-harness/bin/pip install -e 'collection[harness,dev]'

uv venv --python 3.12 /home/youwei/bzh/venvs/tokenmoe-vllm030
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install --python /home/youwei/bzh/venvs/tokenmoe-vllm030/bin/python \
  --torch-backend=cu130 -e ./vllm

export TOKENMOE_HARNESS_VENV=/home/youwei/bzh/venvs/tokenmoe-harness
export TOKENMOE_FORK_VENV=/home/youwei/bzh/venvs/tokenmoe-vllm030
```

fork 安装使用 v0.30.0 编译产物及 torch 2.13/CUDA 13。不要在旧 venv 安装新 fork。GPU 命令显式设置 `CUDA_VISIBLE_DEVICES`，选择空闲设备。

```bash
"$TOKENMOE_HARNESS_VENV/bin/python" -m pytest collection/tests -q
cd vllm
"$TOKENMOE_FORK_VENV/bin/python" -m pytest --confcutdir=tests/tokenmoe tests/tokenmoe -q
```

对齐测试使用本机 `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B/tokenizer.json`，不加载模型权重。fork 测试通过文件加载 recorder，避开 upstream 全局测试夹具的 GPU/模型依赖。

## 配置与命令顺序

模型配置在 `configs/models/`：

| 配置 | 用途 | TP | parser | 待验证 |
|---|---|---:|---|---|
| qwen3-30b-a3b | 本地 A100，YaRN ×4、128K | 2 | qwen3 / hermes | 本地 smoke |
| gpt-oss-120b | Hopper，原生 MXFP4 | 2 | Harmony / openai | 36 个 monolithic 层、等价性 |
| deepseek-v4-flash | Hopper | 4 | deepseek_v4 | router 路径、full-attention KV group、prefix cache |
| dots3-note | Hopper，仅文本、无 MTP | 8 | 默认 / dots | 实际绑定层、parser、prefix cache |

Hopper 配置是 **confirm at smoke** 的起点。来源：[GPT-OSS recipe](https://recipes.vllm.ai/openai/gpt-oss-120b)、[DeepSeek-V4 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)、[dots3 recipe](https://recipes.vllm.ai/dots-studio/dots3-note-prev)。采集约束固定 DP=1、无 EP/投机解码、同步调度；没有照搬 recipe 的 DP/EP/MTP 拓扑。Hopper 的实际 backend 和兼容性须由对应主机验证。

设置 `TOKENMOE_MODEL_ROOT` 为 Hopper 模型下载根目录。Qwen 配置使用已有的本地路径。每次运行使用新目录，`run` 拒绝复用已有输出目录。

每个模型依次执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export TOKENMOE_TRACE_DIR=/path/to/new-model-smoke/engine
collection/scripts/smoke.sh qwen3-30b-a3b

# 50 个固定 SWE-bench 首步 prompt，依次独立启动 off/off/on/on 四次服务。
export TOKENMOE_EQUIV_DIR=/path/to/new-model-equivalence
collection/scripts/equivalence.sh qwen3-30b-a3b

# 每个 instance 一个镜像。不会在 session 启动时隐式拉取。
collection/scripts/pull_images.sh collection/configs/runs/pilot.yaml

# 根据目标模型修改 pilot.yaml 的 model_config。
# pilot 为 20 个 repo 分层实例，N=1 和 N=4 各一个 engine lifetime。
export TOKENMOE_RUN_DIR=/path/to/new-pilot
export TOKENMOE_EQUIVALENCE_RECORD=/path/to/equivalence/static/equivalence/ENGINE_CONFIG_ID.json
collection/scripts/pilot.sh collection/configs/runs/pilot.yaml
```

其它模型替换模型名和 run config 的 `model_config`；不复用不同引擎配置的等价性记录。`full.yaml` 的 500 实例清单为草案，pilot 验收前不运行。2026-09-30 本机仅完成了少量 smoke；交由其他运行者执行三个模型的完整采集时，请按 [交接操作说明](HANDOFF.md) 逐个完成验证与采集。

手动连接已启动的服务时：

```bash
export TOKENMOE_TRACE_DIR=/path/to/engine-root
collection/scripts/serve.sh qwen3-30b-a3b > /path/to/server.log 2>&1 &
server_pid=$!
# 等待 /health 可用；engine root 下的独立目录给出 engine_instance_id。
export TOKENMOE_ENGINE_DIR=/path/to/engine-root/eng_ID
export TOKENMOE_RUN_DIR=/path/to/new-run
"$TOKENMOE_HARNESS_VENV/bin/python" -m tokenmoe_collect run \
  --config collection/configs/runs/local-debug.yaml
kill -TERM "$server_pid"
wait "$server_pid"
"$TOKENMOE_HARNESS_VENV/bin/python" -m tokenmoe_collect finalize "$TOKENMOE_RUN_DIR"
"$TOKENMOE_HARNESS_VENV/bin/python" -m tokenmoe_collect validate "$TOKENMOE_RUN_DIR"
```

工具执行沿用安装版的超时错误结果、UTF-8 replacement、换行转换和 observation 模板。原始合并管道字节不作修改；每次读取记录一个 chunk。长输出头尾各自有 provenance 区间。相同工具结果可由多个重试消费，记录 `llm_request_ids_consuming` 列表。

## 输出与验证

```text
<run>/
  static/                  配置、模型 profile、角色模板、benchmark、equivalence
  raw/harness/<host>-<pid>/ 请求消息增量、工具原始输出与宿主机负载
  raw/engine/<eng>/        引擎 metadata、layer map、steps、requests、routing
  runtime/
    application_runs.jsonl sessions.jsonl llm_requests.jsonl tool_calls.jsonl
    prompt_segments.parquet tool_output_chunks.parquet engine_steps.parquet host_load.parquet
    prompts/ tool_outputs/ tool_args/ routing/<model_profile_id>/
  alignment.json validation.json run_summary.json
```

`finalize` 保留 raw；routing、tool_outputs 和 tool_args 同文件系统硬链接，跨文件系统才复制。文字由引擎 token IDs 解码，保留特殊 token。路由只记录本请求实际计算的 token；`token_positions` 明确位置，支持抢占后缓存造成的缺口；`row_end` 和 `routing_complete` 标记部分请求。最后一个采样 token 没有路由。

`validation.json` 检查规则 1–9、11、12；10/13/14 为 not_applicable。任何适用规则失败均返回非零。没有执行正式 50-prompt 等价性协议的 smoke **仍会报告规则 12 失败**，不得将它改写成通过；这与其余结构规则通过应分别报告。SWE-bench 不打分，benchmark_score 为 null。

## 存储估算

假设每 session 60 步、78K 实际计算 token、累计 2M 请求 token；保留 raw、文件未压缩，单位采用十进制。包含 int32 step_index/token_positions 的额外 0.624 MB/session。

| 模型 | L × K | 每 session | 500 sessions，单并发配置 |
|---|---|---:|---:|
| Qwen3-30B-A3B | 48 × 8 | 54.58 MB | 27.29 GB |
| GPT-OSS-120B | 36 × 4 | 35.86 MB | 17.93 GB |
| DeepSeek-V4-Flash | 43 × 6 | 44.75 MB | 22.37 GB |
| dots3-note | 暂按 45 × 8 | 52.70 MB | 26.35 GB |

dots3 的 45 层来自官方 config 的 46 层减去首个 dense 层，最终以 layer_map 为准。N=1 加一个并发配置约翻倍。prompt 文本和重复 token_ids 的累计存储随 session 步数近似平方增长。

```bash
"$TOKENMOE_HARNESS_VENV/bin/python" -m tokenmoe_collect estimate \
  --model-config collection/configs/models/qwen3-30b-a3b.yaml
"$TOKENMOE_HARNESS_VENV/bin/python" -m tokenmoe_collect estimate \
  --model-config collection/configs/models/qwen3-30b-a3b.yaml --run-dir /path/to/finalized-run
```

实测模式按 inode 去重，报告逻辑文件字节数与实际分配磁盘字节数。

## 尚需用户所在环境完成的步骤

- 将 fork 分支推送到 origin，使父仓库的 submodule commit 可获取。
- 可访问 Docker Hub 或镜像源时拉取 SWE-bench 实例镜像。
- Hopper 模型权重、对应主机安装、smoke、正式等价性与 pilot。
- pilot 检查正常终止率 ≥90%、recorder 开销 <5%、定位率 ≥99%、工具输出和存储估算；通过后确认 full 参数。

本次用户授权了本机环境安装和小规模 GPU 验证；Hopper 与全量数据采集仍未执行。
