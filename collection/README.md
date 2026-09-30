# 数据采集脚本

采集代码位于 `collection/`，包括 mini-swe-agent 请求、工具输出和 vLLM 引擎路由的记录与离线处理。

- [HANDOFF.md](HANDOFF.md)：clone、安装、验证、完整采集和结果交付的操作步骤。
- [data_collect.md](data_collect.md)：采集范围、记录字段、存储格式、校验规则和验收要求。

## 操作脚本

所有 Python 入口都位于 `scripts/`，使用 harness venv 的 Python 执行。参数完整列表可用 `--help` 查看。

| 脚本 | 用途 | 主要参数 | 输出 |
|---|---|---|---|
| [download_assets.py](scripts/download_assets.py) | 下载模型权重和 benchmark | `--models`、`--model-root`、`--dataset-dir`；可固定 revision、指定 endpoint、跳过已有资源 | 权重目录、数据集目录 |
| [prepare_configs.py](scripts/prepare_configs.py) | 将配置模板转换为本机配置 | 路径、`--runtime`、`--concurrency`、`--tp`、种子和运行限制 | `models/*.yaml`、`runs/*-{pilot,full}.yaml` |
| [run_collection.py](scripts/run_collection.py) | 执行一个采集阶段，管理服务启停 | `--stage`、配置文件、`--fork-venv`、`--gpus`、`--port`、输出目录、启动超时 | 阶段日志和采集产物 |
| [summarize_runs.py](scripts/summarize_runs.py) | 汇总校验、定位率和正常提交率 | `--run-dir`（可重复）、`--output` | JSON 汇总；自动检查失败时返回非零 |

前三个脚本支持 `--dry-run`，只打印计划。配置生成拒绝覆盖非空目录，采集阶段拒绝复用已有实验目录。`summarize_runs.py` 不执行人工验收项目，具体要求见[规范 §7.1](data_collect.md#71-实验验收)。

### run_collection.py 的阶段

| `--stage` | 使用的配置 | 行为 |
|---|---|---|
| `smoke` | `--model-config` | 启动服务，发送一条请求，检查捕获层、路由和引擎步，再停止服务 |
| `equiv` | `--model-config`、`--dataset-dir` | 执行规范中的 capture 等价性协议，保存验证记录 |
| `pull-images` | `--config` | 拉取任务清单对应的容器镜像，不启动模型服务 |
| `pilot` | `--config`、`--equivalence-record` | 按配置运行任务和并发档，自动收尾、finalize、validate |
| `full` | `--config`、`--equivalence-record` | 与 pilot 使用同一执行流程，任务规模由传入配置决定 |

`--stage full` 不会自动把任务清单扩展为 500 条；需要传入生成的 `*-full.yaml`。pilot/full 的服务参数必须与等价性记录匹配。

## 配置文件

| 目录或文件 | 内容 |
|---|---|
| `configs/models/*.yaml` | 模型仓库、权重路径、服务参数、采样参数和模型静态信息 |
| `configs/runs/pilot.yaml` | 20 个 repo 分层实例的运行模板 |
| `configs/runs/full.yaml` | 500 个实例的运行模板 |
| `configs/runs/local-debug.yaml` | 本地 Qwen 调试配置 |
| `configs/runs/smoke-local.yaml`、`smoke-agent.yaml`、`configs/smoke-items.json` | 本地容器小任务验证配置 |

模型配置对应 `gpt-oss-120b`、`deepseek-v4-flash`、`dots3-note`；`qwen3-30b-a3b` 用于本地调试。参数来源链接保存在各模型 YAML 中。`prepare_configs.py` 将权重和数据集路径写为绝对路径，并保留模板的任务清单。

## 底层命令与模块

安装 `collection` 后，可通过 `python -m tokenmoe_collect` 调用底层命令：

| 命令 | 功能 |
|---|---|
| `serve`、`smoke`、`equiv` | 模型服务、单请求检查、等价性实验 |
| `run` | 连接已经启动的服务，执行一次 agent 任务清单 |
| `pilot` | 自动管理服务和并发档，完成采集、整理、校验 |
| `pull-images` | 下载运行配置指定的实例镜像 |
| `finalize` | 关联 raw 记录，生成规范规定的存储布局并运行校验 |
| `validate` | 对已整理的数据重新运行校验 |
| `profile` | 查看运行目录内的模型 profile |
| `estimate` | 按模型参数估算存储，或对已整理的数据测量实际占用 |

`tokenmoe_collect/` 中，`client.py`、`executor.py`、`hostload.py`、`minisweagent_adapter.py` 负责运行记录；`static.py`、`launcher.py` 负责静态信息和 session 生命周期；`align.py`、`finalize.py`、`validate.py` 负责离线处理。harness 不导入 vLLM，双方通过 raw 文件交换记录。

### Shell 入口

`serve.sh`、`smoke.sh`、`equivalence.sh` 接收仓库模型配置的名称；`pilot.sh`、`pull_images.sh` 接收运行配置路径。这些入口使用 `TOKENMOE_*` 环境变量传参，并打印执行命令。参数化 Python 入口的运行步骤见 [HANDOFF.md](HANDOFF.md)。

## 开发检查

在已安装的 harness 环境中运行 `pytest collection/tests`。部分对齐和端到端测试使用本机 Qwen tokenizer，路径在 `tests/test_pipeline.py` 中。

fork 的 CPU recorder 测试使用 fork venv，在 `vllm/` 目录执行 `python -m pytest --confcutdir=tests/tokenmoe tests/tokenmoe`。
