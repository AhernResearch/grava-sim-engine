<div align="center">
  <img src="assets/grava_logo.png" alt="GRAVA logo" width="144">
  <h1>GRAVA Sim Engine</h1>
  <p><strong>面向自动驾驶的高效仿真与强化学习奖励引擎。</strong></p>
  <p>
    <a href="https://github.com/AhernResearch/grava"><img alt="Project" src="https://img.shields.io/badge/Project-GRAVA-2563eb"></a>
    <a href="https://arxiv.org/abs/2609.15169"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-6366f1"></a>
    <a href="https://github.com/AhernResearch/grava-train"><img alt="Training" src="https://img.shields.io/badge/Code-GRAVA_Train-8b5cf6"></a>
    <a href="docs/api.md"><img alt="API" src="https://img.shields.io/badge/Read-API-475569"></a>
  </p>
  <p><a href="README.md">English</a> | <strong>简体中文</strong></p>
</div>

## 概览

GRAVA Sim Engine 为自动驾驶强化学习提供高效的仿真与奖励计算。
它对候选轨迹进行仿真，计算驾驶奖励，用于模型训练、rollout 筛选和轨迹评测。
引擎可以部署为独立的 CPU 服务，通过 HTTP 接入。
你可以配合 [GRAVA Train](https://github.com/AhernResearch/grava-train) 使用，
也可以通过同一套 API 接入自己的训练代码。

强化学习需要反复比较同一场景下的多条候选轨迹。我们读取一次场景，再批量完成仿真评分。
把这部分 CPU 计算与模型生成、参数更新分开后，仿真可以部署在单独的机器上，按评分负载
配置资源。仿真资源可以随评分负载增加，模型进程则保留各自的运行环境。

后台进程在服务响应请求的同时准备场景缓存和参考指标，已完成的缓存可在重启后复用。
服务提供与固定版本 NAVSIM 对齐的 PDMS v1，以及 CDS 和驾驶实验中使用的
continuous/discrete 奖励。

## 快速开始

### 1. 安装

下面的 NAVSIM 使用流程采用 Python 3.10。基础包支持 Python 3.10 及更高版本。

```bash
git clone https://github.com/AhernResearch/grava-sim-engine.git
cd grava-sim-engine
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

读取 NAVSIM MetricCache 时，额外安装固定版本的 NAVSIM 和 nuPlan 依赖：

```bash
python -m pip install -e '.[navsim]'
```

准备好的 SceneContext 文件使用基础安装即可。建议评分服务与模型训练使用独立环境；
依赖和环境说明见[安装文档](docs/installation.md)。

### 2. 选择场景源

| 场景源 | 目录结构 | 评分模式 |
| --- | --- | --- |
| NAVSIM MetricCache | `<log_name>/<scenario_type>/<scene_token>/metric_cache.pkl` | PDMS、continuous、discrete |
| 准备好的 SceneContext | `dataset_meta.json` 和 `<log_name>/<scene_token>.pkl` | 默认 CDS；其他模式由元数据声明 |

场景源需要与训练或评测记录中的 `log_name`、`scene_token` 对应。
除了模型使用的 JSONL 和相机图片，仿真还需要这些场景资源。
场景格式见[场景数据说明](docs/data.md)。

使用 NAVSIM 时，设置场景源目录和独立的转换缓存目录：

```bash
export SCENE_ROOT=/path/to/metric_cache_navtest
export CACHE_DIR=/path/to/grava_scene_cache
```

### 3. 启动评分服务

```bash
python scripts/serve.py \
  --dataset "navtest=$SCENE_ROOT" \
  --cache-dir "$CACHE_DIR"
```

服务默认监听 `0.0.0.0:8100`，使用最多四个 HTTP worker 和 32 个预热 worker。
按可用 CPU 资源设置 `--workers` 和 `--warmup-workers`；`--warmup-workers 0`
关闭后台转换。重复传入 `--dataset NAME=PATH` 可配置其他数据集。
缓存复用和进度查询见[预热文档](docs/warmup.md)。

在另一个终端检查服务，并发送仓库中的真实轨迹示例：

```bash
curl --fail-with-body http://localhost:8100/v1/health
curl --fail-with-body http://localhost:8100/v1/score \
  -H 'Content-Type: application/json' --data @examples/score_request.json
```

在仓库目录下执行，且 `SCENE_ROOT` 中需包含示例对应的 NAVSIM 场景。
评估其他场景时，替换[请求文件](examples/score_request.json)中的
`log_name`、`scene_token` 和 `trajectory`。

## 评分 API

| `scoring_mode` | 主分数字段 | 输入 |
| --- | --- | --- |
| `pdms` | `pdm_score` | 轨迹或控制量 |
| `cds` | `cds` | 轨迹，场景需包含关键动作标注 |
| `continuous` | `rl_score` | 轨迹 |
| `discrete` | `rl_score` | 轨迹 |

所有轨迹模式共用 `POST /v1/score` 和 `POST /v1/score/batch`。
轨迹包含八个未来 XY 或 XYH 点，间隔 0.5 秒，覆盖未来四秒。
HTTP 坐标使用 nuPlan 约定：x 向前、y 向左，heading 单位为弧度。
当前自车位姿由服务补入。

Python 客户端可以直接读取同一个示例请求：

```python
import json
from pathlib import Path

from grava_sim_engine.sim_client import SimEngineClient

request = json.loads(Path("examples/score_request.json").read_text())
client = SimEngineClient("http://localhost:8100")
score, metrics = client.score(**request)
print(score)
```

同一场景的多个候选使用 `client.score_batch()`，结果顺序与输入一致。
标量分数保留完整精度；设置 `include_details=True` 可查看逐点进度、边界距离和碰撞诊断。
CDS 还返回 `sample_valid` 和 `invalid_reason`，用来区分不适用样本与有效零分。
计算失败会返回 HTTP 错误。

控制量端点接收八组 `[acceleration, heading_rate]`，单位分别为 m/s² 和 rad/s，
同样间隔 0.5 秒。全部七个端点及奖励配置见 [API 文档](docs/api.md)，
控制量请求见[示例文件](examples/control_request.json)。服务启动后可访问
[交互式 API 文档](http://localhost:8100/docs)，通过 `/v1/datasets` 查看数据集支持的模式，
通过 `/v1/warmup` 查询预热进度。

## 接入 GRAVA Train

[GRAVA Train](https://github.com/AhernResearch/grava-train) 提供模型训练、vLLM rollout
生成、Active RL 和拒绝采样，GRAVA Sim Engine 提供仿真奖励。
训练代码通过 HTTP 调用评分接口，仿真服务与模型进程可以部署在不同机器上。

GRPO 奖励在训练环境中读取以下变量。地址应能被训练 worker 访问，数据集名称需与
服务启动时的配置一致：

```bash
export SIM_ENGINE_URL=http://localhost:8100
export SIM_ENGINE_DATASET=navtest
```

评测和 rollout 脚本都使用 `--sim-url` 指定服务地址。场景数据集在评测脚本中通过
`--dataset` 指定，在 rollout 脚本中通过 `--sim-dataset` 指定。
后续流程见 [GRPO 训练](https://github.com/AhernResearch/grava-train/blob/main/docs/training.md#grpo)、
[Active RL](https://github.com/AhernResearch/grava-train/blob/main/docs/active_rl.md)
和[自蒸馏](https://github.com/AhernResearch/grava-train/blob/main/docs/self_distillation.md)。

## 文档

以下详细指南使用英文。

| 指南 | 内容 |
| --- | --- |
| [安装](docs/installation.md) | 环境、依赖和开发检查 |
| [场景数据](docs/data.md) | MetricCache 和 SceneContext 目录结构 |
| [HTTP API](docs/api.md) | 请求、评分字段、控制量和配置 |
| [预热](docs/warmup.md) | Worker 设置、磁盘缓存和进度 |
| [NAVSIM 对齐](docs/alignment.md) | 固定版本基准、对比脚本和数值回归 |

## 引用

如果本项目对你的研究有帮助，请引用 [GRAVA](https://arxiv.org/abs/2609.15169)：

```bibtex
@misc{liu2026grava,
  title         = {GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving},
  author        = {Xiao Liu and Haoyu Li and Jianghao Leng and Lin Wang and Chao Sun},
  year          = {2026},
  eprint        = {2609.15169},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.15169}
}
```

## 致谢

感谢 [NAVSIM](https://github.com/autonomousvision/navsim) 和
[nuPlan](https://github.com/motional/nuplan-devkit) 团队提供的仿真工具、数据格式和评测协议。
