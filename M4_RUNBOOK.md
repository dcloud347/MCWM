# M4 运行手册

本文说明如何构建 macro-action codebook、运行离线规划，以及在 MineRL 1.0 中运行
receding-horizon MPC smoke test。

## 1. 准备运行环境

### 1.1 仅构建 codebook 或运行离线规划

建议使用 Python 3.9 或 3.10：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[planning,test]'
```

确认关键依赖：

```bash
python3 -c 'import torch, numpy, yaml, PIL; print(torch.__version__)'
```

### 1.2 运行 MineRL 1.0

MineRL 1.0 使用旧版 Gym，需要：

- Python 3.9 或 3.10；
- Java JDK 8；
- 支持图形显示的桌面环境，或 Linux 下使用 `xvfb-run`；
- 不要使用 Python 3.13 或 NumPy 2.x 环境。

确认 Java：

```bash
java -version
javac -version
```

两个命令都应显示 Java 8。然后安装：

```bash
python3 -m pip install -e '.[minerl,test]'
```

确认 MineRL：

```bash
python3 -c 'import gym, minerl; print("MineRL environment ready")'
```

## 2. 准备输入

运行 M4 需要以下文件：

```text
/path/to/vpt-store/
  dataset_manifest.json
  episodes/...

/path/to/checkpoint-00006000.pt       # M2 checkpoint
/path/to/m1-checkpoint.pt             # M1 parent；原路径失效时显式传入
/path/to/current.png                   # 离线规划的当前画面
/path/to/goal.png                      # 目标画面
```

VPT store 必须已经完成 split，且包含 `source=vpt`、`split=train` 的 contractor
episodes。MineRL 在线轨迹不会被加入训练 manifest，也不会用于拟合 codebook。

M2 checkpoint 会记录原始 M1 parent 路径和 SHA-256。如果该路径仍然有效，不需要传
`--m1-checkpoint`；如果 checkpoint 被移动过，需要传入内容相同的 M1 文件。

## 3. 构建 macro-action codebook

从 VPT training split 构建：

```bash
mkdir -p artifacts/m4

PYTHONPATH=src python3 scripts/build_macro_codebook.py \
  /path/to/vpt-store \
  --output artifacts/m4/macro_codebook.json
```

默认行为：

- 把 VPT source-rate 动作聚合成 4 FPS model ticks；
- 每个 macro 包含 2 个 model ticks；
- 默认最多生成 512 个 code；
- 固定保留 no-op、前进、左右转向、跳跃、攻击和使用；
- 记录 manifest hash、训练 split、拟合参数和随机种子；
- 相同输入与配置生成逐字节相同的 JSON。

自定义上限或随机种子：

```bash
PYTHONPATH=src python3 scripts/build_macro_codebook.py \
  /path/to/vpt-store \
  --output artifacts/m4/macro_codebook.json \
  --max-codes 512 \
  --seed 2026
```

构建成功后会输出 code 数量、manifest SHA-256 和 codebook SHA-256。

## 4. 运行离线规划

离线模式不启动 Minecraft。它读取一张当前画面和一张 goal image，执行一次规划，并
把最优 4-macro 计划及首个 2-tick macro 写入 JSON。

### 4.1 GPU 运行

```bash
PYTHONPATH=src python3 scripts/plan_offline.py \
  --checkpoint /path/to/checkpoint-00006000.pt \
  --m1-checkpoint /path/to/m1-checkpoint.pt \
  --codebook artifacts/m4/macro_codebook.json \
  --config configs/plan_m4.yaml \
  --observation /path/to/current.png \
  --goal /path/to/goal.png \
  --device cuda \
  --output artifacts/m4/offline_plan.json
```

如果 M2 checkpoint 中记录的 M1 parent 路径仍然存在，可以删除
`--m1-checkpoint` 参数。

### 4.2 CPU 功能检查

正式模型在 CPU 上会很慢，只建议检查加载和接口：

```bash
PYTHONPATH=src python3 scripts/plan_offline.py \
  --checkpoint /path/to/checkpoint-00006000.pt \
  --codebook artifacts/m4/macro_codebook.json \
  --observation /path/to/current.png \
  --goal /path/to/goal.png \
  --device cpu \
  --output /tmp/m4-offline-smoke.json
```

## 5. 运行 MineRL 在线 MPC

默认配置为：

```text
4 FPS model frequency
2 model ticks / macro
4 macros / plan
8-step latent rollout
64 candidates
8 elites
4 CEM iterations
candidate chunk size = 16
```

运行 10 次“观察—规划—执行 2 ticks—重新观察”：

```bash
PYTHONPATH=src python3 scripts/run_minerl_mpc.py \
  --env-id MineRLBasaltFindCave-v0 \
  --checkpoint /path/to/checkpoint-00006000.pt \
  --m1-checkpoint /path/to/m1-checkpoint.pt \
  --codebook artifacts/m4/macro_codebook.json \
  --config configs/plan_m4.yaml \
  --goal /path/to/goal.png \
  --cycles 10 \
  --device cuda \
  --output artifacts/m4/m4_smoke.json
```

Linux 无显示器时可以尝试：

```bash
xvfb-run -a env PYTHONPATH=src python3 scripts/run_minerl_mpc.py \
  --env-id MineRLBasaltFindCave-v0 \
  --checkpoint /path/to/checkpoint-00006000.pt \
  --codebook artifacts/m4/macro_codebook.json \
  --config configs/plan_m4.yaml \
  --goal /path/to/goal.png \
  --cycles 10 \
  --device cuda \
  --output artifacts/m4/m4_smoke.json
```

环境遇到 `terminated`、`truncated`、异常或手动中断时会安全关闭，并尽量写入报告。

## 6. GPU 与显存配置

正式规划当前使用单张 GPU。

### H100/A100 80GB

直接使用 [configs/plan_m4.yaml](configs/plan_m4.yaml)：

```yaml
cem:
  candidates: 64
  elites: 8
  iterations: 4
  candidate_chunk_size: 16
```

### L40S/A6000 48GB

复制配置后把 chunk 降到 4～8：

```bash
cp configs/plan_m4.yaml /tmp/plan_m4_48gb.yaml
```

编辑 `/tmp/plan_m4_48gb.yaml`：

```yaml
cem:
  candidates: 64
  elites: 8
  iterations: 4
  candidate_chunk_size: 8
```

### RTX 4090/3090 24GB

建议使用：

```yaml
cem:
  candidates: 32
  elites: 4
  iterations: 3
  candidate_chunk_size: 2
```

如果仍然 OOM，把 `candidate_chunk_size` 降为 1。planner 在 OOM 或全部候选无效时会
回退到 legality mask 允许的 no-op，并在报告中记录 `fallback_reason`。

## 7. 查看结果

离线规划结果：

```bash
python3 -m json.tool artifacts/m4/offline_plan.json | less
```

MineRL smoke report：

```bash
python3 -m json.tool artifacts/m4/m4_smoke.json | less
```

重点检查：

- `completed_cycles` 是否达到 10；
- `termination_reason`；
- `fallback_ratio`；
- 每轮的 `cost` 和 `predicted_goal_costs`；
- `selected_macro_codes`；
- `planning_seconds`；
- `reward`、`terminated` 和 `truncated`。

## 8. 运行测试

运行 M4 相关测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest \
  tests.planning.test_macro_codebook \
  tests.planning.test_cem \
  tests.planning.test_online \
  tests.envs.test_minerl1 \
  tests.models.test_ac_predictor -v
```

运行完整测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -v
```

## 9. 常见问题

### `CUDA planning requested but CUDA is unavailable`

当前 PyTorch 没有 CUDA 支持，或进程看不到 GPU。检查：

```bash
python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
nvidia-smi
```

### `codebook and checkpoint use different data manifests`

codebook 和 M2 checkpoint 不是由同一个 dataset manifest 产生。重新使用训练 M2 时的
VPT store 构建 codebook。

### `M1 parent checkpoint hash does not match M2 provenance`

传入了错误的 M1 checkpoint。必须使用训练该 M2 checkpoint 时对应的同一份 M1 文件。

### MineRL 安装失败

确认使用 Python 3.9/3.10、JDK 8，并且没有在同一环境中混装 NumPy 2.x 或
`gymnasium` 来替代 MineRL 所需的旧版 `gym`。

### `all_candidates_invalid`

所有候选都被 camera residual 或 legality 检查拒绝。先确认 GUI planning 关闭、goal
合理且 codebook 正确；然后降低 `initial_residual_std`，例如从 `0.25` 调到 `0.10`。

