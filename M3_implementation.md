# M3 多步 Latent Rollout 实施方案

## 1. 目标与边界

M3 验证 world model 能否把自己的预测持续反馈为下一步输入，在不观察真实未来帧的
情况下模拟更长的未来。M2 checkpoint 已使用 `auto_steps=4` 训练，因此 M3 不重复
实现 4-step rollout，而是：

1. 系统评估 `1/2/4/6/8/10/12/14` 步 open-loop 误差与漂移。
2. 分析移动、交互、camera、hotbar 和 GUI 动作下的长期误差。
3. 用视觉/轨迹不连续扰动验证 surprise 是否在事件点出现峰值。
4. 仅在长 horizon 明显失稳时，从 M2 checkpoint 继续做多步训练。

M3 不包含 CEM、goal selection 或 MineRL 在线执行；这些属于 M4。

## 2. 基线与数据

- 主模型：通过正式 gate 的 M2 `checkpoint-00006000.pt`。
- 数据：沿用同一 canonical validation split 和 M1 frozen encoder。
- provenance：严格验证 M1 parent SHA-256、manifest hash 和 M2 config。
- 对照：使用相同架构、数据和预算的 `auto_steps=1` baseline；比较时必须使用相同
  sample IDs 和动作序列。

M3 evaluation 将 `frames_per_sample` 扩展为 16，因此每条 clip 提供 15 个
transition，并在 4 FPS 下覆盖约 3.75 秒。现有 `context_blocks=16` 足够支持 14-step
rollout，不修改 predictor 结构。所有 horizon 都从真实初始 latent 开始，随后只反馈
predicted latent，禁止中途注入真实 latent。

M3 evaluator 允许只覆盖 evaluation 的 clip 长度；它不得使用 M2 `--eval-only`，因为
后者会严格拒绝 checkpoint 训练配置中的 `frames_per_sample` 发生变化。

## 3. 实现内容

新增：

```text
src/mcwm/diagnostics/rollout.py
src/mcwm/diagnostics/surprise.py
src/mcwm/training/evaluate_m3.py
scripts/evaluate_m3.py
configs/evaluate_m3.yaml
tests/diagnostics/test_rollout.py
tests/diagnostics/test_surprise.py
tests/training/test_m3_evaluation.py
```

`evaluate_m3.py` 只做离线评估，输出 `m3_evaluation.json`，不修改 checkpoint。
报告至少包含：

- 每个 horizon 的 normalized latent L1、cosine 和相对 M2 one-step 的增幅。
- 逐 step error curve，以及 `4→8`、`8→12`、`12→14` 的 drift slope。
- 按 movement、interaction、camera、hotbar、GUI 分桶的样本量和误差。
- predicted/target latent 的 std、effective rank 和 pairwise cosine。
- surprise 原始曲线、扰动点峰值、扰动前基线及无扰动 control。

## 4. Surprise 评估

对固定 validation clips 构造三组输入：

1. 原始连续轨迹。
2. 在中间替换一帧为同 split 的无关帧。
3. 在中间切换到另一个 world/session 的片段。

定义一步 surprise：

```text
s(t) = mean((LN(z_pred(t+1)) - LN(z_target(t+1)))²)
```

扰动实验只用于诊断，不进入训练集。峰值必须出现在扰动点或其后一步，并显著高于
同一样本的扰动前窗口和未扰动 control。

## 5. 执行顺序

1. 先用 16-frame clips 对现有 M2 checkpoint 做
   `1/2/4/6/8/10/12/14` 步评估，不训练。
2. 运行 action buckets 和 surprise 评估，定位误差来源。
3. 训练或确认 `auto_steps=1` baseline，完成 paired comparison。
4. 若当前模型在 14 步仍优于 baseline，则直接保留 M2 权重作为 M3 checkpoint。
5. 若长 horizon 漂移过快，再根据误差拐点选择 `auto_steps=8/12/14`，从 M2 低学习率
   续训，不得改动 M1 encoder。
6. 对候选 checkpoint 重跑完整 M2 gate，防止长 horizon 训练削弱动作敏感性。

## 6. 完成标准

- 1/2/4-step 指标能复现 M2 验收结果，全部 8 个 horizon 数值 finite。
- 4/8/12/14-step open-loop error 均显著低于 `auto_steps=1` baseline 的自由滚动。
- 14-step 误差允许增长，但不能出现突跳、NaN 或 latent collapse。
- surprise 在视觉替换和 world/session 切换点产生统计显著峰值。
- 每个主要动作 bucket 都报告样本量；低覆盖 bucket 明确标为证据不足。
- 最终 checkpoint 重新通过 M2 shuffled/no-op action-sensitivity gate，provenance 完整。

通过后，M3 的产物是一个经过长期 rollout 验证的 latent dynamics checkpoint 和对应
诊断报告，可作为 M4 离线规划与 MineRL MPC smoke test 的输入。
