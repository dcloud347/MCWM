# M4 MineRL Planning Smoke Test 实施方案

## 1. 目标与边界

M4 使用通过 M2/M3 评估的 `checkpoint-00006000.pt`，验证 world model 能否进入
“观察—规划—执行—再观察”的在线闭环。第一版只要求闭环稳定、动作合法和规划代价
可计算，不要求完成 Minecraft 任务或报告 benchmark 成功率。

MineRL 1.0 只作为交互环境。环境帧、动作和轨迹可以写入 smoke-test 日志，但不得加入
训练 manifest、不得用于更新 M1/M2 权重，也不得拟合 macro-action codebook。

## 2. 控制与规划范围

- 模型频率保持与训练数据一致的 4 FPS；环境 wrapper 负责把一个 model tick 映射为
  MineRL action repeat，并记录实际 elapsed time。
- 默认 macro 长度为 2 个 model ticks，规划 4 个 macro，即 8-step latent horizon。
- MPC 每次只执行第一个 macro（2 ticks），随后读取新 observation 并重新规划。
- 首版默认关闭 GUI planning；允许移动、camera、attack/use 和 hotbar，并通过 legality
  mask 禁止互斥移动、无效 hotbar、过大 camera 及不支持的 GUI 组合。
- M3 的 14-step rollout 仅用于诊断；在线默认不直接执行 14-step open-loop 计划。

## 3. Macro-action codebook

仅从 VPT contractor **training split** 提取固定 2-tick 动作块。先按 binary movement、
interaction、hotbar event 和 GUI mode 分组，再对 camera 轨迹做标准化聚类。每个 code
保存：

- 两个 tick 的 canonical binary/categorical 模板；
- camera 均值、标准差和 residual 上限；
- GUI mode、样本数及合法性 metadata。

过滤 padding、损坏动作和低覆盖组合，并显式保留 no-op、前进、转向、跳跃、攻击和
使用等基础 code。产物 `macro_codebook.json` 保存 manifest hash、拟合参数、随机种子和
训练 split provenance；相同输入必须生成相同 codebook。

## 4. Hybrid CEM 与目标函数

CEM 对每个 macro 位置维护 categorical code 分布，对每个 micro tick 的 camera residual
维护二维 Gaussian。每轮采样候选序列、展开成 canonical actions、用 world model 做
8-step rollout，选择 elite 后更新两类分布。初始建议：64 个候选、8 个 elite、4 轮，
并支持 candidate chunking 控制显存。

给定当前 observation/context 与 goal image：

```text
z_goal = E_target(goal)
cost = latent_goal_cost(z_pred[H], z_goal)
     + action_change_penalty
     + camera_residual_penalty
     + invalid_action_penalty
```

`latent_goal_cost` 使用 layer-normalized latent L1 与 cosine distance 的加权和。所有候选
必须产生 finite cost；若本轮全部无效或 OOM，planner 返回经过 legality mask 的 no-op，
同时记录 fallback 原因。在线 goal 由文件或 `GoalProvider` 注入，首版不实现语言目标或
自动子目标生成。

## 5. 在线 MPC 与 MineRL 适配

新增反向动作适配，将 `CanonicalActionTick` 转成 MineRL action dict；输入 observation
统一转换为训练时的 640×360 tensor。MPC 保存最近 16 帧及实际执行动作作为 context，
episode 开始时用首帧和 valid no-op 做确定性的 warm-up padding。

每个 planning cycle 记录 goal cost、候选/elite 数、选中 macro、预测轨迹、规划耗时、
fallback、environment reward 和终止原因。MineRL 的 `terminated`、`truncated`、异常及
手动中断必须安全关闭环境并落盘摘要。

建议新增：

```text
src/mcwm/planning/macro_actions.py
src/mcwm/planning/hybrid_cem.py
src/mcwm/planning/mpc.py
src/mcwm/envs/minerl1.py
scripts/build_macro_codebook.py
scripts/plan_offline.py
scripts/run_minerl_mpc.py
configs/plan_m4.yaml
tests/planning/
tests/envs/test_minerl1.py
```

MineRL 依赖保持 optional；普通单元测试使用 mock environment，不要求启动 Minecraft。

## 6. 实施顺序

1. 实现 codebook builder、序列展开和动作 legality tests。
2. 实现与环境无关的 hybrid CEM，并用 toy cost 验证 elite 更新确实降低代价。
3. 加载 M2 checkpoint，在固定 validation goals 上运行离线 planning smoke test。
4. 实现 MineRL observation/action wrapper 和 mock-env MPC 闭环。
5. 在 MineRL 中先跑 10 个 planning cycles，再扩展到完整短 episode。
6. 根据显存和实时性调整候选数/chunk，不修改 checkpoint；若 8-step 规划不稳，降为
   4-step horizon 并提高 replanning 频率。

## 7. 完成标准

- codebook 只使用 VPT training split，provenance 和合法性检查完整。
- CEM 在固定种子下可复现，并在 toy/offline smoke 中优于初始随机候选的预测代价。
- planner 输出的每个动作都能通过 canonical 与 MineRL action-space 校验。
- MineRL 中至少连续完成 10 次“观察—规划—执行—再规划”，无 NaN、OOM、环境泄漏或
  未处理异常；episode 结束后生成 `m4_smoke.json`。
- 报告规划耗时、fallback 比例、选中 macro 分布和逐 cycle goal cost。
- 不要求 reward 提升或任务成功率；这些属于后续 benchmark，而不是 M4 gate。
