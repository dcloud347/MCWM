# MCWM: Minecraft Joint-Embedding World Model 设计文档

## 0. 文档状态

- 状态：首版实现计划（Benchmark 暂不在范围内）
- 目标环境：Minecraft 1.16.5；训练数据对齐 VPT contractor 格式，未来在线评估可接 MineRL 1.0 environment
- 数据来源：仅使用带动作标签的 VPT contractor demonstrations
- 训练原则：所有模型均由本项目从随机初始化开始训练
- 明确排除：MineRL/BASALT demonstrations、VPT 无标签网络视频、Malmo、MineRL-v0、旧版 Treechop 数据、任何外部预训练权重

本文设计基于 [LeWorldModel](https://arxiv.org/html/2603.19312v1) 的 latent world model 思路，但按需求采用带 EMA target encoder 和 stop-gradient 的两阶段训练，而不是照搬论文中完全端到端、无 EMA/stop-gradient 的训练方式。

## 1. 最终目标

我们要训练一个只依赖第一人称 RGB 画面和键鼠动作的 Minecraft latent world model。给定一段历史画面与动作，模型在低维 latent space 中预测动作执行后的未来状态；后续可在该 latent space 中进行目标图像条件规划。

首期完成标准如下：

1. 能把 VPT contractor demonstrations 转换为时序严格对齐的 canonical 数据格式。
2. 从零训练一个 Minecraft 视觉 encoder，不加载任何外部视觉权重。
3. 从零训练 action encoder 和 action-conditioned predictor。
4. 训练过程不发生 latent collapse，并能通过量化指标检测 collapse。
5. 真实动作的预测误差显著低于动作清零或动作打乱后的误差，证明 predictor 确实使用动作。
6. 能进行离线多步 latent rollout，并输出 rollout error、action sensitivity 和 surprise 曲线。
7. 预留 MineRL 1.0 在线 MPC 接口，但首期不报告任务 Benchmark 分数。

## 2. 非目标

首期不做以下工作：

- 不生成或重建未来像素；模型预测的是 latent，不是视频。
- 不复现 VPT 的 behavioral cloning 或 RL policy。
- 不使用 VPT 官方 policy、IDM、model zoo 或其他 checkpoint。
- 不使用 ImageNet、DINO、I-JEPA、V-JEPA 等外部预训练权重。
- 不兼容 MineRL-v0 的 Minecraft 1.11 数据和旧动作空间。
- 不使用 Malmo 作为数据源或运行环境。
- 不以 ObtainDiamond、Treechop 或 BASALT 成功率作为首期交付条件。
- 不在第一版实现长时程分层规划、语言目标或奖励学习。

## 3. 核心技术决定

### 3.1 两阶段训练，而不是直接端到端训练

整个系统分为两个独立、可验收的阶段。

**阶段 A：Minecraft 视觉预训练**

在 VPT contractor demonstration 视频上训练 masked video JEPA。online encoder 只看到被 mask 的时空 context，predictor 预测完整视频中被 mask 区域的 target latent。target encoder 不参与反向传播，通过 online encoder 的 EMA 更新：

```text
θ̄ ← τθ̄ + (1 − τ)θ
```

target latent 显式 stop-gradient：

```text
p = P_visual(E_θ(mask(x)), M)
ȳ = E_θ̄(x)
L_visual = mean_(i,t)∈M ‖p(i,t) − sg(ȳ(i,t))‖₁
```

其中 `M` 是被遮挡的空间—时间 token 集合；`p(i,t)` 是 predictor 对该位置的预测，`ȳ(i,t)` 是 EMA target encoder 在完整视频上产生的目标。梯度只经过 `p` 分支。

EMA 和 stop-gradient 默认同时启用，不把它们当作二选一。训练完成后使用 EMA encoder 作为本项目自己的 pretrained visual encoder。

**阶段 B：动作条件世界模型训练**

action encoder 和 action-conditioned predictor 全部随机初始化。视觉 encoder 只加载阶段 A checkpoint 中的 EMA target 权重，并在 M2 中永久冻结。每帧复制为一个静态 2-frame tubelet，独立编码且保留全部 spatial tokens：

```text
z(t) = frozen_E_EMA([o(t), o(t)])
```

```text
ẑ(t+1) = P_φ(z(≤t), A(≤t))
```

```text
L_wm = L_teacher_forced + L_autoregressive
```

M2 不维护新的 online/EMA encoder，不对视觉 encoder 反向传播，也不做 spatial mean pooling。

### 3.2 M2 默认使用 normalized latent prediction loss

预测和冻结 target latent 都沿最后一维做无仿射 LayerNorm，然后计算 L1：

```text
L_tf = mean(abs(normalize(ẑ_tf) - normalize(z_target)))
L_ar = mean(abs(normalize(ẑ_ar) - normalize(z_target)))
L_wm = L_tf + L_ar
```

正式首版使用 `auto_steps=2`。SIGReg、IDM 和 pixel reconstruction 都不是 M2 默认 loss；若启用，必须作为独立扩展实验。

### 3.3 所有权重都必须由 MCWM 训练产生

下面的模块均从随机初始化开始训练：

- visual online encoder
- visual EMA target encoder（初始化时复制 online encoder，之后仅 EMA 更新）
- masked-video predictor
- Minecraft action encoder
- action-conditioned latent predictor
- 可选 inverse dynamics head
- 可选 diagnostic decoder
- planner 使用的 macro-action codebook

VPT 仓库只允许用于理解 contractor 数据格式和动作语义。代码可以重新实现兼容逻辑，但不得自动下载或加载 VPT `.weights` / `.model` 文件。

每个 checkpoint 保存以下 provenance：

- git commit
- 完整 resolved config
- 随机种子
- 数据 manifest hash
- parent checkpoint ID
- W&B entity、project、run ID 和 run name
- `external_pretrained=false`

加载 checkpoint 时若缺失 provenance 或发现外部权重标记，训练入口直接报错。本项目不提供加载外部预训练权重的配置开关。

## 4. 数据设计

### 4.1 统一数据范围

| 数据 | 用途 | 是否首期使用 | 备注 |
|---|---|---:|---|
| VPT contractor demonstrations | 视觉预训练、动作条件训练 | 是 | MP4 + 每帧 JSONL 动作；Minecraft 1.16.5 |
| MineRL 1.0 / BASALT demonstrations | 不使用 | 否 | 不进入下载、预处理、训练或评估数据集 |
| VPT 无标签网络视频 | 不使用 | 否 | 不训练或使用 IDM 生成伪标签 |
| MineRL 1.0 environment | 在线 smoke test / 未来规划 | 是 | 仅作为交互环境，不保留其 trajectory 作为训练数据 |

训练语料固定为 VPT contractor demonstrations。它们已经带有键鼠动作 JSONL，因此不需要 VPT 官方 IDM，也不为无标签视频生成伪标签。训练入口必须拒绝包含任何非 VPT episode 的 manifest；数据 manifest 和 checkpoint provenance 必须能证明所有训练 episode 都来自 contractor 集合。

### 4.2 Canonical action schema

不能像参考项目一样把动作简单压成十维向量，因为这会丢失 `use/place`、hotbar、GUI、drop、swap-hands 等关键状态变化。统一动作定义如下：

```text
CanonicalActionTick
  movement: bool[7]
    forward, back, left, right, jump, sneak, sprint
  interaction: bool[7]
    attack, use, drop, pick_item, swap_hands, inventory, esc
  hotbar: int8                  # 0 表示未切换，1..9 表示目标槽位
  camera: float32[2]            # pitch_delta, yaw_delta，单位为度
  cursor: float32[2] | null     # GUI 打开时的归一化鼠标坐标
  gui_open: bool
  valid: bool
  timestamp_ms: int64
  source: enum[vpt]
  label_confidence: float32     # 人工记录动作固定为 1.0
```

训练 sample 使用动作块而不是破坏性聚合：

```text
frames:       uint8[T+1, 3, 360, 640]
action_block: CanonicalActionTick[T, K]
frame_time:   int64[T+1]
valid_mask:   bool[T, K]
metadata:     episode_id, session_id, source, recorder_version
```

`K` 是两个保留帧之间的原始动作 tick 数。视觉预训练遵循 V-JEPA 的 `sample_fps=4`，即按时间戳每 250 ms 保留一帧；`K` 因源视频帧率而异。原始逐 tick 动作始终保留，因此可以通过配置改变采样率而不用重新下载数据。

### 4.3 动作编码

每一个动作 tick 分组件编码：

- binary buttons：每个按键独立 embedding，支持同时按键。
- hotbar：10 类 embedding（未切换 + 1..9）。
- camera：保存原始连续值，经裁剪、mu-law 标准化后输入小型 MLP。
- GUI/cursor：独立 embedding/MLP，并由 `gui_open` mask。
- source、位置、世界坐标和 inventory 等特权信息不输入 world model。

同一个 macro step 内的 `K` 个 tick 由一个两层 micro-action Transformer 汇总为 `a_t`。这样能区分“先转头再攻击”和“先攻击再转头”，也避免只对 camera 求和、binary 求 OR 所造成的动作顺序丢失。

### 4.4 时间对齐规则

数据的核心不变量是：`action_block[t]` 必须恰好包含从 `frame[t]` 到 `frame[t+1]` 之间发生的动作。

预处理步骤：

1. 读取 MP4 的真实 PTS，而不是假设固定帧率。
2. 读取 JSONL 中的 `milli`、`tick`、`serverTick` 等可用时间字段。
3. 以 frame PTS 为边界，将动作分配到半开区间 `[frame_t, frame_{t+1})`。
4. 对 recorder version 特有的 mouse scaling 做版本化修正。
5. 检测缺失 chunk、时间倒退、超长间隔、重复帧和损坏视频。
6. 遇到 discontinuity 就切断 episode，绝不跨断点采样 sequence。
7. GUI 打开时按 VPT 规则把 cursor 合成到画面，同时保留 cursor 数值。
8. 保留真实 no-op。no-op 对学习惯性、掉落、昼夜变化和被动环境动态很重要，不能像 VPT demo loader 那样直接跳过。
9. 修复可检测的 stuck-attack 和 hotbar 切换记录问题，并将修复写入 audit log。

每个 clip 严格满足：

```text
o_0 --A_0--> o_1 --A_1--> ... --A_(T-1)--> o_T
```

将提供一个人工可视化工具，逐帧叠加动作、时间戳和 action block 边界，抽查对齐结果。

### 4.5 图像预处理

- 解码后统一为 RGB `uint8`。
- 模型输入固定为 `640x360`，即 tensor shape 为 `(C=3, H=360, W=640)`。
- VPT contractor 的 360p 数据保持原始 `640x360`，不做常规空间 resize、crop 或 square distortion。
- 若文件不满足 `640x360`，预处理直接标为异常；只有在来源明确且配置了版本化转换规则时才允许转为 `640x360`，不能静默缩放。
- 不做会改变动作语义的水平翻转。
- 亮度、gamma、轻微颜色扰动只用于阶段 A 视觉预训练，阶段 B 默认关闭强增强。
- 原始分辨率、FOV、GUI scale 和任何例外转换参数写入 manifest。
- normalization 由模型入口统一完成，磁盘中保留 `uint8`，避免重复量化。

### 4.6 数据存储

采用两层存储，避免一个不可增量维护的巨大 HDF5：

1. **Canonical episode store**
   - 原始或重封装 MP4
   - Arrow/Parquet action table
   - episode manifest JSON
   - QA/audit report
2. **Training shard cache**
   - WebDataset tar shards
   - 每个 sample 是连续 clip
   - shard 由 manifest hash 和 preprocessing config 唯一命名

数据划分必须按 `session_id/world_id` 完成，不能随机按 clip 划分，否则同一个世界的相邻片段会泄漏到 train 和 validation。

建议首期划分：train 90%、validation 5%、test 5%。test 暂时只用于离线诊断，不作为 Benchmark。

### 4.7 数据质量门槛

预处理完成必须生成以下报告：

- 每个来源的小时数、episode 数、有效 transition 数。
- action component 频率和组合频率。
- no-op 比例、GUI 比例、camera 分布和异常值。
- 帧间隔、动作间隔和 A/V offset 分布。
- discontinuity、损坏文件和修复记录。
- train/val/test 的 session/world 去重结果。
- 随机抽样的 frame/action overlay 视频。

如果 validation 中出现 train 的 session/world，数据构建任务直接失败。

## 5. 模型架构

### 5.1 视觉 encoder

首版使用从零训练的 ViT-Large 级别 encoder：

| 参数 | 默认值 |
|---|---:|
| 输入 | 640 × 360 RGB |
| patch size | 20 |
| tubelet size | 2 frames |
| video tokens | 8 × 32 × 18 = 4608 |
| depth | 24 |
| heads | 16 |
| hidden dim | 1024 |
| MLP hidden dim | 4096 |
| pooled latent dim | 1024 |
| 参数量 | 约 305M |

阶段 A 需要 token-level 输出用于 masked prediction。阶段 B 对滚动 16 帧窗口的 video tokens 做 pooling，得到 1024 维 state latent；首版使用 mean pooling，不额外引入 CLS token。

visual encoder 对齐 V-JEPA 2 的 Video ViT 数据流：先用 `Conv3d(kernel=stride=(2,20,20))` 生成 tubelet tokens，再让所有可见时空 token 在 encoder 内联合 self-attention。阶段 B 因此维护滚动 16 帧 observation window，而不是把 encoder 当作单帧 2D 网络。

输入使用官方 ImageNet normalization：mean `(0.485, 0.456, 0.406)`、std `(0.229, 0.224, 0.225)`。

online 和 EMA target encoder 结构完全相同。target encoder 参数 `requires_grad=false`，并在每次 optimizer step 后更新；不能在 micro-batch/gradient accumulation 中间更新。

EMA momentum 固定为 `0.999`。checkpoint 同时保存 online encoder、EMA encoder 和训练调度器状态。

### 5.2 阶段 A masked-video predictor

`360` 不能被 patch size 16 整除，因此首版使用 `20x20` non-overlapping patches；`640/20=32`、`360/20=18`，无需 padding 或裁剪。

视觉预训练遵循 V-JEPA 的随机 clip 语义：每次访问一个视频时随机选择起点，再以 `sample_fps=4` 按 PTS 保留 16 帧，不预先枚举固定 hop 窗口。2 帧 tubelet 把时间长度降为 8，因此每个 clip 有 `8×18×32=4608` 个 video tokens。encoder 和 predictor 都使用联合时空 self-attention 与 3D RoPE；CUDA 上由 PyTorch SDPA 自动选择 Flash Attention。

mask 使用 V-JEPA 2 的两组 multi-block 配置。第一组把 8 个约占空间 patch 网格 15% 的矩形取并集，第二组把 2 个约占 70% 的矩形取并集；两组的 aspect ratio 都在 `[0.75, 1.5]` 内，temporal scale 固定为 `1.0`，所以每个矩形贯穿全部 8 个 tubelet 时间位置。同一 clip 分别用两套 mask 构造 context，两个 masked-token prediction loss 等权平均；不把 10 个矩形合成一个 mask，也不补删随机 patch 来凑固定比例。

online encoder 在 Transformer 前移除预测区域，只联合编码可见 context tokens；EMA target encoder 对完整 clip 的 4608 个 tokens 只计算一次。阶段 A predictor 对齐官方设置，使用 12 层、model dim 384、12 heads、MLP dim 1536，并为两套 mask 使用两个独立的零初始化 mask token。predictor 将 context 与 target mask tokens 按原视频位置排序后做联合时空 attention，只返回目标位置预测。

640x360 是不可降级的模型输入契约。为控制显存，训练实现默认启用 bf16、scaled-dot-product/Flash Attention、activation checkpointing、FSDP/ZeRO 和 gradient accumulation，而不是降低输入分辨率。

训练结束时保留 EMA visual encoder；阶段 A predictor 不直接用于阶段 B，防止把“补 masked token”和“动作条件未来预测”混成一个不可解释模块。

M1 默认配置实现后的实际参数量为：单份 visual encoder `304,770,048`，masked-video predictor `22,082,944`，阶段 A 可梯度参数 `326,852,992`，包含 online/EMA 两份 encoder 的 checkpoint 总参数 `631,623,040`。这些数值由 meta-tensor 参数预算测试锁定，不依赖近似估算。

### 5.3 Action encoder

Action encoder 完全从零训练，由以下部分组成：

1. component encoders：binary、categorical、continuous、cursor。
2. component fusion MLP：得到 256 维 tick-level action token。
3. 两层 micro-action Transformer（dim 256、8 heads、MLP dim 1024）：聚合一个 frame interval 内的 `K` 个 action tick。
4. projection：输出 1024 维 action embedding，与 predictor conditioning dim 相同。

Action Encoder 的目标参数量约为 2M；实现完成后以实际 `numel()` 为准，允许在 1.5M–3M 内调整。

padding tick 必须有 `valid_mask`；padding 不能被解释为 no-op。

### 5.4 Action-conditioned predictor

Predictor 是一个 causal Transformer，从随机初始化开始训练：

| 参数 | 默认值 |
|---|---:|
| context length | 16 macro steps |
| input latent dim | 1024 |
| depth | 24 |
| model dim | 1024 |
| heads | 16 |
| head dim | 64 |
| MLP hidden dim | 4096 |
| output dim | 1024 |
| dropout | 0.1 |
| conditioning | action-token interleaving |
| 参数量 | 约 305M |

输入为 observation latent history 和 action-block embeddings。每个时间点排列为 `[action token, 576 visual tokens]`；同一 block 内完全互相可见，当前 block 可以看到过去 block，但不能看到未来 block。位置编码使用 3D RoPE。

训练使用 causal mask。teacher forcing 下并行预测每个 next latent：

```text
input:   z_0 ... z_(T-1), A_0 ... A_(T-1)
target:  z_1 ... z_T
output:  zhat_1 ... zhat_T
```

推理 rollout 时使用滑动 context，把预测 latent 作为后续输入。实现中必须分别测试 teacher-forced path 和 autoregressive path，防止只在训练路径正确。

### 5.5 参数预算

参数预算只统计实际推理保留的 frozen visual encoder、action encoder 和 AC predictor，不包括阶段 A predictor、optimizer state、diagnostic decoder 或可选 IDM head。

| 推理模块 | 目标参数量 |
|---|---:|
| EMA Visual Encoder | 约 305M |
| Action Encoder | 约 2M |
| Action-Conditioned Predictor | 约 305M |
| **总计** | **约 612M** |

参数预算以实际 `numel()` 为准；结构修改后由测试防止模型无意中变小或膨胀。

参数与 checkpoint 口径：

| 阶段 | 保存参数 | 可梯度参数 | 说明 |
|---|---:|---:|---|
| 阶段 A | 约 632M | 约 327M | online encoder 305M + EMA encoder 305M + visual predictor 22M |
| 阶段 B | 约 612M | 约 308M | frozen encoder 305M + action encoder 2M + action predictor 305M |
| 最终推理 | 约 612M | 不适用 | frozen encoder、action encoder 和 action predictor |

M2 checkpoint 只保存可训练的 action encoder 和 predictor，并用 path/hash 引用 M1 parent，避免重复保存冻结视觉权重。

### 5.6 M2 latent 诊断

监控项包括：

- latent mean 和 per-dimension std
- covariance off-diagonal norm
- effective rank
- 平均 pairwise cosine similarity
- predicted/target norm 和 effective-rank gap
- true/shuffled/no-op action sensitivity

### 5.7 可选 inverse dynamics head

首版默认不把 inverse dynamics loss 加进主训练目标。先用 action sensitivity test 判断 predictor 是否忽略动作。

若真实动作、打乱动作和清零动作的 prediction error 几乎相同，再启用从零训练的 inverse dynamics head：

```text
Â(t) = D(z(t), z(t+1))
```

其 loss 按动作类型拆分：binary BCE、hotbar CE、camera Huber。此时总损失为：

```text
    L = L_tf + L_ar + βL_IDM
```

该配置属于 Minecraft 扩展实验，必须与默认两项损失分开命名 checkpoint 和 run group。

## 6. 训练方案

### 6.1 阶段 A：视觉预训练

输入只使用 VPT contractor demonstrations 的有效视频片段。阶段 A 不读取动作作为模型输入，但每个视频仍必须来自带标签的 contractor 集合。按 contractor、recorder version 和 session 做分桶验证，避免少数长录像完全主导训练。

初始默认值：

| 参数 | 默认值 |
|---|---:|
| optimizer | AdamW |
| base learning rate | 1e-4 |
| weight decay | 0.05 |
| precision | bf16 |
| gradient clip | 1.0 |
| effective batch | 64 clips（通过 DDP/梯度累积） |
| warmup | 前 5% steps |
| LR schedule | cosine |
| EMA momentum | 0.999 fixed |
| clip frames | 16 |

验收指标：

- validation masked latent loss 稳定下降。
- target latent effective rank 不塌缩。
- online/EMA gap 保持有限且逐渐收敛。
- 冻结 encoder 后，简单 linear probe 能预测 camera-induced motion、GUI open 和粗粒度 scene change。
- nearest-neighbor 检索不会只按亮度、HUD 或 recorder source 聚类。

### 6.2 阶段 B0：最小世界模型 smoke test

使用少量数据和小配置跑通：

- 读入阶段 A 的 EMA visual checkpoint。
- 随机初始化 action encoder 和 predictor。
- 完成 forward、loss、backward 和 checkpoint resume。
- 在 1 个 batch 上过拟合，确认真实动作可降低误差。
- 进行 1、2、4、8 步 autoregressive rollout。

如果无法过拟合一个 batch，不进入大规模训练。

### 6.3 阶段 B1：单步 world model

默认同时优化 teacher-forced normalized L1 和可配置的多步 autoregressive normalized L1。视觉 encoder 使用阶段 A EMA 权重并保持冻结。

初始默认值参考 LeWM 官方训练范围，但针对较长 context 调低 batch：

| 参数 | 默认值 |
|---|---:|
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 0.05 |
| precision | bf16；loss FP32 |
| effective batch | 64 clips（双卡全局 batch） |
| sampled frames | 12 |
| macro action ticks K | 由真实 PTS 决定 |
| autoregressive steps | 6 |
| gradient clip | 1.0 |

### 6.4 阶段 B2：多步 rollout training

M2 正式配置训练 6-step rollout，并保留较短 horizon 指标。后续 M3 可继续加入 8 步及更长 open-loop rollout；它仍归入 prediction loss，不增加新的 loss 家族：

```text
L_auto = (1/6) Σ_h=1..6 ‖LN(ẑ(t+h)) − LN(z_target(t+h))‖₁
L_pred = L_teacher + L_auto
```

rollout 从真实初始 latent 开始，后续严格反馈 predictor 自己的输出。

### 6.5 训练顺序

严格按以下 gate 推进：

1. 数据 schema/unit tests 通过。
2. 100 条 trajectory 的可视化对齐抽检通过。
3. 阶段 A 单 batch overfit 通过。
4. 阶段 A 小数据训练无 collapse。
5. 阶段 A 全量训练并冻结一个版本化 encoder checkpoint。
6. 阶段 B0 单 batch overfit 和 resume test 通过。
7. 阶段 B1 action sensitivity 通过。
8. 阶段 B2 多步 rollout 优于 B1。
9. 离线 goal-reaching planning smoke test。
10. MineRL 1.0 在线 MPC smoke test。

### 6.6 Weights & Biases 训练监控

正式的阶段 A、B0、B1 和 B2 训练统一使用 Weights & Biases（W&B）记录实验。训练入口默认启用 W&B；单元测试、CI 和本地快速 smoke test 可显式设置 `wandb.mode=disabled`，无网络但仍需保留日志时使用 `wandb.mode=offline`，之后再执行 `wandb sync`。禁止因为 W&B 暂时不可用而丢失本地 checkpoint 或中断已开始的训练，核心标量同时写入本地 JSONL 日志。

每个 run 开始时记录：

- 完整 resolved config、git commit、随机种子和数据 manifest hash。
- 当前 milestone、数据来源、训练/验证 split 和 `external_pretrained=false`。
- 模型各模块的参数量、可训练参数量和最终推理参数量。
- 分布式 world size、每卡 batch、梯度累积步数、有效 batch、precision 和设备信息。
- parent checkpoint、W&B run ID，以及是否从 checkpoint resume。

每个 optimizer step 记录以下训练状态；高开销诊断按较低频率记录，频率全部由配置控制：

- total loss 与 teacher-forced、autoregressive prediction loss。
- learning rate、gradient norm、parameter norm 和 AMP loss scale；阶段 A 额外记录 EMA momentum。
- samples/second、tokens/second、data loading time、step time、显存使用量和累计训练时长。
- 阶段 A 的每组 mask ratio、跨帧一致性统计，以及各 contractor/recorder 分桶占比。
- latent mean/std、effective rank、pairwise cosine 和 covariance off-diagonal norm；阶段 A 额外记录 online/EMA cosine gap。

每次 validation 记录 validation loss、collapse diagnostics、linear probe、action sensitivity 和多步 rollout 指标中当前阶段适用的部分。可视化采用固定 validation sample ID，以便跨 run 比较：

- 阶段 A：原始 clip、mask 图和按 token 聚合的 prediction-error heatmap。
- 阶段 B：真实/shuffled/no-op 动作误差对比、rollout error 曲线和 surprise 曲线。

W&B artifact 只保存小型可追溯产物：resolved config、数据 audit 报告、评估汇总和版本化 checkpoint。原始 VPT contractor 视频、完整 action 文件和包含隐私信息的本地路径不得上传。大型 checkpoint 是否上传由 `wandb.log_checkpoints` 控制；无论是否上传，checkpoint 都先原子写入本地目录。

断点恢复规则：

1. checkpoint 中保存 W&B run ID 和最后完成的 optimizer step。
2. 同一次训练恢复时用该 run ID 继续写入，不能创建一个看似全新的 run。
3. 改变数据 manifest、模型结构或关键训练目标时必须创建新 run，并通过 `parent_checkpoint` 关联来源。
4. W&B 中的 step 必须等于 optimizer step；gradient accumulation 的 micro-step 不递增全局 step，也不触发 EMA 更新。

建议配置结构：

```yaml
wandb:
  enabled: true
  mode: online             # online | offline | disabled
  entity: null             # 由命令行、环境变量或私有本地配置提供
  project: mcwm
  group: m1-visual
  name: null               # 默认由 milestone、时间和短 git SHA 生成
  tags: [vpt, contractor, from-scratch]
  log_every_steps: 10
  diagnostics_every_steps: 200
  validation_every_steps: 1000
  log_checkpoints: false
```

API key 不写入仓库、配置、checkpoint 或日志；通过 `wandb login` 或运行环境的 secret 注入。多 GPU 训练只允许 rank 0 初始化和写入 W&B，其他 rank 仅参与指标归约。

## 7. 诊断与验证

Benchmark 暂缓不等于不做验证。没有下面这些诊断，world model 是否学到动作因果关系无法判断。

### 7.1 Prediction metrics

- one-step latent MSE
- h-step rollout latent MSE，`h = 1, 2, 4, 8, 16`
- cosine distance
- 按 action component 分桶的误差
- 按 source、recorder version、GUI/non-GUI 分桶的误差

### 7.2 Action sensitivity

对同一个 observation history 比较：

1. 真实动作
2. batch 内打乱动作
3. 全 no-op
4. camera 方向取反
5. attack/use 互换

至少要求真实动作条件的平均预测误差显著低于打乱和 no-op。camera 方向取反后，预测 latent 的变化方向也应产生可检测差异。

### 7.3 Collapse detection

训练中设置硬性报警：

- latent per-dim std 长时间低于阈值
- effective rank 急剧下降
- batch 内 pairwise cosine 接近 1
- normalized latent loss 出现 NaN 或爆炸
- target encoder 长时间不随 online encoder 更新

发现 collapse 时保存 failure checkpoint 和最近的数据 sample IDs，便于复现。

### 7.4 Physical/semantic probes

仅使用数据中已有但不输入模型的字段做离线 probe，例如：

- camera yaw/pitch delta
- GUI open
- hotbar change
- player displacement（字段存在时）
- inventory/stat change（字段存在时）
- attack/use 后的可见 scene change

probe 只用于判断 latent 包含什么信息，不参与 encoder 训练。

### 7.5 Surprise / violation-of-expectation

定义 surprise：

```text
s(t) = ‖P(z(≤t), A(≤t)) − z_target(t+1)‖₂²
```

构造三类离线轨迹：

- 原始连续轨迹
- 中间插入不相关 frame 的视觉扰动
- 中间切换到另一个 world/session 的物理不连续扰动

模型应在扰动点产生明显 surprise 峰值。

## 8. 规划设计（首期只做 smoke test）

LeWM 原始 CEM 假设较简单的连续动作。Minecraft 是 binary、categorical、continuous 混合动作，不能直接对完整动作向量套高斯 CEM。

### 8.1 Macro-action codebook

从训练集动作块中学习离散 macro-action codebook，但 codebook 聚类器也必须由本项目数据拟合。每个 macro action 保留：

- binary button template
- hotbar/interaction event
- camera mean 和 residual distribution
- GUI/non-GUI mode

规划时 CEM 对 macro-action ID 使用 categorical distribution，对 camera residual 使用 Gaussian distribution，并屏蔽非法组合。

### 8.2 Goal-conditioned latent cost

给定当前 observation 和 goal image：

```text
z(0) = E(o(0)),    z_goal = E(o_goal)
```

使用 predictor rollout 候选动作并最小化：

```text
C = ‖ẑ(H) − z_goal‖₂²
```

首期 goal 从同一条离线 trajectory 的未来 frame 采样，确保 goal 可达。在线 MineRL smoke test 只执行第一 macro step 就重新观察和规划，不一次性执行完整 horizon。

由于第一人称 Minecraft 存在严重部分可观测性，长时程任务后续需要 hierarchical goals 或 memory state；首期不承诺用单张 goal image 完成 ObtainDiamond 一类任务。

## 9. 工程结构

当前 MCWM 目录为空，建议建立如下结构：

```text
MCWM/
├── design.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/
│   │   └── vpt.yaml
│   ├── pretrain_visual.yaml
│   ├── train_world_model.yaml
│   └── plan.yaml
├── src/mcwm/
│   ├── actions/
│   │   ├── schema.py
│   │   ├── vpt_adapter.py
│   │   └── codec.py
│   ├── data/
│   │   ├── manifest.py
│   │   ├── alignment.py
│   │   ├── episode_store.py
│   │   ├── shards.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── visual_encoder.py
│   │   ├── visual_jepa.py
│   │   ├── action_encoder.py
│   │   ├── predictor.py
│   │   ├── sigreg.py
│   │   ├── inverse_dynamics.py
│   │   └── world_model.py
│   ├── training/
│   │   ├── ema.py
│   │   ├── logging.py
│   │   ├── pretrain_visual.py
│   │   ├── train_world_model.py
│   │   └── checkpoint.py
│   ├── planning/
│   │   ├── macro_actions.py
│   │   ├── hybrid_cem.py
│   │   └── mpc.py
│   ├── envs/
│   │   └── minerl1.py
│   └── diagnostics/
│       ├── collapse.py
│       ├── rollout.py
│       ├── probes.py
│       └── surprise.py
├── scripts/
│   ├── prepare_vpt.py
│   ├── build_shards.py
│   ├── audit_data.py
│   ├── pretrain_visual.py
│   ├── train_world_model.py
│   ├── evaluate_offline.py
│   └── run_minerl_mpc.py
└── tests/
    ├── actions/
    ├── data/
    ├── models/
    ├── training/
    └── planning/
```

训练逻辑放在 Python module 和 scripts 中，notebook 只允许作为可选分析工具，不能成为唯一的数据处理路径。

## 10. 关键测试

### 10.1 数据测试

- VPT JSONL 到 CanonicalAction 的逐字段 fixture。
- frame/action off-by-one 测试。
- 时间断点不会跨越采样测试。
- no-op 被保留、padding 与 no-op 可区分。
- GUI cursor 合成位置测试。
- session/world split 无泄漏测试。

### 10.2 模型测试

- visual online/target 初始权重完全一致。
- target encoder 没有梯度。
- optimizer step 后 EMA 公式数值正确。
- gradient accumulation 时每个 optimizer step 只更新一次 EMA。
- causal predictor 看不到未来 token。
- action mask 对 padding 生效。
- normalized latent L1 为 FP32 且有有限梯度。
- one-step 与 autoregressive rollout shape/语义一致。
- block-causal predictor 看不到未来 frame/action，同一 block 内可通信。

### 10.3 Checkpoint 测试

- 保存并恢复 action encoder、predictor、optimizer、scheduler、scaler 和 RNG state。
- FSDP checkpoint 保存各 rank 的 RNG，并在 world size 改变时拒绝直接 resume。
- M1 parent path/hash 不一致时拒绝 resume。
- resume 后下一步 loss 与 uninterrupted run 在容差内一致。
- `external_pretrained=false` provenance 存在。
- 数据 manifest hash 不一致时拒绝静默 resume。

## 11. 里程碑

### M0：工程与数据契约

- 初始化 Python package、配置和测试框架。
- 实现 CanonicalAction 和 VPT contractor 数据 adapter。
- 实现 manifest、对齐、episode store 和数据 audit。
- 产出小规模可复现 fixture dataset。

完成条件：数据测试全部通过，人工 overlay 抽检无明显时序错位。

### M1：自训练视觉 encoder

- 实现 ViT online/EMA encoders。
- 实现 masked spatiotemporal sampling 和 visual predictor。
- 完成小数据 overfit、collapse diagnostics 和 checkpoint resume。
- 接入 W&B，记录 loss、EMA、吞吐、显存、collapse diagnostics 和固定样本可视化。
- 仅在 VPT contractor demonstration 视频上训练首个 visual checkpoint。

完成条件：无 collapse，validation loss 与 probe 指标优于随机 encoder；W&B run 可从 checkpoint 无缝恢复且 provenance 完整。

### M2：动作条件 world model

- 实现 action encoder、action-token block-causal predictor 和 normalized latent loss。
- 加载我们自己的 M1 encoder，其他模块随机初始化。
- 完成 B0/B1 训练与 action sensitivity 诊断。
- 支持单机双卡 FSDP：数据和模型状态按 rank 分片，验证指标跨 rank 汇总，rank 0 独占 W&B 和 checkpoint 写入。

完成条件：真实动作条件显著优于 shuffled/no-op，one-step validation loss 稳定。

### M3：多步 latent rollout

- 实现多 horizon training 和 autoregressive rollout。
- 完成 surprise、probe 和按动作类型分桶的分析。
- 必要时以独立实验启用自训练 IDM head。

完成条件：4/8 步 rollout 优于单步模型直接自由滚动，扰动点出现 surprise 峰值。

### M4：MineRL 1.0 planning smoke test

- 学习 macro-action codebook。
- 实现 hybrid CEM 和 receding-horizon MPC。
- 接入 MineRL 1.0 environment。

完成条件：能够稳定运行观察—规划—执行—再规划闭环；暂不要求任务成功率。

## 12. 已知风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| VPT 数据对齐错误 | 模型看似不使用动作 | PTS 对齐、overlay 抽检、off-by-one fixture |
| no-op 被删除 | 惯性和被动动态学错 | 保留真实 no-op，只过滤损坏 transition |
| 动作空间过度简化 | 无法学习 use/hotbar/GUI | 完整 canonical schema + micro-action encoder |
| latent collapse | std/rank 降低 | frozen EMA target space + rank/std 硬报警 |
| predictor 忽略动作 | shuffled action 误差不变 | action sensitivity gate，必要时自训练 IDM head |
| 长 rollout 漂移 | horizon 增大误差爆炸 | 逐步多步训练、短 horizon MPC、频繁 replan |
| 第一人称部分可观测 | 相同画面对应不同世界状态 | 16-step context；后续引入 memory/hierarchy |
| 数据来源偏差 | latent 按 contractor/recorder 聚类 | contractor/recorder 分桶验证 |
| GUI 与世界控制混杂 | 非法规划动作 | GUI mode 显式编码、macro-action 合法性 mask |
| 训练规模超预算 | 约 612M 推理模型、640×360 输入成本过高 | repeated-frame chunking、20×20 patch、bf16、Flash Attention、梯度累积 |

## 13. 实现时必须保持的原则

1. 训练数据只允许 VPT contractor demonstrations；MineRL 仅可作为未来在线评估环境，环境 trajectory 不得回流训练集。
2. 所有可学习模块全部由 MCWM 自己训练。
3. 默认同时使用 EMA target encoder 与 stop-gradient。
4. action-conditioned predictor 必须随机初始化训练，不能复用 VPT policy 表征。
5. 数据时序正确性优先于模型规模。
6. 不以低 prediction loss 代替 action sensitivity 和 collapse 检查。
7. faithful component 与 Minecraft extension 必须用配置和实验名区分。
8. Benchmark 暂缓，但离线诊断、测试和 provenance 不得暂缓。

## 14. 参考资料

- [LeWorldModel paper](https://arxiv.org/html/2603.19312v1)
- [LeWorldModel official code](https://github.com/lucas-maes/le-wm)
- [OpenAI Video Pre-Training repository](https://github.com/openai/Video-Pre-Training)
- [MineRL 1.0 repository](https://github.com/minerllabs/minerl)
- [I-JEPA paper](https://arxiv.org/abs/2301.08243)
- [参考实现 le-wm-minecraft](https://github.com/Jaslavie/le-wm-minecraft)
