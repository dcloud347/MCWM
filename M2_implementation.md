# M2 V-JEPA 2-AC 实施方案

## 0. 文档定位

本文档记录 M1 Visual Encoder 训练完成后，M2 Action-Conditioned World Model 的实施方案。

M2 以官方 V-JEPA 2-AC 为主线：

1. 每帧画面复制为一个 2-frame tubelet，独立编码为 frame latent。
2. M1 EMA visual encoder 在 M2 中冻结，不反向传播，不维护新的 online/EMA 副本。
3. 保留每帧的全部 spatial tokens，不做 mean pooling。
4. 使用 frame/block-causal Transformer，通过 action token 条件化下一帧 latent prediction。
5. 同时训练 teacher-forced prediction 和 autoregressive rollout prediction。
6. 默认损失对齐官方 V-JEPA 2-AC，使用 normalized latent L1 loss。

Minecraft 的必要适配：

- 用 Minecraft micro-action encoder 替代官方的 7 维 robot-action linear projection。
- 不输入 robot proprioceptive state 和 camera extrinsics，MCWM 只允许 RGB 和键鼠动作作为 world-model 输入。
- 保留 Minecraft GUI、hotbar、camera 和变长原始 action ticks。

## 1. 时序契约

**实现状态：已完成。** 实现见 `src/mcwm/data/world_model_dataset.py`，回归测试见 `tests/data/test_world_model_dataset.py`。

### 1.1 一帧对应一个 latent state

从连续 trajectory 中按 PTS 以 `4 FPS` 采样 `T=8` 帧，与官方 released V-JEPA 2-AC 配置一致：

```text
frames: o_0, o_1, o_2, ... o_7
```

每帧独立复制成一个 tubelet：

```text
o_t -> [o_t, o_t] -> frozen M1 EMA encoder -> z_t
```

M1 encoder 的 `tubelet_size=2`，因此两个相同帧产生一个 temporal slice。对 MCWM 的 `640x360` 输入和 `patch_size=20`：

```text
spatial grid: 18 x 32
tokens/frame: 576
latent dim:   1024
z_t shape:    [576, 1024]
```

不使用重叠视频窗口定义：

```text
z_t     = Encoder(o_(t-15) ... o_t)
z_(t+1) = Encoder(o_(t-14) ... o_(t+1))
```

该方案计算昂贵、相邻 state 高度重叠，也不是官方 V-JEPA 2-AC 的时序定义。

### 1.2 动作对齐

每个 action block 严格包含从当前帧到下一帧之间的全部原始动作：

```text
A_t = actions in [PTS(o_t), PTS(o_(t+1)))
```

一个 8-frame sample：

```text
o_0 --A_0--> o_1 --A_1--> o_2 ... --A_6--> o_7
 z_0           z_1           z_2              z_7
```

训练契约：

```text
input latents: z_0 ... z_6
actions:       A_0 ... A_6
targets:       z_1 ... z_7
```

action block 中的 tick 数 `K` 由真实时间戳决定，不固定为 4。batch collate 时补齐到当前 batch 的 `Kmax`：

```text
movement:       bool[B, 7, Kmax, 7]
interaction:    bool[B, 7, Kmax, 7]
hotbar:         int64[B, 7, Kmax]
camera:         float32[B, 7, Kmax, 2]
cursor:         float32[B, 7, Kmax, 2]
cursor_present: bool[B, 7, Kmax]
gui_open:       bool[B, 7, Kmax]
valid_mask:     bool[B, 7, Kmax]
```

`valid_mask=false` 只表示 padding，不能表示真实 no-op。缺失标签的 interval 不能被静默转换为 no-op。

## 2. Frozen M1 EMA Visual Encoder

**实现状态：代码与 tiny-checkpoint 验证已完成；真实 M1 checkpoint probe 待正式 checkpoint 原子落盘。**

实现见 `src/mcwm/models/frozen_visual_encoder.py` 和 `load_frozen_m1_encoder()`；回归测试见 `tests/models/test_frozen_visual_encoder.py` 与 `tests/models/test_visual_encoder.py`。

M2 只加载 M1 checkpoint 中的 `target_encoder.*`：

```text
M1 target_encoder -> M2 frozen visual encoder
```

不使用 M1 online encoder，不在 M2 维护 online/target 双 encoder，不进行 EMA update。冻结 encoder 始终满足：

```text
requires_grad = false
training mode = eval
forward under torch.no_grad()
not present in optimizer parameter groups
```

默认从完整 M1 checkpoint 严格加载，验证：

- checkpoint 来自 MCWM，且 `external_pretrained=false`。
- visual encoder 参数名和 shape 完全匹配。
- M1 config 使用 `clip_frames=16`、`tubelet_size=2`。
- manifest hash 和 parent checkpoint provenance 完整。
- 使用 `strict=true` 加载。

### 2.1 可变时间长度

M1 Visual Encoder 的 `clip_frames=16` 表示最大运行时帧数。当 `tubelet_size=2` 时支持：

```text
T = 2, 4, 6, 8, 10, 12, 14, 16
```

M2 对每个单帧使用 `T=2` 的重复输入。该运行时改变不修改权重、state-dict key 或 checkpoint 参数 shape。

### 2.2 Repeated-frame latent probe

正式 M2 训练前必须验证真实 M1 checkpoint：

- `T=16` 输出与可变长度改造前保持一致。
- 所有合法 T 的输出都是有限值。
- repeated-frame latent 的 std、effective rank 和 token diversity 没有异常。
- 相邻 Minecraft frame 的 latent 对 camera、GUI 和 scene change 仍敏感。
- fixed validation sample 的 repeated-frame representation 可稳定复现。

如果 probe 不通过，encoder adaptation 必须作为独立实验设计和验收，不静默改变主线。

执行脚本：

```bash
PYTHONPATH=src python3 scripts/probe_m2_encoder.py \
  artifacts/checkpoints/checkpoint-00021600.pt \
  /path/to/canonical-dataset \
  --device cuda \
  --precision bf16 \
  --samples 8 \
  --frame-chunk-size 1 \
  --output artifacts/m2_encoder_probe.json
```

脚本只读 checkpoint 和数据，不更新权重。默认还会依次验证底层 encoder 的
`T=2/4/6/8/10/12/14/16`；显存不足时可先加 `--skip-variable-t`，但正式验收仍需
在训练机器上补跑完整检查。

## 3. M2 Dataset

**实现状态：已完成。**

新增 `src/mcwm/data/world_model_dataset.py`：

- `WorldModelDataset`：按 PTS 选取 8 帧连续画面和 7 个 action blocks。
- `collate_world_model_samples`：对变长 action ticks 做 batch padding。
- 复用 `align_actions_to_frames`，严格合并两个采样画面 PTS 之间的全部原始 action blocks。

输出：

```text
frames:        uint8[B, 8, 3, 360, 640]
action fields: tensors[B, 7, Kmax, ...]
valid_mask:    bool[B, 7, Kmax]
sample_id:     episode and PTS range
```

测试覆盖 frame/action off-by-one、左闭右开边界、discontinuity、变长 `K`、valid no-op 与 padding，以及基于真实 PTS 的 4 FPS 采样。

## 4. Minecraft Action Encoder

**实现状态：已完成。**

新增 `src/mcwm/models/action_encoder.py`。官方 V-JEPA 2-AC 用 linear layer 编码 7 维 robot action；Minecraft action block 是变长、混合类型序列，因此替换为：

1. 14 个 binary component 独立 embedding。
2. hotbar 10 类 embedding。
3. camera clipping、mu-law normalization 和 MLP。
4. cursor MLP，由 `gui_open` 和 `cursor_present` 控制。
5. component fusion 产生 256 维 tick token。
6. 两层 micro-action Transformer 按时间顺序汇总同一 interval 的 ticks。
7. masked pooling 和 projection 产生 1024 维 macro-action token。

```text
A_t: variable ticks -> ActionEncoder -> a_t[1024]
```

测试覆盖同时按键、顺序敏感性、padding invariance、valid no-op、camera 数值稳定性和有限梯度。

## 5. Frame/Block-Causal Predictor

新增 `src/mcwm/models/ac_predictor.py`。

| 参数 | 默认值 |
|---|---:|
| encoder latent dim | 1024 |
| predictor dim | 1024 |
| depth | 24 |
| heads | 16 |
| MLP dim | 4096 |
| frames per sample | 8 |
| spatial tokens per frame | 576 |
| action tokens per transition | 1 |
| attention | frame/block causal |
| positional encoding | 3D RoPE |

先将 visual tokens 和 action embedding 投影到 predictor dim，再按 frame block 排列：

```text
block_t = [action_token_t, 576 visual_tokens_t]
```

Minecraft 不输入官方 robot proprioceptive state token 和 extrinsics token。Block-causal mask 的语义：

- 同一 block 内的 action token 和 spatial tokens 可以互相 attention。
- 当前 block 可以看到全部过去 blocks。
- 任何 token 都不能看到未来 block。

预测器只输出 visual tokens，不预测 action token：

```text
input:  latents[B, 7, 576, 1024] + actions[B, 7, 1024]
output: predicted next latents[B, 7, 576, 1024]
target: next latents[B, 7, 576, 1024]
```

提供两个独立接口：

```python
predict_teacher_forced(latents, actions)
rollout(initial_latent, actions)
```

测试覆盖 causal leakage、block 内通信、输出 token 过滤、teacher-forced 语义，以及 rollout 确实反馈 predicted latent。

## 6. Teacher-Forced 与 Autoregressive Loss

Teacher-forced path 为每个时间位置提供真实 encoder latent：

```text
z_0 + A_0 -> zhat_1
z_1 + A_1 -> zhat_2
...
z_6 + A_6 -> zhat_7
```

Autoregressive path 从真实初始 latent 开始，反馈预测 latent：

```text
z_0    + A_0 -> zhat_1
zhat_1 + A_1 -> zhat_2
zhat_2 + A_2 -> zhat_3
```

首个正式配置对齐官方 released config，使用 `auto_steps=2`。预测和 target 按最后一维做 LayerNorm，默认使用 L1：

```text
L_tf = mean(abs(normalize(zhat_tf) - normalize(z_target_tf)))
L_ar = mean(abs(normalize(zhat_ar) - normalize(z_target_ar)))

L_total = L_tf + L_ar
```

M2 默认不加 SIGReg、IDM 或 pixel reconstruction loss。Frozen M1 target space 不会在 M2 中随 predictor 训练而 collapse。任何额外 loss 都必须作为独立 MCWM extension 配置和 run group。

## 7. World Model 组合层

新增 `src/mcwm/models/world_model.py`，组合 frozen encoder、Minecraft action encoder、AC predictor、frame normalization、repeated-frame tubelet、teacher-forced path 和 rollout path。

```text
frames[B, 8, 3, 360, 640]
  -> flatten to [B*8, 1, 3, 360, 640]
  -> duplicate to [B*8, 2, 3, 360, 640]
  -> frozen encoder under no_grad
  -> reshape z to [B, 8, 576, 1024]

action blocks[B, 7, Kmax, ...]
  -> action encoder
  -> action tokens[B, 7, 1024]

z[:, :-1] + action tokens
  -> block-causal predictor
  -> teacher-forced and autoregressive predictions
  -> normalized latent L1 losses
```

`B*8` 个 repeated-frame tubelets 允许按 `encoder_frame_chunk_size` 分块编码。Encoder 冻结且在 `no_grad` 下运行，不保留 activation graph。

## 8. 训练与 Checkpoint

新增：

```text
configs/train_world_model_tiny.yaml
configs/train_world_model.yaml
scripts/train_world_model.py
src/mcwm/training/train_world_model.py
```

Optimizer 只包含 Minecraft action encoder 和 AC predictor。M2 checkpoint 保存：

- frozen visual encoder state 或可验证的 M1 parent reference
- action encoder 和 AC predictor
- optimizer、scheduler、scaler 和 RNG state
- sampler epoch 和 epoch 内位置
- optimizer step
- M1 parent checkpoint ID/path/hash
- resolved M2 config、data manifest hash 和 W&B run ID

恢复时必须验证 M1 parent checkpoint 和 manifest，禁止将另一个 visual encoder 静默代入原 run。

## 9. B0 Smoke-Test Gate

1. M1 EMA encoder 以 `strict=true` 加载。
2. `T=16` 原有输出保持一致。
3. repeated-frame `T=2` latent probe 通过。
4. frozen encoder 没有梯度，也不在 optimizer 中。
5. dataset 输出 8 帧和 7 个严格对齐的 action blocks。
6. teacher-forced forward/loss/backward 跑通。
7. 2-step autoregressive rollout/loss/backward 跑通。
8. action encoder 和 predictor 可在固定 batch 上过拟合。
9. checkpoint resume 与 uninterrupted run 的下一步 loss 在容差内一致。
10. 真实动作误差低于 batch-shuffled 和 no-op 动作。

Action sensitivity 报告：

```text
gap_shuffled = error_shuffled - error_real
gap_noop     = error_noop - error_real
ratio        = error_real / error_baseline
```

正式 gate 在 validation set 上使用置信区间或 permutation test，不只依赖肉眼观察。

## 10. 诊断与验收

每次 validation 记录：

- teacher-forced latent L1/cosine distance
- 1/2-step autoregressive latent L1/cosine distance
- true/shuffled/no-op action error
- camera reverse 和 attack/use swap 对预测的影响
- 按 movement、interaction、camera、hotbar、GUI 分桶的误差
- frozen target latent 的 std、effective rank 和 pairwise cosine
- predicted latent 与 target latent 的 norm/rank gap
- 吞吐、step time 和峰值显存

固定 validation samples 记录 frame/action overlay、true/shuffled/no-op comparison、teacher-forced/rollout error curve 和 spatial-token error heatmap。

## 11. 预计文件变更

```text
configs/
├── train_world_model_tiny.yaml
└── train_world_model.yaml

scripts/
└── train_world_model.py

src/mcwm/data/
└── world_model_dataset.py

src/mcwm/models/
├── action_encoder.py
├── ac_predictor.py
└── world_model.py

src/mcwm/training/
└── train_world_model.py

tests/data/
└── test_world_model_dataset.py

tests/models/
├── test_action_encoder.py
├── test_ac_predictor.py
└── test_world_model.py

tests/training/
├── test_world_model_checkpoint.py
├── test_world_model_overfit.py
└── test_world_model_resume.py
```

## 12. 实施顺序

1. M1 checkpoint repeated-frame latent probe。
2. 8-frame/7-action-block dataset 及时序测试。
3. Minecraft action encoder 及单元测试。
4. Action-token block-causal mask 和 AC predictor。
5. Teacher-forced path 和 normalized latent L1 loss。
6. Autoregressive rollout path 和 `auto_steps=2`。
7. WorldModel 组合层和 frozen-encoder chunked encoding。
8. Checkpoint、resume、W&B 和本地 JSONL logging。
9. Tiny synthetic B0 和 canonical fixture B0。
10. 真实 M1 checkpoint 固定 batch overfit。
11. Action sensitivity gate。
12. 正式 M2 训练。

## 13. 完成标准

- 每帧通过 repeated-frame tubelet 独立编码，不泄漏未来画面。
- M1 EMA visual encoder 冻结，没有梯度或 EMA update。
- 每帧保留 `576x1024` spatial latent tokens，不做 mean pooling。
- Minecraft action encoder 和 AC predictor 从随机初始化训练。
- Block-causal mask 不泄漏未来 frame/action。
- Teacher-forced 和 autoregressive path 都通过语义、梯度和 resume 测试。
- 固定 batch 可过拟合。
- Validation 上真实动作的预测误差显著低于 shuffled/no-op。
- Checkpoint provenance 可追溯到 M1 parent checkpoint 和数据 manifest。

## 14. 与当前 `design.md` 的差异

本方案已确定使用官方 V-JEPA 2-AC 主线，因此与当前 `design.md` 中的以下 M2 设计不同：

- 不使用滚动 16-frame window 作为单个 state。
- 不将 spatial tokens mean-pool 为单个 1024 维 state。
- 不在 M2 微调 visual encoder。
- 不在 M2 维护 online/EMA visual encoder。
- 不使用 AdaLN-Zero 作为默认 action conditioning，改为 action-token interleaving。
- 默认损失不使用 SIGReg，改为 normalized teacher-forced L1 + autoregressive L1。

实现 M2 前应同步更新 `design.md` 的模型、损失、参数预算、checkpoint 和测试口径，避免两份文档长期矛盾。

## 15. 官方对照资料

- [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2 official repository](https://github.com/facebookresearch/vjepa2)
- [Official V-JEPA 2-AC training loop](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_droid/train.py)
- [Official action-conditioned predictor](https://github.com/facebookresearch/vjepa2/blob/main/src/models/ac_predictor.py)
- [Official released V-JEPA 2-AC config](https://github.com/facebookresearch/vjepa2/blob/main/configs/train/vitg16/droid-256px-8f.yaml)
