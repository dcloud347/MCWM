"""为动作条件世界模型准备视频帧和与之对齐的 VPT 动作。

这个模块完成两件事：

1. :class:`WorldModelDataset` 从一段 episode 中抽取 ``T`` 帧画面，并返回
   相邻采样帧之间的 ``T - 1`` 个动作块。
2. :func:`collate_world_model_samples` 把多个样本组成 batch。因为每段时间内
   的动作 tick 数量可能不同，它会把动作补齐到相同长度，并额外返回
   ``valid_mask`` 标记哪些位置是真实动作。

动作块统一使用半开区间 ``[当前帧时间, 下一帧时间)``。例如采样帧时间是
``0 ms`` 和 ``250 ms``，对应动作块会包含时间戳大于等于 0、且小于 250 的
所有动作。这样边界上的动作只会属于下一个区间，不会被重复使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mcwm.actions.schema import CanonicalActionTick, ActionSource
from .alignment import ActionBlock, align_actions_to_frames
from .dataset import eligible_clip_start_ranges, random_clip_frame_indices_from_ranges
from .episode_store import EpisodeStore
from .manifest import DatasetManifest, EpisodeManifest
from .visual_dataset import decode_frames_at_timestamps


# 普通 int 使用数据集自身的 seed；(index, seed) 允许调用方明确控制本次采样，
# 主要用于分布式训练或测试中可复现地选择同一个 clip。
SampleIndex = Union[int, Tuple[int, int]]


@dataclass(frozen=True)
class WorldModelEpisodeRef:
    """一段可用于动作条件训练的 VPT episode 的只读索引信息。

    这里只保存采样时反复使用的数据，避免每次调用 ``__getitem__`` 都重新读取
    episode、重新做动作对齐和重新计算合法的 clip 起点。
    """

    # episode 的稳定标识，用于生成 sample_id。
    episode_id: str
    # 实际解码的视频文件位置。
    video_path: Path
    # 视频所有帧的时间戳，单位为毫秒，并且严格递增。
    frame_timestamps_ms: Tuple[int, ...]
    # 第 i 项保存原始视频帧 i 到 i+1 之间的全部动作。
    aligned_blocks: Tuple[ActionBlock, ...]
    # 可以完整采出一个 clip 的起点范围，由采样工具负责解释和随机选择。
    clip_start_ranges: Tuple[Tuple[int, int, int], ...]


def _resolve_video_path(root: Path, manifest: EpisodeManifest) -> Path:
    """把 manifest 中的视频路径解析成可用于解码的路径。

    兼容三种数据布局：绝对路径、相对于数据集根目录的路径，以及相对于
    ``episodes/<episode_id>/`` 的路径。如果文件暂时不存在，仍返回第一种
    相对路径候选，让真正的解码步骤给出具体的文件错误。
    """

    configured = Path(manifest.video_path)
    if configured.is_absolute():
        return configured
    candidates = (
        root / configured,
        root / "episodes" / manifest.episode_id / configured,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _actions_between_sampled_frames(
    aligned_blocks: Sequence[ActionBlock],
    frame_indices: Sequence[int],
) -> Tuple[Tuple[CanonicalActionTick, ...], ...]:
    """收集每两张采样帧之间的全部原始动作 tick。

    ``aligned_blocks`` 是按原始视频的相邻帧切分的。例如采样帧下标为
    ``(10, 15, 20)``，这里会分别合并原始 block ``[10:15]`` 和
    ``[15:20]``，最终得到两个变长动作块。

    为了避免把损坏或不连续的数据静默送进模型，本函数还会检查：

    * 采样帧下标必须严格递增；
    * 中间所有原始帧区间必须连续；
    * 已存在的动作 tick 必须是有效的 VPT 标签；完全没有 tick 的连续区间会
      生成一个有效 no-op，让模型学习“玩家没有输入”时的状态变化。
    """

    result = []
    for start, end in zip(frame_indices, frame_indices[1:]):
        if end <= start:
            raise ValueError("sampled frame indices must be strictly increasing")

        # aligned_blocks[i] 表示原始 frame[i] -> frame[i + 1]，因此从
        # frame[start] 走到 frame[end] 正好需要 blocks[start:end]。
        source_blocks = aligned_blocks[start:end]
        if len(source_blocks) != end - start or not all(
            block.continuous for block in source_blocks
        ):
            raise ValueError("sampled transition crosses a discontinuity")

        # 保留每个原始 tick，而不是把整个区间压缩成一个动作。模型之后可以
        # 学习在两张低帧率采样画面之间实际发生的完整操作序列。
        actions = tuple(
            action
            for block in source_blocks
            for action in block.actions
        )
        if not actions:
            # VPT 动作表示当前输入状态，因此连续区间里完全没有记录时按真实
            # no-op 处理。它必须是 valid=True；valid=False 只保留给 batch
            # padding，Action Encoder 才能区分“没有操作”和“没有数据”。
            actions = (
                CanonicalActionTick.noop(
                    source_blocks[0].start_ms,
                    ActionSource.VPT,
                    valid=True,
                ),
            )
        if not all(action.valid for action in actions):
            raise ValueError("sampled transition contains invalid action labels")
        if not all(action.source is ActionSource.VPT for action in actions):
            raise ValueError("sampled transition contains non-VPT action labels")
        result.append(actions)
    return tuple(result)


class WorldModelDataset(Dataset):
    """返回视频帧和帧间严格对齐的变长 VPT 动作块。

    默认一个样本包含 8 帧画面和 7 个动作块。第 ``i`` 个动作块描述
    ``frames[i]`` 到 ``frames[i + 1]`` 之间发生的所有动作。动作块长度不固定，
    因此单个样本暂时保留 Python tuple，组成 batch 时再统一补齐。

    Args:
        root: 数据集根目录，其中应包含 ``dataset_manifest.json``。
        split: 要读取的数据划分，例如 ``"train"`` 或 ``"val"``。
        frames_per_sample: 每个样本抽取的视频帧数，至少为 2。
        sample_fps: 抽取 clip 的目标帧率，不要求等于原视频帧率。
        seed: 整数下标采样时使用的基础随机种子。
        samples_per_video: 每个可用 episode 在一个数据集 epoch 中重复采样几次。
        max_frame_gap_ms: 两个原始视频帧仍被视为连续的最大时间间隔。

    Returns:
        ``__getitem__`` 返回一个字典，其中：

        * ``frames``: ``[T, C, H, W]`` 的视频帧张量；
        * ``frame_timestamps_ms``: ``[T]`` 的帧时间戳；
        * ``action_blocks``: 长度为 ``T - 1`` 的变长动作块 tuple；
        * ``sample_id``: 便于日志和问题定位的样本标识。
    """

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        frames_per_sample: int = 8,
        sample_fps: int = 4,
        seed: int = 0,
        samples_per_video: int = 1,
        max_frame_gap_ms: int = 250,
    ) -> None:
        # 先统一类型，既支持 Path/int，也兼容可安全转换的调用参数。
        self.root = Path(root)
        self.frames_per_sample = int(frames_per_sample)
        self.sample_fps = int(sample_fps)
        self.seed = int(seed)
        self.samples_per_video = int(samples_per_video)
        self.max_frame_gap_ms = int(max_frame_gap_ms)
        if self.frames_per_sample < 2:
            raise ValueError("frames_per_sample must be at least two")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.samples_per_video <= 0:
            raise ValueError("samples_per_video must be positive")

        # manifest 只负责列出 episode 及其 split；详细时间戳和动作存储在
        # EpisodeStore 中。世界模型当前只接受 VPT 承包商动作，避免混用动作
        # 定义或标签质量不同的数据源。
        manifest = DatasetManifest.read(self.root / "dataset_manifest.json")
        selected = tuple(item for item in manifest.episodes if item.split == split)
        if any(item.source is not ActionSource.VPT for item in selected):
            raise ValueError("world-model training only accepts VPT contractor episodes")
        store = EpisodeStore(self.root)
        references: List[WorldModelEpisodeRef] = []
        for episode_manifest in selected:
            episode = store.read_episode(episode_manifest.episode_id)

            # 第一次加载时一次性把动作分配到原始相邻视频帧之间。后续每次随机
            # 采 clip 时只需切片和合并，不必重复执行时间戳对齐。
            aligned = align_actions_to_frames(
                episode.frame_timestamps_ms,
                episode.actions,
                max_frame_gap_ms=self.max_frame_gap_ms,
            )
            start_ranges = eligible_clip_start_ranges(
                episode.frame_timestamps_ms,
                clip_frames=self.frames_per_sample,
                sample_fps=self.sample_fps,
                max_frame_gap_ms=self.max_frame_gap_ms,
            )
            # 太短或中间存在过大时间断点的 episode 可能无法提供完整样本。
            # 跳过它们比在训练过程中随机报错更容易预期。
            if not start_ranges:
                continue
            references.append(
                WorldModelEpisodeRef(
                    episode_id=episode_manifest.episode_id,
                    video_path=_resolve_video_path(self.root, episode_manifest),
                    frame_timestamps_ms=episode.frame_timestamps_ms,
                    aligned_blocks=aligned.blocks,
                    clip_start_ranges=start_ranges,
                )
            )
        if not references:
            raise ValueError(f"no {split!r} VPT episodes can provide a full world-model sample")
        self.references = tuple(references)

    def __len__(self) -> int:
        """返回一个 epoch 中可访问的样本槽位数。

        每个槽位仍会随机选择 clip；它不是预先固定好的视频片段。
        """

        return len(self.references) * self.samples_per_video

    def sample_action_clip(self, index: SampleIndex) -> Dict[str, object]:
        """随机抽取 clip 的时间戳和动作块，但不解码视频。

        ``index`` 可以是普通整数，也可以是 ``(样本下标, 随机种子)``。后一种
        形式让 sampler 能显式控制随机性；相同 episode、参数和种子会选出相同
        的帧下标。这个轻量接口供数据审计使用，确保审计和训练采用完全相同的
        clip 采样与动作聚合规则。
        """

        if isinstance(index, tuple):
            sample_index, sample_seed = index
        else:
            sample_index = int(index)
            sample_seed = self.seed + sample_index

        # samples_per_video 个连续槽位映射到同一个 episode，但可以使用不同
        # seed 从该 episode 中抽到不同 clip。
        reference = self.references[sample_index // self.samples_per_video]
        frame_indices = random_clip_frame_indices_from_ranges(
            reference.frame_timestamps_ms,
            reference.clip_start_ranges,
            clip_frames=self.frames_per_sample,
            sample_fps=self.sample_fps,
            generator=random.Random(sample_seed),
        )

        # 时间戳与动作使用同一组 frame_indices，因此 timestamps[i] 和
        # action_blocks[i] 始终保持一致。
        timestamps_ms = tuple(
            reference.frame_timestamps_ms[index] for index in frame_indices
        )
        action_blocks = _actions_between_sampled_frames(
            reference.aligned_blocks,
            frame_indices,
        )
        return {
            "frame_timestamps_ms": timestamps_ms,
            "action_blocks": action_blocks,
            "sample_id": (
                f"{reference.episode_id}:pts={timestamps_ms[0]}-{timestamps_ms[-1]}ms"
                f"@{self.sample_fps}fps"
            ),
            "video_path": reference.video_path,
        }

    def __getitem__(self, index: SampleIndex) -> Dict[str, object]:
        """随机抽取一个合法 clip，并解码对应画面与动作。"""

        metadata = self.sample_action_clip(index)
        timestamps_ms = metadata["frame_timestamps_ms"]
        frames = decode_frames_at_timestamps(metadata["video_path"], timestamps_ms)
        return {
            "frames": frames,
            "frame_timestamps_ms": torch.tensor(timestamps_ms, dtype=torch.int64),
            "action_blocks": metadata["action_blocks"],
            "sample_id": metadata["sample_id"],
        }


def collate_world_model_samples(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """把多个变长样本整理成可直接送入模型的规则张量。

    设 ``B`` 为 batch 大小、``T`` 为每个样本的帧数、``A = T - 1`` 为帧间
    transition 数、``K`` 为这个 batch 中最长动作块的 tick 数。输出动作张量
    的前三维统一为 ``[B, A, K]``；不足 ``K`` 的位置以 0 补齐，并由
    ``valid_mask=False`` 标出。

    注意，补位不能当成“玩家没有操作”的真实 no-op。真实 no-op 的动作值虽然
    也全为 0，但其 ``valid_mask=True``，仍应参与训练。

    主要输出形状如下：

    * ``frames``: ``[B, T, C, H, W]``；
    * ``frame_timestamps_ms``: ``[B, T]``；
    * ``movement``、``interaction``: ``[B, A, K, 7]``；
    * ``camera``、``cursor``: ``[B, A, K, 2]``；
    * 其余逐 tick 字段和 mask: ``[B, A, K]``。
    """

    if not samples:
        raise ValueError("samples must not be empty")

    # 视频和帧时间戳本来就是定长的，可以直接在最前面增加 batch 维。
    # frames: [B, T, C, H, W]；frame_timestamps: [B, T]。
    frames = torch.stack([sample["frames"] for sample in samples])
    frame_timestamps = torch.stack(
        [sample["frame_timestamps_ms"] for sample in samples]
    )
    transitions = frames.shape[1] - 1
    blocks_by_sample = [sample["action_blocks"] for sample in samples]

    # T 帧之间恰好有 T-1 个 transition，动作块数量必须一一对应。
    if any(len(blocks) != transitions for blocks in blocks_by_sample):
        raise ValueError("each sample must contain one action block per frame transition")
    max_ticks = max(
        len(block)
        for blocks in blocks_by_sample
        for block in blocks
    )
    if max_ticks <= 0:
        raise ValueError("action blocks must contain labeled ticks")

    batch = len(samples)

    # 所有动作张量先以 padding 默认值初始化。形状中的 7 分别对应 schema.py
    # 定义的 7 个移动键和 7 个交互键；camera/cursor 的 2 表示二维坐标。
    movement = torch.zeros(batch, transitions, max_ticks, 7, dtype=torch.bool)
    interaction = torch.zeros(batch, transitions, max_ticks, 7, dtype=torch.bool)
    hotbar = torch.zeros(batch, transitions, max_ticks, dtype=torch.long)
    camera = torch.zeros(batch, transitions, max_ticks, 2, dtype=torch.float32)
    cursor = torch.zeros(batch, transitions, max_ticks, 2, dtype=torch.float32)
    cursor_present = torch.zeros(batch, transitions, max_ticks, dtype=torch.bool)
    gui_open = torch.zeros(batch, transitions, max_ticks, dtype=torch.bool)
    valid_mask = torch.zeros(batch, transitions, max_ticks, dtype=torch.bool)
    action_timestamps = torch.zeros(batch, transitions, max_ticks, dtype=torch.int64)
    label_confidence = torch.zeros(batch, transitions, max_ticks, dtype=torch.float32)

    # 只覆盖真实 tick 所在的位置，末尾未写入的位置自然保留为 padding。
    for batch_index, blocks in enumerate(blocks_by_sample):
        for transition_index, block in enumerate(blocks):
            for tick_index, action in enumerate(block):
                movement[batch_index, transition_index, tick_index] = torch.tensor(
                    action.movement, dtype=torch.bool
                )
                interaction[batch_index, transition_index, tick_index] = torch.tensor(
                    action.interaction, dtype=torch.bool
                )
                hotbar[batch_index, transition_index, tick_index] = action.hotbar
                camera[batch_index, transition_index, tick_index] = torch.tensor(
                    action.camera, dtype=torch.float32
                )
                if action.cursor is not None:
                    cursor[batch_index, transition_index, tick_index] = torch.tensor(
                        action.cursor, dtype=torch.float32
                    )
                    cursor_present[batch_index, transition_index, tick_index] = True
                gui_open[batch_index, transition_index, tick_index] = action.gui_open
                valid_mask[batch_index, transition_index, tick_index] = action.valid
                action_timestamps[batch_index, transition_index, tick_index] = (
                    action.timestamp_ms
                )
                label_confidence[batch_index, transition_index, tick_index] = (
                    action.label_confidence
                )

    # 返回值中不再保留嵌套的 CanonicalActionTick 对象，全部字段都拆成规则
    # 张量，便于移到 GPU 和进行向量化计算。sample_id 只用于日志，保留为列表。
    return {
        "frames": frames,
        "frame_timestamps_ms": frame_timestamps,
        "movement": movement,
        "interaction": interaction,
        "hotbar": hotbar,
        "camera": camera,
        "cursor": cursor,
        "cursor_present": cursor_present,
        "gui_open": gui_open,
        "valid_mask": valid_mask,
        "action_timestamps_ms": action_timestamps,
        "label_confidence": label_confidence,
        "sample_id": [sample["sample_id"] for sample in samples],
    }
