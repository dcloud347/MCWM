"""在视频 patch 网格上生成 V-JEPA 2 风格的块状 mask。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskGeneratorConfig:
    """一组块状 mask 的大小、形状和数量。"""

    spatial_scale: Tuple[float, float]
    temporal_scale: Tuple[float, float]
    aspect_ratio: Tuple[float, float]
    num_blocks: int

    def __post_init__(self) -> None:
        self._validate_range("spatial_scale", self.spatial_scale, upper=1.0)
        self._validate_range("temporal_scale", self.temporal_scale, upper=1.0)
        self._validate_range("aspect_ratio", self.aspect_ratio)
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

    @staticmethod
    def _validate_range(
        name: str,
        values: Tuple[float, float],
        *,
        upper: Optional[float] = None,
    ) -> None:
        if len(values) != 2:
            raise ValueError(f"{name} must contain two values")
        low, high = values
        if low <= 0.0 or high < low or (upper is not None and high > upper):
            raise ValueError(f"invalid {name} range")


def _default_generators() -> Tuple[MaskGeneratorConfig, ...]:
    """返回 V-JEPA 2 在 16 帧预训练中使用的两组默认 mask。"""

    common = {
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
    }
    return (
        MaskGeneratorConfig(
            spatial_scale=(0.15, 0.15),
            num_blocks=8,
            **common,
        ),
        MaskGeneratorConfig(
            spatial_scale=(0.70, 0.70),
            num_blocks=2,
            **common,
        ),
    )


@dataclass(frozen=True)
class MaskConfig:
    """保存多组 mask；每组单独预测，最后平均 loss。"""

    generators: Tuple[MaskGeneratorConfig, ...] = field(
        default_factory=_default_generators
    )

    def __post_init__(self) -> None:
        if not self.generators:
            raise ValueError("at least one mask generator is required")


class SpatiotemporalMaskSampler:
    """生成多组时空块状 mask。

    返回形状是 ``[组数, batch, 时间, 每帧 patch 数]``。同组样本使用相同的
    block 大小，但位置各自随机。多个 block 重叠时取并集。``True`` 表示该
    patch 不给 online encoder 看，而是交给 predictor 预测。
    """

    def __init__(self, grid_size: Tuple[int, int], config: MaskConfig) -> None:
        self.rows, self.columns = grid_size
        self.config = config
        if min(self.rows, self.columns) <= 0:
            raise ValueError("grid dimensions must be positive")

    @staticmethod
    def _randint(low: int, high: int, generator: Optional[torch.Generator]) -> int:
        if high <= low:
            return low
        return int(torch.randint(low, high, (), generator=generator).item())

    @staticmethod
    def _uniform(
        values: Tuple[float, float], generator: Optional[torch.Generator]
    ) -> float:
        low, high = values
        if high == low:
            return low
        draw = float(torch.rand((), generator=generator).item())
        return low + draw * (high - low)

    def _sample_block_size(
        self,
        clip_frames: int,
        config: MaskGeneratorConfig,
        generator: Optional[torch.Generator],
    ) -> Tuple[int, int, int]:
        temporal_scale = self._uniform(config.temporal_scale, generator)
        spatial_scale = self._uniform(config.spatial_scale, generator)

        duration = max(1, int(clip_frames * temporal_scale))
        target_area = max(1.0, self.rows * self.columns * spatial_scale)
        height, width = self._sample_spatial_block_size(
            target_area,
            config.aspect_ratio,
            generator,
        )
        return (
            min(duration, clip_frames),
            height,
            width,
        )

    def _sample_spatial_block_size(
        self,
        target_area: float,
        aspect_ratio: Tuple[float, float],
        generator: Optional[torch.Generator],
    ) -> Tuple[int, int]:
        """在矩形 patch 网格上选择最接近目标面积的整数宽高。

        ``aspect_ratio`` 是 block 高度除以宽度。代码会检查所有放得下的整数
        尺寸，选择面积和宽高比都最接近目标的一个，避免宽屏网格把大 block
        直接裁小。
        """

        ratio_low, ratio_high = aspect_ratio
        feasible_low = max(ratio_low, target_area / (self.columns**2))
        feasible_high = min(ratio_high, (self.rows**2) / target_area)
        if feasible_low <= feasible_high:
            sampled_ratio = self._uniform(
                (feasible_low, feasible_high),
                generator,
            )
        else:
            # 小网格可能没有完全满足目标面积的矩形，此时选择最接近的尺寸。
            sampled_ratio = self._uniform(aspect_ratio, generator)

        candidates = []
        for height in range(1, self.rows + 1):
            for width in range(1, self.columns + 1):
                actual_ratio = height / width
                if ratio_low <= actual_ratio <= ratio_high:
                    actual_area = height * width
                    score = (
                        math.log(actual_area / target_area) ** 2
                        + math.log(actual_ratio / sampled_ratio) ** 2
                    )
                    candidates.append(
                        (
                            score,
                            abs(actual_area - target_area),
                            height,
                            width,
                        )
                    )

        if not candidates:
            raise ValueError(
                "grid has no integer block size satisfying aspect_ratio"
            )
        _, _, height, width = min(candidates)
        return height, width

    def _sample_group_mask(
        self,
        clip_frames: int,
        block_size: Tuple[int, int, int],
        num_blocks: int,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        duration, height, width = block_size
        # 如果 mask 遮住了全部 patch，就重新随机位置，确保 encoder 仍有输入。
        for _ in range(100):
            mask = torch.zeros(
                clip_frames,
                self.rows,
                self.columns,
                dtype=torch.bool,
            )
            for _ in range(num_blocks):
                start = self._randint(0, clip_frames - duration + 1, generator)
                top = self._randint(0, self.rows - height + 1, generator)
                left = self._randint(0, self.columns - width + 1, generator)
                mask[
                    start : start + duration,
                    top : top + height,
                    left : left + width,
                ] = True
            if mask.any() and not mask.all():
                return mask
        raise RuntimeError("could not sample a mask with non-empty context")

    def sample(
        self,
        batch_size: int,
        clip_frames: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """为一个 batch 生成所有 mask 组。"""

        if min(batch_size, clip_frames) <= 0:
            raise ValueError("batch_size and clip_frames must be positive")

        groups = []
        for group_config in self.config.generators:
            block_size = self._sample_block_size(
                clip_frames,
                group_config,
                generator,
            )
            batch_masks = [
                self._sample_group_mask(
                    clip_frames,
                    block_size,
                    group_config.num_blocks,
                    generator,
                )
                for _ in range(batch_size)
            ]
            groups.append(torch.stack(batch_masks).flatten(2))

        result = torch.stack(groups)
        return result.to(device=device) if device is not None else result
