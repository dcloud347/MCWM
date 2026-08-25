"""V-JEPA 2 风格的 multi-block 时空 mask。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskGeneratorConfig:
    """一组取并集的 3D block mask 参数。"""

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
    """V-JEPA 2 16-frame pretraining config 中的两组默认 mask。"""

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
    """按顺序保存独立计算 loss 的 multi-block mask 组。"""

    generators: Tuple[MaskGeneratorConfig, ...] = field(
        default_factory=_default_generators
    )

    def __post_init__(self) -> None:
        if not self.generators:
            raise ValueError("at least one mask generator is required")


class SpatiotemporalMaskSampler:
    """生成 V-JEPA 2 multi-block mask。

    返回值形状为 ``[G, B, T, H*W]``。``G`` 是配置中的 mask 组数；
    每组先为整个 batch 抽一个 block 尺寸，再为每个样本独立抽位置，并把该组
    的所有 block 取并集。``True`` 表示 online encoder 不可见且需要预测。
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
        aspect_ratio = self._uniform(config.aspect_ratio, generator)

        duration = max(1, int(clip_frames * temporal_scale))
        spatial_area = max(1, int(self.rows * self.columns * spatial_scale))
        height = max(1, int(round(math.sqrt(spatial_area * aspect_ratio))))
        width = max(1, int(round(math.sqrt(spatial_area / aspect_ratio))))
        return (
            min(duration, clip_frames),
            min(height, self.rows),
            min(width, self.columns),
        )

    def _sample_group_mask(
        self,
        clip_frames: int,
        block_size: Tuple[int, int, int],
        num_blocks: int,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        duration, height, width = block_size
        # 和官方实现一样，若 block 的并集吃掉了全部 context，就重新抽位置。
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
