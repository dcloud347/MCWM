"""M1 video JEPA 使用的结构化空间—时间 mask。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskConfig:
    """mask 比例及两类结构化 mask 的数量。"""

    ratio: float = 0.75
    spatial_blocks: int = 4
    temporal_tubes: int = 4
    min_block_fraction: float = 0.08
    max_block_fraction: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError("mask ratio must be between 0 and 1")
        if min(self.spatial_blocks, self.temporal_tubes) < 0:
            raise ValueError("block counts cannot be negative")
        if not 0.0 < self.min_block_fraction <= self.max_block_fraction <= 1.0:
            raise ValueError("invalid block fraction range")


class SpatiotemporalMaskSampler:
    """混合大块 spatial mask、连续 temporal tube 和随机 patch。

    返回值形状是 ``[B, T, H*W]``，其中 ``True`` 始终表示“隐藏且需要预测”。
    mask 相比视频很小，所以统一在 CPU 生成；这样同一个 seed 在 CPU 和 CUDA
    训练机上都能得到相同 mask。
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

    def _rectangle(self, generator: Optional[torch.Generator]) -> Tuple[int, int, int, int]:
        total = self.rows * self.columns
        min_area = max(1, round(total * self.config.min_block_fraction))
        max_area = max(min_area, round(total * self.config.max_block_fraction))
        area = self._randint(min_area, max_area + 1, generator)
        aspect_choices = (0.5, 0.75, 1.0, 1.5, 2.0)
        aspect = aspect_choices[self._randint(0, len(aspect_choices), generator)]
        height = max(1, min(self.rows, round((area / aspect) ** 0.5)))
        width = max(1, min(self.columns, round(area / height)))
        top = self._randint(0, self.rows - height + 1, generator)
        left = self._randint(0, self.columns - width + 1, generator)
        return top, left, height, width

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
        mask = torch.zeros(
            batch_size,
            clip_frames,
            self.rows,
            self.columns,
            dtype=torch.bool,
        )
        target_count = max(
            1,
            min(clip_frames * self.rows * self.columns - 1, round(mask[0].numel() * self.config.ratio)),
        )
        for batch_index in range(batch_size):
            # spatial block 贯穿整个 clip，避免模型只复制相邻帧纹理来解题。
            for _ in range(self.config.spatial_blocks):
                top, left, height, width = self._rectangle(generator)
                mask[batch_index, :, top : top + height, left : left + width] = True

            # temporal tube 只遮住一段连续时间内的局部空间区域。
            for _ in range(self.config.temporal_tubes):
                top, left, height, width = self._rectangle(generator)
                start = self._randint(0, clip_frames, generator)
                length = self._randint(1, clip_frames - start + 1, generator)
                mask[
                    batch_index,
                    start : start + length,
                    top : top + height,
                    left : left + width,
                ] = True

            # 最后补齐或删减少量随机位置，保证每个样本的 mask ratio 完全一致。
            flat = mask[batch_index].flatten()
            masked = torch.nonzero(flat, as_tuple=False).flatten()
            if masked.numel() > target_count:
                order = torch.randperm(masked.numel(), generator=generator)
                flat[masked[order[target_count:]]] = False
            elif masked.numel() < target_count:
                visible = torch.nonzero(~flat, as_tuple=False).flatten()
                order = torch.randperm(visible.numel(), generator=generator)
                flat[visible[order[: target_count - masked.numel()]]] = True

        result = mask.flatten(2)
        return result.to(device=device) if device is not None else result
