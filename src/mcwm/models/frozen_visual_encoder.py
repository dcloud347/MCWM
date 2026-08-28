"""使用冻结的 M1 视觉编码器，为世界模型逐帧生成 latent token。

M1 的 :class:`VisualEncoder` 是一个 Video ViT，输入必须包含完整的时间
``tubelet``。但 M2 世界模型需要每一帧各自对应一组视觉 token，不能让编码器
提前混合相邻帧的信息。因此这里会把每张画面在时间维复制
``tubelet_size`` 次，组成内容完全相同的静态 tubelet，再单独送入编码器。

形状变化如下，其中 ``S`` 是每帧的空间 patch 数，``D`` 是特征维度：

``[B, T, 3, H, W] -> [B*T, tubelet_size, 3, H, W] -> [B, T, S, D]``

编码器参数全程冻结，并保持 ``eval`` 模式。通常由 checkpoint 加载工具把 M1
的 EMA target encoder 权重装入本类，M2 训练只使用输出的 latent，不更新 M1。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .visual_encoder import VisualEncoder, VisualEncoderConfig


class FrozenVisualEncoder(nn.Module):
    """冻结 M1 encoder，并把视频中的每一帧独立编码为空间 token。

    Args:
        config: M1 视觉编码器的输入尺寸和网络结构配置。
        pixel_mean: RGB 三个通道的归一化均值。默认使用 ImageNet 均值。
        pixel_std: RGB 三个通道的归一化标准差。默认使用 ImageNet 标准差。

    输入的浮点帧应已经位于 ``[0, 1]``；``uint8`` 帧会自动从 ``[0, 255]``
    转换到 ``[0, 1]``。本类不会检查浮点帧的数值范围。
    """

    def __init__(
        self,
        config: VisualEncoderConfig,
        *,
        pixel_mean: tuple = (0.485, 0.456, 0.406),
        pixel_std: tuple = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()

        # 创建与 M1 配置完全相同的 Video ViT。权重通常会在外部从 M1 的
        # EMA target encoder checkpoint 严格加载。
        self.encoder = VisualEncoder(config)

        # requires_grad_(False) 阻止为参数计算和保存梯度；eval() 固定 dropout 等
        # 具有训练/推理差异的层。两者含义不同，因此都需要设置。
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # 保存为 [1, 1, 3, 1, 1]，可以直接广播到 [B, T, 3, H, W]。
        # buffer 会跟随模型迁移到 CPU/GPU；persistent=False 表示它们是固定常量，
        # 无需写入 state_dict 或 checkpoint。
        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(pixel_std).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @property
    def config(self) -> VisualEncoderConfig:
        """返回内部视觉编码器的配置，避免调用方直接依赖内部层级。"""

        return self.encoder.config

    def train(self, mode: bool = True) -> "FrozenVisualEncoder":
        """切换包装器模式，但让内部视觉编码器始终保持 ``eval``。

        PyTorch 在外层模型调用 ``train()`` 时会递归地把所有子模块切到训练
        模式。如果不覆盖此方法，冻结的 encoder 也会被意外切回训练模式。
        这里仍让包装器的 ``training`` 标志遵循 ``mode``，随后立即把 encoder
        恢复为 ``eval``。
        """

        super().train(mode)
        self.encoder.eval()
        return self

    def normalize_frames(self, frames: Tensor) -> Tensor:
        """把 RGB 视频帧转换成 M1 编码器使用的标准化浮点张量。

        Args:
            frames: 形状通常为 ``[B, T, 3, H, W]`` 的视频帧。允许 ``uint8``
                或浮点类型；其他整数类型会被拒绝。

        Returns:
            与输入设备相同的浮点张量，形状不变，每个 RGB 通道分别执行
            ``(像素值 - mean) / std``。
        """

        if frames.dtype == torch.uint8:
            # float() 会创建新张量，因此这里的 div_ 不会修改调用方传入的
            # uint8 原始帧。
            frames = frames.float().div_(255.0)
        elif not frames.is_floating_point():
            raise TypeError("frames must be uint8 or floating point")

        # 显式匹配输入的 device 和 dtype，兼容 CPU、GPU 以及 autocast 场景。
        mean = self.pixel_mean.to(device=frames.device, dtype=frames.dtype)
        std = self.pixel_std.to(device=frames.device, dtype=frames.dtype)
        return (frames - mean) / std

    @torch.no_grad()
    def forward(self, frames: Tensor, *, frame_chunk_size: Optional[int] = None) -> Tensor:
        """把一批视频帧独立编码成逐帧空间 latent token。

        Args:
            frames: ``[B, T, 3, H, W]`` 张量。``H``、``W`` 必须等于 M1
                配置的图像尺寸，且 ``T`` 至少为 1。
            frame_chunk_size: 每次送入 encoder 的“单帧样本”数量。``None``
                表示一次处理全部 ``B*T`` 帧；设置较小值可以降低显存峰值，
                但会增加循环次数。

        Returns:
            形状为 ``[B, T, S, D]`` 的张量。``S`` 等于
            ``config.patch_count``，``D`` 等于 ``config.dim``。

        ``@torch.no_grad()`` 与参数的 ``requires_grad=False`` 共同保证这条路径
        不构建反向传播图，从而节省 M2 训练时的显存。
        """

        # 先验证维数，再拆包形状，可以给数据管线提供更明确的错误信息。
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, 3, H, W]")
        batch, frame_count, channels, height, width = frames.shape

        # 通道数和分辨率必须与 M1 预训练配置完全一致；这里只允许时间长度 T
        # 改变，因为每一帧稍后都会被独立处理。
        expected = (3, self.config.image_height, self.config.image_width)
        if (channels, height, width) != expected:
            raise ValueError(
                "frames must have shape [B, T, 3, "
                f"{self.config.image_height}, {self.config.image_width}]"
            )
        if frame_count <= 0:
            raise ValueError("frames must contain at least one observation")

        # chunk_size 的单位是帧，而不是视频数量。默认值 B*T 表示不分块。
        if frame_chunk_size is None:
            chunk_size = batch * frame_count
        else:
            chunk_size = int(frame_chunk_size)
            if chunk_size <= 0:
                raise ValueError("frame_chunk_size must be positive")

        # 合并 batch 和时间维，使原视频中的每一帧成为一个独立样本。
        # 中间的长度 1 是临时的时间维：[B*T, 1, 3, H, W]。
        flattened = frames.reshape(batch * frame_count, 1, channels, height, width)
        outputs = []
        for start in range(0, flattened.shape[0], chunk_size):
            chunk = flattened[start : start + chunk_size]

            # Video ViT 的 Conv3d patch embedding 至少需要 tubelet_size 帧。
            # 复制同一画面可以满足输入契约，同时避免引入相邻真实帧的信息。
            tubelets = chunk.repeat(1, self.config.tubelet_size, 1, 1, 1)
            normalized = self.normalize_frames(tubelets)

            # True 保留所有空间 token；False 才会平均池化成一个向量。
            outputs.append(self.encoder(normalized, return_patch_tokens=True))

        # 分块只影响计算过程，输出仍按原来的 B*T 帧顺序拼回去。
        tokens = torch.cat(outputs, dim=0)
        spatial_tokens = self.config.patch_count

        # 每个静态 tubelet 在时间方向只产生 1 格 token，所以总 token 数应该
        # 恰好等于空间 patch 数。该检查可以尽早发现配置或 encoder 行为变化。
        if tokens.shape[1] != spatial_tokens:
            raise RuntimeError("a repeated frame must produce exactly one spatial token grid")

        # 恢复 batch 和原始帧时间维：[B*T, S, D] -> [B, T, S, D]。
        return tokens.reshape(batch, frame_count, spatial_tokens, self.config.dim)


def repeated_frame_metrics(tokens: Tensor) -> Dict[str, Tensor]:
    """计算 repeated-frame 编码结果的轻量健康指标。

    Args:
        tokens: :class:`FrozenVisualEncoder` 输出的 ``[B, T, S, D]`` 张量。

    Returns:
        一个指标字典：

        * ``mean``：所有 token、所有特征维度的整体平均值；
        * ``average_std``：先计算每个特征维度的标准差，再对 ``D`` 维取平均；
        * ``average_token_norm``：每个 token 的 L2 范数均值；
        * ``adjacent_token_cosine``：展平后相邻 token 的平均余弦相似度；
        * ``finite``：全部值是否均为有限数，即不含 NaN 或正负无穷。

    这些指标用于快速发现 latent 全零、尺度异常、表示坍缩或非有限值，并不是
    训练 loss。计算统一转为 float32，避免低精度诊断结果不稳定。
    """

    if tokens.ndim != 4:
        raise ValueError("tokens must have shape [B, T, S, D]")

    # 把 batch、时间和空间位置合并，得到 N 个 D 维 token：[N, D]。
    flat = tokens.float().flatten(0, 2)

    # 这里计算总体标准差（分母为 N），不使用无偏样本方差。
    centered = flat - flat.mean(dim=0, keepdim=True)
    per_dimension_std = centered.square().mean(dim=0).sqrt()

    # 先把每个 token 归一化为单位长度，点积就等于余弦相似度。
    normalized = torch.nn.functional.normalize(flat, dim=-1)
    if normalized.shape[0] > 1:
        adjacent_cosine = (normalized[:-1] * normalized[1:]).sum(dim=-1).mean()
    else:
        # 只有一个 token 时没有“相邻对”，用 1 表示它与自身完全相似。
        adjacent_cosine = torch.ones((), device=tokens.device)

    # 所有值都返回标量 Tensor，方便调用方统一记录到日志系统。
    return {
        "mean": flat.mean(),
        "average_std": per_dimension_std.mean(),
        "average_token_norm": flat.norm(dim=-1).mean(),
        "adjacent_token_cosine": adjacent_cosine,
        "finite": torch.tensor(bool(torch.isfinite(flat).all()), device=tokens.device),
    }
