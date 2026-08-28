"""把变长 Minecraft 动作块编码成固定维度 token。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionEncoderConfig:
    """Minecraft action encoder 的结构配置。"""

    binary_embedding_dim: int = 16
    hotbar_embedding_dim: int = 32
    camera_dim: int = 64
    cursor_dim: int = 64
    component_hidden_dim: int = 512
    tick_dim: int = 256
    transformer_depth: int = 2
    transformer_heads: int = 8
    transformer_mlp_dim: int = 1024
    macro_dim: int = 1024
    dropout: float = 0.0
    camera_clip_degrees: float = 180.0
    camera_mu: float = 255.0

    def __post_init__(self) -> None:
        dimensions = (
            self.binary_embedding_dim,
            self.hotbar_embedding_dim,
            self.camera_dim,
            self.cursor_dim,
            self.component_hidden_dim,
            self.tick_dim,
            self.transformer_depth,
            self.transformer_heads,
            self.transformer_mlp_dim,
            self.macro_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("action encoder dimensions must be positive")
        if self.tick_dim % self.transformer_heads:
            raise ValueError("tick_dim must be divisible by transformer_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.camera_clip_degrees <= 0 or self.camera_mu <= 0:
            raise ValueError("camera normalization values must be positive")


class BinaryComponentEncoder(nn.Module):
    """用互不共享的 embedding 编码 14 个二进制按键。"""

    def __init__(self, components: int = 14, embedding_dim: int = 16) -> None:
        super().__init__()
        if min(components, embedding_dim) <= 0:
            raise ValueError("binary encoder dimensions must be positive")
        self.components = components
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            nn.Embedding(2, embedding_dim) for _ in range(components)
        )

    @property
    def output_dim(self) -> int:
        """返回拼接全部按键 embedding 后的维度。"""

        return self.components * self.embedding_dim

    def forward(self, values: Tensor) -> Tensor:
        """把 ``[..., 14]`` 布尔张量编码成一个拼接特征。"""

        if values.ndim == 0 or values.shape[-1] != self.components:
            raise ValueError(
                f"binary values must have shape [..., {self.components}]"
            )
        if values.dtype != torch.bool:
            raise TypeError("binary values must be a boolean tensor")
        encoded = [
            embedding(values[..., index].long())
            for index, embedding in enumerate(self.embeddings)
        ]
        return torch.cat(encoded, dim=-1)


class HotbarEncoder(nn.Module):
    """编码 0（不切换）和 1～9（切换槽位）共 10 种状态。"""

    def __init__(self, embedding_dim: int = 32) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("hotbar embedding_dim must be positive")
        self.embedding = nn.Embedding(10, embedding_dim)

    def forward(self, hotbar: Tensor) -> Tensor:
        """把 ``[B, A, K]`` hotbar 类别编码成向量。"""

        integer_types = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        if hotbar.dtype not in integer_types:
            raise TypeError("hotbar must be an integer tensor")
        # CanonicalActionTick 在入库时已经保证值属于 [0, 9]。这里不再调用
        # Tensor.item()/bool() 做同步检查，避免每个 GPU batch 都被迫等待 CPU。
        return self.embedding(hotbar.long())


def mu_law_normalize(
    values: Tensor,
    *,
    clip_value: float,
    mu: float = 255.0,
) -> Tensor:
    """裁剪连续值，并用 mu-law 将其压缩到 ``[-1, 1]``。"""

    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")
    if not values.is_floating_point():
        raise TypeError("values must be a floating point tensor")

    # 先除以裁剪范围，得到 [-1, 1]；mu-law 会保留正负号，并放大
    # 小动作之间的差别，同时压缩极大的动作。
    scaled = values.clamp(-clip_value, clip_value) / clip_value
    return scaled.sign() * torch.log1p(mu * scaled.abs()) / math.log1p(mu)


class CameraEncoder(nn.Module):
    """把 ``(pitch_delta, yaw_delta)`` 编码成一个特征向量。"""

    def __init__(
        self,
        output_dim: int = 64,
        *,
        hidden_dim: int = 64,
        clip_degrees: float = 180.0,
        mu: float = 255.0,
    ) -> None:
        super().__init__()
        if min(output_dim, hidden_dim) <= 0:
            raise ValueError("camera encoder dimensions must be positive")
        if clip_degrees <= 0:
            raise ValueError("clip_degrees must be positive")
        if mu <= 0:
            raise ValueError("mu must be positive")

        self.clip_degrees = float(clip_degrees)
        self.mu = float(mu)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, camera: Tensor) -> Tensor:
        """编码最后一维为 2 的 camera 张量，前面的维度保持不变。"""

        if camera.ndim == 0 or camera.shape[-1] != 2:
            raise ValueError("camera must have shape [..., 2]")
        normalized = mu_law_normalize(
            camera,
            clip_value=self.clip_degrees,
            mu=self.mu,
        )
        return self.mlp(normalized)


class CursorEncoder(nn.Module):
    """编码 GUI 光标位置；没有有效光标时输出全零。"""

    def __init__(self, output_dim: int = 64, *, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(output_dim, hidden_dim) <= 0:
            raise ValueError("cursor encoder dimensions must be positive")

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        cursor: Tensor,
        gui_open: Tensor,
        cursor_present: Tensor,
    ) -> Tensor:
        """编码 cursor，并用两个布尔状态共同屏蔽无效位置。"""

        if cursor.ndim == 0 or cursor.shape[-1] != 2:
            raise ValueError("cursor must have shape [..., 2]")
        expected_shape = cursor.shape[:-1]
        if gui_open.shape != expected_shape:
            raise ValueError("gui_open must match cursor leading dimensions")
        if cursor_present.shape != expected_shape:
            raise ValueError("cursor_present must match cursor leading dimensions")
        if gui_open.dtype != torch.bool or cursor_present.dtype != torch.bool:
            raise TypeError("gui_open and cursor_present must be boolean tensors")

        encoded = self.mlp(cursor)
        active = (gui_open & cursor_present).unsqueeze(-1)
        return encoded * active.to(dtype=encoded.dtype)


class ComponentFusion(nn.Module):
    """把四类动作特征合并成一个 tick token。"""

    def __init__(
        self,
        binary_dim: int,
        hotbar_dim: int,
        camera_dim: int,
        cursor_dim: int,
        *,
        tick_dim: int = 256,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        component_dims = (binary_dim, hotbar_dim, camera_dim, cursor_dim)
        if min(*component_dims, tick_dim, hidden_dim) <= 0:
            raise ValueError("fusion dimensions must be positive")

        self.component_dims = component_dims
        self.mlp = nn.Sequential(
            nn.Linear(sum(component_dims), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, tick_dim),
            nn.LayerNorm(tick_dim),
        )

    def forward(
        self,
        binary: Tensor,
        hotbar: Tensor,
        camera: Tensor,
        cursor: Tensor,
    ) -> Tensor:
        """拼接同一个 tick 的四类特征，并输出 256 维 token。"""

        components = (binary, hotbar, camera, cursor)
        leading_shape = binary.shape[:-1]
        for name, value, expected_dim in zip(
            ("binary", "hotbar", "camera", "cursor"),
            components,
            self.component_dims,
        ):
            if value.ndim == 0 or value.shape[-1] != expected_dim:
                raise ValueError(f"{name} must have last dimension {expected_dim}")
            if value.shape[:-1] != leading_shape:
                raise ValueError("all component leading dimensions must match")

        return self.mlp(torch.cat(components, dim=-1))


def _sinusoidal_positions(length: int, dim: int, reference: Tensor) -> Tensor:
    """生成动态长度的位置编码，让 Transformer 能区分 tick 顺序。"""

    positions = torch.arange(length, device=reference.device, dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=reference.device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, dim, device=reference.device, dtype=torch.float32)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, : dim // 2].cos()
    return encoding.to(dtype=reference.dtype)


class MicroActionTransformer(nn.Module):
    """按时间顺序处理一个 interval 内的多个 tick token。"""

    def __init__(
        self,
        dim: int = 256,
        *,
        depth: int = 2,
        heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(dim, depth, heads, mlp_dim) <= 0:
            raise ValueError("transformer dimensions must be positive")
        if dim % heads:
            raise ValueError("transformer dim must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.dim = dim
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(dim),
        )

    def forward(self, tick_tokens: Tensor, valid_mask: Tensor) -> Tensor:
        """编码 ``[B, A, K, D]``，并让 padding 位置保持为零。"""

        if tick_tokens.ndim != 4 or tick_tokens.shape[-1] != self.dim:
            raise ValueError(f"tick_tokens must have shape [B, A, K, {self.dim}]")
        if valid_mask.shape != tick_tokens.shape[:-1]:
            raise ValueError("valid_mask must have shape [B, A, K]")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean tensor")
        if tick_tokens.shape[2] <= 0:
            raise ValueError("each interval must contain at least one tick slot")

        batch, intervals, ticks, _ = tick_tokens.shape
        flat_tokens = tick_tokens.reshape(batch * intervals, ticks, self.dim)
        flat_mask = valid_mask.reshape(batch * intervals, ticks)

        # padding 本身不能被当作真实的 no-op。这里只给有效 tick 添加
        # 位置编码，attention 也不会读取无效 tick。
        active = flat_mask.unsqueeze(-1).to(dtype=flat_tokens.dtype)
        positions = _sinusoidal_positions(ticks, self.dim, flat_tokens)
        encoded_input = (flat_tokens + positions.unsqueeze(0)) * active

        # 全 padding 的 interval 会让 attention 出现 NaN。临时放行一个
        # 零 token，计算结束后仍按原 mask 清零，不会产生假的动作信息。
        safe_mask = flat_mask.clone()
        empty_intervals = ~safe_mask.any(dim=1)
        safe_mask[empty_intervals, 0] = True
        encoded = self.encoder(
            encoded_input,
            src_key_padding_mask=~safe_mask,
        )
        encoded = encoded * active
        return encoded.reshape(batch, intervals, ticks, self.dim)


class MinecraftActionEncoder(nn.Module):
    """把变长 Minecraft ActionBlock 编码成 1024 维动作 token。"""

    def __init__(self, config: ActionEncoderConfig = ActionEncoderConfig()) -> None:
        super().__init__()
        self.config = config
        self.binary_encoder = BinaryComponentEncoder(
            components=14,
            embedding_dim=config.binary_embedding_dim,
        )
        self.hotbar_encoder = HotbarEncoder(config.hotbar_embedding_dim)
        self.camera_encoder = CameraEncoder(
            output_dim=config.camera_dim,
            hidden_dim=config.camera_dim,
            clip_degrees=config.camera_clip_degrees,
            mu=config.camera_mu,
        )
        self.cursor_encoder = CursorEncoder(
            output_dim=config.cursor_dim,
            hidden_dim=config.cursor_dim,
        )
        self.fusion = ComponentFusion(
            binary_dim=self.binary_encoder.output_dim,
            hotbar_dim=config.hotbar_embedding_dim,
            camera_dim=config.camera_dim,
            cursor_dim=config.cursor_dim,
            tick_dim=config.tick_dim,
            hidden_dim=config.component_hidden_dim,
        )
        self.micro_transformer = MicroActionTransformer(
            dim=config.tick_dim,
            depth=config.transformer_depth,
            heads=config.transformer_heads,
            mlp_dim=config.transformer_mlp_dim,
            dropout=config.dropout,
        )
        self.projection = nn.Linear(config.tick_dim, config.macro_dim)

    def forward(
        self,
        movement: Tensor,
        interaction: Tensor,
        hotbar: Tensor,
        camera: Tensor,
        cursor: Tensor,
        gui_open: Tensor,
        cursor_present: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """返回 ``[B, A, 1024]`` macro-action token。"""

        if movement.ndim != 4 or movement.shape[-1] != 7:
            raise ValueError("movement must have shape [B, A, K, 7]")
        if interaction.shape != movement.shape:
            raise ValueError("interaction must match movement shape")
        tick_shape = movement.shape[:-1]
        if hotbar.shape != tick_shape:
            raise ValueError("hotbar must have shape [B, A, K]")
        for name, value in (("camera", camera), ("cursor", cursor)):
            if value.shape != tick_shape + (2,):
                raise ValueError(f"{name} must have shape [B, A, K, 2]")
        for name, value in (
            ("gui_open", gui_open),
            ("cursor_present", cursor_present),
            ("valid_mask", valid_mask),
        ):
            if value.shape != tick_shape:
                raise ValueError(f"{name} must have shape [B, A, K]")

        binary = self.binary_encoder(torch.cat((movement, interaction), dim=-1))
        hotbar_features = self.hotbar_encoder(hotbar)
        camera_features = self.camera_encoder(camera)
        cursor_features = self.cursor_encoder(cursor, gui_open, cursor_present)
        tick_tokens = self.fusion(
            binary,
            hotbar_features,
            camera_features,
            cursor_features,
        )
        tick_tokens = self.micro_transformer(tick_tokens, valid_mask)

        # 只平均真实 tick。真实 no-op 的 valid_mask=True，所以仍会参与编码；
        # 人工 padding 的 valid_mask=False，不会影响 macro-action token。
        weights = valid_mask.unsqueeze(-1).to(dtype=tick_tokens.dtype)
        tick_count = weights.sum(dim=2).clamp_min(1.0)
        pooled = (tick_tokens * weights).sum(dim=2) / tick_count
        macro_actions = self.projection(pooled)

        # 数据集正常不会产生空 ActionBlock；这里仍让全 padding interval 输出零，
        # 避免 projection bias 把它伪装成一个真实动作。
        has_ticks = valid_mask.any(dim=2, keepdim=True)
        return macro_actions * has_ticks.to(dtype=macro_actions.dtype)
