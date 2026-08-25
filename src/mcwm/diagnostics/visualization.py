"""把固定验证样本画成适合上传 W&B 的小型 M1 诊断图。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def visual_pretraining_images(
    frames: Tensor,
    target_mask: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    grid_size: Tuple[int, int],
) -> List[object]:
    """依次返回原图、mask 叠加图和 latent prediction error 热力图。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("visual diagnostics require `pip install mcwm[train]`") from exc
    image_tensor = frames[0, 0].detach().cpu()
    if image_tensor.dtype != torch.uint8:
        image_tensor = image_tensor.float().clamp(0, 1).mul(255).byte()
    image = image_tensor.permute(1, 2, 0).numpy()
    rows, columns = grid_size
    # Multi-block 训练返回 [G, B, T, N]；诊断图展示第一套 mask。
    if target_mask.ndim == 4:
        target_mask = target_mask[0]
    if prediction.ndim == 5:
        prediction = prediction[0]
    if target_mask.shape[-1] != rows * columns:
        raise ValueError("mask patch count does not match grid_size")
    height, width = image.shape[:2]
    patch_height, patch_width = height // rows, width // columns

    mask = target_mask[0, 0].reshape(rows, columns).detach().cpu()
    pixel_mask = mask.repeat_interleave(patch_height, 0).repeat_interleave(patch_width, 1)
    masked = image.copy()
    masked[pixel_mask.numpy()] = (
        0.25 * masked[pixel_mask.numpy()] + 0.75 * np.array([255, 80, 80])
    ).astype(np.uint8)

    error = (prediction[0, 0].float() - target[0, 0].float()).abs().mean(dim=-1)
    error = error.reshape(rows, columns).detach().cpu()
    error = (error - error.min()) / (error.max() - error.min()).clamp_min(1e-8)
    pixel_error = error.repeat_interleave(patch_height, 0).repeat_interleave(patch_width, 1)
    heatmap = image.astype(np.float32)
    heatmap[..., 0] = heatmap[..., 0] * 0.4 + pixel_error.numpy() * 255 * 0.6
    heatmap[..., 1:] *= 0.4
    return [
        Image.fromarray(image),
        Image.fromarray(masked),
        Image.fromarray(heatmap.clip(0, 255).astype(np.uint8)),
    ]
