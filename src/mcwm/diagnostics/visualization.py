"""把验证样本画成原图、mask 图和预测误差图。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def visual_pretraining_images(
    frames: Tensor,
    target_mask: Tensor,
    prediction: object,
    target: Tensor,
    *,
    prediction_indices: Optional[object] = None,
    grid_size: Tuple[int, int],
) -> List[object]:
    """返回原图、实际预测区域和特征预测误差热力图。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("visual diagnostics require `pip install mcwm[train]`") from exc
    image_tensor = frames[0, 0].detach().cpu()
    if image_tensor.dtype != torch.uint8:
        image_tensor = image_tensor.float().clamp(0, 1).mul(255).byte()
    image = image_tensor.permute(1, 2, 0).numpy()
    rows, columns = grid_size
    # mask 形状是 [组, batch, 时间, patch]；这里只展示第一组和第一帧。
    if target_mask.ndim == 4:
        target_mask = target_mask[0]
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    if not isinstance(prediction, Tensor):
        raise TypeError("prediction must contain tensors")
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

    if prediction_indices is not None:
        if isinstance(prediction_indices, (tuple, list)):
            prediction_indices = prediction_indices[0]
        if not isinstance(prediction_indices, Tensor):
            raise TypeError("prediction_indices must contain tensors")
        indices = prediction_indices[0]
        flat_target = target[0].flatten(0, 1)
        target_values = flat_target.index_select(0, indices)
        predicted_error = (
            prediction[0].float() - target_values.float()
        ).abs().mean(dim=-1)
        dense_error = predicted_error.new_zeros(target.shape[1] * rows * columns)
        dense_error.scatter_(0, indices, predicted_error)
        error = dense_error[: rows * columns]
    else:
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
