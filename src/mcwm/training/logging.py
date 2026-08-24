"""训练指标始终写本地 JSONL，并可由 rank 0 同步到 W&B。"""

from __future__ import annotations

import json
from pathlib import Path
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence


class TrainingLogger:
    """一个很薄的日志层；W&B 故障不会让本地训练记录消失。"""

    def __init__(
        self,
        output_dir: Path,
        *,
        config: Mapping[str, Any],
        wandb_config: Mapping[str, Any],
        run_id: Optional[str] = None,
        rank: int = 0,
    ) -> None:
        self.rank = int(rank)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_path = self.output_dir / "metrics.jsonl"
        self.wandb_run = None
        # 多卡时只有 rank 0 创建 W&B run，避免每张卡产生一个实验。
        if self.rank != 0 or not wandb_config.get("enabled", True):
            return
        mode = str(wandb_config.get("mode", "online"))
        if mode == "disabled":
            return
        try:
            import wandb  # type: ignore

            self.wandb_run = wandb.init(
                entity=wandb_config.get("entity"),
                project=wandb_config.get("project", "mcwm"),
                group=wandb_config.get("group"),
                name=wandb_config.get("name"),
                tags=list(wandb_config.get("tags", [])),
                mode=mode,
                config=dict(config),
                id=run_id,
                resume="allow" if run_id else None,
                dir=str(self.output_dir),
            )
        except Exception as exc:  # W&B 不可用时仍继续保留本地日志。
            warnings.warn(f"W&B initialization failed; continuing with local JSONL: {exc}")

    @property
    def run_id(self) -> Optional[str]:
        return getattr(self.wandb_run, "id", None)

    @property
    def run_name(self) -> Optional[str]:
        return getattr(self.wandb_run, "name", None)

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        """以 optimizer step 为横轴，同时写 JSONL 和 W&B。"""

        if self.rank != 0:
            return
        record: Dict[str, Any] = {"optimizer_step": int(step)}
        record.update(
            {
                key: value.item() if hasattr(value, "item") else value
                for key, value in metrics.items()
            }
        )
        with self.local_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(dict(metrics), step=step)
            except Exception as exc:
                warnings.warn(f"W&B logging failed at step {step}: {exc}")

    def log_images(self, key: str, images: Sequence[Any], *, step: int) -> None:
        """只向 W&B 上传固定验证样本的小型诊断图。"""

        if self.rank != 0 or self.wandb_run is None:
            return
        try:
            import wandb  # type: ignore

            self.wandb_run.log({key: [wandb.Image(image) for image in images]}, step=step)
        except Exception as exc:
            warnings.warn(f"W&B image logging failed at step {step}: {exc}")

    def finish(self) -> None:
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as exc:
                warnings.warn(f"W&B finish failed: {exc}")
