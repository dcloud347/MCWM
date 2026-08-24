"""Optional exact MP4 PTS extraction through PyAV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    frame_timestamps_ms: Tuple[int, ...]


def probe_video(path: Path, *, expected_width: int = 640, expected_height: int = 360) -> VideoProbe:
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise RuntimeError("video PTS extraction requires `pip install mcwm[video]`") from exc

    timestamps = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path}")
        stream = container.streams.video[0]
        width, height = int(stream.codec_context.width), int(stream.codec_context.height)
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"video resolution is {width}x{height}; expected {expected_width}x{expected_height}"
            )
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"decoded frame lacks PTS/time_base in {path}")
            timestamps.append(int(round(float(frame.pts * frame.time_base) * 1000.0)))
    if len(timestamps) < 2:
        raise ValueError(f"video must contain at least two frames: {path}")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"video PTS must be strictly increasing: {path}")
    return VideoProbe(width, height, tuple(timestamps))
