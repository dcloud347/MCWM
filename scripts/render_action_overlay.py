#!/usr/bin/env python3
"""Render a 640x360 MP4 with canonical action/timestamp overlays (optional deps)."""

from argparse import ArgumentParser
from pathlib import Path

from mcwm.actions.schema import INTERACTION_NAMES, MOVEMENT_NAMES
from mcwm.data.alignment import align_actions_to_frames
from mcwm.data.episode_store import EpisodeStore


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("episode_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("overlay rendering requires `pip install mcwm[overlay]`") from exc

    episode = EpisodeStore(args.root).read_episode(args.episode_id)
    video_path = Path(episode.manifest.video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (640, 360),
    )
    aligned = align_actions_to_frames(episode.frame_timestamps_ms, episode.actions)
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame_index >= episode.manifest.frame_count:
            break
        if frame.shape[:2] != (360, 640):
            raise SystemExit(f"frame {frame_index} is {frame.shape[1]}x{frame.shape[0]}, expected 640x360")
        lines = [f"frame={frame_index} pts={episode.frame_timestamps_ms[frame_index]}ms"]
        if frame_index < len(aligned.blocks):
            block = aligned.blocks[frame_index]
            for tick in block.actions:
                active = [name for name, value in zip(MOVEMENT_NAMES, tick.movement) if value]
                active.extend(name for name, value in zip(INTERACTION_NAMES, tick.interaction) if value)
                if tick.hotbar:
                    active.append(f"hotbar.{tick.hotbar}")
                if tick.camera != (0.0, 0.0):
                    active.append(f"camera={tick.camera}")
                lines.append(f"{tick.timestamp_ms}: {'+'.join(active) if active else 'noop'}")
                if tick.cursor is not None:
                    point = (int(tick.cursor[0] * 639), int(tick.cursor[1] * 359))
                    cv2.drawMarker(frame, point, (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
            if not block.continuous:
                lines.append("DISCONTINUITY")
        for line_index, line in enumerate(lines[:12]):
            cv2.putText(
                frame,
                line,
                (8, 18 + line_index * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    print(f"wrote overlay to {args.output}")


if __name__ == "__main__":
    main()

