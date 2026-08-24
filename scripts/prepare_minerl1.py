#!/usr/bin/env python3
"""Ingest one MineRL 1.0 segment into a canonical MCWM episode store."""

from argparse import ArgumentParser
from pathlib import Path

from mcwm.data.episode_store import EpisodeStore
from mcwm.data.ingest import ingest_minerl_episode


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument(
        "--frame-timestamps",
        type=Path,
        help="Optional JSON PTS list; omit to extract exact PTS from MP4 with PyAV",
    )
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--world-id", required=True)
    parser.add_argument("--recorder-version", default="1.0")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    args = parser.parse_args()
    manifest = ingest_minerl_episode(
        EpisodeStore(args.output),
        episode_id=args.episode_id,
        session_id=args.session_id,
        world_id=args.world_id,
        recorder_version=args.recorder_version,
        video_path=args.video,
        action_path=args.actions,
        frame_timestamps_path=args.frame_timestamps,
        split=args.split,
    )
    print(f"ingested {manifest.episode_id}: {manifest.frame_count} frames, {manifest.action_count} actions")


if __name__ == "__main__":
    main()
