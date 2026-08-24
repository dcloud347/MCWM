#!/usr/bin/env python3
"""Extract exact frame presentation timestamps from a 640x360 MP4."""

from argparse import ArgumentParser
import json
from pathlib import Path

from mcwm.data.video import probe_video


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    probe = probe_video(args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "width": probe.width,
                "height": probe.height,
                "timestamps_ms": probe.frame_timestamps_ms,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(probe.frame_timestamps_ms)} frame timestamps to {args.output}")


if __name__ == "__main__":
    main()

