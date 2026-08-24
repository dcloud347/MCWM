#!/usr/bin/env python3
"""Build the deterministic, codec-free M0 fixture episode store."""

from argparse import ArgumentParser
from pathlib import Path

from mcwm.data.fixtures import build_fixture_store


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    store = build_fixture_store(args.output)
    manifest = store.write_dataset_manifest()
    print(f"wrote {len(manifest.episodes)} fixture episodes to {args.output}")
    print(f"manifest_sha256={manifest.content_hash}")


if __name__ == "__main__":
    main()

