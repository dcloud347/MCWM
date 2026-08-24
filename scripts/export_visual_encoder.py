#!/usr/bin/env python3
"""从 M1 checkpoint 导出 EMA visual encoder。"""

import argparse
from pathlib import Path

from mcwm.training.checkpoint import export_ema_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(export_ema_encoder(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
