# Repository Guidelines

## Project Structure & Module Organization

MCWM is a Python 3.9+ package using a `src/` layout. Library code lives under `src/mcwm/`: `actions/` normalizes VPT and MineRL inputs, `data/` handles ingestion and manifests, `models/` contains the visual JEPA components, `training/` owns pretraining and checkpoints, and `diagnostics/` provides collapse checks and probes. Keep command-line wrappers in `scripts/`, YAML experiment settings in `configs/`, and tests in the matching `tests/<area>/` directory. `design.md` documents architecture and milestones. Datasets, checkpoints, and generated artifacts belong outside version control (`data/` and `artifacts/` are ignored).

## Build, Test, and Development Commands

- `python3 -m pip install -e '.[train,test]'` installs the package plus training and test dependencies in editable mode.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v` runs the complete suite, including dependency-aware skips.
- `PYTHONPATH=src python3 scripts/build_fixture_dataset.py /tmp/mcwm-fixture` creates a small canonical dataset for local checks.
- `PYTHONPATH=src python3 scripts/audit_data.py /tmp/mcwm-fixture` validates dataset contracts and split integrity.
- `PYTHONPATH=src python3 scripts/pretrain_visual.py --config configs/pretrain_visual_tiny.yaml --synthetic --max-steps 2` runs a CPU training smoke test.

There is no separate build step; setuptools is configured through `pyproject.toml`.

## Coding Style & Naming Conventions

Follow existing PEP 8-style Python: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase module constants. Add type hints to public interfaces and concise docstrings where behavior or data contracts are not obvious. Prefer immutable dataclasses and `pathlib.Path` where consistent with neighboring code. No formatter or linter is currently configured, so keep imports grouped and lines readable, and avoid drive-by formatting.

## Testing Guidelines

Tests use `unittest` classes and assertions; `pytest` is optional but can collect the same files. Name files `test_<feature>.py`, classes `<Feature>Test`, and methods `test_<behavior>`. Place regressions beside the affected subsystem. Exercise error paths and invariants, especially 640x360 resolution, temporal alignment, leakage-safe splits, checkpoint resume, and optional-dependency behavior.

## Commit & Pull Request Guidelines

Recent history uses short, lowercase, imperative subjects such as `add initial implementation...` and `update README...`. Keep each commit focused. Pull requests should explain the motivation and behavior change, list commands run, link relevant issues, and call out configuration or data-contract changes. Include logs or screenshots only when training metrics or visual diagnostics are affected; never commit datasets, credentials, W&B secrets, or checkpoints.
