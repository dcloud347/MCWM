# MCWM

Minecraft joint-embedding world model trained from VPT contractor and MineRL 1.0 data.

The repository is currently at milestone M0: data contracts, action adapters, temporal alignment, manifests, an episode store, auditing, and deterministic fixtures. Model training is intentionally not part of M0.

## Requirements

- Python 3.9+
- No third-party dependency is required for the M0 core and tests.
- `pyarrow` is optional for Parquet export.
- `av` is optional for extracting exact MP4 presentation timestamps.
- Video decoding/overlay tooling will use the optional `overlay` dependencies.

## Run tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Build and audit the fixture dataset

```bash
PYTHONPATH=src python3 scripts/build_fixture_dataset.py /tmp/mcwm-fixture
PYTHONPATH=src python3 scripts/audit_data.py /tmp/mcwm-fixture
```

For real MP4 ingestion, install `mcwm[video]`; `prepare_vpt.py` and
`prepare_minerl1.py` then extract PTS directly when `--frame-timestamps` is omitted.


See [design.md](design.md) for the full architecture and milestone plan.
