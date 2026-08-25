# MCWM

Minecraft joint-embedding world model trained from VPT contractor and MineRL 1.0 data.

The repository contains the M0 data contracts and the M1 visual-pretraining implementation. M1 uses a from-scratch tubelet Video ViT, an EMA target encoder with stop-gradient, structured video masks, and a joint spatiotemporal predictor. No pretrained model weights are loaded.

## Requirements

- Python 3.9+
- No third-party dependency is required for the M0 core and tests.
- `pyarrow` is optional for Parquet export.
- `av` is optional for extracting exact MP4 presentation timestamps.
- Video decoding/overlay tooling will use the optional `overlay` dependencies.
- M1 training dependencies are installed with `pip install -e '.[train,test]'`.

## Run tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Model tests are skipped with an explicit message when PyTorch is not installed.

## Build and audit the fixture dataset

```bash
PYTHONPATH=src python3 scripts/build_fixture_dataset.py /tmp/mcwm-fixture
PYTHONPATH=src python3 scripts/audit_data.py /tmp/mcwm-fixture
```

For real MP4 ingestion, install `mcwm[video]`; `prepare_vpt.py` and
`prepare_minerl1.py` then extract PTS directly when `--frame-timestamps` is omitted.

## M1 visual pretraining

Log in to W&B once on the training host. For one machine with two H100 GPUs,
use the dedicated configuration:

```bash
wandb login
torchrun --standalone --nproc-per-node=2 -m mcwm.training.pretrain_visual \
  --config configs/pretrain_visual_2xh100.yaml \
  --data-root /path/to/canonical-dataset \
  --output-dir /path/to/checkpoints/m1-visual-2xh100
```

The two-H100 configuration uses a per-GPU batch of 4 and eight gradient
accumulation steps, giving an effective batch of 64 clips. The generic
`configs/pretrain_visual.yaml` remains available for other GPU counts.
Both configurations keep the formal 640x360 input and follow V-JEPA's video
sampling contract: each video access chooses one random 16-frame clip and
samples it at 4 FPS from frame timestamps. They use a Video ViT-Base encoder with
2-frame tubelets, joint spatiotemporal attention, 3D RoPE, bf16, activation
checkpointing, FSDP, EMA, and W&B logging.
Masking follows the V-JEPA 2 two-group setup: one prediction task unions eight
15% full-duration spatial blocks, the other unions two 70% full-duration
blocks, and their per-sample losses are averaged equally.
Formal training uses 300 optimizer iterations per epoch and defaults to 20
epochs (6,000 steps).
The checked parameter counts are 86,899,968 per visual encoder and 21,886,080
for the M1 predictor (195,686,016 parameters saved during phase A).
With one CUDA process the same configuration automatically runs without a
distributed wrapper. Use `wandb.mode=offline` in the YAML on a host without
network access, or pass `--wandb-mode offline`, then sync that run later.

Run a CPU smoke test with the tiny configuration (this is not a valid formal
checkpoint):

```bash
PYTHONPATH=src python3 scripts/pretrain_visual.py \
  --config configs/pretrain_visual_tiny.yaml --synthetic
```

Resume by passing `--resume /path/to/checkpoint.pt`. The checkpoint restores the
online encoder, EMA encoder, predictor, optimizer, scheduler, AMP scaler, each
rank's RNG state, optimizer step, data manifest hash, and W&B run ID.

Export the EMA encoder after M1:

```bash
PYTHONPATH=src python3 scripts/export_visual_encoder.py \
  /path/to/checkpoint.pt /path/to/mcwm-visual-ema.safetensors
```

Compare its frozen linear probes with a random encoder:

```bash
PYTHONPATH=src python3 scripts/evaluate_visual_probe.py \
  /path/to/checkpoint.pt /path/to/canonical-dataset \
  --output /path/to/probe-report.json --log-wandb
```

Training artifacts and datasets remain outside the repository by default.
The repository does not include or claim a fully trained M1 checkpoint; that
artifact is produced on the CUDA training host from the configured VPT/MineRL
1.0 dataset.


See [design.md](design.md) for the full architecture and milestone plan.
