# MCWM

MCWM is a from-scratch, joint-embedding world model for Minecraft 1.16.5. It
learns from first-person RGB video and keyboard/mouse actions without loading
external pretrained weights.

The repository implements the data layer, visual pretraining stage, and M2
action-conditioned latent world-model training stack. No trained checkpoint
is distributed, and the planning stack remains future work.

## Project status

| Milestone | Scope                                                                                    | Status                                            |
|-----------|------------------------------------------------------------------------------------------|---------------------------------------------------|
| M0        | Canonical actions, ingestion, manifests, alignment, audits, and fixtures                 | Implemented                                       |
| M1        | From-scratch visual encoder, masked-video predictor, EMA, diagnostics, and checkpointing | Implemented; no trained checkpoint is distributed |
| M2        | Action encoder, block-causal predictor, rollout loss, training, and diagnostics            | Implemented; formal training/gates pending         |
| M3        | Multi-step latent rollout                                                                | Planned                                           |
| M4        | Online planning smoke test in MineRL (environment only)                                  | Planned                                           |

MCWM is a research codebase under active development, not a pretrained model
release or a complete Minecraft agent.

**Training data policy:** MCWM uses only labeled VPT contractor
demonstrations. MineRL/BASALT demonstrations and unlabeled VPT internet videos
are outside the training-data scope. A future MineRL integration may provide
an online evaluation environment, but it must not contribute training data.

## What is implemented

- A source-independent action schema covering movement, interaction, hotbar,
  camera, cursor, GUI state, validity, timestamps, and label confidence.
- A VPT contractor adapter that preserves real no-op actions.
- Exact frame/action alignment based on MP4 presentation timestamps (PTS).
- Leakage-aware dataset manifests grouped by session and world.
- Dataset auditing and rendered action overlays for manual QA.
- A Video ViT masked-video JEPA trained entirely from random initialization.
- EMA target updates, stop-gradient targets, structured spatiotemporal masks,
  collapse diagnostics, W&B logging, and resumable checkpoints.
- Frozen linear probes and export of the trained EMA visual encoder.
- A frozen-M1 repeated-frame encoder and timestamp-aligned M2 dataset.
- Minecraft micro-action encoding with padding-safe temporal aggregation.
- A frame/block-causal latent predictor with teacher-forced and autoregressive loss.
- M2 training, parent-verified checkpoint resume, B0 smoke gates, and action-sensitivity diagnostics.

The formal M1 model uses 16 frames sampled at 4 FPS and keeps the native
`640x360` Minecraft frame. Its Video ViT-Large encoder has 304,770,048
parameters; the masked-video predictor has 22,082,944 parameters.

## Requirements

- Python 3.9 or newer
- PyTorch 2.1 or newer for model training
- CUDA hardware for formal M1 training

Install the package and the development/training dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[train,test]'
```

Optional dependency groups are also available for narrower workflows:

| Extra | Purpose |
| --- | --- |
| `train` | PyTorch training, video decoding, W&B, probes, and export |
| `test` | Test collection with pytest |
| `video` | Exact MP4 PTS extraction with PyAV |
| `overlay` | Action-overlay rendering with OpenCV |
| `parquet` | Canonical action export with PyArrow |

## Quick start

Run the complete test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -v
```

Tests that require an unavailable optional dependency are skipped with an
explicit reason.

Build and audit a tiny canonical dataset:

```bash
PYTHONPATH=src python3 scripts/build_fixture_dataset.py /tmp/mcwm-fixture
PYTHONPATH=src python3 scripts/audit_data.py /tmp/mcwm-fixture
```

Run the deterministic two-step CPU training smoke test:

```bash
PYTHONPATH=src python3 scripts/pretrain_visual.py \
  --config configs/pretrain_visual_tiny.yaml \
  --synthetic
```

Run the M2 synthetic smoke training after installing the training dependencies:

```bash
PYTHONPATH=src python3 scripts/train_world_model.py \
  --config configs/train_world_model_tiny.yaml \
  --synthetic --max-steps 2
```

Run formal M2 training on one node with two H100 SXM GPUs:

```bash
wandb login
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 \
  -m mcwm.training.train_world_model \
  --config configs/train_world_model_2xh100_sxm.yaml
```

For this FSDP configuration, `data.batch_size` is per GPU and
`optimizer.effective_batch_size` is global across both GPUs.

Audit how many clips sampled by the M2 training configuration contain actions:

```bash
PYTHONPATH=src python3 scripts/audit_world_model_clips.py \
  --config configs/train_world_model_2xh100_sxm.yaml \
  --sampling-epochs 5 \
  --output artifacts/world_model_clip_action_audit.json
```

The audit follows the training sampler and action alignment but skips video
decoding. It reports action coverage at clip, transition, and raw-tick levels,
plus movement, interaction, camera, hotbar, GUI, and cursor categories.

Tiny configurations are only for tests and local smoke checks. Their outputs
are not valid formal M1 or M2 checkpoints.

## Data pipeline

MCWM stores each recording as a canonical episode while leaving the original
MP4 in place:

```text
<dataset-root>/
├── dataset_manifest.json
└── episodes/
    └── <episode-id>/
        ├── manifest.json
        ├── actions.jsonl
        ├── frame_timestamps.json
        └── audit.json            # present when the adapter emits QA metadata
```

Every action block is assigned to the half-open interval between two frames:

```text
frame[t] -- actions in [pts[t], pts[t+1]) --> frame[t+1]
```

The data contract requires exactly `640x360` RGB video. A canonical manifest
references the source MP4 instead of copying it, so do not move or delete the
video after ingestion. Formal training validates the complete manifest and
refuses to start if it contains a non-VPT episode.

### Ingest one recording

VPT contractor data:

```bash
PYTHONPATH=src python3 scripts/prepare_vpt.py \
  --output /path/to/canonical \
  --video /path/to/episode.mp4 \
  --actions /path/to/episode.jsonl \
  --episode-id episode-001 \
  --session-id session-001 \
  --world-id world-001 \
  --recorder-version 7.x \
  --split train
```

The command extracts exact frame PTS with PyAV. Pass
`--frame-timestamps timestamps.json` only when timestamps have already been
extracted separately.

Audit the resulting store and write a machine-readable report:

```bash
PYTHONPATH=src python3 scripts/audit_data.py /path/to/canonical \
  --output /path/to/canonical/audit-report.json
```

The audit exits with a non-zero status when session/world leakage is found.
Inspect all reported timing or discontinuity issues before training. For a
visual alignment check, render an episode with its actions overlaid:

```bash
PYTHONPATH=src python3 scripts/render_action_overlay.py \
  /path/to/canonical episode-001 /tmp/episode-001-overlay.mp4
```

See [DATA_PREPARATION.md](DATA_PREPARATION.md) for the reproducible VPT subset
download, conversion, audit, and single-H200 workflow.

## Visual pretraining

The M1 objective uses an online encoder for masked context, an EMA target
encoder for the complete clip, and a predictor for the masked tubelet tokens:

```text
masked clip ──> online encoder ──> visual predictor ──> predicted latents
full clip   ──> EMA encoder ─────────────────────────> target latents
```

Only the online branch and predictor receive gradients. The target branch is
stop-gradient and updated from the online encoder with EMA. All model weights
start inside MCWM; checkpoints explicitly record `external_pretrained=false`.

### Training configurations

| Configuration | Intended use |
| --- | --- |
| `configs/pretrain_visual_tiny.yaml` | CPU smoke test only |
| `configs/pretrain_visual.yaml` | Generic formal CUDA run |
| `configs/pretrain_visual_1xh200.yaml` | One H200 |
| `configs/pretrain_visual_2xh100.yaml` | Two H100 GPUs with FSDP |
| `configs/pretrain_visual_4xh100_sxm.yaml` | Four H100 SXM GPUs with FSDP |

Start a single-process run:

```bash
wandb login
PYTHONPATH=src python3 -m mcwm.training.pretrain_visual \
  --config configs/pretrain_visual_1xh200.yaml \
  --data-root /path/to/canonical \
  --output-dir /path/to/checkpoints/m1-visual
```

Start the two-H100 configuration:

```bash
wandb login
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 \
  -m mcwm.training.pretrain_visual \
  --config configs/pretrain_visual_2xh100.yaml \
  --data-root /path/to/canonical \
  --output-dir /path/to/checkpoints/m1-visual-2xh100
```

Use `--wandb-mode offline` on a host without network access, or
`--wandb-mode disabled` for local checks. Training metrics are also written to
local JSONL logs.

### Resume a run

```bash
PYTHONPATH=src python3 -m mcwm.training.pretrain_visual \
  --config configs/pretrain_visual_1xh200.yaml \
  --data-root /path/to/canonical \
  --output-dir /path/to/checkpoints/m1-visual \
  --resume /path/to/checkpoint-00001800.pt
```

A checkpoint restores the online and EMA encoders, predictor, optimizer,
scheduler, AMP scaler, per-rank RNG state, optimizer step, W&B run identity,
and sampler progress. Resume validates the dataset manifest hash and critical
training semantics before loading weights.

### Export and evaluate the encoder

Export only the EMA visual encoder to SafeTensors:

```bash
PYTHONPATH=src python3 scripts/export_visual_encoder.py \
  /path/to/checkpoint.pt \
  /path/to/mcwm-visual-ema.safetensors
```

Compare frozen linear probes from the trained encoder with a random encoder:

```bash
PYTHONPATH=src python3 scripts/evaluate_visual_probe.py \
  /path/to/checkpoint.pt \
  /path/to/canonical \
  --output /path/to/probe-report.json \
  --log-wandb
```

## Repository layout

```text
src/mcwm/
├── actions/       # canonical action schema and source adapters
├── data/          # ingestion, manifests, alignment, datasets, and audits
├── diagnostics/   # collapse metrics, probes, and visualizations
├── models/        # Video ViT, masking, and visual JEPA components
└── training/      # configuration, EMA, checkpointing, logging, and training

configs/           # data and experiment YAML files
scripts/           # command-line workflows
tests/             # unittest suites mirroring the package layout
design.md          # architecture decisions and future milestones
```

Datasets, checkpoints, W&B credentials, and generated artifacts must remain
outside version control. The repository ignores `data/` and `artifacts/` by
default.

## Design constraints

- VPT contractor demonstrations are the only supported training dataset;
  MineRL/BASALT demonstrations and unlabeled VPT videos are out of scope.
- Minecraft 1.16.5 is the target game version. MineRL may be used later as an
  online evaluation environment, never as a source of training trajectories.
- Training resolution is fixed at `640x360`; ingestion rejects incompatible
  manifests instead of silently resizing them.
- Dataset splits must be grouped by session/world, never by individual clip.
- Real no-op ticks are training data and must not be discarded.
- External pretrained weights, including VPT policy/IDM weights and generic
  visual checkpoints, are not accepted by the training pipeline.

For the full model rationale, M2 contracts, tests, and milestone acceptance
criteria, read [M2_implementation.md](M2_implementation.md) and
[design.md](design.md).
