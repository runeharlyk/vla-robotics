# HPC Usage

DTU HPC uses LSF. Submit with `bsub < jobs/script.sh`. Max walltime: 24h.

## Available GPUs

| Queue | Nodes | GPUs per node | VRAM |
| ----- | ----- | ------------- | ---- |
| gpua100 | 4 + 6 | 2 × A100 | 40 GB or 80 GB |
| gpul40s | 6 | 2 × L40s | 48 GB |
| gpuv100 | 6 + 8 + 3 | 2–4 × V100 | 16 GB or 32 GB |
| gpua10 | 1 | 2 × A10 | 24 GB |
| gpua40 | 1 | 2 × A40 | 48 GB (NVlink) |

## LSF Directives

```sh
#BSUB -J job_name
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -oo logs/job_name/%J.out
```

- **`rusage[mem=X]`** — RAM per core (4 cores × 8GB = 32GB total).
- **`-n`** — Number of cores.

### Selecting GPU VRAM

| Need | Add to script |
| ---- | ------------- |
| V100 32 GB (gpuv100) | `#BSUB -R "select[gpu32gb]"` |
| V100 with NVlink (gpuv100) | `#BSUB -R "select[sxm2]"` |
| A100 80 GB (gpua100) | `#BSUB -R "select[gpu80gb]"` |
| 2 GPUs | `#BSUB -n 8` and `#BSUB -gpu "num=2:mode=exclusive_process"` |

## Interactive Shells

| Command | GPUs |
| ------- | ---- |
| `voltash` | 2 × V100 16 GB |
| `sxm2sh` | 4 × V100 32 GB |
| `a100sh` | 2 × A100 40 GB |

## Project Setup

Jobs source `jobs/_env.sh` (sets `VLA_WORK3`, `HF_HOME`, `WANDB_DIR`, loads cuda/12.2, runs `uv sync`).

**Validate before submitting:**

```sh
uv run huggingface-cli download facebook/dinov2-large facebook/vjepa2-vitg-fpc64-384-ssv2
uv run python scripts/validate_hpc_setup.py --world-model all
```

## Creating Jobs

```sh
invoke create-job --config configs/smolvla_eval_libero_all.yaml --gpu a100
```

Profiles in `jobs/_profiles.yaml`; actions in `jobs/_actions.yaml`.
