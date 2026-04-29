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
uv run invoke submit-train --experiment fpo_t5_v28_control --profile l40s-16
uv run invoke submit-eval --experiment spatial_current_protocol --profile l40s-16
```

Experiment recipes live in `configs/train_srpo/experiment/` and `configs/evaluate/experiment/`.
Profiles live in `jobs/_profiles.yaml`. Generated submission scripts are written to `jobs/generated/`.

## Queue Limits

CPU queues (non-GPU).
GPU queue limits differ per queue.

| Parameter | `hpc` (LSF) | `thinlinc` (app-node) |
| --------- | ----------- | --------------------- |
| Default queue | Yes | No |
| Max walltime | 72 h | 48 h |
| Max CPU time | N/A | 24 h |
| Max nodes/job | N/A | 1 |
| Max processes/job | 100 | 1 |
| Max processes/queue | 100–120 | 64 |
| Max processes/node | 20 or 24 | 1 |
| Default walltime | 15 min | 48 h |

## Job Submission & Monitoring

```sh
bsub < submit.sh                     # submit
bstat                                # queue status overview
bjobs <jobid>                        # job status
bjobs <jobid>[idx]                   # single array element
bkill <jobid>                        # kill job / whole array
bkill <jobid>[1-5,212,334]           # kill selected array elements
nodestat -F hpc                      # list available node models/features
```

## Enforced Defaults

LSF emits a warning and fills in defaults when these are missing.
Set them explicitly to avoid surprises.

| Option | Default when missing |
| ------ | -------------------- |
| `-J` (job name) | `NONAME` |
| `-W` (walltime) | `15` (15 minutes) |
| `-o` / `-oo` (stdout) | `jobname_%J.out` |
| `-M` (mem limit) | matches `rusage[mem=X]` or `1024MB` |
| `-R "rusage[mem=X]"` | matches `-M` or `1024MB` |
| `-R "span[...]"` | `span[hosts=1]` when `-n > 1` |

## Memory Flags

Both accept `KB`/`MB`/`GB`/`TB` with no space between number and unit.

- **`-R "rusage[mem=X]"`** — per-core reservation used for scheduling.
  With `-n 4 -R "span[hosts=1] rusage[mem=4GB]"` the job needs a host with ≥ 16 GB free.
- **`-M X`** — per-process kill threshold.
  With `-n 4 -M 5GB` the job is killed above 20 GB total on a single host, or above 10 GB per host with `span[ptile=2]`.

## Core Distribution (`span`)

| Directive | Meaning |
| --------- | ------- |
| `span[hosts=1]` | All cores on one host (required for non-MPI multi-core). |
| `span[ptile=N]` | Reserve groups of N cores, each group on a separate host (MPI). |
| `span[block=N]` | Groups of N cores; multiple groups may land on the same host (MPI). |

## Selecting CPU / Node Features

```sh
#BSUB -R "select[model == XeonE5_2660v3]"   # specific CPU model
#BSUB -R "select[avx2]"                     # requires AVX2
```

Use `nodestat -F <queue>` to list available models and features.

## Job Arrays

Run N jobs sharing the same template.
Each element is fully independent and may run out of order.

```sh
#BSUB -J My_array[1-25]%5
#BSUB -oo logs/My_array/%J_%I.out
#BSUB -eo logs/My_array/%J_%I.err

./my_program > Out_${LSB_JOBINDEX}.out
```

- `[1-25]` — indices 1..25.
  Also supports lists (`[1,23,45-67]`) and strides (`[1-21:2]`).
- `%5` — cap on simultaneously running elements (be nice to other users).
- `%J` / `%I` — jobid / array index in `#BSUB` directives.
- `$LSB_JOBID` / `$LSB_JOBINDEX` — same values inside the script body.
- Disable per-element email for large arrays: `#BSUB -env "LSB_JOB_REPORT_MAIL=N"`.

## Notifications

| Directive | Behaviour |
| --------- | --------- |
| `#BSUB -B` | Email when the job starts. |
| `#BSUB -N` | Email when the job ends. |
| `#BSUB -Ne` | Email only on failure. |
| `#BSUB -u addr` | Override recipient (must be a real address). |
| `#BSUB -env "LSB_JOB_REPORT_MAIL=N"` | Disable job-report email entirely. |

## Output Files

- `-o` / `-e` append; `-oo` / `-eo` overwrite.
- Omitting `-e` merges stderr into the `-o` file.
- Paths ending in `/` are treated as directories; LSF writes `<jobid>.out` there.
- Avoid spaces, Danish letters, and special characters in job names and file paths.
