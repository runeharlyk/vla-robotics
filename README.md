# VLA Robotics

## Setup

This project targets `Python 3.11.14`.

```bash
git clone https://github.com/runeharlyk/vla-robotics
cd vla-robotics
uv sync
uv sync --dev
```

### Windows LIBERO

Native Windows LIBERO installs need two extra pieces beyond a normal `uv sync`:

```powershell
uv run python scripts/setup_libero.py --install
uv run python -m vla visualize --checkpoint HuggingFaceVLA/smolvla_libero --simulator libero --suite spatial
```

`scripts/setup_libero.py --install` downloads the upstream `libero` source distribution, installs the package into the active virtualenv, and fetches the missing LIBERO assets that upstream expects at runtime. It also creates the non-interactive LIBERO config file that upstream otherwise prompts for on first import.

On Windows the setup script now:

- installs the `libero` package directly into the active `.venv`
- downloads the missing LIBERO scene assets into that installed package
- writes the LIBERO config under `.libero\config.yaml` in the repo by default
- uses `.hf-cache\` in the repo for Hugging Face downloads unless `HF_HOME` is already set
- patches `robosuite` for Windows so `MUJOCO_GL=wgl` works without the Linux EGL path

If you want datasets/config outside the repo-local defaults:

```powershell
uv run python scripts/setup_libero.py --install --config-dir C:\libero-config --datasets-dir D:\libero-data
```

## Results

Latest results are tracking in:
- `results/` for raw training and eval results
- [docs/smolvla_libero_eval.md](docs/smolvla_libero_eval.md) for SmolVLA LIBERO results
- [docs/fpo_hyperparameter_experiments.md](docs/fpo_hyperparameter_experiments.md) for FPO hyperparameter experiments

Current SmolVLA LIBERO results:

| Suite | Success (SFT) | Success (RL) | Episodes |
| ----- | ------------- | ------------ | -------- |
| `libero_spatial` | `80.9%` | `78.8%` | `1000` |
| `libero_object` | `86.3%` | - | `1000` |

![LIBERO Spatial Comparison](assets/libero_spatial_comparison.png)

<!-- Regenerate plots from committed eval results:

```bash
uv run python -m vla.utils.plot_results --results-dir results/evals --suite spatial
``` -->

## Experiments And HPC Jobs

Training and eval runs are defined as Hydra configs, while `jobs/` only contains shared LSF environment/profile helpers and a few legacy SFT/eval wrappers.

- Train configs: `configs/train_srpo/experiment/`
- Eval configs: `configs/evaluate/experiment/`
- Queue profiles: `jobs/_profiles.yaml`
- Generated submit scripts: `jobs/generated/` (gitignored)

List and inspect configured experiments:

```bash
uv run invoke list-experiments --kind train
uv run invoke list-experiments --kind eval
uv run invoke list-training-runs --experiment fpo_t5_v28_control
uv run invoke list-unrun-experiments
```

Create a validated LSF submit script:

```bash
# Train from a training experiment config.
uv run invoke submit-train --experiment fpo_t5_v28_control --profile l40s-16

# Evaluate the base SFT checkpoint from an eval experiment config.
uv run invoke submit-eval --experiment spatial_sft_seeded --profile a10-10h

# Evaluate a checkpoint produced by a training experiment.
uv run invoke submit-eval --experiment fpo_t5_v28_control --checkpoint best --profile a10-10h
```

`submit-train` always reads from `configs/train_srpo/experiment/`. `submit-eval` first checks whether `--experiment` names a training experiment; if it does, it finds the matching local training record, resolves `--checkpoint best`, `last`, or `best-rollout`, and checks that the selected checkpoint is visible before real submission. If no training experiment matches, `submit-eval` falls back to `configs/evaluate/experiment/`, which is used for SFT baseline evals and explicit comparison protocols such as `spatial_current_protocol`.

The submit tasks validate that configs exist, Hydra composes, generated CLI arguments match the underlying Typer entrypoint, the profile exists, and expected HPC prerequisites are visible.

On HPC you can prepare the shell first:

```bash
source jobs/_env.sh
uv run --no-sync invoke submit-train --experiment fpo_t5_v28_control --profile l40s-16
uv run --no-sync invoke submit-eval --experiment spatial_sft_seeded --profile a10-10h
uv run --no-sync invoke submit-eval --experiment fpo_t5_v28_control --checkpoint best --profile a10-10h
```

Generated jobs use `uv run --no-sync` after sourcing `jobs/_env.sh`, because `_env.sh` already runs `uv sync`.

## Studies

### Perturbation study

We did a study to explore and understand how the model performs under different perturbations.

The models action sensitivity to language instructions and visual changes is shown below:

![Language Action Sensitivity](assets/language_action_sensitivity.png)

![Coming soon]()

This is quantified by looking the models succesrate and mean episode length under different perturbations.

### Attention study

- Visual study: `src/vla/diagnostics/` contains cross-attention, self-attention, and Grad-CAM analysis.

### Reward study

Explores [SRPO](https://arxiv.org/abs/2511.15605) as a way to improve training signal.

Goal:

- Quantify Progress Monotonicity
- Quantify the difficulty of differentiating between demonstrations, successful and failed trajectories, and random trajectories.
- Test encoding methods: per-frame mean pool, clip-based and [siiRL](https://github.com/sii-research/siiRL/blob/main/siirl/utils/embodied/video_emb.py) implementation.
