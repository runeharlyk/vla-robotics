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
$env:UV_CACHE_DIR='.uv-cache'
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

The `vla` CLI now also defaults to repo-local runtime folders on Windows unless you override them explicitly:

- `.hf-cache\` for Hugging Face downloads
- `.wandb\` for Weights & Biases files
- `.tmp\` for temporary files

If you want datasets/config outside the repo-local defaults:

```powershell
uv run python scripts/setup_libero.py --install --config-dir C:\libero-config --datasets-dir D:\libero-data
```

### Windows LIBERO-Plus

RLinf’s `LIBERO-plus` flow matches the upstream benchmark: it is intended as a drop-in replacement for the base `libero` package, but you must install the `LIBERO-plus` source tree together with its own benchmark assets.

```powershell
$env:UV_CACHE_DIR='.uv-cache'
git clone https://github.com/sylvestf/LIBERO-plus.git
# unzip the upstream LIBERO-plus assets into LIBERO-plus\libero\libero\assets first
uv run python scripts/setup_libero.py --install --source-dir .\LIBERO-plus --variant libero-plus
uv run python -m vla visualize --checkpoint HuggingFaceVLA/smolvla_libero --simulator libero --suite spatial
```

If you prefer Invoke:

```powershell
uv run invoke setup-libero --install --source-dir .\LIBERO-plus --variant libero-plus
uv run invoke visualize --suite spatial
```

`LIBERO-PRO` is closer to an evaluation extension than a pure drop-in replacement. This repo can install a local `LIBERO-PRO` source tree with `--source-dir ... --variant libero-pro`, but you still need the extra benchmark `bddl/init` files populated in that source tree before evaluating its perturbation suites.

### RoboCasa

RoboCasa is not a drop-in dependency in this repo's shared `.venv`.

As of `2026-04-20`, the upstream RoboCasa `v1.0` release expects:

- `numpy==2.2.5`
- `mujoco==3.3.1`
- `robosuite>=1.5.2`

This repo currently pins and uses a different shared simulator stack for LIBERO and ManiSkill, so the safe setup path is a **separate RoboCasa environment** plus a repo-local RoboCasa source / asset checkout.

Bootstrap the source tree and kitchen assets into `.robocasa-src/` with:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/setup_robocasa.py --install
```

That script:

- downloads the official RoboCasa source tree into `.robocasa-src/`
- copies `robocasa/macros.py` to `robocasa/macros_private.py`
- downloads the current Hugging Face kitchen assets into the source tree
- patches the local texture / fixture registries so the current asset release works
- probes whether the **current** Python environment is actually compatible

If the script reports `Workspace ready: True` but compatibility failures, that is expected in the shared repo environment. Use the prepared `.robocasa-src/` tree from a dedicated Python `3.11` environment that matches the upstream RoboCasa requirements.

Repo-native RL training is now wired in for RoboCasa as `--simulator robocasa`, but the supported training mode is currently `--mode sparse_rl` from the isolated RoboCasa environment. `srpo` still depends on a RoboCasa demo replay path that is not implemented yet.

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
