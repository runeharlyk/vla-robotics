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