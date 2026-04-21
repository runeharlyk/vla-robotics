# RoboCasa Setup

This repo now includes `scripts/setup_robocasa.py` to stage the official RoboCasa source tree and current kitchen assets into the workspace without mutating the shared LIBERO / ManiSkill environment.

## Why separate setup is required

Upstream RoboCasa `v1.0` expects a different simulator stack than this repo's shared `.venv`:

- `numpy==2.2.5`
- `mujoco==3.3.1`
- `robosuite>=1.5.2`

The current workspace uses the versions required for the existing LIBERO and ManiSkill flows instead, so RoboCasa should be treated as an **isolated environment**.

## Stage source and assets

From the repo root:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/setup_robocasa.py --install
```

That command downloads:

- the official RoboCasa source archive into `.robocasa-src/`
- the current Hugging Face kitchen asset packs into `.robocasa-src/robocasa/models/assets/`
- a local `macros_private.py` file next to the upstream `macros.py`
- the local generative-texture / fixture-registry patch via `scripts/fix_robocasa_gentex.py`

## What the script checks

After staging the files, the script probes the active Python environment and reports:

- whether the workspace files are present (`Workspace ready`)
- whether the active Python stack matches RoboCasa's upstream requirements
- whether the robosuite composite-controller API is available

Typical output in the shared repo environment is:

- `Workspace ready: True`
- compatibility failures for `numpy` / `robosuite`

That means the workspace files are staged correctly, but you still need a dedicated Python environment to actually import and run RoboCasa.

## RL training status

Repo-native RoboCasa RL is now wired into `scripts/train_srpo.py`, but the supported mode is currently `sparse_rl`.

Example:

```powershell
$env:PYTHONPATH='src'
.robocasa-venv\Scripts\python.exe scripts/train_srpo.py `
  --simulator robocasa `
  --mode sparse_rl `
  --env PickPlaceCounterToCabinet `
  --robocasa-layout 20 `
  --robocasa-style 58 `
  --iterations 10 `
  --trajs-per-task 2 `
  --num-rollout-envs 1 `
  --checkpoint HuggingFaceVLA/smolvla_libero
```

`srpo` is still blocked for RoboCasa because this repo does not yet have a RoboCasa demo-seeding / replay pipeline equivalent to the existing LIBERO flow.

## Recommended next step

Create a separate Python `3.11` environment for RoboCasa, then point it at the staged source tree:

```powershell
$env:ROBOCASA_SOURCE_PATH = (Resolve-Path .\.robocasa-src)
```

Use the upstream RoboCasa installation guidance for the dedicated environment:

- RoboCasa GitHub README: https://github.com/robocasa/robocasa
- RoboCasa docs: https://robocasa.ai/docs/build/html/index.html

The dedicated environment should satisfy the upstream version requirements before attempting to import RoboCasa.
