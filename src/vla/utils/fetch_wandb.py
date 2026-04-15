import json

import typer
import wandb
from tqdm import tqdm

from vla.constants import RESULTS_DIR
from vla.results_registry import sanitize_name

app = typer.Typer(help="Fetch runs from WandB and reconstruct local JSON results.")


def reconstruct_eval_record(run) -> dict:
    """Map a WandB eval run back to the local eval_record JSON format."""
    cfg = run.config
    summary = run.summary._json_dict
    suite = cfg.get("suite", "spatial")

    # Reconstruct the task_metrics list from the flat summary keys
    task_metrics = []
    for k, v in summary.items():
        if k.startswith(f"eval/{suite}/task_") and k.endswith("/success_rate"):
            # extract task_id string "0", "1", etc.
            part = k.split("/")[2]
            task_id_str = part.replace("task_", "")
            try:
                task_id = int(task_id_str)
                task_metrics.append({"task_id": task_id, "success_rate": v})
            except ValueError:
                pass

    # Sort for consistency
    task_metrics = sorted(task_metrics, key=lambda x: x["task_id"])

    return {
        "record_type": "evaluation",
        "wandb_run_name": run.name,
        "checkpoint": cfg.get("checkpoint", ""),
        "simulator": cfg.get("simulator", "libero"),
        "suite": suite,
        "success_rate": summary.get(f"eval/{suite}/overall/success_rate", 0.0),
        "mean_reward": summary.get(f"eval/{suite}/overall/mean_reward", 0.0),
        "task_metrics": task_metrics,
        # Plot_results relies on these for labeling
        "training_save_dir": cfg.get("checkpoint_dir") or "",
        "training_method": _determine_eval_method(run.name, cfg),
    }


def _determine_eval_method(run_name: str, cfg: dict) -> str:
    """Robustly determine if an eval run is SFT or Sparse RL based on wandb metadata."""
    name_lower = run_name.lower()
    ckpt_dir = str(cfg.get("checkpoint_dir", "")).lower()

    if cfg.get("method"):
        return cfg.get("method")

    if "sft" in name_lower or "sft" in ckpt_dir or not ckpt_dir:
        return "sft"
    elif "srpo" in name_lower or "srpo" in ckpt_dir:
        return "srpo"

    return "sparse_rl"


def reconstruct_training_record(run) -> dict:
    """Map a WandB training run back to the local training_record JSON format."""
    return {
        "record_type": "training",
        "wandb_run_name": run.name,
        "method": run.config.get("mode", run.config.get("method", "sparse_rl")),
        "save_dir": run.config.get("save_dir", ""),
        "checkpoint": run.config.get("checkpoint", ""),
        "suite": run.config.get("suite", ""),
        "config": run.config,
    }


@app.command()
def sync(
    project: str = typer.Argument(..., help="WandB project (e.g., vla-libero-eval or srpo-smolvla)"),
    entity: str = typer.Option(None, "--entity", "-e", help="WandB entity/username"),
    record_type: str = typer.Option("eval", "--type", "-t", help="Dataset type: 'eval' or 'training'"),
):
    """
    Fetch all runs from a WandB project and recreate them as local JSON files
    in results/evals/ or results/training/ for offline plotting.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    typer.echo(f"Fetching {record_type} runs from {path}...")

    try:
        runs = api.runs(path)
    except Exception as e:
        typer.secho(f"Failed to fetch runs: {e}", fg=typer.colors.RED)
        raise typer.Exit(1) from e

    if not runs:
        typer.secho("No runs found.", fg=typer.colors.YELLOW)
        return

    out_folder = RESULTS_DIR / ("evals" if record_type == "eval" else "training")
    out_folder.mkdir(parents=True, exist_ok=True)
    count = 0

    for run in tqdm(runs, desc=f"Writing JSONs to {out_folder.name}/"):
        name = sanitize_name(run.name)
        if not name:
            continue

        json_path = out_folder / f"{name}.json"

        record = reconstruct_eval_record(run) if record_type == "eval" else reconstruct_training_record(run)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        count += 1

    typer.secho(f"\nSuccessfully synced {count} {record_type} runs cleanly to:", fg=typer.colors.GREEN)
    typer.echo(f"  {out_folder}/")


if __name__ == "__main__":
    app()
