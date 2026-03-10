import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


from smolvla_language_pilot.language_class import (
        LanguageRunResult,
        load_llm_bundle,
        load_policy_bundle,
        run_language_sensitivity_for_rollout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run language sensitivity via language_class.py and save LLM_language-style plots."
    )
    parser.add_argument("--rollout", default="smolvla_language_pilot/rollout.h5")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-variants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="smolvla_language_pilot/outputs")
    return parser.parse_args()


def plot_heatmap(result: LanguageRunResult, output_dir: Path) -> Path:
    type_labels = list(result.variant_type_means.keys())
    type_curves = torch.stack([result.variant_type_means[k] for k in type_labels])

    plt.figure(figsize=(10, 5))
    plt.imshow(type_curves.numpy(), aspect="auto")
    plt.yticks(range(len(type_labels)), type_labels)
    plt.colorbar(label="Relative L2")
    plt.xlabel("Timestep")
    plt.ylabel("Variant")
    plt.title("Language Variant Sensitivity Heatmap")
    plt.tight_layout()

    path = output_dir / "language_heatmap.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_variant_type_mean(result: LanguageRunResult, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    x = np.arange(len(result.mean_curve))

    for variant_type, curve in result.variant_type_means.items():
        std = result.variant_type_stds[variant_type]
        plt.plot(x, curve.numpy(), label=variant_type)
        plt.fill_between(x, (curve - std).numpy(), (curve + std).numpy(), alpha=0.2)

    plt.xlabel("Timestep")
    plt.ylabel("Relative Action Divergence")
    plt.title("Mean Divergence per Instruction Variant Type")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "language_variant_type_mean.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_boxplot(result: LanguageRunResult, output_dir: Path) -> Path:
    plt.figure(figsize=(6, 5))
    plt.boxplot(result.boxplot_data.values(), labels=result.boxplot_data.keys())
    plt.ylabel("Trajectory Mean Divergence")
    plt.title("Language Robustness per Instruction Type")
    plt.tight_layout()

    path = output_dir / "language_boxplot.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_lss(result: LanguageRunResult, output_dir: Path) -> Path:
    plt.figure(figsize=(6, 5))
    plt.bar(result.lss_scores.keys(), result.lss_scores.values())
    plt.ylabel("Language Sensitivity Score")
    plt.title("Language Sensitivity by Instruction Type")
    plt.tight_layout()

    path = output_dir / "language_lss.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_divergence_with_frame(result: LanguageRunResult, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    x = np.arange(len(result.overall_mean_curve))
    for variant_type, curve in result.variant_type_means.items():
        ax1.plot(x, curve.numpy(), label=variant_type)

    ax1.axvline(result.peak_timestep, color="black", linestyle="--", label="peak divergence")
    ax1.set_ylabel("Relative Action Divergence")
    ax1.set_title("Language Sensitivity with Peak Frame")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(result.peak_frame)
    ax2.set_title(f"Rollout frame at timestep {result.peak_timestep}")
    ax2.axis("off")

    plt.tight_layout()
    path = output_dir / "language_divergence_with_frame.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading policy and LLM bundles...")
    policy_bundle = load_policy_bundle(checkpoint=args.checkpoint, device=args.device)
    llm_bundle = load_llm_bundle(llm_model=args.llm_model)

    print("Running rollout language sensitivity...")
    result = run_language_sensitivity_for_rollout(
        rollout_path=args.rollout,
        policy_bundle=policy_bundle,
        llm_bundle=llm_bundle,
        n_variants=args.n_variants,
        seed=args.seed,
    )

    print("Saving plots...")
    saved = [
        plot_heatmap(result, output_dir),
        plot_variant_type_mean(result, output_dir),
        plot_boxplot(result, output_dir),
        plot_lss(result, output_dir),
        plot_divergence_with_frame(result, output_dir),
    ]

    print("Language Sensitivity Score (LSS):")
    for variant_type, score in result.lss_scores.items():
        print(f"- {variant_type}: {score:.3f}")

    print("Saved files:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
