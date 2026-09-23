"""Figures from runs/sweep.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _mean_std(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["model", "strength"]).agg(
        unmatched=("unmatched", "mean"), std=("unmatched", "std"), ssim=("ssim", "mean")
    )


def main_result(df: pd.DataFrame, out: Path) -> None:
    """Share of same-person pairs unmatched vs edit budget, one line per model, SSIM on a second axis."""
    stats = _mean_std(df[df.condition == "pgd"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in stats.groupby("model"):
        ax.errorbar(g.strength, g.unmatched, yerr=g["std"], marker="o", capsize=3, label=model)
    ax.set(xlabel="edit budget ε (L∞, /255)", ylabel="same-person pairs no longer matched", ylim=(-0.02, 1.02))
    ax.set_xscale("symlog", linthresh=1)
    ax2 = ax.twinx()
    ssim = stats.groupby("strength").ssim.mean()
    ax2.plot(ssim.index, ssim.values, "k--", alpha=0.6, label="SSIM")
    ax2.set(ylabel="SSIM (dashed)", ylim=(0, 1.02))
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(out / "main_result.png", dpi=200)
    plt.close(fig)


def edit_comparison(df: pd.DataFrame, out: Path) -> None:
    """PGD vs random noise vs PGD followed by JPEG, one panel per model."""
    conditions = ["pgd", "noise", "pgd+jpeg75"]
    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        for cond in conditions:
            g = _mean_std(df[(df.condition == cond) & (df.model == model)]).reset_index()
            ax.plot(g.strength, g.unmatched, marker="o", label=cond)
        ax.set(title=model, xlabel="ε (/255)", xscale="symlog")
    axes[0, 0].set_ylabel("unmatched share")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "edit_comparison.png", dpi=200)
    plt.close(fig)


def free_edits(df: pd.DataFrame, out: Path) -> None:
    """JPEG quality and blur sigma: what any user could do for free."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cond, label in [(axes[0], "jpeg", "JPEG quality"), (axes[1], "blur", "Gaussian blur σ (px)")]:
        stats = _mean_std(df[df.condition == cond]).reset_index()
        for model, g in stats.groupby("model"):
            ax.plot(g.strength, g.unmatched, marker="o", label=model)
        ax.set(xlabel=label, ylabel="unmatched share", title=cond)
    axes[0].invert_xaxis()
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "free_edits.png", dpi=200)
    plt.close(fig)


def plot_all(csv: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    main_result(df, out)
    edit_comparison(df, out)
    free_edits(df, out)
    print(f"figures written to {out}")
