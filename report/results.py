"""Experiment results loaded from runs/, shared by the report generators (generate.py, ieee.py)."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


sweep = pd.read_csv(RUNS / "sweep.csv")
smoke = pd.read_csv(RUNS / "smoke.csv")
validate = pd.read_csv(RUNS / "validate.csv").set_index("model")
log = (RUNS / "train.log").read_text(errors="replace")
epochs = [
    tuple(map(float, m))
    for m in re.findall(r"^epoch (\d+): loss (\S+) train acc (\S+) LFW (\S+) \+/- (\S+)", log, re.M)
]
# Held-out-identity verification accuracy per epoch; train.py keeps the first maximum.
val = [float(v) for v in re.findall(r"^epoch \d+: .* val (\S+) best", log, re.M)]
saved = val.index(max(val))  # 0-based epoch of the saved surrogate
header = re.search(r"(\d+) images, (\d+) identities, (\d+) steps/epoch", log).groups()

MODELS = ["surrogate", "casia", "vggface2", "arcface", "lbph"]

agg = sweep.groupby(["condition", "strength", "model"]).agg(m=("unmatched", "mean"), s=("unmatched", "std"))
ssim = sweep[sweep.model == "surrogate"].groupby(["condition", "strength"]).ssim.mean()
n_pairs = 3000  # all same-person pairs of View 2
n_folds = sweep.fold.nunique()


def pg(e, m):
    """Unmatched share for the main sweep (surrogate attacker, PGD)."""
    return agg.loc[("pgd", e, m)].m


# Robustness: the main sweep (seed 0) plus two more seeds, surrogate as attacker.
seeds = pd.concat(
    [
        sweep.assign(attacker="surrogate", seed=0),
        *(pd.read_csv(RUNS / "sweeps" / f"surrogate_seed{k}.csv") for k in (1, 2)),
    ]
)
spread = (
    seeds.groupby(["condition", "strength", "model", "seed"])
    .unmatched.mean()
    .groupby(["condition", "strength", "model"])
    .agg(["mean", "std"])
)
max_seed_std = spread["std"].groupby("condition").max()
arc_seed_std = spread.xs("arcface", level="model")["std"].max()

# Every differentiable model as the attacker (seed 0).
ATTACKERS = ["surrogate", "casia", "vggface2", "arcface"]
att = pd.concat(
    [
        sweep.assign(attacker="surrogate", seed=0),
        *(pd.read_csv(RUNS / "sweeps" / f"attacker_{a}.csv") for a in ATTACKERS[1:]),
    ]
)
xg = att.groupby(["condition", "strength", "attacker", "model"]).unmatched
xm, xs = xg.mean(), xg.std()
xssim = att[att.model == "surrogate"].groupby(["condition", "strength", "attacker"]).ssim.mean()


def X(e, a, t, c="pgd"):
    """Unmatched share of target t when the edit was built against attacker a."""
    return xm.loc[(c, e, a, t)]


off_diag_eps4 = max((X(4.0, a, t), a, t) for a in ATTACKERS for t in ATTACKERS if a != t)
best_vs_arc = max(ATTACKERS[:3], key=lambda a: X(16.0, a, "arcface"))

# Example edits (report/examples.py).
EX = RUNS / "examples"
exs = pd.read_csv(EX / "examples.csv")
people = exs.groupby("example").person.first()


def fooled(k, att, e):
    c = exs[(exs.example == k) & (exs.attacker == att) & (exs.eps == e) & ~exs.matched]
    return ", ".join(c.target) or "none"


def matrix_rows(eps, cond="pgd"):
    rows = [["Attacker ↓ · target →"] + MODELS + ["SSIM"]]
    hl = []
    for i, a in enumerate(ATTACKERS):
        rows.append(
            [a]
            + [f"{pct(X(eps, a, t, cond))} ± {pct(xs.loc[(cond, eps, a, t)])}" for t in MODELS]
            + [f"{xssim.loc[(cond, eps, a)]:.3f}"]
        )
        hl.append((i + 1, 1 + MODELS.index(a)))
    return rows, hl


def attacker_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, a in zip(axes, ATTACKERS, strict=True):
        for t in MODELS:
            g = xm.loc["pgd"].xs(a, level="attacker").xs(t, level="model")
            ax.plot(
                g.index, g.values, marker="o", label=t, lw=2.6 if t == a else 1.4, color=f"C{sorted(MODELS).index(t)}"
            )
        ax.set(title=f"attacker: {a}", xlabel="ε (/255)", xscale="symlog", ylim=(-0.02, 1.02))
    axes[0].set_ylabel("unmatched share")
    axes[0].legend(title="target", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cond_table(cond, strength_label, fmt_strength, show_std=True, extra_ssim=True):
    strengths = sorted(sweep[sweep.condition == cond].strength.unique())
    rows = [[strength_label] + MODELS + (["SSIM"] if extra_ssim else [])]
    for s in strengths:
        r = [fmt_strength(s)]
        for m in MODELS:
            v = agg.loc[(cond, s, m)]
            r.append(f"{pct(v.m)} ± {pct(v.s)}" if show_std else pct(v.m))
        if extra_ssim:
            r.append(f"{ssim.loc[(cond, s)]:.3f}")
        rows.append(r)
    return rows


def eps_fmt(s):
    return f"{s:g}/255"
