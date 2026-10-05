"""Example edits for the report: attacker x epsilon image grids, scored on every target model.

Picks same-person pairs that every deep model matches before editing, edits photo `a` with PGD for each
attacker at each budget (seed 0, same settings as the sweep), and records whether each target still
matches the edited `a` to the clean `b` under that model's fold threshold.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from facetransfer import attack
from facetransfer.evaluate import Workspace, embed_all, fold_thresholds, pair_distances, ssim, to_float

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "examples"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ["surrogate", "casia", "vggface2", "arcface", "lbph"]
TAG = {"surrogate": "S", "casia": "C", "vggface2": "V", "arcface": "A", "lbph": "L"}
ATTACKERS = ["surrogate", "casia", "vggface2", "arcface"]
EPS = [0, 2, 4, 8, 16]
N_EXAMPLES = 3
CROP = (slice(30, 220), slice(30, 220))  # LFW photos are centred on the face; show the face region

ws = Workspace(ROOT / "data", ROOT / "runs", "cuda")
p = ws.pairs
models = {}
for name in TARGETS:
    model, affines = ws.model(name)
    emb = embed_all(model, ws.images, affines, ws.device)
    dist = pair_distances(model, emb, p)
    models[name] = dict(model=model, affines=affines, emb=emb, dist=dist, thr=fold_thresholds(dist, p.same, p.fold))

# Same-person pairs every deep model gets right before editing; one per fold, different people.
ok = p.same.copy()
for name in ["surrogate", "casia", "vggface2", "arcface"]:
    ok &= models[name]["dist"] <= models[name]["thr"][p.fold]
rng = np.random.default_rng(0)
chosen = []
for f in rng.permutation(10):
    idx = rng.permutation(np.flatnonzero(ok & (p.fold == f)))[0]
    chosen.append(int(idx))
    if len(chosen) == N_EXAMPLES:
        break


def person(i):
    return p.paths[p.a[i]].parent.name.replace("_", " ")


rows = []
for k, i in enumerate(chosen):
    a, b, fold = p.a[i], p.b[i], p.fold[i]
    clean = ws.images[a : a + 1]
    edited = {}
    for att in ATTACKERS:
        model, affines = models[att]["model"], models[att]["affines"]
        for e in EPS:
            torch.manual_seed(0)
            x = attack.pgd(model, to_float(clean, ws.device), affines[a : a + 1].to(ws.device), e / 255, steps=20)
            edited[(att, e)] = attack.quantise(x).cpu()
            s = float(ssim(edited[(att, e)], clean)[0])
            for t in TARGETS:
                m = models[t]
                ea = embed_all(m["model"], edited[(att, e)], m["affines"][a : a + 1], ws.device)
                d = float(m["model"].distance(ea, m["emb"][b : b + 1]).item())
                rows.append(
                    dict(
                        example=k,
                        person=person(i),
                        attacker=att,
                        eps=e,
                        target=t,
                        distance=d,
                        threshold=float(m["thr"][fold]),
                        matched=d <= m["thr"][fold],
                        ssim=s,
                    )
                )
    df = pd.DataFrame(rows)
    ex = df[df.example == k]

    def show(ax, img_u8, title=None):
        ax.imshow(img_u8[0].permute(1, 2, 0).numpy()[CROP])
        ax.set_xticks([]), ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)

    def tags(ax, ex, att, e):
        cell = ex[(ex.attacker == att) & (ex.eps == e)].set_index("target")
        for j, t in enumerate(TARGETS):
            fooled = not cell.loc[t, "matched"]
            x0 = 0.02 + j * 0.196
            ax.add_patch(
                Rectangle(
                    (x0, -0.2),
                    0.18,
                    0.15,
                    transform=ax.transAxes,
                    clip_on=False,
                    facecolor="#c0392b" if fooled else "#e8e8e8",
                    edgecolor="#555555",
                    lw=0.5,
                )
            )
            ax.text(
                x0 + 0.09,
                -0.125,
                TAG[t],
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if fooled else "#333333",
                fontweight="bold",
            )
        ax.text(
            0.5,
            -0.29,
            f"SSIM {cell['ssim'].iloc[0]:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=6.5,
            color="#444444",
        )

    # Figure 1: edited photos with target tags.
    fig, axes = plt.subplots(len(ATTACKERS), len(EPS) + 1, figsize=(11, 10.2))
    for r, att in enumerate(ATTACKERS):
        show(axes[r, 0], ws.images[b : b + 1], "comparison photo b" if r == 0 else None)
        axes[r, 0].set_ylabel(f"attacker:\n{att}", fontsize=10)
        for c, e in enumerate(EPS):
            show(axes[r, c + 1], edited[(att, e)], (f"ε = {e}/255" if e else "original a (ε = 0)") if r == 0 else None)
            tags(axes[r, c + 1], ex, att, e)
    fig.suptitle(f"{person(i)}: photo a edited by PGD against each attacker", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.04, wspace=0.08, hspace=0.42)
    fig.savefig(OUT / f"example_{k}.png", dpi=170)
    plt.close(fig)

    # Figure 2: the perturbations alone, each scaled to its own budget so the pattern is visible.
    fig, axes = plt.subplots(len(ATTACKERS), len(EPS) - 1, figsize=(9, 9.4))
    for r, att in enumerate(ATTACKERS):
        axes[r, 0].set_ylabel(f"attacker:\n{att}", fontsize=10)
        for c, e in enumerate(EPS[1:]):
            delta = (edited[(att, e)].float() - clean.float())[0] / e  # in [-1, 1]
            axes[r, c].imshow(((delta * 0.5 + 0.5).clamp(0, 1)).permute(1, 2, 0).numpy()[CROP])
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"ε = {e}/255", fontsize=9)
    fig.suptitle(f"{person(i)}: the edit alone (edited − original, scaled by 1/ε)", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.02, wspace=0.05, hspace=0.08)
    fig.savefig(OUT / f"perturbation_{k}.png", dpi=170)
    plt.close(fig)
    print("example", k, person(i), "pair", i, "fold", fold)

pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)
print(
    pd.DataFrame(rows)
    .pivot_table(index=["example", "attacker", "eps"], columns="target", values="matched")
    .astype(int)
    .to_string()
)
