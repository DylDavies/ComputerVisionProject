"""Threshold calibration, clean accuracy (validate) and the edit-budget sweep."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

from facetransfer import attack
from facetransfer.align import load_meta
from facetransfer.data import Pairs, load_images, load_pairs
from facetransfer.models import PUBLISHED_ACCURACY, FaceModel, load_model


def best_threshold(dist: np.ndarray, same: np.ndarray) -> float:
    """Threshold t maximising accuracy of the rule `same iff dist <= t`."""
    order = np.argsort(dist)
    d, s = dist[order], same[order]
    # Accepting the first k pairs: correct = same among them + different among the rest.
    correct = np.concatenate([[0], np.cumsum(s)]) + np.concatenate([[(~s).sum()], (~s).sum() - np.cumsum(~s)])
    k = int(np.argmax(correct))
    if k == 0:
        return float(d[0]) - 1e-6
    if k == len(d):
        return float(d[-1])
    return float((d[k - 1] + d[k]) / 2)


def fold_thresholds(dist: np.ndarray, same: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Standard LFW protocol: each fold's threshold is fitted on the other nine."""
    return np.array([best_threshold(dist[fold != f], same[fold != f]) for f in np.unique(fold)])


def fold_accuracy(dist: np.ndarray, same: np.ndarray, fold: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    correct = (dist <= thresholds[fold]) == same
    return np.array([correct[fold == f].mean() for f in np.unique(fold)])


def batches(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def to_float(images_u8: torch.Tensor, device: str) -> torch.Tensor:
    return images_u8.to(device).float() / 255


@torch.no_grad()
def embed_all(model: FaceModel, images_u8: torch.Tensor, affines: torch.Tensor, device: str, bs: int = 256):
    return torch.cat(
        [model.embed(to_float(images_u8[s], device), affines[s].to(device)) for s in batches(len(images_u8), bs)]
    )


def pair_distances(model: FaceModel, emb: torch.Tensor, pairs: Pairs) -> np.ndarray:
    return model.distance(emb[pairs.a], emb[pairs.b]).cpu().numpy()


def ssim(a_u8: torch.Tensor, b_u8: torch.Tensor) -> np.ndarray:
    a, b = a_u8.permute(0, 2, 3, 1).numpy(), b_u8.permute(0, 2, 3, 1).numpy()
    return np.array([structural_similarity(x, y, channel_axis=-1) for x, y in zip(a, b, strict=True)])


class Workspace:
    """LFW pairs, photos, cached face geometry and the per-model affines derived from it."""

    def __init__(self, data_dir: Path, runs_dir: Path, device: str):
        self.data_dir, self.runs_dir, self.device = data_dir, runs_dir, device
        self.pairs = load_pairs(data_dir)
        self.images = load_images(self.pairs.paths)
        self.meta = load_meta(data_dir / "lfw_meta.npz")

    def model(self, name: str) -> tuple[FaceModel, torch.Tensor]:
        model = load_model(name, self.device, self.runs_dir, self.data_dir / "weights")
        affines = torch.from_numpy(model.crop(self.meta, tuple(self.images.shape[2:])))
        return model, affines


def clean_accuracy(model: FaceModel, affines: torch.Tensor, ws: Workspace) -> np.ndarray:
    dist = pair_distances(model, embed_all(model, ws.images, affines, ws.device), ws.pairs)
    p = ws.pairs
    return fold_accuracy(dist, p.same, p.fold, fold_thresholds(dist, p.same, p.fold))


def validate(ws: Workspace, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        model, affines = ws.model(name)
        acc = clean_accuracy(model, affines, ws)
        rows.append(
            {"model": name, "accuracy": acc.mean(), "std": acc.std(), "published": PUBLISHED_ACCURACY.get(name)}
        )
        print(f"{name:>10}: {acc.mean():.4f} +/- {acc.std():.4f}  (published {rows[-1]['published']})")
    print(f"MTCNN misses (central fallback used): {(~ws.meta['detected']).sum()} / {len(ws.meta['detected'])}")
    return pd.DataFrame(rows)


def sweep(
    ws: Workspace,
    names: list[str],
    attacker: str,
    eps_list: list[float],
    jpeg_qualities: list[int],
    blur_sigmas: list[float],
    steps: int,
    limit: int | None,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Edit photo `a` of each same-person pair; record the share each model stops matching, per fold."""
    torch.manual_seed(seed)
    p = ws.pairs
    targets = np.flatnonzero(p.same)
    if limit:
        targets = np.sort(np.random.default_rng(seed).permutation(targets)[:limit])
    a, b, fold = p.a[targets], p.b[targets], p.fold[targets]

    # Clean embeddings and fold thresholds for every test model.
    models = {}
    for name in names:
        model, affines = ws.model(name)
        emb = embed_all(model, ws.images, affines, ws.device)
        dist = pair_distances(model, emb, p)
        models[name] = (model, affines, emb[b], fold_thresholds(dist, p.same, p.fold))
    attack_model, attack_affines = (models[attacker][:2]) if attacker in models else ws.model(attacker)

    clean = ws.images[a]
    rows = []

    def record(condition: str, strength: float, edited: torch.Tensor) -> None:
        sims = ssim(edited, clean)
        for name, (model, affines, emb_b, thr) in models.items():
            emb_a = embed_all(model, edited, affines[a], ws.device)
            unmatched = model.distance(emb_a, emb_b).cpu().numpy() > thr[fold]
            for f in np.unique(fold):
                m = fold == f
                rows.append(
                    {
                        "model": name,
                        "condition": condition,
                        "strength": strength,
                        "fold": int(f),
                        "unmatched": unmatched[m].mean(),
                        "ssim": sims[m].mean(),
                    }
                )

    def edit(fn, source: torch.Tensor = clean) -> torch.Tensor:
        out = [
            attack.quantise(fn(to_float(source[s], ws.device), attack_affines[a[s]].to(ws.device))).cpu()
            for s in batches(len(source), batch_size)
        ]
        return torch.cat(out)

    for eps255 in tqdm(eps_list, desc="eps"):
        eps = eps255 / 255
        edited = edit(lambda x, m, eps=eps: attack.pgd(attack_model, x, m, eps, steps=steps))
        record("pgd", eps255, edited)
        record("pgd+jpeg75", eps255, edit(lambda x, m: attack.jpeg(x, 75), source=edited))
        record("noise", eps255, edit(lambda x, m, eps=eps: attack.random_noise(x, eps)))
    for q in jpeg_qualities:
        record("jpeg", q, edit(lambda x, m, q=q: attack.jpeg(x, q)))
    for sigma in blur_sigmas:
        record("blur", sigma, edit(lambda x, m, sigma=sigma: attack.blur(x, sigma)))
    return pd.DataFrame(rows)
