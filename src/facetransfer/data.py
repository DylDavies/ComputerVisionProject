"""Dataset download and loading: LFW View 2 pairs (test) and CASIA-WebFace (surrogate training)."""

import tarfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from torchvision.io import ImageReadMode, decode_image, decode_jpeg, read_file
from tqdm import tqdm

# The UMass LFW host is frequently offline; these are the maintainers' figshare mirrors.
LFW_IMAGES_URL = "https://ndownloader.figshare.com/files/5976018"  # lfw.tgz
LFW_PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"  # pairs.txt (View 2)


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def download_lfw(data_dir: Path) -> None:
    tgz = download(LFW_IMAGES_URL, data_dir / "lfw.tgz")
    download(LFW_PAIRS_URL, data_dir / "pairs.txt")
    if not (data_dir / "lfw").exists():
        with tarfile.open(tgz) as t:
            t.extractall(data_dir, filter="data")


@dataclass
class Pairs:
    """LFW View 2: 10 folds x (300 same + 300 different) pairs, as published (mislabels left in)."""

    paths: list[Path]  # unique images referenced by the pairs
    a: np.ndarray  # index into paths
    b: np.ndarray
    same: np.ndarray  # bool
    fold: np.ndarray  # 0..9

    def __len__(self) -> int:
        return len(self.a)


def load_pairs(data_dir: Path) -> Pairs:
    lines = (data_dir / "pairs.txt").read_text().split("\n")
    n_folds, n_per = map(int, lines[0].split())

    def img(name: str, i: str) -> Path:
        return data_dir / "lfw" / name / f"{name}_{int(i):04d}.jpg"

    left, right, same = [], [], []
    for line in lines[1 : 1 + 2 * n_folds * n_per]:
        parts = line.split()
        if len(parts) == 3:
            left.append(img(parts[0], parts[1]))
            right.append(img(parts[0], parts[2]))
            same.append(True)
        else:
            left.append(img(parts[0], parts[1]))
            right.append(img(parts[2], parts[3]))
            same.append(False)
    paths = sorted(set(left) | set(right))
    index = {p: i for i, p in enumerate(paths)}
    return Pairs(
        paths=paths,
        a=np.array([index[p] for p in left]),
        b=np.array([index[p] for p in right]),
        same=np.array(same),
        fold=np.repeat(np.arange(n_folds), 2 * n_per),
    )


def load_images(paths: list[Path]) -> torch.Tensor:
    """Decode photos to a uint8 tensor [N,3,H,W] (LFW photos are all 250x250)."""
    return torch.stack([decode_image(str(p), mode=ImageReadMode.RGB) for p in tqdm(paths, desc="decode")])


class CasiaLoader:
    """CASIA-WebFace batches as (uint8 images [B,3,112,112] on device, labels), one folder per identity.

    Images are pre-aligned to 112x112. Files are read and decoded on a thread pool, one batch ahead.
    No DataLoader workers: on Windows each is a full process holding ~1.5GB of commit memory, which
    exhausted the paging file. Not nvJPEG either: torchvision's GPU decode races with queued training
    kernels on the default stream and corrupted the weights (NaN loss within a few steps).
    """

    def __init__(self, root: Path, batch_size: int, device: str, threads: int = 16, holdout: int = 0, seed: int = 0):
        if not root.is_dir():
            raise SystemExit(f"CASIA-WebFace not found at {root}: expected {root}/<identity>/<image>.jpg (see README)")
        everyone = sorted(d.name for d in root.iterdir() if d.is_dir())
        # `holdout` identities (fixed by seed) are never trained on: they form the open-set validation set.
        held = set(np.random.default_rng(seed).permutation(everyone)[:holdout].tolist())
        self.classes = [c for c in everyone if c not in held]
        self.held, self.root = sorted(held), root
        self.paths, labels = [], []
        for label, name in enumerate(tqdm(self.classes, desc="scan")):
            files = sorted((root / name).glob("*.jpg"))
            self.paths += [str(p) for p in files]
            labels += [label] * len(files)
        self.labels = torch.tensor(labels)
        self.batch_size, self.device = batch_size, device
        self.readers = ThreadPoolExecutor(threads)
        self.prefetcher = ThreadPoolExecutor(1)

    def __len__(self) -> int:
        return len(self.paths) // self.batch_size  # drop the last partial batch

    def val_set(self, per_id: int = 10, n_pairs: int = 3000, seed: int = 0):
        """Held-out identities as fixed verification pairs: (images uint8 [N,3,112,112], a, b, same)."""
        files, ids = [], []
        for i, name in enumerate(self.held):
            found = sorted((self.root / name).glob("*.jpg"))[:per_id]
            files += [str(p) for p in found]
            ids += [i] * len(found)
        ids = np.array(ids)
        rng = np.random.default_rng(seed)
        # Same pairs: pick an image, then another image of the same identity (random draws almost never collide).
        first = {i: np.flatnonzero(ids == i) for i in np.unique(ids)}
        a_same = rng.choice(np.flatnonzero(np.bincount(ids)[ids] > 1), n_pairs)
        b_same = np.array([rng.choice(first[ids[x]][first[ids[x]] != x]) for x in a_same])
        a, b = rng.choice(len(ids), (2, n_pairs))
        ok = ids[a] != ids[b]
        a_diff, b_diff = a[ok], b[ok]
        same = np.r_[np.ones(len(a_same), bool), np.zeros(len(a_diff), bool)]
        images = torch.stack(list(self.readers.map(self._decode, files)))
        return images, np.r_[a_same, a_diff], np.r_[b_same, b_diff], same

    @staticmethod
    def _decode(path: str) -> torch.Tensor:
        return decode_jpeg(read_file(path), mode=ImageReadMode.RGB)

    def _load(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.stack(list(self.readers.map(self._decode, [self.paths[i] for i in idx]))).pin_memory()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        order = torch.randperm(len(self.paths))
        batches = [order[k * self.batch_size : (k + 1) * self.batch_size] for k in range(len(self))]
        pending = self.prefetcher.submit(self._load, batches[0])
        for k, idx in enumerate(batches):
            images = pending.result()
            if k + 1 < len(batches):
                pending = self.prefetcher.submit(self._load, batches[k + 1])
            yield images.to(self.device, non_blocking=True), self.labels[idx].to(self.device)
