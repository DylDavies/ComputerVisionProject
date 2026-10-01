# Do edits that fool one face recognition model also fool models they were never built for?

COMS4036A project. We train a face recognition network from scratch (the **surrogate**), edit one photo in each
LFW View 2 same-person pair with PGD against it, and measure how often four models we never used while editing
also stop matching the pair.

| Model | Role | Differs from surrogate in |
|---|---|---|
| `surrogate` | InceptionResnetV1, softmax CE, CASIA-WebFace, trained here | — (only model used to build edits) |
| `vggface2` | facenet-pytorch InceptionResnetV1 | training faces |
| `casia` | facenet-pytorch InceptionResnetV1 | nothing but the seed |
| `arcface` | InsightFace buffalo_l ResNet-50 (WebFace600K) | architecture, objective, faces |
| `lbph` | Local Binary Pattern Histograms | no deep learning |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.8 wheels; works on RTX 50xx).

```bash
uv sync
```

`facenet-pytorch` pins an old torch/numpy; `pyproject.toml` overrides those pins (see `[tool.uv]`).

## Data

- **LFW + View 2 pairs** and all pretrained weights: downloaded by `uv run ft prepare` (figshare mirror, the UMass host
  is often down). The six known mislabelled pairs are left in, as the maintainers ask.
- **CASIA-WebFace** (surrogate training): download a 112x112 ArcFace-aligned JPEG copy (e.g. from Kaggle) and place it
  as `data/casia/<identity>/<image>.jpg`. It must be aligned with the standard InsightFace 5-point template, which is
  the geometry the surrogate also sees on LFW.

## Pipeline

```bash
uv run ft prepare     # LFW, MTCNN boxes/landmarks cached to data/lfw_meta.npz, weights
uv run ft validate --models vggface2 casia arcface lbph   # clean accuracy vs published: run before any edits
uv run ft train       # surrogate -> runs/surrogate.pt (prints LFW accuracy each epoch)
uv run ft sweep       # all conditions -> runs/sweep.csv
uv run ft plot        # runs/figures/*.png
```

Smoke test without the surrogate: `uv run ft sweep --attacker vggface2 --models vggface2 casia --limit 100 --eps 0 8`.

## Design

- **One geometry path.** MTCNN runs once on each clean photo. Every model's crop (facenet box crop at 160px with margin 32,
  ArcFace/LBPH 5-point alignment at 112px, surrogate alignment at 160px) is a 2x3 affine applied by
  `align.warp` (`grid_sample`). Edits are made on the full 250x250 photo, and each model crops it its own way;
  the surrogate's crop is differentiable, so it sits inside the PGD graph. Landmarks come from the clean photo.
- **Thresholds.** Each model gets its own threshold per fold, fitted on the other nine folds of clean pairs
  (standard LFW protocol).
- **Threat model.** PGD (L-inf, random start) pushes the edited photo away from *its own* clean embedding: the
  user doesn't know which photo it will be compared against. Edited photos are quantised to uint8 before scoring.
- **Conditions** (`runs/sweep.csv`, one row per model/condition/strength/fold): `pgd`, `noise` (random +/-eps),
  `pgd+jpeg75` (compression after editing), `jpeg` (quality), `blur` (sigma). `pgd` at eps 0 is the clean baseline.
  SSIM against the clean photo is recorded for every row.

```
src/facetransfer/
  align.py     detection cache, affines, warp
  data.py      LFW / CASIA download and loading
  models.py    the five models behind one interface
  train.py     surrogate training
  attack.py    PGD and baseline edits
  evaluate.py  thresholds, validate, sweep
  plots.py     figures
  cli.py       `ft` command
tests/         uv run pytest
```
