# Do edits that fool one face recognition model also fool models they were never built for?

COMS4036A project. We train a face recognition network from scratch (the **surrogate**), edit one photo in each
LFW View 2 same-person pair with PGD against it, and measure how often four models we never used while editing
also stop matching the pair.

| Model | Role | Differs from surrogate in |
|---|---|---|
| `surrogate` | InceptionResnetV1, softmax CE, CASIA-WebFace, trained here | - |
| `vggface2` | facenet-pytorch InceptionResnetV1 | training faces |
| `casia` | facenet-pytorch InceptionResnetV1 | nothing but the seed |
| `arcface` | InsightFace buffalo_l ResNet-50 (WebFace600K), ONNX converted to torch | architecture, objective, faces |
| `lbph` | Local Binary Pattern Histograms | no deep learning |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.8 wheels; works on RTX 50xx).

```bash
uv sync
```

`facenet-pytorch` pins an old torch/numpy; `pyproject.toml` overrides those pins (see `[tool.uv]`).

## Data

- **LFW + View 2 pairs** and all pretrained weights: downloaded by `uv run ft prepare` (figshare mirror, the UMass host
  is often down). The published View 2 pairs are used unchanged, with no corrections.
- **CASIA-WebFace** (surrogate training; only `ft train` needs it, every other command works without it). This is
  not downloaded automatically. We used the Kaggle copy
  [`debarghamitraroy/casia-webface`](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) (3.4 GB; licence
  listed there as "Database: Open Database, Contents: Database Contents"). Download it from the page (Kaggle login
  required), or with the
  [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

  ```bash
  kaggle datasets download -d debarghamitraroy/casia-webface -p data --unzip
  ```

  Then arrange it as `data/casia/<identity>/<image>.jpg`, one folder per identity, directly under `data/casia/`
  (if the archive nests the identity folders one level deeper, move them up). Our copy has 112x112 JPEGs, aligned with
  the standard InsightFace 5-point template (the geometry the surrogate also sees on LFW), 10,572 identity folders and
  490,623 images; with the default 500 held-out identities `ft train` should start by printing
  `465498 images, 10072 identities`, so you can check yours. A differently aligned copy
  will lower the surrogate's accuracy without raising any error.

## Pipeline

```bash
uv run ft prepare     # LFW, MTCNN boxes/landmarks cached to data/lfw_meta.npz, weights
uv run ft validate --models vggface2 casia arcface lbph   # clean accuracy vs published: run before any edits
uv run ft train       # surrogate -> runs/surrogate.pt (prints LFW accuracy each epoch)
uv run ft sweep       # all conditions -> runs/sweep.csv
uv run ft plot        # runs/figures/*.png
uv run ft watch       # live training dashboard (run in a second terminal)
```

### Architecture control (ResNet-50 surrogate)

A second surrogate with the same data, loss, crop and training recipe but a torchvision ResNet-50 backbone, so it
differs from `surrogate` only in architecture. It is opt-in: `surrogate_r50` is not in the default `--models` list.

```bash
uv run ft train --arch resnet50 > runs/train_r50.log           # -> runs/surrogate_r50.pt
uv run ft validate --models surrogate surrogate_r50
uv run ft sweep --attacker surrogate_r50 --models surrogate surrogate_r50 casia vggface2 arcface lbph \
    --conditions pgd --out runs/sweeps/attacker_surrogate_r50.csv
uv run ft sweep --attacker surrogate --models surrogate_r50 --conditions pgd --out runs/sweeps/r50_target_surrogate.csv
# ... and the same for --attacker casia / vggface2 / arcface, to score surrogate_r50 as a target
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
