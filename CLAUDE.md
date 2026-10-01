# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

COMS4036A research project (proposal: `CV Project Proposal.pdf`). Question: do PGD edits built against one face
recognition model (a surrogate we train from scratch) also fool models never used while editing? Measured on LFW
View 2 same-person pairs, swept over L-inf edit budget, against 4 test models: facenet-pytorch InceptionResnetV1
(`vggface2`, `casia`), InsightFace ArcFace ONNX (`arcface`), and `lbph`.

## Commands

```bash
uv sync                                   # create/update .venv (Python 3.12)
uv run pytest                             # all tests
uv run pytest tests/test_core.py::test_pgd_respects_budget_and_moves_embedding   # single test
uv run ruff check && uv run ruff format   # lint / format (line length 120)
uv run ft prepare && uv run ft validate   # pipeline order: prepare, train, validate, sweep, plot (see README.md)
uv run ft sweep --attacker vggface2 --models vggface2 casia --limit 100 --eps 0 8   # smoke test, no surrogate needed
```

The user often has other jobs training on the GPU: ask before running anything GPU-heavy (`validate`, `train`, `sweep`).

## Architecture

- **Single geometry path (`align.py`).** MTCNN runs once on clean LFW photos → `data/lfw_meta.npz` (boxes, 5 landmarks).
  Every model's crop is a 2x3 pixel-space affine (photo → model input) applied by `warp()` via `grid_sample`.
  Edits are made on the full 250x250 photo; each model crops it its own way (`FaceModel.crop`). Because `warp` is
  differentiable, the surrogate's crop is inside the PGD graph. Landmarks from the clean photo are reused for edits.
  `box_affines` deliberately replicates facenet-pytorch's `extract_face` (margin in output px, int truncation).
- **Model interface (`models.py`).** `FaceModel.embed(images_float01 [B,3,H,W], affines) -> embeddings` plus
  `distance()` (cosine for deep models, chi-square for LBPH). Normalisation lives inside each model. Only
  `TorchFaceNet` is differentiable; `ArcFace` (onnxruntime) and `LBPH` are query-only. `load_model(name, ...)` is the registry.
- **Protocol (`evaluate.py`).** Per model, per fold thresholds fitted on the other 9 folds of *clean* pairs;
  these thresholds are then reused for edited pairs. `sweep` edits photo `a` of each same-person pair, quantises to
  uint8, and records per model/condition/strength/fold the unmatched share + SSIM → `runs/sweep.csv`.
  PGD target = the photo's own clean embedding (attacker doesn't know the comparison photo — a confirmed design decision).
- **Surrogate (`train.py`).** InceptionResnetV1 + softmax CE on CASIA-WebFace (`data/casia/<id>/*.jpg`, 112x112
  InsightFace-aligned, resized to 160). Same objective as the facenet test models on purpose, so ArcFace is the
  "different objective" model. Checkpoint: `runs/surrogate.pt` (`state_dict`, `num_classes`); the embedding path
  is used by setting `net.classify = False`.

## Gotchas

- Dependency pins in `pyproject.toml` are load-bearing: torch from the `cu128` index (RTX 50xx needs torch>=2.7);
  `[tool.uv] override-dependencies` lifts facenet-pytorch's old torch/numpy/pillow pins; `onnxruntime-gpu<1.24`
  because newer builds need CUDA 13 and silently fall back to CPU.
- LFW is fetched from figshare (UMass host is often down), not via torchvision's `LFWPairs`.
- `ft validate` is the gate before any attack results are trusted: deep models must be within ~1pt of
  `PUBLISHED_ACCURACY`. Last run: vggface2 99.37%, casia 98.73% (margin 32 was best of 0/14/32/44),
  arcface 99.85% (published 99.83%), lbph 67.0%; MTCNN detected all 7,701 photos.
- `data/` and `runs/` are git-ignored (datasets, weights, outputs).
