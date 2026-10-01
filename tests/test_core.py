import numpy as np
import torch

from facetransfer.align import ARCFACE_TEMPLATE, box_affines, landmark_affines, warp
from facetransfer.attack import pgd
from facetransfer.evaluate import best_threshold, fold_accuracy, fold_thresholds
from facetransfer.models import FaceModel


def test_best_threshold_separates_clean_split():
    dist = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    same = np.array([True, True, True, False, False, False])
    t = best_threshold(dist, same)
    assert 0.3 < t < 0.7


def test_fold_thresholds_perfect_accuracy():
    rng = np.random.default_rng(0)
    same = np.tile([True, False], 50)
    dist = np.where(same, rng.uniform(0, 0.4, 100), rng.uniform(0.6, 1, 100))
    fold = np.repeat(np.arange(10), 10)
    acc = fold_accuracy(dist, same, fold, fold_thresholds(dist, same, fold))
    assert acc.shape == (10,) and np.all(acc == 1)


def test_box_warp_matches_integer_crop():
    img = torch.rand(1, 3, 50, 50)
    # A 20px box resized to 20px must equal plain slicing.
    m = torch.from_numpy(box_affines(np.array([[10, 15, 30, 35]], np.float32), 20, 0, (50, 50)))
    assert torch.allclose(warp(img, m, 20), img[:, :, 15:35, 10:30], atol=1e-5)


def test_landmark_affine_maps_landmarks_to_template():
    lm = ARCFACE_TEMPLATE * 1.7 + 40
    m = landmark_affines(lm[None], 112)[0]
    mapped = lm @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, ARCFACE_TEMPLATE, atol=1e-3)


class TinyModel(FaceModel):
    differentiable = True

    def __init__(self):
        super().__init__("tiny", 8, None)
        self.proj = torch.nn.Linear(3 * 8 * 8, 16)

    def embed(self, images, affines):
        return torch.nn.functional.normalize(self.proj(warp(images, affines, 8).flatten(1)), dim=-1)


def test_pgd_respects_budget_and_moves_embedding():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.rand(4, 3, 16, 16)
    m = torch.from_numpy(box_affines(np.array([[0, 0, 16, 16]] * 4, np.float32), 8, 0, (16, 16)))
    eps = 8 / 255
    adv = pgd(model, x, m, eps, steps=10)
    assert (adv - x).abs().max() <= eps + 1e-6
    assert adv.min() >= 0 and adv.max() <= 1
    moved = model.distance(model.embed(adv, m), model.embed(x, m))
    assert (moved > 0).all()


def test_watch_parses_training_log(tmp_path):
    from facetransfer.watch import render

    log = tmp_path / "train.log"
    log.write_text(
        "100 images, 10 identities, 50 steps/epoch, 2 epochs\n"
        "epoch 0 step 0/50 loss 2.300\n"
        "epoch 0: loss 2.000 train acc 0.100 LFW 0.8000 +/- 0.0100\n"
        "epoch 1 step 25/50 loss 1.500\n"
    )
    out = render(log, epochs_default=30, rate=[])
    assert "epoch  2/2" in out and "1.500" in out and "0.8000" in out and "<- best" in out
