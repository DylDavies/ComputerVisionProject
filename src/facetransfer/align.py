"""Face geometry: detect once on clean photos, then express every model's crop as a 2x3 affine.

All affines map pixel coordinates of the full LFW photo to pixel coordinates of a model's input
(pixel centres at integer positions). `warp` applies them with grid_sample, so the crop is
differentiable and PGD edits live in photo space.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import SimilarityTransform

# Standard InsightFace 5-point template for a 112x112 crop: eyes, nose, mouth corners.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def detect(images: torch.Tensor, device: str, batch_size: int = 64) -> dict[str, np.ndarray]:
    """Run MTCNN on uint8 images [N,3,H,W]; keep the face nearest the image centre.

    Returns boxes [N,4], landmarks [N,5,2] and `detected` [N]. LFW photos are centred on the
    labelled person, so a miss falls back to a fixed central box and a template-shaped landmark set.
    """
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, device=device)
    n, _, h, w = images.shape
    centre = np.array([w / 2, h / 2])
    boxes = np.zeros((n, 4), np.float32)
    landmarks = np.zeros((n, 5, 2), np.float32)
    detected = np.zeros(n, bool)
    fallback_box = np.array([w * 0.28, h * 0.28, w * 0.72, h * 0.72], np.float32)
    s = (fallback_box[2] - fallback_box[0]) / 112
    fallback_lm = ARCFACE_TEMPLATE * s + fallback_box[:2]

    for start in range(0, n, batch_size):
        batch = images[start : start + batch_size].permute(0, 2, 3, 1).numpy()
        batch_boxes, _, batch_lms = mtcnn.detect(batch, landmarks=True)
        for i, (b, lm) in enumerate(zip(batch_boxes, batch_lms, strict=True)):
            k = start + i
            if b is None:
                boxes[k], landmarks[k] = fallback_box, fallback_lm
                continue
            j = np.argmin(np.linalg.norm((b[:, :2] + b[:, 2:]) / 2 - centre, axis=1))
            boxes[k], landmarks[k], detected[k] = b[j], lm[j], True
    return {"boxes": boxes, "landmarks": landmarks, "detected": detected}


def save_meta(path: Path, meta: dict[str, np.ndarray]) -> None:
    np.savez(path, **meta)


def load_meta(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return dict(f)


def landmark_affines(landmarks: np.ndarray, size: int) -> np.ndarray:
    """Similarity transforms mapping each landmark set onto the ArcFace template scaled to `size`."""
    dst = ARCFACE_TEMPLATE * (size / 112)
    return np.stack([SimilarityTransform.from_estimate(lm, dst).params[:2] for lm in landmarks]).astype(np.float32)


def box_affines(boxes: np.ndarray, size: int, margin: int, image_hw: tuple[int, int]) -> np.ndarray:
    """Axis-aligned crop+resize, replicating facenet-pytorch's `extract_face` (margin in output pixels)."""
    h, w = image_hw
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    mx, my = margin * bw / (size - margin), margin * bh / (size - margin)
    x1 = np.floor(np.maximum(boxes[:, 0] - mx / 2, 0))
    y1 = np.floor(np.maximum(boxes[:, 1] - my / 2, 0))
    x2 = np.floor(np.minimum(boxes[:, 2] + mx / 2, w))
    y2 = np.floor(np.minimum(boxes[:, 3] + my / 2, h))
    sx, sy = size / (x2 - x1), size / (y2 - y1)
    m = np.zeros((len(boxes), 2, 3), np.float32)
    m[:, 0, 0], m[:, 1, 1] = sx, sy
    # Resize with pixel centres: u = sx * (x - x1 + 0.5) - 0.5.
    m[:, 0, 2] = sx * (0.5 - x1) - 0.5
    m[:, 1, 2] = sy * (0.5 - y1) - 0.5
    return m


def _to_normalised(size_hw: tuple[int, int], device) -> torch.Tensor:
    """3x3 matrix mapping pixel coords to grid_sample's [-1,1] coords (align_corners=False)."""
    h, w = size_hw
    return torch.tensor([[2 / w, 0, 1 / w - 1], [0, 2 / h, 1 / h - 1], [0, 0, 1]], device=device)


def warp(images: torch.Tensor, affines: torch.Tensor, size: int) -> torch.Tensor:
    """Crop float images [B,C,H,W] with pixel-space affines [B,2,3] (photo -> crop) to [B,C,size,size]."""
    b, c, h, w = images.shape
    m = torch.cat([affines, affines.new_tensor([0, 0, 1]).expand(b, 1, 3)], dim=1)
    # grid_sample wants, for each output location, where to read in the input (normalised coords).
    theta = (
        _to_normalised((h, w), images.device)
        @ torch.linalg.inv(m)
        @ torch.linalg.inv(_to_normalised((size, size), images.device))
    )
    grid = F.affine_grid(theta[:, :2], [b, c, size, size], align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
