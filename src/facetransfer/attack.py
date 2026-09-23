"""Photo edits: PGD against the surrogate, and the free baselines (random noise, JPEG, blur).

All functions take/return float photos [B,3,H,W] in [0,1]; `quantise` turns them into the uint8
photo a user would actually post.
"""

import math

import torch
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.functional import gaussian_blur

from facetransfer.models import FaceModel


def pgd(
    model: FaceModel,
    images: torch.Tensor,
    affines: torch.Tensor,
    eps: float,
    steps: int = 20,
    step_size: float | None = None,
) -> torch.Tensor:
    """L-inf PGD with random start, maximising the model's distance to each photo's own clean embedding."""
    if eps == 0:
        return images
    step_size = step_size or 2.5 * eps / steps
    with torch.no_grad():
        target = model.embed(images, affines)
    # Random start: the clean photo is a minimum of the distance, so its gradient is zero.
    x = (images + torch.empty_like(images).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x.requires_grad_(True)
        loss = model.distance(model.embed(x, affines), target).sum()
        (grad,) = torch.autograd.grad(loss, x)
        x = x.detach() + step_size * grad.sign()
        x = (images + (x - images).clamp(-eps, eps)).clamp(0, 1)
    return x.detach()


def random_noise(images: torch.Tensor, eps: float) -> torch.Tensor:
    """Random +/-eps per pixel: same L-inf budget as PGD, no optimisation."""
    return (images + eps * torch.randint_like(images, 0, 2).mul(2).sub(1)).clamp(0, 1)


def jpeg(images: torch.Tensor, quality: int) -> torch.Tensor:
    encoded = encode_jpeg(list(quantise(images).cpu()), quality=quality)
    return torch.stack(decode_jpeg(encoded)).to(images.device).float() / 255


def blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian_blur(images, kernel_size=2 * math.ceil(3 * sigma) + 1, sigma=sigma)


def quantise(images: torch.Tensor) -> torch.Tensor:
    return (images * 255).round().clamp(0, 255).to(torch.uint8)
