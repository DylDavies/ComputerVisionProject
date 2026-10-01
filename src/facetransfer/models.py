"""The five face models behind one interface: crop geometry, embed(), distance().

`embed` always takes float photos [B,3,H,W] in [0,1] plus per-photo affines, so every model does
its own cropping from the same edited photo.
"""

import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from skimage.feature import local_binary_pattern

from facetransfer.align import ARCFACE_TEMPLATE, CASIA_MTCNN_TEMPLATE, box_affines, landmark_affines, warp
from facetransfer.data import download

MODEL_NAMES = ("surrogate", "vggface2", "casia", "arcface", "lbph")
SURROGATE_SIZE = 160

# Published LFW accuracy, used by `ft validate` to catch alignment/preprocessing mistakes.
PUBLISHED_ACCURACY = {"vggface2": 0.9965, "casia": 0.9905, "arcface": 0.9983}

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

AffineFn = Callable[[dict[str, np.ndarray], tuple[int, int]], np.ndarray]


def landmark_crop(size: int, template: np.ndarray = ARCFACE_TEMPLATE) -> AffineFn:
    return lambda meta, hw: landmark_affines(meta["landmarks"], size, template)


def box_crop(size: int, margin: int) -> AffineFn:
    return lambda meta, hw: box_affines(meta["boxes"], size, margin, hw)


# Matches the geometry of the CASIA crops the surrogate is trained on (resized 112 -> 160).
surrogate_crop = landmark_crop(SURROGATE_SIZE, CASIA_MTCNN_TEMPLATE)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - (a * b).sum(-1)


class FaceModel:
    differentiable = False

    def __init__(self, name: str, size: int, crop: AffineFn):
        self.name, self.size, self.crop = name, size, crop

    def embed(self, images: torch.Tensor, affines: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return cosine_distance(a, b)


class TorchFaceNet(FaceModel):
    """InceptionResnetV1 with facenet-pytorch's fixed standardisation; our surrogate or pretrained."""

    differentiable = True

    def __init__(self, name: str, net: torch.nn.Module, size: int, crop: AffineFn):
        super().__init__(name, size, crop)
        self.net = net.eval().requires_grad_(False)
        self.net.classify = False

    def embed(self, images, affines):
        x = warp(images, affines, self.size) * 255
        return F.normalize(self.net((x - 127.5) / 128), dim=-1)


class ArcFace(FaceModel):
    """InsightFace buffalo_l recognition model (ResNet-50, ArcFace loss, WebFace600K) via onnxruntime."""

    def __init__(self, onnx_path: Path, device: str):
        import onnxruntime as ort

        super().__init__("arcface", 112, landmark_crop(112))
        providers = ["CUDAExecutionProvider"] if device.startswith("cuda") else []
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # the model declares output batch 1; every batched run would warn
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=[*providers, "CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    @torch.no_grad()
    def embed(self, images, affines):
        x = (warp(images, affines, self.size) * 255 - 127.5) / 127.5
        out = self.session.run(None, {self.input: x.float().cpu().numpy()})[0]
        return F.normalize(torch.from_numpy(out).to(images.device), dim=-1)


class LBPH(FaceModel):
    """Local Binary Pattern Histograms: 8x8 grid of non-rotation-invariant uniform LBP (59 bins)."""

    grid, bins = 8, 59

    def __init__(self):
        super().__init__("lbph", 112, landmark_crop(112))

    @torch.no_grad()
    def embed(self, images, affines):
        crops = warp(images, affines, self.size)
        grey = (crops * torch.tensor([0.299, 0.587, 0.114], device=crops.device)[:, None, None]).sum(1)
        grey = (grey * 255).round().byte().cpu().numpy()
        cell = self.size // self.grid
        feats = []
        for g in grey:
            codes = local_binary_pattern(g, P=8, R=1, method="nri_uniform").astype(np.int64)
            cells = codes.reshape(self.grid, cell, self.grid, cell).transpose(0, 2, 1, 3).reshape(-1, cell * cell)
            hist = np.stack([np.bincount(c, minlength=self.bins) for c in cells]).astype(np.float32)
            feats.append((hist / hist.sum(1, keepdims=True)).ravel())
        return torch.from_numpy(np.stack(feats)).to(images.device)

    def distance(self, a, b):
        return ((a - b) ** 2 / (a + b + 1e-10)).sum(-1)  # chi-square


def arcface_onnx(weights_dir: Path) -> Path:
    onnx = weights_dir / "w600k_r50.onnx"
    if not onnx.exists():
        with zipfile.ZipFile(download(BUFFALO_URL, weights_dir / "buffalo_l.zip")) as z:
            onnx.write_bytes(z.read("w600k_r50.onnx"))
    return onnx


def load_surrogate_net(checkpoint: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    net = InceptionResnetV1(classify=True, num_classes=ckpt["num_classes"])
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device)


def load_model(name: str, device: str, runs_dir: Path, weights_dir: Path) -> FaceModel:
    match name:
        case "surrogate":
            net = load_surrogate_net(runs_dir / "surrogate.pt", device)
            return TorchFaceNet(name, net, SURROGATE_SIZE, surrogate_crop)
        case "vggface2" | "casia":
            net = InceptionResnetV1(pretrained="vggface2" if name == "vggface2" else "casia-webface")
            return TorchFaceNet(name, net.to(device), 160, box_crop(160, margin=32))
        case "arcface":
            return ArcFace(arcface_onnx(weights_dir), device)
        case "lbph":
            return LBPH()
    raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")
