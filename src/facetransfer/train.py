"""Train the surrogate from scratch: InceptionResnetV1 + softmax cross-entropy on CASIA-WebFace.

Same architecture and objective as the facenet-pytorch test models, so what separates the surrogate
from test model 2 (CASIA) is only the random seed, and from test model 1 only the training faces.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader

from facetransfer.data import casia
from facetransfer.evaluate import Workspace, clean_accuracy
from facetransfer.models import SURROGATE_SIZE, TorchFaceNet, landmark_crop


def train(
    casia_dir: Path,
    ws: Workspace,
    epochs: int,
    batch_size: int,
    lr: float,
    workers: int,
) -> None:
    device = ws.device
    dataset = casia(casia_dir, SURROGATE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    num_classes = len(dataset.classes)
    net = InceptionResnetV1(classify=True, num_classes=num_classes).to(device, memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.05)
    print(f"{len(dataset)} images, {num_classes} identities, {len(loader)} steps/epoch")

    lfw_affines = torch.from_numpy(landmark_crop(SURROGATE_SIZE)(ws.meta, tuple(ws.images.shape[2:])))
    ws.runs_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        net.train()
        net.classify = True
        total, correct, loss_sum = 0, 0, 0.0
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last).float()
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = net((x - 127.5) / 128)
                loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.3f}", flush=True)

        # TorchFaceNet flips the net to eval/embedding mode; restored at the top of the next epoch.
        acc = clean_accuracy(TorchFaceNet("surrogate", net, SURROGATE_SIZE, None), lfw_affines, ws)
        net.requires_grad_(True)
        torch.save({"state_dict": net.state_dict(), "num_classes": num_classes}, ws.runs_dir / "surrogate.pt")
        print(
            f"epoch {epoch}: loss {loss_sum / total:.3f} train acc {correct / total:.3f} "
            f"LFW {acc.mean():.4f} ± {acc.std():.4f}",
            flush=True,
        )
