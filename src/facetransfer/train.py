"""Train the surrogate from scratch: InceptionResnetV1 + softmax cross-entropy on CASIA-WebFace.

Same architecture and objective as the facenet-pytorch test models, so what separates the surrogate
from test model 2 (CASIA) is only the random seed, and from test model 1 only the training faces.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from facetransfer.data import CasiaLoader
from facetransfer.evaluate import Workspace, clean_accuracy
from facetransfer.models import SURROGATE_SIZE, TorchFaceNet, surrogate_crop


def train(
    casia_dir: Path,
    ws: Workspace,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    device = ws.device
    loader = CasiaLoader(casia_dir, batch_size, device)
    num_classes = len(loader.classes)
    net = InceptionResnetV1(classify=True, num_classes=num_classes).to(device, memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.05)
    print(f"{len(loader.paths)} images, {num_classes} identities, {len(loader)} steps/epoch")

    lfw_affines = torch.from_numpy(surrogate_crop(ws.meta, tuple(ws.images.shape[2:])))
    ws.runs_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        net.train()
        net.classify = True
        correct, loss_sum = torch.zeros((), device=device), torch.zeros((), device=device)
        for step, (x, y) in enumerate(loader):
            # 112 -> 160 to match the facenet input size; random horizontal flip.
            x = F.interpolate(x.float(), size=SURROGATE_SIZE, mode="bilinear", align_corners=False)
            flip = torch.rand(len(x), device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = net((x - 127.5) / 128)
                loss = F.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                torch.save({"x": x.cpu(), "y": y.cpu()}, ws.runs_dir / "nan_batch.pt")
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}; batch saved to nan_batch.pt")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            correct += (logits.argmax(1) == y).sum()
            loss_sum += loss.detach() * len(y)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.3f}", flush=True)

        # TorchFaceNet flips the net to eval/embedding mode; restored at the top of the next epoch.
        acc = clean_accuracy(TorchFaceNet("surrogate", net, SURROGATE_SIZE, None), lfw_affines, ws)
        net.requires_grad_(True)
        torch.save({"state_dict": net.state_dict(), "num_classes": num_classes}, ws.runs_dir / "surrogate.pt")
        total = len(loader) * batch_size
        print(
            f"epoch {epoch}: loss {loss_sum.item() / total:.3f} train acc {correct.item() / total:.3f} "
            f"LFW {acc.mean():.4f} +/- {acc.std():.4f}",
            flush=True,
        )
