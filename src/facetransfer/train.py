"""Train a surrogate from scratch with softmax cross-entropy on CASIA-WebFace.

The default (`inception`) is InceptionResnetV1, the same architecture, objective and data as the facenet-pytorch
CASIA test model; it differs from that model in crop and training procedure, and from the VGGFace2 model also in
training faces. `resnet50` trains the same recipe with a torchvision ResNet-50 backbone, so the two surrogates
differ only in architecture.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from facetransfer.data import CasiaLoader
from facetransfer.evaluate import Workspace, best_threshold, clean_accuracy
from facetransfer.models import SURROGATE_ARCHS, SURROGATE_SIZE, TorchFaceNet, surrogate_crop, surrogate_net


@torch.no_grad()
def val_accuracy(net, images_u8, a, b, same, device: str) -> float:
    """Verification accuracy on held-out identities (threshold fitted on the same pairs, used only to rank epochs)."""
    net.eval()
    net.classify = False
    emb = []
    for s in range(0, len(images_u8), 256):
        x = F.interpolate(images_u8[s : s + 256].to(device).float(), size=SURROGATE_SIZE, mode="bilinear")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            emb.append(
                F.normalize(net(((x - 127.5) / 128).contiguous(memory_format=torch.channels_last)).float(), dim=-1)
            )
    emb = torch.cat(emb)
    dist = (1 - (emb[a] * emb[b]).sum(-1)).cpu().numpy()
    return float(((dist <= best_threshold(dist, same)) == same).mean())


def train(
    casia_dir: Path,
    ws: Workspace,
    epochs: int,
    batch_size: int,
    lr: float,
    arch: str = "inception",
    holdout: int = 500,
    patience: int = 0,
) -> None:
    device = ws.device
    loader = CasiaLoader(casia_dir, batch_size, device, holdout=holdout)
    val = loader.val_set() if holdout else None
    num_classes = len(loader.classes)
    name = SURROGATE_ARCHS[arch]
    net = surrogate_net(arch, num_classes).to(device, memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.05)
    print(f"{len(loader.paths)} images, {num_classes} identities, {len(loader)} steps/epoch, {epochs} epochs ({arch})")

    lfw_affines = torch.from_numpy(surrogate_crop(ws.meta, tuple(ws.images.shape[2:])))
    ws.runs_dir.mkdir(parents=True, exist_ok=True)
    best, stale = -1.0, 0
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
        acc = clean_accuracy(TorchFaceNet(name, net, SURROGATE_SIZE, None), lfw_affines, ws)
        net.requires_grad_(True)
        # Model selection uses the held-out CASIA identities, never LFW (the test set). With no holdout, keep the last.
        score = val_accuracy(net, *val, device) if val else float(epoch)
        if score > best:
            best, stale = score, 0
            ckpt = {"state_dict": net.state_dict(), "num_classes": num_classes, "arch": arch}
            torch.save(ckpt, ws.runs_dir / f"{name}.pt")
        else:
            stale += 1
        total = len(loader) * batch_size
        print(
            f"epoch {epoch}: loss {loss_sum.item() / total:.3f} train acc {correct.item() / total:.3f} "
            f"LFW {acc.mean():.4f} +/- {acc.std():.4f} val {score:.4f} best {best:.4f}",
            flush=True,
        )
        if patience and stale >= patience:
            print(f"early stop: no val improvement for {patience} epochs", flush=True)
            break
