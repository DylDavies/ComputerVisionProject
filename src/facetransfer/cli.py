"""`ft` command line: prepare -> train -> validate -> sweep -> plot."""

import argparse
from pathlib import Path

import torch

from facetransfer.evaluate import CONDITIONS
from facetransfer.models import MODEL_NAMES


def cmd_prepare(args) -> None:
    from facetransfer.align import detect, save_meta
    from facetransfer.data import download_lfw, load_images, load_pairs
    from facetransfer.models import arcface_onnx, load_model

    download_lfw(args.data)
    pairs = load_pairs(args.data)
    print(f"{len(pairs)} pairs over {len(pairs.paths)} photos")
    meta_path = args.data / "lfw_meta.npz"
    if not meta_path.exists():
        meta = detect(load_images(pairs.paths), args.device)
        save_meta(meta_path, meta)
        print(f"MTCNN missed {(~meta['detected']).sum()} photos (central fallback used)")
    arcface_onnx(args.data / "weights")
    for name in ("vggface2", "casia"):
        load_model(name, "cpu", args.runs, args.data / "weights")  # fetches facenet-pytorch weights


def _workspace(args):
    from facetransfer.evaluate import Workspace

    return Workspace(args.data, args.runs, args.device)


def cmd_train(args) -> None:
    from facetransfer.train import train

    train(args.casia, _workspace(args), args.epochs, args.batch_size, args.lr)


def cmd_validate(args) -> None:
    from facetransfer.evaluate import validate

    args.runs.mkdir(parents=True, exist_ok=True)
    validate(_workspace(args), args.models).to_csv(args.runs / "validate.csv", index=False)


def cmd_sweep(args) -> None:
    from facetransfer.evaluate import sweep

    df = sweep(
        _workspace(args),
        names=args.models,
        attacker=args.attacker,
        eps_list=args.eps,
        jpeg_qualities=args.jpeg,
        blur_sigmas=args.blur,
        steps=args.steps,
        limit=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
        conditions=tuple(args.conditions),
    )
    args.runs.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out or args.runs / "sweep.csv", index=False)
    summary = df.groupby(["condition", "strength", "model"]).unmatched.mean().unstack("model")
    print(summary.round(3).to_string())


def cmd_plot(args) -> None:
    from facetransfer.plots import plot_all

    plot_all(args.csv or args.runs / "sweep.csv", args.runs / "figures")


def cmd_watch(args) -> None:
    from facetransfer.watch import watch

    watch(args.log or args.runs / "train.log", args.interval, args.epochs)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ft", description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("prepare", help="download LFW + weights, detect faces once")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="train the surrogate on CASIA-WebFace")
    p.add_argument("--casia", type=Path, default=Path("data/casia"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("validate", help="clean LFW accuracy per model vs published")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("sweep", help="edit-budget sweep over all conditions")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--attacker", choices=MODEL_NAMES, default="surrogate")
    p.add_argument("--eps", nargs="+", type=float, default=[0, 0.5, 1, 2, 4, 8, 16], help="L-inf, in /255")
    p.add_argument("--jpeg", nargs="+", type=int, default=[90, 75, 50, 30])
    p.add_argument("--blur", nargs="+", type=float, default=[0.5, 1, 2, 3])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--limit", type=int, help="only this many same-person pairs (smoke tests)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("plot", help="figures from a sweep CSV")
    p.add_argument("--csv", type=Path)
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("watch", help="live dashboard for a training run")
    p.add_argument("--log", type=Path, help="default: <runs>/train.log")
    p.add_argument("--interval", type=float, default=2.0, help="refresh seconds")
    p.add_argument("--epochs", type=int, default=30, help="total epochs, if the log header doesn't say")
    p.set_defaults(fn=cmd_watch)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
