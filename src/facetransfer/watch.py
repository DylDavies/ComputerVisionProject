"""Live terminal dashboard for a training run: parses runs/train.log and polls nvidia-smi."""

import re
import subprocess
import sys
import time
from pathlib import Path

HEADER = re.compile(r"(\d+) images, (\d+) identities, (\d+) steps/epoch(?:, (\d+) epochs)?")
STEP = re.compile(r"^epoch (\d+) step (\d+)/(\d+) loss (\S+)", re.M)
EPOCH = re.compile(r"^epoch (\d+): loss (\S+) train acc (\S+) LFW (\S+) (?:\+/-|±) (\S+)", re.M)
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    values = [v for v in values if v == v][-width:]  # drop NaN
    if not values:
        return ""
    lo, hi = min(values), max(values)
    return "".join(SPARK[int((v - lo) / (hi - lo or 1) * (len(SPARK) - 1))] for v in values)


def bar(fraction: float, width: int = 40) -> str:
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled) + f" {fraction:6.1%}"


def duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def gpu() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.split(",")
        util, used, total, temp, power = (x.strip() for x in out)
        return f"util {util}%  mem {int(used) / 1024:.1f}/{int(total) / 1024:.1f} GB  {temp}°C  {float(power):.0f} W"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "nvidia-smi unavailable"


def render(log: Path, epochs_default: int, rate: list[tuple[float, int]]) -> str:
    text = log.read_text(errors="replace") if log.exists() else ""
    header = HEADER.search(text)
    steps = STEP.findall(text)
    done = EPOCH.findall(text)
    lines = [f"facetransfer training  ·  {log}  ·  {time.strftime('%H:%M:%S')}", ""]
    if not header:
        return "\n".join([*lines, "waiting for training to start..."])

    n_images, n_ids, per_epoch, n_epochs = header.groups()
    per_epoch, n_epochs = int(per_epoch), int(n_epochs or epochs_default)
    lines.append(f"data   {int(n_images):,} images, {int(n_ids):,} identities, {per_epoch} steps/epoch")

    if steps:
        epoch, step, _, loss = steps[-1]
        global_step = int(epoch) * per_epoch + int(step)
        if not rate or rate[-1][1] != global_step:
            rate.append((time.time(), global_step))
        total = n_epochs * per_epoch
        lines += [
            "",
            f"epoch  {int(epoch) + 1}/{n_epochs}   {bar(int(step) / per_epoch)}",
            f"total  {bar(global_step / total)}",
            f"loss   {float(loss):.3f}   {sparkline([float(s[3]) for s in steps])}",
        ]
        if len(rate) >= 2:
            (t0, s0), (t1, s1) = rate[0], rate[-1]
            sps = (s1 - s0) / (t1 - t0)
            eta = (total - global_step) / sps if sps > 0 else float("inf")
            img_s = sps * int(n_images) / per_epoch
            lines.append(f"speed  {sps:.1f} steps/s  ({img_s:,.0f} img/s)   ETA {duration(eta)}")
        else:
            lines.append("speed  measuring (needs two logged steps while watching)...")

    lines += ["", f"gpu    {gpu()}"]
    age = time.time() - log.stat().st_mtime
    if "Traceback" in text:
        tail = text[text.rindex("Traceback") :].strip().splitlines()[-1]
        lines += ["", f"!! CRASHED: {tail}"]
    elif done and int(done[-1][0]) + 1 == n_epochs:
        lines += ["", "finished"]
    elif age > 180:
        lines += ["", f"!! log not updated for {duration(age)} (stalled or stopped?)"]

    if done:
        best = max(float(d[3]) for d in done)
        lines += ["", "epoch   loss   train acc   LFW acc", "-----  ------  ---------  ----------------"]
        for e, loss, acc, lfw, std in done[-15:]:
            mark = "  <- best" if float(lfw) == best else ""
            lines.append(
                f"{int(e) + 1:>5}  {float(loss):6.3f}  {float(acc):9.3f}  {float(lfw):.4f} ± {float(std):.4f}{mark}"
            )
        lines.append(f"LFW    {sparkline([float(d[3]) for d in done])}")
    return "\n".join(lines)


def watch(log: Path, interval: float, epochs_default: int) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rate: list[tuple[float, int]] = []
    try:
        while True:
            sys.stdout.write("\033[H\033[J" + render(log, epochs_default, rate) + "\n\n(ctrl+c to exit)\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
