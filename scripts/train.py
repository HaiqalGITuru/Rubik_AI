"""
Train CubeNet on a CPU (laptop) or a GPU (Google Colab).

Examples (from the project root):
    python -m scripts.train                      # default training on CPU (small model)
    python -m scripts.train --steps 2000         # quick test
    python -m scripts.train --resume             # resume from the last checkpoint

    # Large model on a GPU (see notebooks/train_colab.ipynb):
    python -m scripts.train --device cuda --hidden1 5000 --hidden2 1000 --blocks 4 \\
        --scrambles 1000 --steps 100000 --cosine --save models/cubenet_large.pt --resume

Press Ctrl+C at any time: the model is saved automatically and can be resumed with --resume.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from rubik import Cube
from rubik.data import generate_batch
from rubik.model import CubeNet, pick_device, save_model
from rubik.solver import beam_search


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000, help="total training steps")
    p.add_argument("--scrambles", type=int, default=200, help="scrambles per batch")
    p.add_argument("--max-depth", type=int, default=26, help="scramble length (26 = God's number in QTM)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cosine", action="store_true", help="decay the learning rate (cosine schedule) until the end")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--hidden1", type=int, default=1024)
    p.add_argument("--hidden2", type=int, default=512)
    p.add_argument("--blocks", type=int, default=4, help="number of residual blocks")
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000, help="evaluate + save a checkpoint every N steps")
    p.add_argument("--save", default="models/cubenet.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--time-limit", type=float, default=0,
                   help="stop cleanly + save a checkpoint after N hours (0 = no limit). "
                        "Useful on Kaggle (12-hour sessions) and Colab.")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def quick_eval(model, rng, depths=(5, 10, 15, 20), n=10, beam_width=64):
    """Share of scrambles solved at several difficulty levels."""
    result = {}
    for d in depths:
        ok = 0
        for _ in range(n):
            cube = Cube()
            cube.scramble(d, rng=rng, quarter_turns_only=True)
            ok += beam_search(model, cube, beam_width=beam_width, max_depth=d + 10).solved
        result[d] = ok / n
    model.train()
    return result


def lr_at(step, args):
    if not args.cosine:
        return args.lr
    t = min(1.0, step / args.steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda"  # mixed precision: much faster on a GPU
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    log_path = os.path.splitext(args.save)[0] + "_log.csv"

    start_step = 0
    ckpt = None
    if args.resume and os.path.exists(args.save):
        ckpt = torch.load(args.save, map_location="cpu")
        model = CubeNet(**ckpt["config"])  # architecture follows the checkpoint
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = CubeNet(hidden1=args.hidden1, hidden2=args.hidden2, num_blocks=args.blocks)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"Resuming from step {start_step}")

    # Different random data after a resume (do not repeat the same batches)
    rng = np.random.default_rng(args.seed + start_step)

    if start_step == 0:  # new training run -> start a new log file
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "accuracy", "lr", "solve_d5", "solve_d10", "solve_d15", "solve_d20"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} | device: {device}"
          f"{' (mixed precision)' if use_amp else f' | CPU threads: {args.threads}'} | "
          f"samples per batch: {args.scrambles * args.max_depth:,}")

    def checkpoint(step):
        save_model(model, args.save, optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(), step=step)

    model.train()
    step = start_step
    t_start = time.perf_counter()
    t_log = time.perf_counter()
    loss_sum = acc_sum = 0.0
    try:
        for step in range(start_step + 1, args.steps + 1):
            lr = lr_at(step, args)
            for g in optimizer.param_groups:
                g["lr"] = lr

            X, y, _ = generate_batch(args.scrambles, args.max_depth, rng)
            X = torch.from_numpy(X).to(device, non_blocking=True).long()
            y = torch.from_numpy(y).to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(X)
                loss = F.cross_entropy(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            acc_sum += (logits.argmax(1) == y).float().mean().item()

            if step % args.log_every == 0:
                dt = time.perf_counter() - t_log
                loss_avg, acc_avg = loss_sum / args.log_every, acc_sum / args.log_every
                remaining = (args.steps - step) * dt / args.log_every
                print(f"step {step:>6} | loss {loss_avg:.4f} | accuracy {acc_avg:.3f} | lr {lr:.1e} | "
                      f"{args.log_every / dt:.1f} steps/s | ~{remaining / 60:.0f} min left", flush=True)
                eval_row = ["", "", "", ""]
                if step % args.eval_every == 0:
                    res = quick_eval(model, rng)
                    print("   evaluation (success rate per scramble depth): "
                          + "  ".join(f"d{d}={r:.0%}" for d, r in res.items()), flush=True)
                    eval_row = [res[d] for d in (5, 10, 15, 20)]
                    checkpoint(step)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{loss_avg:.4f}", f"{acc_avg:.4f}", f"{lr:.2e}", *eval_row])
                loss_sum = acc_sum = 0.0
                t_log = time.perf_counter()
            if args.time_limit and time.perf_counter() - t_start > args.time_limit * 3600:
                print(f"Time limit of {args.time_limit} h reached, stopping cleanly.", flush=True)
                break
    except KeyboardInterrupt:
        print("\nTraining stopped.")

    checkpoint(step)
    print(f"Model saved to {args.save} (step {step})")


if __name__ == "__main__":
    main()