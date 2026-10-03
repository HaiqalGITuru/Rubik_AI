"""
Training DeepCubeA dengan Deep Approximate Value Iteration (DAVI).

Contoh:
    python -m scripts.train_deepcubea --steps 2000                       # uji cepat di CPU (model kecil)
    python -m scripts.train_deepcubea --device cuda --hidden1 5000 --hidden2 1000 --blocks 4 \\
        --scrambles 400 --steps 50000 --cosine --save models/deepcubea_large.pt --resume

Bisa dihentikan (Ctrl+C / tombol stop Colab) dan dilanjutkan dengan --resume.
"""

import argparse
import copy
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from rubik import Cube
from rubik.data import generate_batch
from rubik.deepcubea import davi_targets, weighted_astar
from rubik.model import CubeNet, pick_device, save_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--scrambles", type=int, default=200, help="scramble per batch (x max-depth = jumlah posisi)")
    p.add_argument("--max-depth", type=int, default=30, help="panjang scramble maksimum (paper: 30)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--update-every", type=int, default=1000,
                   help="salin jaringan ke target network setiap N langkah")
    p.add_argument("--hidden1", type=int, default=1024)
    p.add_argument("--hidden2", type=int, default=512)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save", default="models/deepcubea.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def quick_eval(model, rng, depths=(5, 10, 15, 20), n=10, batch_size=100):
    result = {}
    for d in depths:
        ok = 0
        for _ in range(n):
            cube = Cube()
            cube.scramble(d, rng=rng, quarter_turns_only=True)
            ok += weighted_astar(model, cube, batch_size=batch_size, max_iterations=40).solved
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
    use_amp = device.type == "cuda"
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    log_path = os.path.splitext(args.save)[0] + "_log.csv"

    ckpt = None
    if args.resume and os.path.exists(args.save):
        ckpt = torch.load(args.save, map_location="cpu")
        model = CubeNet(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = CubeNet(hidden1=args.hidden1, hidden2=args.hidden2, num_blocks=args.blocks, num_moves=1)
    model.to(device)

    target = copy.deepcopy(model).eval()
    if ckpt is not None and "target_state_dict" in ckpt:
        target.load_state_dict(ckpt["target_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step, updates = 0, 0
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_step, updates = ckpt["step"], ckpt.get("target_updates", 0)
        print(f"Melanjutkan dari langkah {start_step}")

    rng = np.random.default_rng(args.seed + start_step)
    if start_step == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "target_mean", "target_max", "lr", "target_updates",
                                    "solve_d5", "solve_d10", "solve_d15", "solve_d20"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeepCubeA | parameter: {n_params:,} | perangkat: {device}"
          f"{' (mixed precision)' if use_amp else ''} | posisi per batch: {args.scrambles * args.max_depth:,}")

    def checkpoint(step):
        save_model(model, args.save, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                   target_state_dict=target.state_dict(), step=step, target_updates=updates,
                   method="deepcubea")

    model.train()
    step = start_step
    t_log = time.perf_counter()
    loss_sum = tmean_sum = 0.0
    tmax = 0.0
    try:
        for step in range(start_step + 1, args.steps + 1):
            lr = lr_at(step, args)
            for g in optimizer.param_groups:
                g["lr"] = lr

            X, _, _ = generate_batch(args.scrambles, args.max_depth, rng)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                y = davi_targets(target, X)
            Xt = torch.from_numpy(X).to(device).long()
            yt = torch.from_numpy(y).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(Xt).float().squeeze(1)
                loss = F.mse_loss(pred, yt)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            tmean_sum += float(y.mean())
            tmax = max(tmax, float(y.max()))

            if step % args.update_every == 0:  # perbarui target network
                target.load_state_dict(model.state_dict())
                updates += 1

            if step % args.log_every == 0:
                dt = time.perf_counter() - t_log
                la, tm = loss_sum / args.log_every, tmean_sum / args.log_every
                remaining = (args.steps - step) * dt / args.log_every
                print(f"step {step:>6} | loss {la:.4f} | target rata2 {tm:5.2f} maks {tmax:5.1f} | "
                      f"update target {updates} | lr {lr:.1e} | {args.log_every / dt:.1f} step/dtk | "
                      f"sisa ~{remaining / 60:.0f} mnt", flush=True)
                ev = ["", "", "", ""]
                if step % args.eval_every == 0:
                    res = quick_eval(model, rng)
                    print("   evaluasi (tingkat berhasil per jarak scramble): "
                          + "  ".join(f"d{d}={r:.0%}" for d, r in res.items()), flush=True)
                    ev = [res[d] for d in (5, 10, 15, 20)]
                    checkpoint(step)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{la:.4f}", f"{tm:.3f}", f"{tmax:.2f}", f"{lr:.2e}", updates, *ev])
                loss_sum = tmean_sum = 0.0
                tmax = 0.0
                t_log = time.perf_counter()
    except KeyboardInterrupt:
        print("\nTraining dihentikan.")

    checkpoint(step)
    print(f"Model disimpan di {args.save} (langkah {step})")


if __name__ == "__main__":
    main()
