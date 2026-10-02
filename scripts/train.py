"""
Training CubeNet di CPU.

Contoh (dari folder root proyek):
    python -m scripts.train                      # training default
    python -m scripts.train --steps 2000         # uji cepat
    python -m scripts.train --resume             # lanjutkan dari checkpoint terakhir

Tekan Ctrl+C kapan saja: model otomatis disimpan, dan bisa dilanjutkan dengan --resume.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from rubik import Cube
from rubik.data import generate_batch
from rubik.model import CubeNet, save_model
from rubik.solver import beam_search


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000, help="total langkah training")
    p.add_argument("--scrambles", type=int, default=200, help="jumlah scramble per batch")
    p.add_argument("--max-depth", type=int, default=26, help="panjang scramble (26 = God's number QTM)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save", default="models/cubenet.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def quick_eval(model, rng, depths=(5, 10, 15, 20), n=10, beam_width=64):
    """Persentase scramble yang berhasil diselesaikan pada beberapa tingkat kesulitan."""
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


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    log_path = os.path.splitext(args.save)[0] + "_log.csv"

    model = CubeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0

    if args.resume and os.path.exists(args.save):
        ckpt = torch.load(args.save, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Melanjutkan dari langkah {start_step}")
    if start_step == 0:  # training baru -> buat file log baru
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "accuracy", "solve_d5", "solve_d10", "solve_d15", "solve_d20"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter model: {n_params:,} | thread CPU: {args.threads} | "
          f"sampel per batch: {args.scrambles * args.max_depth:,}")

    def checkpoint(step):
        save_model(model, args.save, optimizer=optimizer.state_dict(), step=step)

    model.train()
    step = start_step
    t_log = time.perf_counter()
    loss_sum = acc_sum = 0.0
    try:
        for step in range(start_step + 1, args.steps + 1):
            X, y, _ = generate_batch(args.scrambles, args.max_depth, rng)
            X, y = torch.from_numpy(X).long(), torch.from_numpy(y)

            logits = model(X)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            acc_sum += (logits.argmax(1) == y).float().mean().item()

            if step % args.log_every == 0:
                dt = time.perf_counter() - t_log
                loss_avg, acc_avg = loss_sum / args.log_every, acc_sum / args.log_every
                remaining = (args.steps - step) * dt / args.log_every
                print(f"step {step:>6} | loss {loss_avg:.4f} | akurasi {acc_avg:.3f} | "
                      f"{args.log_every / dt:.1f} step/dtk | sisa ~{remaining / 60:.0f} mnt")
                eval_row = ["", "", "", ""]
                if step % args.eval_every == 0:
                    res = quick_eval(model, rng)
                    print("   evaluasi (tingkat berhasil per jarak scramble): "
                          + "  ".join(f"d{d}={r:.0%}" for d, r in res.items()))
                    eval_row = [res[d] for d in (5, 10, 15, 20)]
                    checkpoint(step)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{loss_avg:.4f}", f"{acc_avg:.4f}", *eval_row])
                loss_sum = acc_sum = 0.0
                t_log = time.perf_counter()
    except KeyboardInterrupt:
        print("\nTraining dihentikan.")

    checkpoint(step)
    print(f"Model disimpan di {args.save} (langkah {step})")


if __name__ == "__main__":
    main()
