"""
Evaluasi model AI dan bandingkan dengan Kociemba (baseline).

Contoh:
    python -m scripts.evaluate                          # 20 scramble acak penuh
    python -m scripts.evaluate --n 50 --beam 1024
    python -m scripts.evaluate --depth 15               # scramble 15 langkah
"""

import argparse
import time

import numpy as np

from rubik import Cube
from rubik.model import load_model
from rubik.solver import beam_search, compress_moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/cubenet.pt")
    p.add_argument("--n", type=int, default=20, help="jumlah kubus yang diuji")
    p.add_argument("--depth", type=int, default=100, help="panjang scramble (100 = acak penuh)")
    p.add_argument("--beam", type=int, default=512, help="lebar beam search")
    p.add_argument("--max-depth", type=int, default=40)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    try:
        import kociemba
    except ImportError:
        kociemba = None

    model = load_model(args.model)
    rng = np.random.default_rng(args.seed)

    ai_ok, ai_len, ai_time, k_len, k_time = 0, [], [], [], []
    for i in range(1, args.n + 1):
        cube = Cube()
        cube.scramble(args.depth, rng=rng)

        res = beam_search(model, cube, beam_width=args.beam, max_depth=args.max_depth)
        line = f"[{i:>3}/{args.n}] AI: "
        if res.solved:
            sol = compress_moves(res.moves)
            assert cube.copy().apply_sequence(sol).is_solved()
            ai_ok += 1
            ai_len.append(len(sol))
            ai_time.append(res.time_sec)
            line += f"{len(sol):>2} langkah, {res.time_sec:6.2f} dtk"
        else:
            line += f"GAGAL ({res.time_sec:.2f} dtk)"

        if kociemba:
            t0 = time.perf_counter()
            ks = kociemba.solve(cube.to_facelet_string()).split()
            k_time.append(time.perf_counter() - t0)
            k_len.append(len(ks))
            line += f" | Kociemba: {len(ks):>2} langkah"
        print(line)

    print("\n=========== RINGKASAN ===========")
    print(f"AI (beam {args.beam}): berhasil {ai_ok}/{args.n} ({ai_ok / args.n:.0%})")
    if ai_len:
        print(f"   rata-rata {np.mean(ai_len):.1f} langkah (HTM), {np.mean(ai_time):.2f} dtk")
    if k_len:
        print(f"Kociemba:      berhasil {args.n}/{args.n} (100%)")
        print(f"   rata-rata {np.mean(k_len):.1f} langkah (HTM), {np.mean(k_time) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
