"""
Evaluate the AI model and compare it with Kociemba (baseline).

Examples:
    python -m scripts.evaluate                          # 20 fully random scrambles
    python -m scripts.evaluate --n 50 --beam 1024
    python -m scripts.evaluate --depth 15               # 15-move scrambles
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
    p.add_argument("--n", type=int, default=20, help="number of cubes to test")
    p.add_argument("--depth", type=int, default=100, help="scramble length (100 = fully random)")
    p.add_argument("--beam", type=int, default=512, help="beam search width")
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
            line += f"{len(sol):>2} moves, {res.time_sec:6.2f} s"
        else:
            line += f"FAILED ({res.time_sec:.2f} s)"

        if kociemba:
            t0 = time.perf_counter()
            ks = kociemba.solve(cube.to_facelet_string()).split()
            k_time.append(time.perf_counter() - t0)
            k_len.append(len(ks))
            line += f" | Kociemba: {len(ks):>2} moves"
        print(line)

    print("\n============ SUMMARY ============")
    print(f"AI (beam {args.beam}): solved {ai_ok}/{args.n} ({ai_ok / args.n:.0%})")
    if ai_len:
        print(f"   mean {np.mean(ai_len):.1f} moves (HTM), {np.mean(ai_time):.2f} s")
    if k_len:
        print(f"Kociemba:      solved {args.n}/{args.n} (100%)")
        print(f"   mean {np.mean(k_len):.1f} moves (HTM), {np.mean(k_time) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
