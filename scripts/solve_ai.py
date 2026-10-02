"""
Selesaikan kubus memakai AI (untuk dicoba dengan kubus fisik).

    python -m scripts.solve_ai                         # scramble acak
    python -m scripts.solve_ai "R U R' U' F2 D"        # scramble sendiri
    python -m scripts.solve_ai --beam 2048 "R U F"     # beam lebih lebar

Pegang kubus: PUTIH di atas, HIJAU menghadap Anda, sepanjang scramble & solusi.
"""

import argparse

from rubik import Cube
from rubik.model import load_model
from rubik.solver import beam_search, compress_moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scramble", nargs="*")
    p.add_argument("--model", default="models/cubenet.pt")
    p.add_argument("--beam", type=int, default=1024)
    args = p.parse_args()

    cube = Cube()
    if args.scramble:
        scramble = " ".join(args.scramble)
        cube.apply_sequence(scramble)
    else:
        scramble = " ".join(cube.scramble(20))

    print("Pegang kubus: PUTIH di atas, HIJAU menghadap Anda.\n")
    print("1) Lakukan scramble ini:\n   ", scramble, "\n")
    print(cube.render())

    print(f"\nAI sedang berpikir (beam {args.beam})...")
    res = beam_search(load_model(args.model), cube, beam_width=args.beam)
    if not res.solved:
        print(f"AI belum berhasil ({res.time_sec:.1f} dtk). Coba --beam lebih besar atau latih model lebih lama.")
        return

    sol = compress_moves(res.moves)
    print(f"\n2) Lakukan solusi AI ini ({len(sol)} langkah, {res.time_sec:.2f} dtk, "
          f"{res.nodes:,} state dievaluasi):\n   ", " ".join(sol))
    print("\nVerifikasi di simulator:", "SELESAI" if cube.apply_sequence(sol).is_solved() else "GAGAL")


if __name__ == "__main__":
    main()
