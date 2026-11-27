"""
Solve a cube with the AI (to try on a physical cube).

    python -m scripts.solve_ai                         # random scramble
    python -m scripts.solve_ai "R U R' U' F2 D"        # your own scramble
    python -m scripts.solve_ai --beam 2048 "R U F"     # wider beam

Hold the cube with WHITE on top and GREEN facing you for both the scramble and the solution.
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

    print("Hold the cube: WHITE on top, GREEN facing you.\n")
    print("1) Apply this scramble:\n   ", scramble, "\n")
    print(cube.render())

    print(f"\nThe AI is thinking (beam {args.beam})...")
    res = beam_search(load_model(args.model), cube, beam_width=args.beam)
    if not res.solved:
        print(f"The AI did not find a solution ({res.time_sec:.1f} s). Try a larger --beam or train the model longer.")
        return

    sol = compress_moves(res.moves)
    print(f"\n2) Apply the AI's solution ({len(sol)} moves, {res.time_sec:.2f} s, "
          f"{res.nodes:,} states evaluated):\n   ", " ".join(sol))
    print("\nSimulator check:", "SOLVED" if cube.apply_sequence(sol).is_solved() else "FAILED")


if __name__ == "__main__":
    main()
