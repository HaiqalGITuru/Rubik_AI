"""
Try the Kociemba solver on a PHYSICAL cube.

Usage (from the project root):
    python -m scripts.solve                      # random scramble
    python -m scripts.solve "R U R' U' F2 D"     # your own scramble

How to hold the cube DURING both the scramble AND the solution (do not rotate the whole cube!):
    - White on TOP, green facing YOU (front)
    - Then red is on the right, orange on the left, blue at the back, yellow at the bottom
"""

import sys
import time

import kociemba

from rubik import Cube


def main():
    cube = Cube()

    if len(sys.argv) > 1:
        scramble = " ".join(sys.argv[1:])
        cube.apply_sequence(scramble)
    else:
        scramble = " ".join(cube.scramble(20))

    print("Hold the cube: WHITE on top, GREEN facing you.")
    print("Start from a SOLVED cube.\n")
    print("1) Apply this scramble:")
    print("   ", scramble, "\n")
    print("Your cube should now look like this:")
    print("(U=up, F=front, R=right, B=back, L=left, D=down)\n")
    print(cube.render())

    t0 = time.perf_counter()
    solution = kociemba.solve(cube.to_facelet_string())
    dt = (time.perf_counter() - t0) * 1000

    print(f"\n2) Apply this solution ({len(solution.split())} moves, {dt:.1f} ms):")
    print("   ", solution)
    print("\nSimulator check:", "SOLVED" if cube.apply_sequence(solution).is_solved() else "FAILED")


if __name__ == "__main__":
    main()
