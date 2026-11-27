"""Simulator demo. Run from the project root:  python -m scripts.demo"""

import time

import numpy as np

from rubik import Cube, invert_sequence


def main():
    rng = np.random.default_rng()

    cube = Cube()
    print("Solved cube:\n")
    print(cube.render())

    scramble = cube.scramble(20, rng=rng)
    print("\nScramble:", " ".join(scramble), "\n")
    print(cube.render())
    print("\nSolved?", cube.is_solved())

    # "Cheat": solve by reversing the scramble (only to check the simulator)
    cube.apply_sequence(invert_sequence(scramble))
    print("\nAfter reversing the scramble, solved?", cube.is_solved())

    # Non-AI baseline: Kociemba two-phase
    try:
        import kociemba

        # Note: this is a NEW scramble (different from the one above), only to
        # measure Kociemba's speed. To try it on a physical cube,
        # use:  python -m scripts.solve
        cube = Cube()
        scramble = cube.scramble(25, rng=rng)
        print("\nNew scramble for Kociemba:", " ".join(scramble))
        t0 = time.perf_counter()
        solution = kociemba.solve(cube.to_facelet_string())
        dt = (time.perf_counter() - t0) * 1000
        n = len(solution.split())
        print(f"\nKociemba: {n} moves in {dt:.1f} ms -> {solution}")
        print("Valid?", cube.apply_sequence(solution).is_solved())
    except ImportError:
        print("\n(Install 'kociemba' to see the baseline: pip install kociemba)")

    # Simulator speed (matters when generating training data)
    cube = Cube()
    t0 = time.perf_counter()
    for _ in range(100_000):
        cube.apply("R")
    dt = time.perf_counter() - t0
    print(f"\nSimulator speed: {100_000 / dt:,.0f} moves/second")


if __name__ == "__main__":
    main()