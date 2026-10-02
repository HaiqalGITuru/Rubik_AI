"""Demo simulator. Jalankan dari folder root proyek:  python -m scripts.demo"""

import time

import numpy as np

from rubik import Cube, invert_sequence


def main():
    rng = np.random.default_rng()

    cube = Cube()
    print("Kubus selesai:\n")
    print(cube.render())

    scramble = cube.scramble(20, rng=rng)
    print("\nScramble:", " ".join(scramble), "\n")
    print(cube.render())
    print("\nSelesai?", cube.is_solved())

    # "Curang": selesaikan dengan membalik scramble (hanya untuk mengecek simulator)
    cube.apply_sequence(invert_sequence(scramble))
    print("\nSetelah membalik scramble, selesai?", cube.is_solved())

    # Pembanding (baseline) non-AI: Kociemba two-phase
    try:
        import kociemba

        # Catatan: ini acakan BARU (berbeda dari di atas), hanya untuk
        # mengukur kecepatan Kociemba. Untuk uji coba dengan kubus fisik,
        # gunakan:  python -m scripts.solve
        cube = Cube()
        scramble = cube.scramble(25, rng=rng)
        print("\nScramble baru untuk Kociemba:", " ".join(scramble))
        t0 = time.perf_counter()
        solution = kociemba.solve(cube.to_facelet_string())
        dt = (time.perf_counter() - t0) * 1000
        n = len(solution.split())
        print(f"\nKociemba: {n} langkah dalam {dt:.1f} ms -> {solution}")
        print("Valid?", cube.apply_sequence(solution).is_solved())
    except ImportError:
        print("\n(Install 'kociemba' untuk melihat pembanding: pip install kociemba)")

    # Kecepatan simulator (penting saat membuat data training)
    cube = Cube()
    t0 = time.perf_counter()
    for _ in range(100_000):
        cube.apply("R")
    dt = time.perf_counter() - t0
    print(f"\nKecepatan simulator: {100_000 / dt:,.0f} gerakan/detik")


if __name__ == "__main__":
    main()