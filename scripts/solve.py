"""
Uji coba solver dengan kubus FISIK.

Pemakaian (dari folder root proyek):
    python -m scripts.solve                      # buat scramble acak
    python -m scripts.solve "R U R' U' F2 D"     # pakai scramble sendiri

Cara memegang kubus SELAMA scramble DAN solusi (jangan diputar seluruhnya!):
    - Putih di ATAS, hijau menghadap ANDA (depan)
    - Maka merah di kanan, oranye di kiri, biru di belakang, kuning di bawah
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

    print("Pegang kubus: PUTIH di atas, HIJAU menghadap Anda.")
    print("Mulai dari kubus yang SUDAH SELESAI.\n")
    print("1) Lakukan scramble ini:")
    print("   ", scramble, "\n")
    print("Kubus Anda sekarang harus terlihat seperti ini:")
    print("(U=atas, F=depan, R=kanan, B=belakang, L=kiri, D=bawah)\n")
    print(cube.render())

    t0 = time.perf_counter()
    solution = kociemba.solve(cube.to_facelet_string())
    dt = (time.perf_counter() - t0) * 1000

    print(f"\n2) Lakukan solusi ini ({len(solution.split())} langkah, {dt:.1f} ms):")
    print("   ", solution)
    print("\nVerifikasi di simulator:", "SELESAI" if cube.apply_sequence(solution).is_solved() else "GAGAL")


if __name__ == "__main__":
    main()
