"""
Selesaikan kubus 4x4 dengan METODE REDUKSI (3 tahap):
    1. AI reduksi  : pusat + pasangan tepi        (models/reduksi_4x4.pt)
    2. Paritas     : rumus tetap bila perlu
    3. AI 3x3      : selesaikan sebagai 3x3 besar (models/cubenet_large_infer.pt)

    python -m scripts.solve_4x4_reduksi                         # scramble acak 40 gerakan WCA
    python -m scripts.solve_4x4_reduksi "Rw U2 Fw' R2 D ..."    # scramble sendiri (notasi WCA)
    python -m scripts.solve_4x4_reduksi --depth 15              # scramble AI pendek (uji coba)
"""

import argparse

import numpy as np

from rubik.model import load_model
from rubik.nxn import CubeN
from rubik.reduction import solve_reduction
from rubik.solver import compress_moves
from scripts.solve_ai_4x4 import to_wca

WCA_FACES = ["U", "D", "R", "L", "F", "B", "Uw", "Rw", "Fw"]
AXIS = {"U": 0, "D": 0, "Uw": 0, "R": 1, "L": 1, "Rw": 1, "F": 2, "B": 2, "Fw": 2}


def wca_scramble(length: int = 40, rng=None) -> list[str]:
    """Scramble acak gaya WCA (random-move): tidak memutar sisi yang sama dua kali berturut-turut,
    dan tidak tiga kali berturut-turut pada sumbu yang sama."""
    rng = rng or np.random.default_rng()
    out = []
    while len(out) < length:
        f = WCA_FACES[int(rng.integers(len(WCA_FACES)))]
        prev = [m.rstrip("'2") for m in out[-2:]]
        if prev and prev[-1] == f:
            continue
        if len(prev) == 2 and AXIS[prev[0]] == AXIS[prev[1]] == AXIS[f]:
            continue
        out.append(f + ["", "'", "2"][int(rng.integers(3))])
    return out


def wca_len(moves) -> int:
    return len(compress_moves(list(moves)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scramble", nargs="*")
    p.add_argument("--reduce-model", default="models/reduksi_4x4.pt")
    p.add_argument("--model3", default="models/cubenet_large_infer.pt")
    p.add_argument("--beam", type=int, default=1024, help="lebar pencarian tahap reduksi")
    p.add_argument("--beam3", type=int, default=1024, help="lebar pencarian tahap 3x3")
    p.add_argument("--depth", type=int, default=0, help="pakai scramble AI sepanjang ini (0 = scramble WCA 40)")
    args = p.parse_args()

    cube = CubeN(4)
    if args.scramble:
        scramble = " ".join(args.scramble)
        cube.apply_sequence(scramble)
    elif args.depth:
        scramble = " ".join(to_wca(compress_moves(cube.scramble(args.depth))))
    else:
        scramble = " ".join(wca_scramble())
        cube.apply_sequence(scramble)

    print("1) Pegang kubus: putih di atas, hijau di depan. Lakukan scramble ini:\n   ", scramble)
    print("\n2) Putar SELURUH kubus (tanpa memutar lapisan) sampai sudut KUNING-BIRU-ORANYE ada di")
    print("   bawah-belakang-kiri: kuning menghadap bawah, biru belakang, oranye kiri.\n")
    print(cube.render())

    print(f"\nAI sedang berpikir (reduksi beam {args.beam}, 3x3 beam {args.beam3})...")
    res = solve_reduction(load_model(args.reduce_model), load_model(args.model3), cube,
                          beam_reduce=args.beam, beam_3x3=args.beam3)
    if not res.solved:
        print(f"Belum berhasil: {res.message} ({res.time_sec:.1f} dtk)")
        return

    t1, t2, t3 = to_wca(res.reduce_moves), to_wca(res.parity_moves), list(res.solve3_moves)
    total = compress_moves(t1 + t2 + t3)
    par = [n for n, f in zip(["OLL parity (tepi terbalik)", "PLL parity (tepi tertukar)"], res.parity) if f]
    print(f"\n3) Solusi ({len(total)} gerakan WCA, {res.time_sec:.2f} dtk):")
    print(f"   Tahap 1 - reduksi ({len(t1)}):", " ".join(t1))
    print(f"   Tahap 2 - paritas ({len(t2)}):", " ".join(t2) if t2 else "tidak perlu",
          f"[{', '.join(par)}]" if par else "")
    print(f"   Tahap 3 - 3x3     ({len(t3)}):", " ".join(t3))
    ok = cube.copy().apply_sequence(total).is_solved()
    print("\nVerifikasi di simulator:", "SELESAI" if ok else "GAGAL")


if __name__ == "__main__":
    main()
