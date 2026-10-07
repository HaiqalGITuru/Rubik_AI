"""
Selesaikan kubus 4x4 memakai AI (untuk dicoba dengan kubus fisik).

    python -m scripts.solve_ai_4x4                        # scramble acak 15 gerakan
    python -m scripts.solve_ai_4x4 --depth 25             # lebih sulit
    python -m scripts.solve_ai_4x4 "R 2U F' 3R U2"        # scramble sendiri

Notasi lapisan (SiGN):
    U  = lapisan teratas, 2U = lapisan kedua dari atas, 3U = lapisan ketiga dari atas,
    semuanya diputar searah U (searah jarum jam dilihat dari atas). Sama untuk R dan F.
    Notasi WCA (D, L, B, Uw, Rw, x, y, z, ...) juga bisa dipakai untuk scramble.
"""

import argparse

from rubik.model import load_model
from rubik.nxn import CubeN
from rubik.solver import beam_search, compress_moves

OPPOSITE = {"U": "D", "R": "L", "F": "B"}
INV = {"": "'", "'": "", "2": "2"}


def _split(move: str):
    """'2U'' -> (2, 'U', "'")"""
    layer = int(move[0]) if move[0].isdigit() else 1
    rest = move[1:] if move[0].isdigit() else move
    return layer, rest[0], rest[1:]


def to_wca(moves: list[str]) -> list[str]:
    """Ubah notasi lapisan (SiGN) ke notasi WCA yang biasa dipakai speedcuber.
        2U  = Uw U'   (dua lapisan atas, lalu kembalikan lapisan teratas)
        3U  = Dw' D   (lapisan ketiga dari atas = lapisan kedua dari bawah)
        R 2R (searah) = Rw
    Sudut kuning-biru-oranye tetap tidak berpindah, jadi cara memegang kubus tidak berubah."""
    out, i = [], 0
    while i < len(moves):
        layer, face, suf = _split(moves[i])
        if i + 1 < len(moves):  # gabungkan X dan 2X yang searah menjadi Xw
            l2, f2, s2 = _split(moves[i + 1])
            if f2 == face and s2 == suf and {layer, l2} == {1, 2}:
                out.append(f"{face}w{suf}")
                i += 2
                continue
        if layer == 1:
            out.append(f"{face}{suf}")
        elif layer == 2:
            out += [f"{face}w{suf}", f"{face}{INV[suf]}"]
        else:  # layer 3 pada 4x4
            o = OPPOSITE[face]
            out += [f"{o}w{INV[suf]}", f"{o}{suf}"]
        i += 1
    return compress_moves(out)  # rapikan, misalnya "U2 U" -> "U'"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scramble", nargs="*")
    p.add_argument("--model", default="models/cubenet_4x4.pt")
    p.add_argument("--beam", type=int, default=1024)
    p.add_argument("--depth", type=int, default=15, help="panjang scramble acak")
    p.add_argument("--sign", action="store_true", help="tampilkan notasi lapisan (2U, 3R, ...) saja")
    args = p.parse_args()

    cube = CubeN(4)
    if args.scramble:
        scramble = " ".join(args.scramble)
        cube.apply_sequence(scramble)
    else:
        sign = compress_moves(cube.scramble(args.depth))
        scramble = " ".join(sign if args.sign else to_wca(sign))

    print("Pegang kubus dengan sudut KUNING-BIRU-ORANYE di bawah-belakang-kiri:")
    print("kuning di bawah, biru di belakang, oranye di kiri (putih di atas, hijau di depan).\n")
    print("1) Lakukan scramble ini:\n   ", scramble, "\n")
    print(cube.render())

    print(f"\nAI sedang berpikir (beam {args.beam})...")
    res = beam_search(load_model(args.model), cube, beam_width=args.beam, max_depth=60)
    if not res.solved:
        print(f"AI belum berhasil ({res.time_sec:.1f} dtk). Coba --beam lebih besar atau scramble lebih pendek.")
        return
    sol = compress_moves(res.moves)
    wca = to_wca(sol)
    print(f"\n2) Lakukan solusi AI ini ({len(sol)} langkah, {res.time_sec:.2f} dtk):")
    if args.sign:
        print("   ", " ".join(sol))
    else:
        print("    notasi WCA :", " ".join(wca))
        print("    (notasi lapisan:", " ".join(sol) + ")")
    check = cube.copy().apply_sequence(sol).is_solved() and cube.copy().apply_sequence(wca).is_solved()
    print("\nVerifikasi di simulator:", "SELESAI" if check else "GAGAL")
    print("\nCatatan: kalau scramble memakai D, L, B, gerakan lebar (Uw, Rw, ...) atau rotasi (x, y, z),")
    print("sudut kuning-biru-oranye bisa ikut berpindah. Sebelum memulai solusi, putar SELURUH kubus")
    print("(tanpa memutar lapisan) sampai sudut itu kembali di bawah-belakang-kiri: kuning menghadap")
    print("bawah, biru menghadap belakang, oranye menghadap kiri.")


if __name__ == "__main__":
    main()