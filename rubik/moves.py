"""
Definisi gerakan kubus rubik 3x3.

Alih-alih menulis tabel permutasi secara manual (rawan salah ketik),
permutasi setiap gerakan DIHITUNG dari geometri 3D:

- Setiap stiker punya posisi (x, y, z) dengan nilai -1, 0, atau 1
  dan vektor normal (arah hadap stiker).
- Sumbu koordinat: x -> kanan (R), y -> atas (U), z -> depan (F).
- Gerakan searah jarum jam (dilihat dari sisi wajah tersebut)
  = rotasi -90 derajat terhadap normal wajah itu.

Urutan stiker (indeks 0..53) mengikuti format facelet Kociemba:
U(0-8), R(9-17), F(18-26), D(27-35), L(36-44), B(45-53).
Di dalam tiap wajah, indeks berurutan per baris (kiri->kanan, atas->bawah):

        0 1 2
        3 4 5
        6 7 8
"""

from __future__ import annotations

import numpy as np

FACES = "URFDLB"

FACE_NORMALS: dict[str, tuple[int, int, int]] = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}


def _sticker_position(face: str, row: int, col: int) -> tuple[int, int, int]:
    """Posisi 3D stiker berdasarkan wajah, baris, dan kolom pada jaring (net) kubus."""
    if face == "U":   # dilihat dari atas, B di atas, F di bawah
        return (col - 1, 1, row - 1)
    if face == "D":   # dilihat dari bawah, F di atas, B di bawah
        return (col - 1, -1, 1 - row)
    if face == "F":   # dilihat dari depan
        return (col - 1, 1 - row, 1)
    if face == "B":   # dilihat dari belakang, R di kiri
        return (1 - col, 1 - row, -1)
    if face == "R":   # dilihat dari kanan, F di kiri
        return (1, 1 - row, 1 - col)
    if face == "L":   # dilihat dari kiri, B di kiri
        return (-1, 1 - row, col - 1)
    raise ValueError(f"Wajah tidak dikenal: {face}")


# Daftar semua stiker: (posisi, normal)
STICKERS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    (_sticker_position(face, r, c), FACE_NORMALS[face])
    for face in FACES
    for r in range(3)
    for c in range(3)
]
_INDEX = {sticker: i for i, sticker in enumerate(STICKERS)}


def _rotate_cw(v: tuple[int, int, int], axis: tuple[int, int, int]) -> tuple[int, int, int]:
    """Rotasi vektor v sebesar -90 derajat terhadap sumbu (searah jarum jam dilihat dari ujung sumbu).

    Rumus Rodrigues dengan theta = -90 derajat:  v' = -(a x v) + a (a . v)
    """
    a = np.array(axis)
    v = np.array(v)
    out = -np.cross(a, v) + a * np.dot(a, v)
    return tuple(int(x) for x in out)


def _quarter_turn_perm(face: str) -> np.ndarray:
    """Permutasi untuk putaran seperempat searah jarum jam pada satu wajah.

    Konvensi: state_baru = state_lama[perm]
    """
    axis = FACE_NORMALS[face]
    perm = np.arange(54)
    for src, (pos, normal) in enumerate(STICKERS):
        if np.dot(pos, axis) == 1:  # stiker berada di lapisan yang diputar
            dest = _INDEX[(_rotate_cw(pos, axis), _rotate_cw(normal, axis))]
            perm[dest] = src
    return perm


def _build_move_table() -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for face in FACES:
        p1 = _quarter_turn_perm(face)
        p2 = p1[p1]
        p3 = p2[p1]
        table[face] = p1          # searah jarum jam
        table[face + "2"] = p2    # setengah putaran
        table[face + "'"] = p3    # berlawanan jarum jam
    return table


MOVE_PERMS: dict[str, np.ndarray] = _build_move_table()

# 18 gerakan (Half-Turn Metric) dan 12 gerakan seperempat (Quarter-Turn Metric).
# QUARTER_MOVES biasanya dipakai sebagai kelas output jaringan saraf.
ALL_MOVES: list[str] = [f + s for f in FACES for s in ("", "'", "2")]
QUARTER_MOVES: list[str] = [f + s for f in FACES for s in ("", "'")]

MOVE_TO_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}


def inverse_move(move: str) -> str:
    """Kebalikan dari satu gerakan: U <-> U', U2 tetap U2."""
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[0]
    return move + "'"


def parse_sequence(seq: str | list[str]) -> list[str]:
    """Ubah string notasi seperti "R U R' U'" menjadi list gerakan dan validasi."""
    moves = seq.split() if isinstance(seq, str) else list(seq)
    for m in moves:
        if m not in MOVE_PERMS:
            raise ValueError(f"Gerakan tidak valid: {m!r}")
    return moves


def invert_sequence(seq: str | list[str]) -> list[str]:
    """Kebalikan dari satu urutan gerakan (dibalik urutannya, tiap gerakan diinvers)."""
    return [inverse_move(m) for m in reversed(parse_sequence(seq))]
