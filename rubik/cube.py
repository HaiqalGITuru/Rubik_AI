"""
Simulator kubus rubik 3x3 berbasis representasi stiker (54 elemen).

Warna (skema standar):
    0 = putih (U), 1 = merah (R), 2 = hijau (F),
    3 = kuning (D), 4 = oranye (L), 5 = biru (B)
"""

from __future__ import annotations

import numpy as np

from .moves import ALL_MOVES, FACES, MOVE_PERMS, QUARTER_MOVES, parse_sequence

COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"]
_ANSI = {
    0: "\033[97m",          # putih
    1: "\033[91m",          # merah
    2: "\033[92m",          # hijau
    3: "\033[93m",          # kuning
    4: "\033[38;5;208m",    # oranye
    5: "\033[94m",          # biru
}
_RESET = "\033[0m"

SOLVED_STATE = np.repeat(np.arange(6, dtype=np.uint8), 9)


class Cube:
    def __init__(self, state: np.ndarray | None = None):
        if state is None:
            self.state = SOLVED_STATE.copy()
        else:
            state = np.asarray(state, dtype=np.uint8)
            if state.shape != (54,):
                raise ValueError("State harus berukuran 54")
            self.state = state.copy()

    # ------------------------------------------------------------------ #
    # Operasi dasar
    # ------------------------------------------------------------------ #
    def copy(self) -> "Cube":
        return Cube(self.state)

    def apply(self, move: str) -> "Cube":
        """Terapkan satu gerakan (in-place). Mengembalikan self agar bisa di-chain."""
        self.state = self.state[MOVE_PERMS[move]]
        return self

    def apply_sequence(self, seq: str | list[str]) -> "Cube":
        """Terapkan urutan gerakan, contoh: cube.apply_sequence("R U R' U'")."""
        for m in parse_sequence(seq):
            self.state = self.state[MOVE_PERMS[m]]
        return self

    def is_solved(self) -> bool:
        # Tiap wajah harus satu warna (tidak bergantung orientasi kubus)
        faces = self.state.reshape(6, 9)
        return bool(np.all(faces == faces[:, [4]]))

    def scramble(
        self,
        n_moves: int = 25,
        rng: np.random.Generator | None = None,
        quarter_turns_only: bool = False,
    ) -> list[str]:
        """Acak kubus sebanyak n_moves. Mengembalikan daftar gerakan yang dipakai.

        Gerakan pada wajah yang sama berturut-turut dihindari (misal U lalu U'),
        karena itu hanya membuang langkah.
        """
        rng = rng or np.random.default_rng()
        pool = QUARTER_MOVES if quarter_turns_only else ALL_MOVES
        moves: list[str] = []
        last_face = None
        while len(moves) < n_moves:
            m = pool[rng.integers(len(pool))]
            if m[0] == last_face:
                continue
            moves.append(m)
            last_face = m[0]
        self.apply_sequence(moves)
        return moves

    # ------------------------------------------------------------------ #
    # Konversi (berguna untuk machine learning & pembanding)
    # ------------------------------------------------------------------ #
    def to_one_hot(self) -> np.ndarray:
        """Vektor one-hot berukuran 54 x 6 = 324, siap jadi input jaringan saraf."""
        return np.eye(6, dtype=np.float32)[self.state].reshape(-1)

    def to_facelet_string(self) -> str:
        """Format string Kociemba, misal 'UUUUUUUUURRR...'.

        Bisa langsung dipakai: kociemba.solve(cube.to_facelet_string())
        """
        centers = self.state.reshape(6, 9)[:, 4]
        color_to_face = {int(c): FACES[i] for i, c in enumerate(centers)}
        return "".join(color_to_face[int(c)] for c in self.state)

    # ------------------------------------------------------------------ #
    # Tampilan
    # ------------------------------------------------------------------ #
    def render(self, color: bool = True) -> str:
        """Tampilkan jaring (net) kubus:

                U
              L F R B
                D
        """
        faces = {f: self.state[i * 9:(i + 1) * 9].reshape(3, 3) for i, f in enumerate(FACES)}

        def cell(c: int) -> str:
            ch = COLOR_NAMES[c]
            return f"{_ANSI[c]}{ch}{_RESET}" if color else ch

        def row(face: str, r: int) -> str:
            return " ".join(cell(int(c)) for c in faces[face][r])

        pad = " " * 7
        lines = [pad + row("U", r) for r in range(3)]
        lines += [
            "  ".join(row(f, r) for f in ("L", "F", "R", "B")) for r in range(3)
        ]
        lines += [pad + row("D", r) for r in range(3)]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render(color=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cube) and np.array_equal(self.state, other.state)

    def __hash__(self) -> int:
        return hash(self.state.tobytes())
