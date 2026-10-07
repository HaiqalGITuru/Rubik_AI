"""
Simulator kubus N x N (dipakai untuk 4x4), berbasis stiker seperti simulator 3x3.

Perbedaan penting 4x4 dibanding 3x3: tidak ada pusat yang tetap, sehingga posisi yang
sama bisa muncul dalam 24 orientasi. Solusinya: SUDUT DBL DIKUNCI.
    - Sudut kuning-biru-oranye selalu berada di bawah-belakang-kiri (Down-Back-Left).
    - AI hanya memakai gerakan yang tidak pernah memindahkan sudut itu:
      lapisan 1..N-1 dihitung dari sisi U, R, dan F (notasi SiGN: U, 2U, 3U, R, 2R, ...).
      "2U" = lapisan kedua dari atas, diputar searah U.
    - Setelah gerakan lain (D, L, B, Uw, rotasi x/y/z, ...), state dinormalisasi dengan
      memutar seluruh kubus sampai sudut DBL kembali ke tempatnya.

Koordinat memakai bilangan bulat "digandakan": pusat kubus kecil berada di
-(N-1), -(N-3), ..., N-1 (untuk 4x4: -3, -1, 1, 3). Urutan wajah & stiker sama dengan 3x3:
U, R, F, D, L, B; tiap wajah N x N, per baris.
"""

from __future__ import annotations

import itertools
import re
from functools import lru_cache

import numpy as np

FACES = "URFDLB"
NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}
COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"]
_ANSI = {0: "\033[97m", 1: "\033[91m", 2: "\033[92m", 3: "\033[93m", 4: "\033[38;5;208m", 5: "\033[94m"}
_RESET = "\033[0m"


def _rotate(v, axis, quarter_turns: int):
    """Rotasi vektor bulat v sebanyak quarter_turns x (-90 derajat) terhadap axis (searah jarum jam)."""
    a = np.array(axis)
    v = np.array(v)
    for _ in range(quarter_turns % 4):
        v = -np.cross(a, v) + a * np.dot(a, v)
    return tuple(int(x) for x in v)


class NxNPuzzle:
    """Spesifikasi kubus N x N: posisi stiker, permutasi gerakan, dan gerakan untuk AI."""

    def __init__(self, n: int):
        self.n = n
        self.num_stickers = 6 * n * n
        m = n - 1
        pos_fn = {
            "U": lambda r, c: (2 * c - m, m, 2 * r - m),
            "D": lambda r, c: (2 * c - m, -m, m - 2 * r),
            "F": lambda r, c: (2 * c - m, m - 2 * r, m),
            "B": lambda r, c: (m - 2 * c, m - 2 * r, -m),
            "R": lambda r, c: (m, m - 2 * r, m - 2 * c),
            "L": lambda r, c: (-m, m - 2 * r, 2 * c - m),
        }
        self.stickers = [(pos_fn[f](r, c), NORMALS[f]) for f in FACES for r in range(n) for c in range(n)]
        self.index = {s: i for i, s in enumerate(self.stickers)}
        self.solved = np.repeat(np.arange(6, dtype=np.uint8), n * n)

        # Gerakan untuk AI: lapisan 1..N-1 dari U, R, F (sudut DBL tidak pernah bergerak)
        self.quarter_moves: list[str] = []
        perms = []
        for face in "URF":
            for layer in range(1, n):
                prefix = "" if layer == 1 else str(layer)
                for suffix, turns in (("", 1), ("'", 3)):
                    self.quarter_moves.append(f"{prefix}{face}{suffix}")
                    perms.append(self.layer_perm(face, (layer,), turns))
        self.quarter_perms = np.stack(perms)
        self.num_moves = len(self.quarter_moves)
        self.inverse_idx = np.array([self.quarter_moves.index(self.inverse(mv)) for mv in self.quarter_moves])

        # Stiker sudut DBL (untuk normalisasi orientasi)
        self.dbl_idx = [i for i, (p, _) in enumerate(self.stickers) if p == (-m, -m, -m)]
        self.dbl_colors = self.solved[self.dbl_idx]
        self.rotations = [self._rotation_perm(r) for r in self._all_rotations()]

    # ------------------------------------------------------------------ #
    def layer_perm(self, face: str, layers: tuple[int, ...], turns: int) -> np.ndarray:
        """Permutasi memutar lapisan-lapisan (1 = paling luar) dari sisi `face`.
        Konvensi sama dengan 3x3: state_baru = state_lama[perm]."""
        axis = NORMALS[face]
        m = self.n - 1
        levels = {m - 2 * (k - 1) for k in layers}
        perm = np.arange(self.num_stickers)
        for src, (pos, normal) in enumerate(self.stickers):
            if int(np.dot(pos, axis)) in levels:
                dest = self.index[(_rotate(pos, axis, turns), _rotate(normal, axis, turns))]
                perm[dest] = src
        return perm

    def _all_rotations(self):
        """24 rotasi seluruh kubus, sebagai kombinasi putaran x, y, z."""
        seen, out = set(), []
        for a, b, c in itertools.product(range(4), repeat=3):
            key = []
            for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                v = _rotate(v, NORMALS["R"], a)
                v = _rotate(v, NORMALS["U"], b)
                v = _rotate(v, NORMALS["F"], c)
                key.append(v)
            if tuple(key) not in seen:
                seen.add(tuple(key))
                out.append((a, b, c))
        return out

    def _rotation_perm(self, rot) -> np.ndarray:
        a, b, c = rot
        all_layers = tuple(range(1, self.n + 1))
        p = np.arange(self.num_stickers)
        for face, k in (("R", a), ("U", b), ("F", c)):
            if k:
                p = p[self.layer_perm(face, all_layers, k)]
        return p

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Putar seluruh kubus sampai sudut kuning-biru-oranye kembali di DBL."""
        if np.array_equal(state[self.dbl_idx], self.dbl_colors):
            return state
        for p in self.rotations:
            s = state[p]
            if np.array_equal(s[self.dbl_idx], self.dbl_colors):
                return s
        raise ValueError("State tidak valid: sudut DBL tidak ditemukan")

    # ------------------------------------------------------------------ #
    @staticmethod
    def inverse(move: str) -> str:
        if move.endswith("2"):
            return move
        return move[:-1] if move.endswith("'") else move + "'"

    _TOKEN = re.compile(r"^(\d*)([URFDLBurfdlbxyz])(w?)(['2]?)$")

    @lru_cache(maxsize=None)
    def token_perm(self, token: str) -> np.ndarray:
        """Permutasi fisik untuk satu token notasi (SiGN/WCA):
        U, U', U2, 2U (lapisan ke-2), Uw / u (2 lapisan), 3Rw (3 lapisan), x/y/z (seluruh kubus)."""
        mt = self._TOKEN.match(token)
        if not mt:
            raise ValueError(f"Gerakan tidak valid: {token!r}")
        num, face, wide, suffix = mt.groups()
        turns = {"": 1, "'": 3, "2": 2}[suffix]
        if face in "xyz":
            face = {"x": "R", "y": "U", "z": "F"}[face]
            layers = tuple(range(1, self.n + 1))
        else:
            if face.islower():
                face, wide = face.upper(), "w"
            k = int(num) if num else (2 if wide else 1)
            if not 1 <= k <= self.n:
                raise ValueError(f"Lapisan tidak valid untuk {self.n}x{self.n}: {token!r}")
            layers = tuple(range(1, k + 1)) if wide else (k,)
        return self.layer_perm(face, layers, turns)


@lru_cache(maxsize=None)
def get_puzzle(n: int) -> NxNPuzzle:
    return NxNPuzzle(n)


class CubeN:
    """Kubus N x N. Selalu disimpan dalam orientasi baku (sudut DBL terkunci)."""

    def __init__(self, n: int = 4, state: np.ndarray | None = None):
        self.puzzle = get_puzzle(n)
        self.n = n
        self.state = self.puzzle.solved.copy() if state is None else np.asarray(state, dtype=np.uint8).copy()

    def copy(self) -> "CubeN":
        return CubeN(self.n, self.state)

    def apply(self, token: str) -> "CubeN":
        self.state = self.puzzle.normalize(self.state[self.puzzle.token_perm(token)])
        return self

    def apply_sequence(self, seq) -> "CubeN":
        tokens = seq.replace("’", "'").split() if isinstance(seq, str) else list(seq)
        for t in tokens:
            self.apply(t)
        return self

    def is_solved(self) -> bool:
        faces = self.state.reshape(6, -1)
        return bool(np.all(faces == faces[:, [0]]))

    def scramble(self, n_moves: int = 40, rng=None, **_) -> list[str]:
        """Acak memakai gerakan AI (sudut DBL tetap), tanpa membatalkan gerakan sebelumnya."""
        rng = rng or np.random.default_rng()
        p = self.puzzle
        moves, last = [], None
        while len(moves) < n_moves:
            i = int(rng.integers(p.num_moves))
            if last is not None and i == p.inverse_idx[last]:
                continue
            if len(moves) >= 2 and moves[-1] == moves[-2] == p.quarter_moves[i]:
                continue
            moves.append(p.quarter_moves[i])
            last = i
        self.apply_sequence(moves)
        return moves

    def to_one_hot(self) -> np.ndarray:
        return np.eye(6, dtype=np.float32)[self.state].reshape(-1)

    def render(self, color: bool = True) -> str:
        n = self.n
        faces = {f: self.state[i * n * n:(i + 1) * n * n].reshape(n, n) for i, f in enumerate(FACES)}
        cell = lambda c: f"{_ANSI[c]}{COLOR_NAMES[c]}{_RESET}" if color else COLOR_NAMES[c]
        row = lambda f, r: " ".join(cell(int(c)) for c in faces[f][r])
        pad = " " * (2 * n + 1)
        lines = [pad + row("U", r) for r in range(n)]
        lines += ["  ".join(row(f, r) for f in "LFRB") for r in range(n)]
        lines += [pad + row("D", r) for r in range(n)]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render(color=False)
