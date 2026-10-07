"""
Generator data training (self-supervised, tanpa dataset eksternal).

Ide (EfficientCube, Takano 2023):
    Mulai dari kubus selesai, lakukan gerakan acak satu per satu.
    Untuk setiap state yang dilewati, label-nya adalah KEBALIKAN dari
    gerakan terakhir yang membawa kubus ke state itu.
    Jadi jaringan belajar: "dari state ini, gerakan apa yang membawa
    saya selangkah lebih dekat ke kondisi selesai?"

Semua scramble dibuat paralel dengan NumPy, sehingga sangat cepat.
"""

from __future__ import annotations

import numpy as np

from .cube import SOLVED_STATE
from .moves import MOVE_PERMS, QUARTER_MOVES, inverse_move

# Tabel permutasi 12 gerakan seperempat, shape (12, 54)
QUARTER_PERMS = np.stack([MOVE_PERMS[m] for m in QUARTER_MOVES])
NUM_MOVES = len(QUARTER_MOVES)

# INVERSE_IDX[i] = indeks gerakan kebalikan dari gerakan ke-i
INVERSE_IDX = np.array([QUARTER_MOVES.index(inverse_move(m)) for m in QUARTER_MOVES])


def apply_moves_batch(states: np.ndarray, move_idx: np.ndarray, perms: np.ndarray = QUARTER_PERMS) -> np.ndarray:
    """Terapkan satu gerakan berbeda ke setiap state dalam batch.

    states:   (N, S) uint8   (S = 54 untuk 3x3, 96 untuk 4x4)
    move_idx: (N,)    indeks gerakan
    """
    return np.take_along_axis(states, perms[move_idx], axis=1)


def _random_moves(n: int, depth: int, rng: np.random.Generator,
                  num_moves: int = NUM_MOVES, inverse_idx: np.ndarray = INVERSE_IDX) -> np.ndarray:
    """Buat (n, depth) gerakan acak tanpa langkah sia-sia:
    - tidak langsung membatalkan gerakan sebelumnya (R lalu R')
    - tidak memutar wajah yang sama 3x berturut-turut (R R R = R')
    """
    moves = np.empty((n, depth), dtype=np.int64)
    for t in range(depth):
        m = rng.integers(num_moves, size=n)
        while True:
            bad = np.zeros(n, dtype=bool)
            if t >= 1:
                bad |= m == inverse_idx[moves[:, t - 1]]
            if t >= 2:
                bad |= (m == moves[:, t - 1]) & (m == moves[:, t - 2])
            if not bad.any():
                break
            m[bad] = rng.integers(num_moves, size=int(bad.sum()))
        moves[:, t] = m
    return moves


def generate_batch(
    num_scrambles: int,
    max_depth: int,
    rng: np.random.Generator | None = None,
    puzzle=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buat satu batch data training.

    Mengembalikan:
        X:      (num_scrambles * max_depth, 54) uint8  -> state kubus
        y:      (num_scrambles * max_depth,)    int64  -> label gerakan (0..11)
        depths: (num_scrambles * max_depth,)    int64  -> jarak scramble (1..max_depth)
    """
    rng = rng or np.random.default_rng()
    if puzzle is None:  # default: kubus 3x3
        perms, inv, solved = QUARTER_PERMS, INVERSE_IDX, SOLVED_STATE
    else:               # misalnya NxNPuzzle untuk 4x4
        perms, inv, solved = puzzle.quarter_perms, puzzle.inverse_idx, puzzle.solved
    S = len(solved)
    moves = _random_moves(num_scrambles, max_depth, rng, len(perms), inv)

    states = np.tile(solved, (num_scrambles, 1))
    X = np.empty((max_depth, num_scrambles, S), dtype=np.uint8)
    y = np.empty((max_depth, num_scrambles), dtype=np.int64)
    for t in range(max_depth):
        states = apply_moves_batch(states, moves[:, t], perms)
        X[t] = states
        y[t] = inv[moves[:, t]]

    depths = np.repeat(np.arange(1, max_depth + 1), num_scrambles)
    return X.reshape(-1, S), y.reshape(-1), depths