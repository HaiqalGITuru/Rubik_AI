"""
Training data generator (self-supervised, no external dataset).

Idea (EfficientCube, Takano 2023):
    Start from a solved cube and apply random moves one at a time.
    For every state visited, the label is the INVERSE of the last move
    that led the cube to that state.
    So the network learns: "from this state, which move brings me
    one step closer to solved?"

All scrambles are generated in parallel with NumPy, so it is very fast.
"""

from __future__ import annotations

import numpy as np

from .cube import SOLVED_STATE
from .moves import MOVE_PERMS, QUARTER_MOVES, inverse_move

# Permutation table of the 12 quarter turns, shape (12, 54)
QUARTER_PERMS = np.stack([MOVE_PERMS[m] for m in QUARTER_MOVES])
NUM_MOVES = len(QUARTER_MOVES)

# INVERSE_IDX[i] = index of the inverse of move i
INVERSE_IDX = np.array([QUARTER_MOVES.index(inverse_move(m)) for m in QUARTER_MOVES])


def apply_moves_batch(states: np.ndarray, move_idx: np.ndarray, perms: np.ndarray = QUARTER_PERMS) -> np.ndarray:
    """Apply a (possibly different) move to every state in a batch.

    states:   (N, 54) uint8
    move_idx: (N,)    move indices
    """
    return np.take_along_axis(states, perms[move_idx], axis=1)


def _random_moves(n: int, depth: int, rng: np.random.Generator,
                  num_moves: int = NUM_MOVES, inverse_idx: np.ndarray = INVERSE_IDX) -> np.ndarray:
    """Generate (n, depth) random moves without wasted moves:
    - never directly undo the previous move (R then R')
    - never turn the same face 3 times in a row (R R R = R')
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one batch of training data.

    Returns:
        X:      (num_scrambles * max_depth, 54) uint8  -> cube states
        y:      (num_scrambles * max_depth,)    int64  -> move labels (0..11)
        depths: (num_scrambles * max_depth,)    int64  -> scramble depth (1..max_depth)
    """
    rng = rng or np.random.default_rng()
    perms, inv, solved = QUARTER_PERMS, INVERSE_IDX, SOLVED_STATE
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