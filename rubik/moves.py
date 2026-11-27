"""
Move definitions for the 3x3 Rubik's cube.

Instead of writing permutation tables by hand (easy to mistype),
each move's permutation is COMPUTED from 3D geometry:

- Every sticker has a position (x, y, z) with values -1, 0 or 1
  and a normal vector (the direction the sticker faces).
- Axes: x -> right (R), y -> up (U), z -> front (F).
- A clockwise move (seen from that face)
  = a -90 degree rotation about the face normal.

Sticker order (indices 0..53) follows Kociemba's facelet format:
U(0-8), R(9-17), F(18-26), D(27-35), L(36-44), B(45-53).
Within each face, indices run row by row (left->right, top->bottom):

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
    """3D position of a sticker from its face, row and column on the cube net."""
    if face == "U":   # seen from above, B at the top, F at the bottom
        return (col - 1, 1, row - 1)
    if face == "D":   # seen from below, F at the top, B at the bottom
        return (col - 1, -1, 1 - row)
    if face == "F":   # seen from the front
        return (col - 1, 1 - row, 1)
    if face == "B":   # seen from the back, R on the left
        return (1 - col, 1 - row, -1)
    if face == "R":   # seen from the right, F on the left
        return (1, 1 - row, 1 - col)
    if face == "L":   # seen from the left, B on the left
        return (-1, 1 - row, col - 1)
    raise ValueError(f"Unknown face: {face}")


# All stickers: (position, normal)
STICKERS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    (_sticker_position(face, r, c), FACE_NORMALS[face])
    for face in FACES
    for r in range(3)
    for c in range(3)
]
_INDEX = {sticker: i for i, sticker in enumerate(STICKERS)}


def _rotate_cw(v: tuple[int, int, int], axis: tuple[int, int, int]) -> tuple[int, int, int]:
    """Rotate vector v by -90 degrees about an axis (clockwise seen from the axis tip).

    Rodrigues' formula with theta = -90 degrees:  v' = -(a x v) + a (a . v)
    """
    a = np.array(axis)
    v = np.array(v)
    out = -np.cross(a, v) + a * np.dot(a, v)
    return tuple(int(x) for x in out)


def _quarter_turn_perm(face: str) -> np.ndarray:
    """Permutation for a clockwise quarter turn of one face.

    Convention: new_state = old_state[perm]
    """
    axis = FACE_NORMALS[face]
    perm = np.arange(54)
    for src, (pos, normal) in enumerate(STICKERS):
        if np.dot(pos, axis) == 1:  # sticker lies in the turning layer
            dest = _INDEX[(_rotate_cw(pos, axis), _rotate_cw(normal, axis))]
            perm[dest] = src
    return perm


def _build_move_table() -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for face in FACES:
        p1 = _quarter_turn_perm(face)
        p2 = p1[p1]
        p3 = p2[p1]
        table[face] = p1          # clockwise
        table[face + "2"] = p2    # half turn
        table[face + "'"] = p3    # counter-clockwise
    return table


MOVE_PERMS: dict[str, np.ndarray] = _build_move_table()

# 18 moves (Half-Turn Metric) and 12 quarter turns (Quarter-Turn Metric).
# QUARTER_MOVES are used as the output classes of the neural network.
ALL_MOVES: list[str] = [f + s for f in FACES for s in ("", "'", "2")]
QUARTER_MOVES: list[str] = [f + s for f in FACES for s in ("", "'")]

MOVE_TO_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}


def inverse_move(move: str) -> str:
    """Inverse of one move: U <-> U', U2 stays U2."""
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[0]
    return move + "'"


def parse_sequence(seq: str | list[str]) -> list[str]:
    """Turn a notation string like "R U R' U'" into a validated list of moves."""
    moves = seq.split() if isinstance(seq, str) else list(seq)
    for m in moves:
        if m not in MOVE_PERMS:
            raise ValueError(f"Invalid move: {m!r}")
    return moves


def invert_sequence(seq: str | list[str]) -> list[str]:
    """Inverse of a move sequence (reversed order, each move inverted)."""
    return [inverse_move(m) for m in reversed(parse_sequence(seq))]
