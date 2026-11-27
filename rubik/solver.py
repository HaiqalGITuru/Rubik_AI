"""
Beam search solver guided by a neural network.

How it works:
    1. Start from the scrambled state (the beam holds 1 candidate).
    2. For every candidate in the beam, the model gives probabilities for the 12 moves.
    3. Score of a new candidate = sum of log-probabilities along its path.
    4. Keep only the best `beam_width` candidates and drop duplicate states.
    5. Repeat until a candidate is solved or max_depth is reached.

A wider beam = more reliable and shorter solutions, but slower.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .cube import Cube
from .data import INVERSE_IDX, NUM_MOVES, QUARTER_PERMS
from .moves import QUARTER_MOVES


@dataclass
class SolveResult:
    solved: bool
    moves: list[str]
    time_sec: float
    nodes: int  # number of states evaluated by the model


def _is_solved_batch(states: np.ndarray) -> np.ndarray:
    faces = states.reshape(len(states), 6, -1)
    return np.all(faces == faces[:, :, :1], axis=(1, 2))


def _dedupe(states: np.ndarray) -> np.ndarray:
    """Indices of unique rows (first occurrence kept)."""
    keys = np.ascontiguousarray(states).view(np.dtype((np.void, states.shape[1]))).ravel()
    _, idx = np.unique(keys, return_index=True)
    return np.sort(idx)


@torch.no_grad()
def beam_search(model: torch.nn.Module, cube: Cube, beam_width: int = 256, max_depth: int = 40) -> SolveResult:
    model.eval()
    t0 = time.perf_counter()
    if cube.is_solved():
        return SolveResult(True, [], 0.0, 0)

    perms, inverse_idx, names = QUARTER_PERMS, INVERSE_IDX, QUARTER_MOVES
    n_moves = len(names)

    states = cube.state[None, :].copy()          # (B, 54)
    paths = np.zeros((1, 0), dtype=np.int64)     # (B, depth)
    scores = np.zeros(1, dtype=np.float64)       # (B,)
    nodes = 0
    device = next(model.parameters()).device     # CPU or GPU, wherever the model lives

    for _ in range(max_depth):
        x = torch.from_numpy(states).to(device).long()
        logp = torch.log_softmax(model(x).float(), dim=1).cpu().numpy()
        nodes += len(states)

        cand = scores[:, None] + logp            # (B, 12)
        if paths.shape[1] > 0:                   # forbid undoing the last move
            cand[np.arange(len(states)), inverse_idx[paths[:, -1]]] = -np.inf

        flat = cand.ravel()
        k = min(beam_width, np.isfinite(flat).sum())
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top])]        # sort by best score first

        parent, move = top // n_moves, top % n_moves
        new_states = np.take_along_axis(states[parent], perms[move], axis=1)
        new_paths = np.concatenate([paths[parent], move[:, None]], axis=1)
        new_scores = flat[top]

        solved = _is_solved_batch(new_states)
        if solved.any():
            best = np.flatnonzero(solved)[0]
            moves = [names[i] for i in new_paths[best]]
            return SolveResult(True, moves, time.perf_counter() - t0, nodes)

        keep = _dedupe(new_states)
        states, paths, scores = new_states[keep], new_paths[keep], new_scores[keep]

    return SolveResult(False, [], time.perf_counter() - t0, nodes)


def compress_moves(moves: list[str]) -> list[str]:
    """Merge consecutive moves on the same face: R R -> R2, R R' -> (removed)."""
    out: list[tuple[str, int]] = []
    for m in moves:
        if m.endswith("'"):
            face, amount = m[:-1], 3
        elif m.endswith("2"):
            face, amount = m[:-1], 2
        else:
            face, amount = m, 1
        if out and out[-1][0] == face:
            amount = (out[-1][1] + amount) % 4
            out.pop()
            if amount == 0:
                continue
        out.append((face, amount))
    return [f + {1: "", 2: "2", 3: "'"}[a] for f, a in out]