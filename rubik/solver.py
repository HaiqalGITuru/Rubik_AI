"""
Solver berbasis beam search yang dipandu jaringan saraf.

Cara kerja:
    1. Mulai dari state teracak (beam berisi 1 kandidat).
    2. Untuk setiap kandidat di beam, model memberi probabilitas 12 gerakan.
    3. Skor kandidat baru = jumlah log-probabilitas sepanjang jalurnya.
    4. Simpan hanya `beam_width` kandidat terbaik, buang state duplikat.
    5. Ulangi sampai ada kandidat yang selesai atau mencapai max_depth.

Beam lebih lebar = lebih akurat & solusi lebih pendek, tapi lebih lambat.
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
    nodes: int  # jumlah state yang dievaluasi model


def _is_solved_batch(states: np.ndarray) -> np.ndarray:
    faces = states.reshape(-1, 6, 9)
    return np.all(faces == faces[:, :, 4:5], axis=(1, 2))


def _dedupe(states: np.ndarray) -> np.ndarray:
    """Indeks baris unik (kemunculan pertama dipertahankan)."""
    keys = np.ascontiguousarray(states).view(np.dtype((np.void, states.shape[1]))).ravel()
    _, idx = np.unique(keys, return_index=True)
    return np.sort(idx)


@torch.no_grad()
def beam_search(model: torch.nn.Module, cube: Cube, beam_width: int = 256, max_depth: int = 40) -> SolveResult:
    model.eval()
    t0 = time.perf_counter()
    if cube.is_solved():
        return SolveResult(True, [], 0.0, 0)

    states = cube.state[None, :].copy()          # (B, 54)
    paths = np.zeros((1, 0), dtype=np.int64)     # (B, depth)
    scores = np.zeros(1, dtype=np.float64)       # (B,)
    nodes = 0
    device = next(model.parameters()).device     # CPU atau GPU, mengikuti model

    for _ in range(max_depth):
        x = torch.from_numpy(states).to(device).long()
        logp = torch.log_softmax(model(x).float(), dim=1).cpu().numpy()
        nodes += len(states)

        cand = scores[:, None] + logp            # (B, 12)
        if paths.shape[1] > 0:                   # larang membatalkan gerakan terakhir
            cand[np.arange(len(states)), INVERSE_IDX[paths[:, -1]]] = -np.inf

        flat = cand.ravel()
        k = min(beam_width, np.isfinite(flat).sum())
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top])]        # urutkan dari skor terbaik

        parent, move = top // NUM_MOVES, top % NUM_MOVES
        new_states = np.take_along_axis(states[parent], QUARTER_PERMS[move], axis=1)
        new_paths = np.concatenate([paths[parent], move[:, None]], axis=1)
        new_scores = flat[top]

        solved = _is_solved_batch(new_states)
        if solved.any():
            best = np.flatnonzero(solved)[0]
            moves = [QUARTER_MOVES[i] for i in new_paths[best]]
            return SolveResult(True, moves, time.perf_counter() - t0, nodes)

        keep = _dedupe(new_states)
        states, paths, scores = new_states[keep], new_paths[keep], new_scores[keep]

    return SolveResult(False, [], time.perf_counter() - t0, nodes)


def compress_moves(moves: list[str]) -> list[str]:
    """Gabungkan gerakan berurutan pada wajah yang sama: R R -> R2 (lebih mudah dibaca)."""
    out: list[tuple[str, int]] = []
    for m in moves:
        face, amount = m[0], 3 if m.endswith("'") else 1
        if out and out[-1][0] == face:
            amount = (out[-1][1] + amount) % 4
            out.pop()
            if amount == 0:
                continue
        out.append((face, amount))
    return [f + {1: "", 2: "2", 3: "'"}[a] for f, a in out]