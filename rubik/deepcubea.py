"""
DeepCubeA (Agostinelli dkk., 2019), versi yang diperkecil.

Perbedaan utama dengan EfficientCube:
    - EfficientCube : jaringan memprediksi GERAKAN (policy, 12 kelas) -> beam search
    - DeepCubeA     : jaringan memprediksi JARAK ke kondisi selesai (value / cost-to-go, 1 angka)
                      -> batch weighted A* search

Training memakai Deep Approximate Value Iteration (DAVI):
    target(s) = 0                                   jika s sudah selesai
    target(s) = min_a [ 1 + J_target(anak(s, a)) ]  selain itu
dengan J_target = salinan jaringan "lama" yang diperbarui berkala (target network).
"""

from __future__ import annotations

import heapq
import time

import numpy as np
import torch

from .cube import Cube
from .data import NUM_MOVES, QUARTER_PERMS
from .moves import QUARTER_MOVES
from .solver import SolveResult


def is_solved_batch(states: np.ndarray) -> np.ndarray:
    faces = states.reshape(-1, 6, 9)
    return np.all(faces == faces[:, :, 4:5], axis=(1, 2))


def expand(states: np.ndarray) -> np.ndarray:
    """Semua anak dari setiap state: (N, 54) -> (N, 12, 54)."""
    return states[:, QUARTER_PERMS]


@torch.no_grad()
def predict_cost(model: torch.nn.Module, states: np.ndarray, batch: int = 32768) -> np.ndarray:
    """Perkiraan jumlah langkah menuju selesai (>= 0) untuk setiap state."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(states), batch):
        x = torch.from_numpy(states[i:i + batch]).to(device).long()
        out.append(model(x).float().squeeze(1).clamp_min(0).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def davi_targets(target_model: torch.nn.Module, states: np.ndarray) -> np.ndarray:
    """Target DAVI: min atas 12 gerakan dari (1 + perkiraan jarak anak). State selesai -> 0."""
    n = len(states)
    children = expand(states).reshape(-1, 54)
    cost = predict_cost(target_model, children)
    cost[is_solved_batch(children)] = 0.0
    targets = 1.0 + cost.reshape(n, NUM_MOVES).min(axis=1)
    targets[is_solved_batch(states)] = 0.0
    return targets.astype(np.float32)


@torch.no_grad()
def weighted_astar(
    model: torch.nn.Module,
    cube: Cube,
    batch_size: int = 1000,
    weight: float = 0.6,
    max_iterations: int = 400,
    max_nodes: int = 10_000_000,
) -> SolveResult:
    """Batch Weighted A* (BWAS).

    Prioritas setiap posisi: f = weight * g + h
        g = jumlah langkah yang sudah ditempuh, h = perkiraan jarak dari jaringan.
    weight < 1 membuat pencarian lebih "rakus" (cepat, solusi sedikit lebih panjang).
    Setiap iterasi, `batch_size` posisi dengan f terkecil dikembangkan sekaligus,
    sehingga jaringan bisa mengevaluasi banyak anak dalam satu batch.
    """
    model.eval()
    t0 = time.perf_counter()
    if cube.is_solved():
        return SolveResult(True, [], 0.0, 0)

    # Penyimpanan node: state (bytes), induk, gerakan, g
    node_state: list[bytes] = [cube.state.tobytes()]
    node_parent = [-1]
    node_move = [-1]
    node_g = [0]
    best_g = {node_state[0]: 0}
    h0 = predict_cost(model, cube.state[None])[0]
    open_heap = [(float(h0), 0)]
    evaluated = 1

    def path_to(idx: int) -> list[str]:
        moves = []
        while node_parent[idx] != -1:
            moves.append(QUARTER_MOVES[node_move[idx]])
            idx = node_parent[idx]
        return moves[::-1]

    for _ in range(max_iterations):
        # Ambil batch_size node terbaik (lewati entri usang yang sudah punya g lebih kecil)
        batch = []
        while open_heap and len(batch) < batch_size:
            _, idx = heapq.heappop(open_heap)
            if node_g[idx] == best_g[node_state[idx]]:
                batch.append(idx)
        if not batch:
            break

        parents = np.frombuffer(b"".join(node_state[i] for i in batch), dtype=np.uint8).reshape(-1, 54)
        children = expand(parents).reshape(-1, 54)
        solved = is_solved_batch(children)

        # Saring anak yang sudah pernah dicapai dengan g yang sama atau lebih kecil
        keep, keys, gs, par, mv = [], [], [], [], []
        for k in range(len(children)):
            p = batch[k // NUM_MOVES]
            g = node_g[p] + 1
            key = children[k].tobytes()
            if best_g.get(key, 1 << 30) <= g:
                continue
            best_g[key] = g
            if solved[k]:
                node_state.append(key); node_parent.append(p); node_move.append(k % NUM_MOVES); node_g.append(g)
                return SolveResult(True, path_to(len(node_state) - 1), time.perf_counter() - t0, evaluated)
            keep.append(k); keys.append(key); gs.append(g); par.append(p); mv.append(k % NUM_MOVES)

        if not keep:
            continue
        h = predict_cost(model, children[keep])
        evaluated += len(keep)
        for key, g, p, m, hv in zip(keys, gs, par, mv, h):
            node_state.append(key); node_parent.append(p); node_move.append(m); node_g.append(g)
            heapq.heappush(open_heap, (weight * g + float(hv), len(node_state) - 1))
        if len(node_state) > max_nodes:
            break

    return SolveResult(False, [], time.perf_counter() - t0, evaluated)
