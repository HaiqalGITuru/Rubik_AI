"""
Metode REDUKSI untuk kubus 4x4: ubah 4x4 menjadi "3x3 besar", lalu selesaikan sebagai 3x3.

Tiga tahap:
    1. Reduksi (AI)   : pusat tiap sisi sewarna + setiap pasangan tepi menyatu.
                        Dikerjakan jaringan saraf khusus yang hanya melihat stiker pusat & tepi.
    2. Paritas (rumus): 4x4 tereduksi kadang mustahil sebagai 3x3 (satu tepi terbalik, atau
                        dua tepi tertukar). Diperbaiki dengan rumus tetap seperti cara manusia.
    3. 3x3 (AI)       : kubus dipetakan ke 3x3 virtual dan diselesaikan model 3x3 yang sudah ada.
                        Gerakannya (U, D, R, L, F, B) langsung berlaku sebagai lapisan luar 4x4.

Modul ini berisi semua alat bantunya: pengecekan "sudah tereduksi", pembuat data latih,
pemetaan 4x4 -> 3x3, deteksi paritas, dan solver tiga tahap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .cube import Cube
from .data import apply_moves_batch, generate_batch_from
from .moves import MOVE_PERMS
from .nxn import CubeN, NxNPuzzle, get_puzzle
from .solver import beam_search, compress_moves

# Rumus paritas (sudah diverifikasi di simulator, lihat tests/test_reduction.py)
# "2R" = lapisan kedua dari kanan (slice), "Uw" = dua lapisan atas.
OLL_PARITY = "2R2 B2 U2 2L U2 2R' U2 2R U2 F2 2R F2 2L' B2 2R2"  # membalik tepi UF
PLL_PARITY = "2R2 U2 2R2 Uw2 2R2 2U2"                                # menukar tepi UF dan UB


# ---------------------------------------------------------------------- #
# Indeks stiker
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReductionSpec:
    n: int
    center_groups: np.ndarray   # (6, (n-2)^2)        stiker pusat per sisi
    edge_pairs: np.ndarray      # (24, n-2)           stiker tepi sejajar per sisi wajah
    center_probe: np.ndarray    # (6,)                satu stiker pusat per sisi
    corner_idx: np.ndarray      # stiker sudut (tidak relevan untuk reduksi)
    input_idx: np.ndarray       # stiker pusat + tepi (input jaringan reduksi)
    valid_center_codes: frozenset  # 24 susunan warna pusat yang sah (rotasi dari susunan baku)


@lru_cache(maxsize=None)
def get_spec(n: int = 4) -> ReductionSpec:
    p = get_puzzle(n)
    nn = n * n
    inner = range(1, n - 1)
    centers, edges, corners = [], [], []
    for f in range(6):
        idx = lambda r, c: f * nn + r * n + c
        centers.append([idx(r, c) for r in inner for c in inner])
        edges += [[idx(0, c) for c in inner], [idx(n - 1, c) for c in inner],
                  [idx(r, 0) for r in inner], [idx(r, n - 1) for r in inner]]
        corners += [idx(0, 0), idx(0, n - 1), idx(n - 1, 0), idx(n - 1, n - 1)]
    centers = np.array(centers)
    edges = np.array(edges)
    probe = centers[:, 0]
    # Susunan pusat yang sah = 24 rotasi seluruh kubus dari kubus selesai
    codes = set()
    for a, b, c in p._all_rotations():
        s = p.solved[p._rotation_perm((a, b, c))]
        codes.add(_code(s[probe][None])[0])
    input_idx = np.sort(np.concatenate([centers.ravel(), edges.ravel()]))
    return ReductionSpec(n, centers, edges, probe, np.array(sorted(corners)), input_idx, frozenset(codes))


def _code(colors: np.ndarray) -> np.ndarray:
    """(B, 6) warna pusat -> satu bilangan per baris."""
    return (colors.astype(np.int64) * (6 ** np.arange(colors.shape[1]))).sum(1)


def is_reduced_batch(states: np.ndarray, spec: ReductionSpec | None = None) -> np.ndarray:
    """True kalau pusat tiap sisi sewarna, semua pasangan tepi menyatu,
    dan susunan warna pusat sah (bukan susunan cermin yang mustahil di 3x3)."""
    spec = spec or get_spec(4)
    c = states[:, spec.center_groups]                              # (B, 6, k)
    ok = np.all(c == c[:, :, :1], axis=(1, 2))
    e = states[:, spec.edge_pairs]                                 # (B, 24, n-2)
    ok &= np.all(e == e[:, :, :1], axis=(1, 2))
    if ok.any():
        codes = _code(states[ok][:, spec.center_probe])
        ok[np.flatnonzero(ok)] = np.isin(codes, list(spec.valid_center_codes))
    return ok


def is_reduced(cube: CubeN) -> bool:
    return bool(is_reduced_batch(cube.state[None], get_spec(cube.n))[0])


# ---------------------------------------------------------------------- #
# Permutasi urutan gerakan (dalam bingkai baku: sudut DBL terkunci)
# ---------------------------------------------------------------------- #
def sequence_perm(puzzle: NxNPuzzle, tokens) -> np.ndarray:
    """Permutasi gabungan untuk urutan token apa pun (termasuk D, L, B, Uw, x, ...)."""
    tokens = tokens.split() if isinstance(tokens, str) else list(tokens)
    perm = np.arange(puzzle.num_stickers)
    for t in tokens:  # diterapkan pada kubus fisik (pegangan tetap)
        perm = perm[puzzle.token_perm(t)]
    # normalisasi: putar seluruh kubus sampai stiker DBL kembali ke posisinya
    for r in puzzle.rotations:
        q = perm[r]
        if np.array_equal(q[puzzle.dbl_idx], puzzle.dbl_idx):
            return q
    raise ValueError("sudut DBL tidak ditemukan")


@lru_cache(maxsize=None)
def reduced_generators(n: int = 4) -> np.ndarray:
    """Gerakan yang menjaga kubus tetap tereduksi (seperti gerakan 3x3):
    lapisan luar U, R, F, dan lapisan tengah 2+3 sekaligus. Masing-masing dua arah."""
    p = get_puzzle(n)
    mids = " ".join(str(k) for k in range(2, n))
    gens = []
    for f in "URF":
        outer = p.token_perm(f)
        mid = np.arange(p.num_stickers)
        for k in range(2, n):
            mid = mid[p.token_perm(f"{k}{f}")]
        for perm in (outer, mid):
            gens += [perm, np.argsort(perm)]  # searah dan kebalikannya
    return np.stack(gens)


@lru_cache(maxsize=None)
def parity_perms(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    p = get_puzzle(n)
    return sequence_perm(p, OLL_PARITY), sequence_perm(p, PLL_PARITY)


def random_reduced_states(num: int, rng: np.random.Generator, n: int = 4, walk: int = 40) -> np.ndarray:
    """Kubus tereduksi acak: jalan acak dengan gerakan 'ala 3x3', lalu paritas acak."""
    p = get_puzzle(n)
    gens = reduced_generators(n)
    states = np.tile(p.solved, (num, 1))
    for _ in range(walk):
        states = apply_moves_batch(states, rng.integers(len(gens), size=num), gens)
    oll, pll = parity_perms(n)
    for perm in (oll, pll):
        mask = rng.random(num) < 0.5
        states[mask] = states[mask][:, perm]
    # acak lagi supaya tepi yang terbalik/tertukar tidak selalu di UF/UB
    for _ in range(walk // 2):
        states = apply_moves_batch(states, rng.integers(len(gens), size=num), gens)
    return states


def generate_reduction_batch(num_scrambles: int, max_depth: int, rng: np.random.Generator, n: int = 4):
    """Data latih AI reduksi: mulai dari kubus TEREDUKSI acak (bukan kubus selesai),
    acak dengan gerakan AI, label = kebalikan gerakan terakhir (sama seperti EfficientCube)."""
    p = get_puzzle(n)
    start = random_reduced_states(num_scrambles, rng, n)
    return generate_batch_from(start, max_depth, rng, p.quarter_perms, p.inverse_idx)


# ---------------------------------------------------------------------- #
# 4x4 tereduksi -> 3x3 virtual
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _map_to_3x3(n: int = 4) -> np.ndarray:
    """Indeks stiker NxN yang mewakili tiap stiker 3x3 (sudut, satu stiker tepi, satu pusat)."""
    pick = [0, 1, n - 1]
    return np.array([f * n * n + pick[r] * n + pick[c] for f in range(6) for r in range(3) for c in range(3)])


def to_3x3_state(state4: np.ndarray, n: int = 4) -> np.ndarray:
    return state4[_map_to_3x3(n)]


P3 = get_puzzle(3)
CENTER3 = np.arange(6) * 9 + 4


def _cubie_tables():
    """Posisi sudut & tepi 3x3 beserta stiker-stikernya (dari geometri simulator)."""
    groups: dict = {}
    for i, (pos, _) in enumerate(P3.stickers):
        groups.setdefault(pos, []).append(i)
    corners, edges = [], []
    for pos, idx in groups.items():
        nz = sum(1 for x in pos if x)
        if nz == 3:
            corners.append(idx)
        elif nz == 2:
            # stiker acuan orientasi tepi: yang menghadap U/D, kalau tidak ada yang menghadap F/B
            idx = sorted(idx, key=lambda i: (P3.stickers[i][1][1] == 0, P3.stickers[i][1][2] == 0))
            edges.append(idx)
    return corners, edges


_CORNERS, _EDGES = _cubie_tables()
_SOLVED3 = P3.solved
_CORNER_ID = {frozenset(_SOLVED3[c]): k for k, c in enumerate(_CORNERS)}
_EDGE_ID = {frozenset(_SOLVED3[e]): k for k, e in enumerate(_EDGES)}
_EDGE_REF = [_SOLVED3[e[0]] for e in _EDGES]  # warna acuan tiap kepingan tepi


def _perm_parity(perm: list[int]) -> int:
    seen, parity = [False] * len(perm), 0
    for i in range(len(perm)):
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length:
            parity ^= (length - 1) & 1
    return parity


def standardize_3x3(state3: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Putar 3x3 virtual sampai pusatnya baku (putih atas, hijau depan).
    Mengembalikan (state baku, permutasi rotasi), atau None kalau susunan pusat tidak sah."""
    for r in P3.rotations:
        s = state3[r]
        if np.array_equal(s[CENTER3], np.arange(6)):
            return s, r
    return None


def parity_of(state3: np.ndarray) -> tuple[bool, bool] | None:
    """(OLL parity?, PLL parity?) untuk 3x3 virtual berpusat baku. None = keadaan rusak."""
    try:
        cperm = [_CORNER_ID[frozenset(state3[c])] for c in _CORNERS]
        eperm = [_EDGE_ID[frozenset(state3[e])] for e in _EDGES]
    except KeyError:
        return None
    if len(set(cperm)) != 8 or len(set(eperm)) != 12:
        return None
    flips = sum(int(state3[e[0]] != _EDGE_REF[eperm[k]]) for k, e in enumerate(_EDGES))
    return bool(flips % 2), _perm_parity(cperm) != _perm_parity(eperm)


@lru_cache(maxsize=None)
def _face_move_by_perm() -> dict:
    return {MOVE_PERMS[m].tobytes(): m for m in MOVE_PERMS}


def map_moves_back(moves: list[str], rot: np.ndarray) -> list[str]:
    """Solusi 3x3 dalam bingkai yang diputar -> gerakan fisik (U, D, R, L, F, B)."""
    inv = np.argsort(rot)
    table = _face_move_by_perm()
    out = []
    for m in moves:
        phys = rot[MOVE_PERMS[m][inv]]
        out.append(table[phys.tobytes()])
    return out


# ---------------------------------------------------------------------- #
# Solver tiga tahap
# ---------------------------------------------------------------------- #
@dataclass
class ReductionResult:
    solved: bool
    reduce_moves: list[str] = field(default_factory=list)   # notasi AI (SiGN): U, 2U, 3R', ...
    parity_moves: list[str] = field(default_factory=list)   # rumus paritas (token fisik)
    solve3_moves: list[str] = field(default_factory=list)   # U, D, R, L, F, B (lapisan luar)
    parity: tuple[bool, bool] = (False, False)
    time_reduce: float = 0.0
    time_3x3: float = 0.0
    nodes: int = 0
    message: str = ""

    @property
    def moves(self) -> list[str]:
        return self.reduce_moves + self.parity_moves + self.solve3_moves

    @property
    def time_sec(self) -> float:
        return self.time_reduce + self.time_3x3


def solve_reduction(reduce_model, model3, cube: CubeN, beam_reduce: int = 1024, beam_3x3: int = 1024,
                    max_depth_reduce: int = 60) -> ReductionResult:
    spec = get_spec(cube.n)
    res = ReductionResult(False)
    goal = lambda s: is_reduced_batch(s, spec)

    # 1. Reduksi
    r1 = beam_search(reduce_model, cube, beam_width=beam_reduce, max_depth=max_depth_reduce, goal_fn=goal)
    res.time_reduce, res.nodes = r1.time_sec, r1.nodes
    if not r1.solved:
        res.message = "Tahap reduksi gagal (pusat/tepi belum tersusun)."
        return res
    res.reduce_moves = compress_moves(r1.moves)
    work = cube.copy().apply_sequence(res.reduce_moves)

    # 2. Paritas
    std = standardize_3x3(to_3x3_state(work.state, cube.n))
    if std is None:
        res.message = "Susunan pusat tidak sah setelah reduksi."
        return res
    par = parity_of(std[0])
    if par is None:
        res.message = "Keadaan 3x3 virtual tidak dikenali."
        return res
    res.parity = par
    if par[0]:
        res.parity_moves += OLL_PARITY.split()
    if par[1]:
        res.parity_moves += PLL_PARITY.split()
    work.apply_sequence(res.parity_moves)

    # 3. Selesaikan sebagai 3x3
    std = standardize_3x3(to_3x3_state(work.state, cube.n))
    if std is None or parity_of(std[0]) != (False, False):
        res.message = "Paritas tidak berhasil diperbaiki."
        return res
    state3, rot = std
    r3 = beam_search(model3, Cube(state3), beam_width=beam_3x3, max_depth=40)
    res.time_3x3 = r3.time_sec
    res.nodes += r3.nodes
    if not r3.solved:
        res.message = "Tahap 3x3 gagal."
        return res
    res.solve3_moves = compress_moves(map_moves_back(r3.moves, rot))
    res.solved = cube.copy().apply_sequence(res.moves).is_solved()
    if not res.solved:
        res.message = "Solusi gabungan tidak menyelesaikan kubus (bug)."
    return res
