"""
Set kubus uji (benchmark) yang TETAP, dipakai oleh semua eksperimen.

Dengan kubus yang persis sama, perbandingan antar eksperimen (lebar beam,
model kecil vs besar, EfficientCube vs DeepCubeA) menjadi adil dan bisa diulang.

File disimpan di benchmarks/benchmark_<n>.json dan sebaiknya ikut di-commit ke Git.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .cube import Cube

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"


def benchmark_path(n: int) -> Path:
    return BENCH_DIR / f"benchmark_{n}.json"


def create_benchmark(n: int = 100, depth: int = 25, seed: int = 2026) -> dict:
    """Buat n kubus teracak penuh (depth gerakan HTM) + solusi Kociemba sebagai pembanding."""
    try:
        import kociemba
    except ImportError:
        kociemba = None

    if kociemba:  # panggilan pertama memuat tabel (lambat), jangan ikut terukur
        kociemba.solve(Cube().apply_sequence("R U").to_facelet_string())

    rng = np.random.default_rng(seed)
    cubes = []
    for i in range(n):
        cube = Cube()
        scramble = cube.scramble(depth, rng=rng)
        item = {"id": i, "scramble": " ".join(scramble)}
        if kociemba:
            t0 = time.perf_counter()
            sol = kociemba.solve(cube.to_facelet_string()).split()
            item["kociemba_moves"] = " ".join(sol)
            item["kociemba_length"] = len(sol)
            item["kociemba_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        cubes.append(item)
    return {"n": n, "depth": depth, "seed": seed, "cubes": cubes}


def load_or_create_benchmark(n: int = 100) -> dict:
    path = benchmark_path(n)
    if path.exists():
        return json.loads(path.read_text())
    BENCH_DIR.mkdir(exist_ok=True)
    bench = create_benchmark(n)
    path.write_text(json.dumps(bench, indent=1))
    print(f"Set kubus uji baru dibuat: {path}")
    return bench
