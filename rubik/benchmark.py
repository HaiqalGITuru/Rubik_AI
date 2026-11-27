"""
A FIXED benchmark set of test cubes, shared by every experiment.

Using exactly the same cubes makes comparisons between experiments (beam width,
small vs. large model, EfficientCube vs. DeepCubeA) fair and reproducible.

The file is stored at benchmarks/benchmark_<n>.json and should be committed to Git.
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
    """Create n fully scrambled cubes (depth HTM moves) plus Kociemba's solution as a baseline."""
    try:
        import kociemba
    except ImportError:
        kociemba = None

    if kociemba:  # the first call loads lookup tables (slow); keep it out of the timing
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
    print(f"New benchmark set created: {path}")
    return bench
