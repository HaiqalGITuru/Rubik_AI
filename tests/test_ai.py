"""Test generator data & solver (tidak butuh model terlatih)."""

import numpy as np
import pytest

from rubik import Cube
from rubik.data import QUARTER_PERMS, apply_moves_batch, generate_batch
from rubik.moves import QUARTER_MOVES
from rubik.solver import compress_moves


def test_bentuk_batch():
    X, y, d = generate_batch(8, 26, np.random.default_rng(0))
    assert X.shape == (8 * 26, 54) and y.shape == (8 * 26,) and d.shape == (8 * 26,)
    assert y.min() >= 0 and y.max() < 12


def test_label_membawa_ke_state_sebelumnya():
    """Menerapkan label pada state ke-t harus menghasilkan state ke-(t-1)."""
    n, depth = 16, 10
    X, y, _ = generate_batch(n, depth, np.random.default_rng(1))
    X, y = X.reshape(depth, n, 54), y.reshape(depth, n)
    for t in range(depth):
        back = apply_moves_batch(X[t], y[t])
        expected = X[t - 1] if t > 0 else np.tile(Cube().state, (n, 1))
        assert np.array_equal(back, expected)


def test_apply_moves_batch_sama_dengan_cube():
    for i, m in enumerate(QUARTER_MOVES):
        c = Cube().apply_sequence("R U F")
        batch = apply_moves_batch(c.state[None], np.array([i]))
        assert np.array_equal(batch[0], c.apply(m).state)


def test_compress_moves():
    assert compress_moves(["R", "R", "U", "U'", "F'"]) == ["R2", "F'"]
    assert compress_moves(["R", "R", "R"]) == ["R'"]


def test_beam_search_dengan_model_acak():
    torch = pytest.importorskip("torch")
    from rubik.model import CubeNet
    from rubik.solver import beam_search

    torch.manual_seed(0)
    model = CubeNet(hidden1=64, hidden2=32, num_blocks=1)
    cube = Cube().apply_sequence("R U")
    res = beam_search(model, cube, beam_width=200, max_depth=4)
    assert res.solved  # beam 200 cukup untuk mencakup semua jalur 2 langkah
    assert cube.apply_sequence(res.moves).is_solved()


def test_benchmark_deterministik():
    """Set kubus uji harus selalu sama untuk seed yang sama."""
    from rubik.benchmark import create_benchmark
    a, b = create_benchmark(5), create_benchmark(5)
    assert [c["scramble"] for c in a["cubes"]] == [c["scramble"] for c in b["cubes"]]
    for c in a["cubes"]:
        assert not Cube().apply_sequence(c["scramble"]).is_solved()