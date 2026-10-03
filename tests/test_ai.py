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


class _ConstModel:
    """Model palsu: selalu memprediksi jarak `c` (untuk menguji DeepCubeA tanpa training)."""

    def __init__(self, c=0.0):
        import torch
        self.c = c
        self.p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.p])

    def eval(self):
        return self

    def __call__(self, x):
        import torch
        return torch.full((x.shape[0], 1), self.c)


def test_davi_target():
    from rubik.deepcubea import davi_targets
    states = np.stack([Cube().state, Cube().apply("R").state, Cube().apply_sequence("R U").state])
    # Dengan prediksi jarak 5 untuk semua anak (kecuali yang selesai = 0):
    t = davi_targets(_ConstModel(5.0), states)
    assert t[0] == 0.0          # sudah selesai
    assert t[1] == 1.0          # satu gerakan lagi -> anak selesai -> 1 + 0
    assert t[2] == 6.0          # tidak ada anak yang selesai -> 1 + 5


def test_weighted_astar_menemukan_solusi():
    pytest.importorskip("torch")
    from rubik.deepcubea import weighted_astar
    cube = Cube().apply_sequence("R U F'")
    res = weighted_astar(_ConstModel(0.0), cube, batch_size=2000, weight=1.0)
    assert res.solved and len(res.moves) == 3   # h=0, weight=1 -> pencarian optimal
    assert cube.apply_sequence(res.moves).is_solved()