"""Test simulator kubus N x N (4x4)."""

import numpy as np
import pytest

from rubik.moves import MOVE_PERMS
from rubik.nxn import CubeN, get_puzzle


def test_3x3_generik_sama_dengan_simulator_3x3():
    p = get_puzzle(3)
    for m, perm in MOVE_PERMS.items():
        assert np.array_equal(p.token_perm(m), perm), m


def test_4x4_gerakan_ai():
    p = get_puzzle(4)
    assert p.num_stickers == 96 and p.num_moves == 18
    for i, m in enumerate(p.quarter_moves):
        perm = p.quarter_perms[i]
        assert sorted(perm) == list(range(96))
        assert np.array_equal(perm[p.dbl_idx], p.dbl_idx), f"{m} memindahkan sudut DBL"
        c = CubeN(4)
        for _ in range(4):
            c.apply(m)
        assert c.is_solved()


def test_4x4_jumlah_stiker_berpindah():
    p = get_puzzle(4)
    assert (p.token_perm("U") != np.arange(96)).sum() == 32   # 16 di wajah + 16 di sisi
    assert (p.token_perm("2U") != np.arange(96)).sum() == 16  # lapisan dalam: hanya sisi


@pytest.mark.parametrize("seq", ["y", "x2 z'", "Uw 3U D'", "D D'", "Rw r'", "Rw Lw' x'"])
def test_4x4_rotasi_dan_notasi_wca(seq):
    # Semua urutan ini setara dengan rotasi seluruh kubus (atau tidak mengubah apa pun)
    assert CubeN(4).apply_sequence(seq).is_solved()


def test_4x4_scramble_lalu_invers():
    p = get_puzzle(4)
    rng = np.random.default_rng(1)
    for _ in range(20):
        c = CubeN(4)
        moves = c.scramble(40, rng=rng)
        assert not c.is_solved()
        c.apply_sequence([p.inverse(m) for m in reversed(moves)])
        assert c.is_solved()


def test_4x4_data_dan_beam_search():
    torch = pytest.importorskip("torch")
    from rubik.data import apply_moves_batch, generate_batch
    from rubik.model import CubeNet
    from rubik.solver import beam_search

    p = get_puzzle(4)
    X, y, _ = generate_batch(4, 6, np.random.default_rng(0), p)
    X, y = X.reshape(6, 4, 96), y.reshape(6, 4)
    for t in range(1, 6):  # label membawa kembali ke state sebelumnya
        assert np.array_equal(apply_moves_batch(X[t], y[t], p.quarter_perms), X[t - 1])

    torch.manual_seed(0)
    model = CubeNet(64, 32, 1, num_moves=p.num_moves, num_stickers=p.num_stickers)
    cube = CubeN(4).apply_sequence("2R F")
    res = beam_search(model, cube, beam_width=400, max_depth=4)
    assert res.solved and cube.apply_sequence(res.moves).is_solved()
