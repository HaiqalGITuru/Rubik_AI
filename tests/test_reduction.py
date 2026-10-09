"""Test metode reduksi 4x4 (tanpa model reduksi terlatih)."""

from pathlib import Path

import numpy as np
import pytest

from rubik import Cube
from rubik.data import generate_batch
from rubik.model import CubeNet
from rubik.nxn import CubeN, get_puzzle
from rubik.reduction import (
    OLL_PARITY, PLL_PARITY, generate_reduction_batch, get_spec, is_reduced, is_reduced_batch,
    map_moves_back, parity_of, random_reduced_states, sequence_perm, standardize_3x3, to_3x3_state,
)

ROOT = Path(__file__).resolve().parent.parent
P4 = get_puzzle(4)
SPEC = get_spec(4)


def test_indeks_stiker():
    assert SPEC.center_groups.shape == (6, 4)
    assert SPEC.edge_pairs.shape == (24, 2)
    assert len(SPEC.corner_idx) == 24
    assert len(SPEC.input_idx) == 72
    assert len(SPEC.valid_center_codes) == 24
    assert not set(SPEC.input_idx) & set(SPEC.corner_idx)


def test_kubus_selesai_tereduksi():
    assert is_reduced(CubeN(4))
    for alg in ["U R F D L B", "R U R' U' F2 D L' B", "x y R2 z"]:
        assert is_reduced(CubeN(4).apply_sequence(alg)), alg


def test_gerakan_lapisan_dalam_merusak_reduksi():
    for alg in ["2R", "Rw", "2U'", "Uw2 R"]:
        assert not is_reduced(CubeN(4).apply_sequence(alg)), alg


def test_susunan_pusat_cermin_ditolak():
    s = P4.solved.copy()
    u, d = SPEC.center_groups[0], SPEC.center_groups[3]
    s[u], s[d] = 3, 0  # tukar pusat putih & kuning saja: susunan cermin
    assert not is_reduced_batch(s[None])[0]


def test_sequence_perm_sama_dengan_cubeN():
    rng = np.random.default_rng(1)
    for alg in [OLL_PARITY, PLL_PARITY, "D L' B2 Rw x 3Fw'"]:
        c = CubeN(4)
        c.scramble(20, rng=rng)
        expect = c.copy().apply_sequence(alg).state
        assert np.array_equal(c.state[sequence_perm(P4, alg)], expect), alg


def test_rumus_paritas():
    oll = CubeN(4).apply_sequence(OLL_PARITY)
    pll = CubeN(4).apply_sequence(PLL_PARITY)
    assert is_reduced(oll) and is_reduced(pll)
    assert (oll.state != P4.solved).sum() == 4   # hanya satu pasangan tepi terbalik
    assert (pll.state != P4.solved).sum() == 4   # tepi UF dan UB tertukar
    assert parity_of(standardize_3x3(to_3x3_state(oll.state))[0]) == (True, False)
    assert parity_of(standardize_3x3(to_3x3_state(pll.state))[0]) == (False, True)


def test_kubus_tereduksi_acak():
    rng = np.random.default_rng(2)
    states = random_reduced_states(400, rng)
    assert is_reduced_batch(states).all()
    seen = set()
    for s in states:
        std = standardize_3x3(to_3x3_state(s))
        assert std is not None
        par = parity_of(std[0])
        assert par is not None
        seen.add(par)
        # rumus paritas tetap menjaga reduksi & memperbaiki paritas di posisi mana pun
        c = CubeN(4, s)
        if par[0]:
            c.apply_sequence(OLL_PARITY)
        if par[1]:
            c.apply_sequence(PLL_PARITY)
        assert is_reduced(c)
        assert parity_of(standardize_3x3(to_3x3_state(c.state))[0]) == (False, False)
    assert seen == {(False, False), (True, False), (False, True), (True, True)}


def test_3x3_biasa_tanpa_paritas():
    rng = np.random.default_rng(3)
    for _ in range(50):
        c = Cube()
        c.scramble(30, rng=rng)
        assert parity_of(c.state) == (False, False)


def test_peta_gerakan_kembali():
    """Solusi di bingkai 3x3 yang diputar harus menyelesaikan kubus 4x4 fisik."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        c4 = CubeN(4)
        c4.apply_sequence(list(rng.choice(["U", "D'", "R2", "L", "F'", "B", "x", "y'"], size=15)))
        state3, rot = standardize_3x3(to_3x3_state(c4.state))
        # "solusi" 3x3: urutan acak lalu kebalikannya, diterapkan pada state berotasi
        seq = list(rng.choice(["U", "R'", "F2", "D", "L'", "B"], size=10))
        cube3 = Cube(state3).apply_sequence(seq)
        from rubik.moves import invert_sequence
        sol = invert_sequence(seq)
        phys = map_moves_back(sol, rot)
        # terapkan kebalikan pada 4x4: (c4 + seq fisik) lalu solusi fisik = c4 semula
        seq_phys = map_moves_back(seq, rot)
        a = c4.copy().apply_sequence(seq_phys + phys)
        assert np.array_equal(a.state, c4.state)
        assert np.array_equal(standardize_3x3(to_3x3_state(c4.copy().apply_sequence(seq_phys).state))[0],
                              cube3.state)


def test_data_latih_reduksi():
    X, y, d = generate_reduction_batch(50, 10, np.random.default_rng(5))
    assert X.shape == (500, 96) and y.max() < 18
    # langkah pertama dari kubus tereduksi + gerakan berlabel = kembali tereduksi
    first = X[:50]
    back = np.take_along_axis(first, P4.quarter_perms[y[:50]], axis=1)
    assert is_reduced_batch(back).all()


def test_model_dengan_input_idx():
    import torch
    m = CubeNet(64, 32, 1, num_moves=18, num_stickers=96, input_idx=SPEC.input_idx)
    out = m(torch.from_numpy(np.tile(P4.solved, (3, 1))))
    assert out.shape == (3, 18)
    m2 = CubeNet(**m.config)
    assert m2.config["input_idx"] == m.config["input_idx"]


@pytest.mark.skipif(not (ROOT / "models" / "cubenet_large_infer.pt").exists(), reason="model 3x3 tidak ada")
def test_tahap_paritas_dan_3x3_dengan_model_asli():
    """Tahap 2 + 3: dari 4x4 tereduksi acak (dengan paritas) sampai selesai."""
    from rubik.model import load_model
    from rubik.reduction import solve_reduction

    class Diam(CubeNet):  # model reduksi palsu; tidak dipanggil karena kubus sudah tereduksi
        pass

    model3 = load_model(str(ROOT / "models" / "cubenet_large_infer.pt"))
    dummy = Diam(16, 16, 1, num_moves=18, num_stickers=96)
    rng = np.random.default_rng(6)
    for s in random_reduced_states(5, rng):
        res = solve_reduction(dummy, model3, CubeN(4, s), beam_3x3=256)
        assert res.solved, res.message
        assert res.reduce_moves == []
