"""Unit test simulator. Jalankan dengan:  pytest -v"""

import numpy as np
import pytest

from rubik import ALL_MOVES, Cube, invert_sequence
from rubik.moves import MOVE_PERMS

FACE_MOVES = list("URFDLB")


def test_cube_baru_sudah_selesai():
    assert Cube().is_solved()


@pytest.mark.parametrize("move", ALL_MOVES)
def test_setiap_gerakan_adalah_permutasi_valid(move):
    assert sorted(MOVE_PERMS[move]) == list(range(54))


@pytest.mark.parametrize("face", FACE_MOVES)
def test_empat_kali_putar_kembali_semula(face):
    c = Cube()
    for _ in range(4):
        c.apply(face)
    assert c.is_solved()


@pytest.mark.parametrize("face", FACE_MOVES)
def test_gerakan_lalu_inversnya(face):
    assert Cube().apply(face).apply(face + "'").is_solved()
    assert Cube().apply(face + "2").apply(face + "2").is_solved()
    assert not Cube().apply(face).is_solved()


@pytest.mark.parametrize("face", FACE_MOVES)
def test_seperempat_putaran_memindahkan_20_stiker(face):
    # 8 stiker di wajah itu (pusat tetap) + 12 stiker di sisi samping
    perm = MOVE_PERMS[face]
    assert np.sum(perm != np.arange(54)) == 20


def test_U_memindahkan_baris_atas_R_ke_F():
    c = Cube().apply("U")
    assert list(c.state[18:21]) == [1, 1, 1]  # baris atas F jadi merah (dari R)


def test_R_memindahkan_kolom_kanan_D_ke_F():
    c = Cube().apply("R")
    assert [c.state[i] for i in (20, 23, 26)] == [3, 3, 3]  # kuning dari D


def test_sexy_move_enam_kali_kembali_semula():
    c = Cube()
    for _ in range(6):
        c.apply_sequence("R U R' U'")
    assert c.is_solved()


def test_orde_RU_adalah_105():
    # Fakta matematis terkenal: (R U) harus diulang 105 kali untuk kembali semula
    c = Cube()
    for i in range(1, 106):
        c.apply_sequence("R U")
        if c.is_solved():
            break
    assert i == 105


def test_scramble_lalu_invers_menyelesaikan():
    rng = np.random.default_rng(42)
    for _ in range(50):
        c = Cube()
        moves = c.scramble(30, rng=rng)
        assert np.bincount(c.state, minlength=6).tolist() == [9] * 6
        c.apply_sequence(invert_sequence(moves))
        assert c.is_solved()


def test_invert_sequence():
    assert invert_sequence("R U' F2") == ["F2", "U", "R'"]


def test_one_hot():
    c = Cube()
    c.scramble(10)
    x = c.to_one_hot()
    assert x.shape == (324,)
    assert x.sum() == 54


def test_facelet_string_kubus_selesai():
    assert Cube().to_facelet_string() == "".join(f * 9 for f in "URFDLB")


def test_gerakan_tidak_valid():
    with pytest.raises(ValueError):
        Cube().apply_sequence("R X")


def test_validasi_dengan_kociemba():
    """Validasi silang: solusi dari solver Kociemba harus menyelesaikan kubus kita."""
    kociemba = pytest.importorskip("kociemba")
    rng = np.random.default_rng(7)
    for _ in range(10):
        c = Cube()
        c.scramble(25, rng=rng)
        solution = kociemba.solve(c.to_facelet_string())
        assert c.apply_sequence(solution).is_solved()
