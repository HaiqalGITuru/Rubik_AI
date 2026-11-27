"""Simulator unit tests. Run with:  pytest -v"""

import numpy as np
import pytest

from rubik import ALL_MOVES, Cube, invert_sequence
from rubik.moves import MOVE_PERMS

FACE_MOVES = list("URFDLB")


def test_new_cube_is_solved():
    assert Cube().is_solved()


@pytest.mark.parametrize("move", ALL_MOVES)
def test_every_move_is_a_valid_permutation(move):
    assert sorted(MOVE_PERMS[move]) == list(range(54))


@pytest.mark.parametrize("face", FACE_MOVES)
def test_four_turns_return_to_start(face):
    c = Cube()
    for _ in range(4):
        c.apply(face)
    assert c.is_solved()


@pytest.mark.parametrize("face", FACE_MOVES)
def test_move_then_inverse(face):
    assert Cube().apply(face).apply(face + "'").is_solved()
    assert Cube().apply(face + "2").apply(face + "2").is_solved()
    assert not Cube().apply(face).is_solved()


@pytest.mark.parametrize("face", FACE_MOVES)
def test_quarter_turn_moves_20_stickers(face):
    # 8 stickers on the face itself (the center stays) + 12 stickers on the sides
    perm = MOVE_PERMS[face]
    assert np.sum(perm != np.arange(54)) == 20


def test_U_moves_top_row_of_R_to_F():
    c = Cube().apply("U")
    assert list(c.state[18:21]) == [1, 1, 1]  # top row of F becomes red (from R)


def test_R_moves_right_column_of_D_to_F():
    c = Cube().apply("R")
    assert [c.state[i] for i in (20, 23, 26)] == [3, 3, 3]  # yellow from D


def test_sexy_move_six_times_returns_to_start():
    c = Cube()
    for _ in range(6):
        c.apply_sequence("R U R' U'")
    assert c.is_solved()


def test_order_of_RU_is_105():
    # Well-known fact: (R U) must be repeated 105 times to return to the start
    c = Cube()
    for i in range(1, 106):
        c.apply_sequence("R U")
        if c.is_solved():
            break
    assert i == 105


def test_scramble_then_inverse_solves():
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


def test_facelet_string_of_solved_cube():
    assert Cube().to_facelet_string() == "".join(f * 9 for f in "URFDLB")


def test_invalid_move():
    with pytest.raises(ValueError):
        Cube().apply_sequence("R X")


def test_cross_check_with_kociemba():
    """Cross-check: Kociemba's solutions must solve our simulated cube."""
    kociemba = pytest.importorskip("kociemba")
    rng = np.random.default_rng(7)
    for _ in range(10):
        c = Cube()
        c.scramble(25, rng=rng)
        solution = kociemba.solve(c.to_facelet_string())
        assert c.apply_sequence(solution).is_solved()
