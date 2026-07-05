import numpy as np
import pytest
from stochastix.time_grid import TimeGrid


def test_time_grid_perfect_alignment():
    """
    Test Case 1: The distance between observations is a perfect multiple of dt_max.
    """
    obsv_dates = [1.0, 2.0]
    dt_max = 0.5

    time_grid = TimeGrid(obsv_dates, dt_max)
    full_grid, obsv_indices = time_grid.build()

    expected_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    expected_indices = np.array([2, 4])

    np.testing.assert_allclose(
        full_grid, expected_grid, err_msg="Grid mismatch on perfect alignment"
    )
    np.testing.assert_array_equal(
        obsv_indices, expected_indices, err_msg="Indices mismatch on perfect alignment"
    )


def test_time_grid_misalignment():
    """
    Test Case 2: The distance requires subdivision into smaller identical steps to respect dt_max.
    """
    obsv_dates = [0.4, 1.0]
    dt_max = 0.3

    time_grid = TimeGrid(obsv_dates, dt_max)
    full_grid, obsv_indices = time_grid.build()

    expected_grid = np.array([0.0, 0.2, 0.4, 0.7, 1.0])
    expected_indices = np.array([2, 4])

    np.testing.assert_allclose(
        full_grid, expected_grid, err_msg="Grid mismatch on misalignment case"
    )
    np.testing.assert_array_equal(
        obsv_indices, expected_indices, err_msg="Indices mismatch on misalignment case"
    )


def test_time_grid_manual_exercise():
    """
    Test Case 3: The manual exercise validating the varying dt_actual per segment.
    Segment 1 (0 to 0.5) requires 2 steps of 0.25.
    Segment 2 (0.5 to 1.25) requires 2 steps of 0.375.
    """
    obsv_dates = [0.5, 1.25]
    dt_max = 0.4

    time_grid = TimeGrid(obsv_dates, dt_max)
    full_grid, obsv_indices = time_grid.build()

    expected_grid = np.array([0.0, 0.25, 0.5, 0.875, 1.25])
    expected_indices = np.array([2, 4])

    np.testing.assert_allclose(
        full_grid, expected_grid, err_msg="Grid mismatch on manual exercise case"
    )
    np.testing.assert_array_equal(
        obsv_indices,
        expected_indices,
        err_msg="Indices mismatch on manual exercise case",
    )


def test_time_grid_preserves_contract_dates():
    """
    Validates the core constraint: the exact observation dates MUST exist in the output grid
    at the exact locations specified by obsv_indices.
    """
    obsv_dates = np.array([0.33, 0.77, 1.5, 2.1])
    dt_max = 0.2

    time_grid = TimeGrid(obsv_dates, dt_max)
    full_grid, obsv_indices = time_grid.build()

    extracted_dates = full_grid[obsv_indices]
    np.testing.assert_allclose(
        extracted_dates,
        obsv_dates,
        err_msg="The extracted dates do not match the required observation dates",
    )
