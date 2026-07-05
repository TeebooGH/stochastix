"""
Time Grid Generation Module.

This module resolves the dimensional conflict between the financial observation requirements
(Option Contract) and the numerical convergence constraints (Simulation Scheme).

Mathematically, it computes the union of two temporal subdivisions:
1. The macroscopic grid: Observation dates mandated by the financial product (e.g., [0.5, 1.0]).
2. The microscopic grid: Maximum time step h (dt_max) mandated by the SDE numerical resolution.

The resulting subdivision ensures that no segment exceeds dt_max, while exactly
intersecting every required observation date. It also computes the index mapping
so the contract can extract its required states in O(1) time.

Usage example:
    >>> obsv_dates = np.array([0.5, 1.25])
    >>> dt_max = 0.4
    >>> grid_builder = TimeGrid(obsv_dates, dt_max)
    >>> full_grid, obsv_indices = grid_builder.build()
    >>> print(full_grid)
    [0.   0.25 0.5  0.875 1.25]
    >>> print(obsv_indices)
    [2 4]
"""

import numpy as np


class TimeGrid:
    def __init__(self, obsv_grid, dt_max):
        """
        Constructor for the TimeGrid orchestrator.

        Args:
            obsv_grid (np.ndarray or list): Increasing sequence of observation times t_k > 0
                                            mandated by the OptionContract.
            dt_max (float): Maximum time step h allowed by the SimulationScheme
                            to ensure numerical convergence.
        """
        self.obsv_grid = obsv_grid
        self.dt_max = dt_max

    def build(self):
        """
        Constructs the unified temporal subdivision.

        Returns:
            tuple:
                - full_time_grid (np.ndarray): The complete array of simulation times.
                - obsv_indices (np.ndarray): The integer indices corresponding to the
                                             original observation dates in the full grid.
        """
        full_time_grid = [0.0]
        obsv_indices = []

        current_index = 0
        previous_time = 0.0

        for target_time in self.obsv_grid:
            distance = target_time - previous_time
            required_steps = int(np.ceil(distance / self.dt_max))

            # required_steps + 1 yields bounds, slicing [1:] drops the overlapping previous_time
            segment_grid = np.linspace(previous_time, target_time, required_steps + 1)[
                1:
            ]
            full_time_grid.extend(segment_grid)

            current_index += required_steps
            obsv_indices.append(current_index)

            previous_time = target_time

        return np.array(full_time_grid), np.array(obsv_indices)
