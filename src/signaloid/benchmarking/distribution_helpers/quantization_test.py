#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

import unittest

import numpy as np

from signaloid.benchmarking.distribution_helpers.quantization import (
    _asymptotically_optimal_wasserstein_p_representation,
    _weighted_quantile,
)


class TestWeightedQuantile(unittest.TestCase):
    def test_uniform_weights_match_linear_interpolation(self) -> None:
        """Equal weights reduce to an interp over the empirical CDF grid.

        For n equally-weighted points, ``_weighted_quantile`` interpolates over
        the cumulative-weight grid ``[1/n, 2/n, ..., 1]`` (right-edge of each
        step), so the expected output is ``np.interp`` against that grid -- a
        deterministic analytic reference, not ``np.quantile`` (which uses the
        midpoint grid by default).
        """
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        n = len(values)
        cumulative_grid = np.arange(1, n + 1) / n
        expected = np.interp(quantiles, cumulative_grid, np.sort(values))
        np.testing.assert_allclose(_weighted_quantile(values, quantiles), expected)

    def test_median_matches_interp_on_cumulative_grid(self) -> None:
        """The 0.5 quantile interpolates the right-edge cumulative-weight grid.

        For 5 equally-weighted points the cumulative grid is
        [.2, .4, .6, .8, 1.0]; interp(0.5, grid, sorted_values) falls halfway
        between the .4 -> -1.0 and .6 -> 0.0 knots, i.e. -0.5.
        """
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = _weighted_quantile(values, [0.5])
        np.testing.assert_allclose(result, np.array([-0.5]))

    def test_unsorted_input_is_sorted_internally(self) -> None:
        """Output is invariant to input ordering (values are argsorted)."""
        ordered = _weighted_quantile([0.0, 1.0, 2.0], [0.5])
        shuffled = _weighted_quantile([2.0, 0.0, 1.0], [0.5])
        np.testing.assert_allclose(ordered, shuffled)


class TestAsymptoticRepresentation(unittest.TestCase):
    def test_preserves_total_mass(self) -> None:
        """The collapsed masses always sum to 1.0."""
        positions = np.linspace(0.0, 10.0, 50)
        masses = np.ones_like(positions)
        _, new_masses = _asymptotically_optimal_wasserstein_p_representation(
            positions, masses, n_dirac_deltas=8, p=1
        )
        self.assertAlmostEqual(float(np.sum(new_masses)), 1.0)
        self.assertEqual(len(new_masses), 8)

    def test_deterministic(self) -> None:
        """Same inputs give bit-identical outputs (pure-numpy, no RNG)."""
        positions = np.linspace(-1.0, 1.0, 40)
        masses = np.linspace(1.0, 2.0, 40)
        first = _asymptotically_optimal_wasserstein_p_representation(
            positions, masses, n_dirac_deltas=6, p=1
        )
        second = _asymptotically_optimal_wasserstein_p_representation(
            positions, masses, n_dirac_deltas=6, p=1
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_collapsed_positions_within_support(self) -> None:
        """Collapsed positions stay within the original support range."""
        positions = np.linspace(3.0, 7.0, 30)
        masses = np.ones_like(positions)
        new_positions, _ = _asymptotically_optimal_wasserstein_p_representation(
            positions, masses, n_dirac_deltas=5, p=1
        )
        self.assertGreaterEqual(float(np.min(new_positions)), 3.0)
        self.assertLessEqual(float(np.max(new_positions)), 7.0)


if __name__ == "__main__":
    unittest.main()
