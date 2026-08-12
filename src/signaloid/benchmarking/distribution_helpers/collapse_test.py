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

from signaloid.benchmarking.distribution_helpers.collapse import (
    _collapse_asymptotically_optimal_w1,
)
from signaloid.distributional.distributional import DistributionalValue


def _uniform_distribution(n: int) -> DistributionalValue:
    """A deterministic equally-weighted distribution on ``[0, 1]``."""
    positions = list(np.linspace(0.0, 1.0, n))
    masses = [1.0] * n
    return DistributionalValue.from_weighted_samples(positions, masses)


class TestCollapseAsymptoticallyOptimalW1(unittest.TestCase):
    def test_preserves_total_mass(self) -> None:
        """Collapsing to N deltas keeps total mass at 1.0 with exactly N deltas."""
        dist = _uniform_distribution(64)
        collapsed = _collapse_asymptotically_optimal_w1(dist, n_dirac_deltas=8)
        self.assertEqual(len(collapsed.dirac_deltas), 8)
        self.assertAlmostEqual(float(np.sum(collapsed.masses)), 1.0)

    def test_deterministic(self) -> None:
        """Same input collapses to identical positions and masses (no RNG)."""
        first = _collapse_asymptotically_optimal_w1(
            _uniform_distribution(64), n_dirac_deltas=8
        )
        second = _collapse_asymptotically_optimal_w1(
            _uniform_distribution(64), n_dirac_deltas=8
        )
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.masses, second.masses)

    def test_collapsed_positions_within_support(self) -> None:
        """Collapsed positions stay within the original support ``[0, 1]``."""
        collapsed = _collapse_asymptotically_optimal_w1(
            _uniform_distribution(64), n_dirac_deltas=8
        )
        self.assertGreaterEqual(float(np.min(collapsed.positions)), 0.0)
        self.assertLessEqual(float(np.max(collapsed.positions)), 1.0)

    def test_fewer_than_two_deltas_raises(self) -> None:
        with self.assertRaises(ValueError):
            _collapse_asymptotically_optimal_w1(
                _uniform_distribution(16), n_dirac_deltas=1
            )


if __name__ == "__main__":
    unittest.main()
