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

import math
import unittest
from typing import TypedDict

import numpy as np
import ot  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore

from .conftest import gaussianMotherTTRTestcases, gaussianSamples

from signaloid.distributional_distance.wasserstein import (
    wasserstein_1_uxhw_wrapper,
    wasserstein_p_distance,
)
from signaloid.distributional.distributional import DistributionalValue


class TestWasserstein1(unittest.TestCase):
    """Wasserstein-1 distance from an Athens/weighted distribution to its mother."""

    def test_wasserstein_1_bootleg_zero(self) -> None:
        """W1 is ~0 between a weighted distribution and a ground truth
        resampled from its own (position, mass) pairs."""

        class TestCase(TypedDict):
            positions: list[float]
            masses: list[float]

        testcases: list[TestCase] = [
            {"positions": [1, 2, 3, 4, 5], "masses": [0.1, 0.2, 0.1, 0.4, 0.2]},
            {"positions": [10, 20, 30, 40, 50], "masses": [0.1, 0.2, 0.1, 0.4, 0.2]},
            {"positions": [10, 20, 30, 40, 50], "masses": [0.3, 0.1, 0.1, 0.3, 0.2]},
            {
                "positions": [-100, 20, 500, 1000, 0],
                "masses": [0.3, 0.1, 0.1, 0.3, 0.2],
            },
        ]

        for tc in testcases:
            with self.subTest(positions=tc["positions"], masses=tc["masses"]):
                sample_arrays = []
                for pos, mass in zip(tc["positions"], tc["masses"]):
                    # Blow up the sample array
                    sample_arrays.append(
                        np.full(shape=math.floor(mass * 100), fill_value=pos)
                    )

                test_dist = DistributionalValue.from_weighted_samples(
                    tc["positions"], tc["masses"]
                )
                samples = np.concatenate((sample_arrays), axis=None)
                ground_truth = DistributionalValue.from_samples(samples)
                result = wasserstein_1_uxhw_wrapper(test_dist, ground_truth)

                self.assertAlmostEqual(result, 0)

    def test_wasserstein_1_gaussian_mother_zero(self) -> None:
        """W1 from a Gaussian Athens representation to its mother samples
        is near zero (within a loose tolerance)."""
        testcases = gaussianMotherTTRTestcases
        samples = gaussianSamples

        ground_truth = DistributionalValue.from_samples(samples)

        for tc in testcases:
            with self.subTest(size=len(tc["positions"])):
                test_dist = DistributionalValue.from_weighted_samples(
                    tc["positions"], tc["masses"]
                )
                result = wasserstein_1_uxhw_wrapper(test_dist, ground_truth)

                self.assertAlmostEqual(
                    result,
                    0,
                    delta=0.3,
                    msg=f"Assertion failed for Athens-{len(tc['positions'])}",
                )

    # For uniform(0, 1): dd_pos(i) = (i + 0.5) / N for i in [0, N).
    #
    # The DD positions of an Athens of size N for a uniform(a, a + L) are:
    # positions = [a + (i + 0.5) * L / N for i in range(N)]
    #
    # The Wasserstein distance between a uniform distribution with range L
    # and its Athens of size N is W(L, N) = L / (4N).
    def test_wasserstein_1_uniform_mother_zero(self) -> None:
        """W1 between a uniform(a, a + L) and its size-N Athens matches the
        closed form L / (4N)."""
        rng = np.random.default_rng(20240624)
        for a in np.arange(-5, 5, 1):  # Uniform start point
            for L in [0.5, 1, 2]:  # Uniform length
                samples = rng.uniform(a, a + L, size=100_000)

                for N in [4, 8, 16, 32, 64, 128]:  # Representation size
                    with self.subTest(a=a, L=L, N=N):
                        positions = [a + (i + 0.5) * L / N for i in range(N)]
                        masses = [1 / N] * N

                        # Check against formula
                        expected_distance = L / (4 * N)

                        ground_truth = DistributionalValue.from_samples(samples)
                        test_dist = DistributionalValue.from_weighted_samples(
                            np.asarray(positions), masses
                        )
                        result = wasserstein_1_uxhw_wrapper(test_dist, ground_truth)

                        self.assertAlmostEqual(
                            result,
                            expected_distance,
                            delta=0.05,
                            msg=f"Expectation failed for a={a},L={L}, N={N}",
                        )


class TestWassersteinPDistanceParity(unittest.TestCase):
    """`wasserstein_p_distance` matches the scipy / POT calls it replaces.

    Pins numerical parity on the exact input shapes the benchmarking call
    sites use: `equivalent_mc_utils._distance_func` (raw weighted arrays,
    ``u_weights`` may be `None`) and the `adversary_distance` W1/W2 loops
    (unweighted samples vs an optionally weighted ground truth). W1 is
    checked against ``scipy.stats.wasserstein_distance``. W2 against
    ``sqrt(ot.wasserstein_1d(..., p=2))``.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(seed=20260701)
        # Unweighted samples (the adversary-loop `u` and the `_distance_func`
        # convergence samples, whose `u_weights` is None).
        self.u_samples = rng.standard_normal(2000)
        # Weighted ground truth (positions + non-uniform masses).
        self.v_positions = rng.standard_normal(300) + 0.4
        v_masses = rng.random(300) + 0.05
        self.v_masses = v_masses / v_masses.sum()
        # A fully weighted `u` for the general _distance_func signature.
        u_masses = rng.random(2000) + 0.05
        self.u_masses = u_masses / u_masses.sum()

    def test_w1_unweighted_both_matches_scipy(self) -> None:
        """p=1, u_masses=None and v_masses=None (adversary unweighted branch)."""
        expected = float(wasserstein_distance(self.u_samples, self.v_positions))
        actual = wasserstein_p_distance(
            self.u_samples, None, self.v_positions, None, p=1
        )
        self.assertAlmostEqual(actual, expected, places=9)

    def test_w1_weighted_v_matches_scipy(self) -> None:
        """p=1, unweighted u vs weighted v (adversary weighted branch)."""
        expected = float(
            wasserstein_distance(
                u_values=self.u_samples,
                v_values=self.v_positions,
                v_weights=self.v_masses,
            )
        )
        actual = wasserstein_p_distance(
            self.u_samples, None, self.v_positions, self.v_masses, p=1
        )
        self.assertAlmostEqual(actual, expected, places=9)

    def test_w1_both_weighted_matches_scipy(self) -> None:
        """p=1, both sides weighted (general _distance_func signature)."""
        expected = float(
            wasserstein_distance(
                self.u_samples, self.v_positions, self.u_masses, self.v_masses
            )
        )
        actual = wasserstein_p_distance(
            self.u_samples, self.u_masses, self.v_positions, self.v_masses, p=1
        )
        self.assertAlmostEqual(actual, expected, places=9)

    def test_w2_unweighted_both_matches_ot(self) -> None:
        """p=2, u_masses=None and v_masses=None (adversary unweighted branch)."""
        expected = float(
            np.sqrt(ot.wasserstein_1d(self.u_samples, self.v_positions, p=2))
        )
        actual = wasserstein_p_distance(
            self.u_samples, None, self.v_positions, None, p=2
        )
        self.assertAlmostEqual(actual, expected, places=9)

    def test_w2_weighted_v_matches_ot(self) -> None:
        """p=2, unweighted u vs weighted v (adversary weighted branch)."""
        expected = float(
            np.sqrt(
                ot.wasserstein_1d(
                    u_values=self.u_samples,
                    v_values=self.v_positions,
                    v_weights=self.v_masses,
                    p=2,
                )
            )
        )
        actual = wasserstein_p_distance(
            self.u_samples, None, self.v_positions, self.v_masses, p=2
        )
        self.assertAlmostEqual(actual, expected, places=9)

    def test_w2_both_weighted_matches_ot(self) -> None:
        """p=2, both sides weighted (general _distance_func signature)."""
        expected = float(
            np.sqrt(
                ot.wasserstein_1d(
                    self.u_samples,
                    self.v_positions,
                    self.u_masses,
                    self.v_masses,
                    p=2,
                )
            )
        )
        actual = wasserstein_p_distance(
            self.u_samples, self.u_masses, self.v_positions, self.v_masses, p=2
        )
        self.assertAlmostEqual(actual, expected, places=9)


if __name__ == "__main__":
    unittest.main()
