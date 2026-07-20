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

"""Tests for signaloid.distributional_distance.ks_distance."""

import unittest
from typing import Sequence

import numpy as np

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance.ks_distance import (
    _ks_distance,
    kolmogorov_smirnov_distance_uxhw_wrapper,
)


def _dv_from_weighted_samples(
    positions: Sequence[float], masses: Sequence[float]
) -> DistributionalValue:
    dirac_deltas = [
        DiracDelta(position=float(p), mass=float(m)) for p, m in zip(positions, masses)
    ]
    return DistributionalValue(dirac_deltas=dirac_deltas)


class TestKolmogorovSmirnovDistance(unittest.TestCase):
    """Tests for the KS distance (sup_x |F_u(x) - F_v(x)|)."""

    def test_self_distance_is_zero(self) -> None:
        """KS(X, X) = 0."""
        dv = _dv_from_weighted_samples([-1.0, 0.5, 2.5], [0.2, 0.5, 0.3])
        self.assertAlmostEqual(
            kolmogorov_smirnov_distance_uxhw_wrapper(dv, dv), 0.0, places=12
        )

    def test_two_distinct_diracs_yields_one(self) -> None:
        """KS distance between two single Diracs at different positions
        is exactly 1 — the CDFs are 0/1 step functions that disagree by
        the full unit between min(a, b) and max(a, b)."""
        for a, b in ((0.0, 1.0), (-5.0, 5.0), (-1.5, -1.0)):
            with self.subTest(a=a, b=b):
                u = _dv_from_weighted_samples([a], [1.0])
                v = _dv_from_weighted_samples([b], [1.0])
                self.assertAlmostEqual(
                    kolmogorov_smirnov_distance_uxhw_wrapper(u, v),
                    1.0,
                    places=12,
                )

    def test_bounded_in_unit_interval(self) -> None:
        """KS distance is always in [0, 1] since both CDFs are in [0, 1]."""
        rng = np.random.default_rng(seed=20260520)
        for _ in range(20):
            n_u = int(rng.integers(2, 20))
            n_v = int(rng.integers(2, 20))
            u_pos = rng.standard_normal(n_u) * 2
            v_pos = rng.standard_normal(n_v) * 2 + 0.5
            u_m = rng.uniform(0.1, 1.0, n_u)
            v_m = rng.uniform(0.1, 1.0, n_v)
            u = _dv_from_weighted_samples(list(u_pos), list(u_m))
            v = _dv_from_weighted_samples(list(v_pos), list(v_m))
            d = kolmogorov_smirnov_distance_uxhw_wrapper(u, v)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_known_two_point_example(self) -> None:
        """Hand-computed: u = {0:0.5, 1:0.5}, v = {0:1.0}.

        Right-continuous step CDFs:
        F_u(x) = 0 for x<0, 0.5 for 0≤x<1, 1 for x≥1.
        F_v(x) = 0 for x<0, 1   for x≥0.
        |F_u − F_v| at the union {0, 1} is {0.5, 0}.
        KS = 0.5.
        """
        u = _dv_from_weighted_samples([0.0, 1.0], [0.5, 0.5])
        v = _dv_from_weighted_samples([0.0], [1.0])
        self.assertAlmostEqual(
            kolmogorov_smirnov_distance_uxhw_wrapper(u, v), 0.5, places=12
        )

    def test_known_triangle_example(self) -> None:
        """Hand-computed three-point case with overlapping support.

        u = {0:0.5, 2:0.5}, v = {0:0.25, 1:0.5, 2:0.25}.
        F_u at {0,1,2} = {0.5, 0.5, 1.0}; F_v = {0.25, 0.75, 1.0}.
        |diff| = {0.25, 0.25, 0.0}. KS = 0.25.
        """
        u = _dv_from_weighted_samples([0.0, 2.0], [0.5, 0.5])
        v = _dv_from_weighted_samples([0.0, 1.0, 2.0], [0.25, 0.5, 0.25])
        self.assertAlmostEqual(
            kolmogorov_smirnov_distance_uxhw_wrapper(u, v), 0.25, places=12
        )

    def test_dirac_vs_two_point_overlap(self) -> None:
        """Dirac at a vs equal-mix of {a, b}: F_u(a)=1, F_v(a)=0.5.

        Maximum gap is 0.5 in the half-open interval [a, b).
        """
        u = _dv_from_weighted_samples([0.0], [1.0])
        v = _dv_from_weighted_samples([0.0, 3.0], [0.5, 0.5])
        self.assertAlmostEqual(
            kolmogorov_smirnov_distance_uxhw_wrapper(u, v), 0.5, places=12
        )

    def test_two_equal_mass_uniform_discretisations_half_overlap(self) -> None:
        """Two equal-mass integer grids shifted by half their length.

        u_pos = {0, 1, …, 2N-1}, v_pos = u_pos + N, each with N points
        in [0, 2N-1] resp. [N, 3N-1] at equal mass 1/(2N). At x = 2N-1
        (the last u point), F_u = 1, F_v = N/(2N) = 0.5 → diff = 0.5.
        Integer positions avoid float-precision artefacts at shared
        grid points. KS = 0.5 exactly.
        """
        n = 200
        u_pos = np.arange(2 * n, dtype=np.float64)
        v_pos = u_pos + float(n)
        masses = np.full(2 * n, 1.0 / (2 * n))
        u = _dv_from_weighted_samples(list(u_pos), list(masses))
        v = _dv_from_weighted_samples(list(v_pos), list(masses))
        self.assertAlmostEqual(
            kolmogorov_smirnov_distance_uxhw_wrapper(u, v), 0.5, places=12
        )

    def test_symmetry(self) -> None:
        """KS is symmetric: KS(u, v) = KS(v, u)."""
        rng = np.random.default_rng(seed=20260521)
        for _ in range(10):
            n_u = int(rng.integers(2, 10))
            n_v = int(rng.integers(2, 10))
            u = _dv_from_weighted_samples(
                list(rng.standard_normal(n_u)),
                list(rng.uniform(0.1, 1.0, n_u)),
            )
            v = _dv_from_weighted_samples(
                list(rng.standard_normal(n_v)),
                list(rng.uniform(0.1, 1.0, n_v)),
            )
            d_uv = kolmogorov_smirnov_distance_uxhw_wrapper(u, v)
            d_vu = kolmogorov_smirnov_distance_uxhw_wrapper(v, u)
            self.assertAlmostEqual(d_uv, d_vu, places=12)

    def test_triangle_inequality(self) -> None:
        """KS is a metric: KS(u, w) ≤ KS(u, v) + KS(v, w)."""
        rng = np.random.default_rng(seed=20260522)
        for _ in range(10):
            u = _dv_from_weighted_samples(
                list(rng.standard_normal(5)), list(rng.uniform(0.1, 1.0, 5))
            )
            v = _dv_from_weighted_samples(
                list(rng.standard_normal(5)), list(rng.uniform(0.1, 1.0, 5))
            )
            w = _dv_from_weighted_samples(
                list(rng.standard_normal(5)), list(rng.uniform(0.1, 1.0, 5))
            )
            d_uv = kolmogorov_smirnov_distance_uxhw_wrapper(u, v)
            d_vw = kolmogorov_smirnov_distance_uxhw_wrapper(v, w)
            d_uw = kolmogorov_smirnov_distance_uxhw_wrapper(u, w)
            # Tolerance covers floating-point slop at the metric bound.
            self.assertLessEqual(d_uw, d_uv + d_vw + 1e-12)

    def test_permutation_invariance(self) -> None:
        """KS depends on the (position, mass) multiset, not input order."""
        sorted_pos = [-1.0, 0.5, 1.5]
        sorted_w = [0.1, 0.6, 0.3]
        permuted_pos = [1.5, -1.0, 0.5]
        permuted_w = [0.3, 0.1, 0.6]
        gt = _dv_from_weighted_samples([0.0, 1.0, 2.0], [0.3, 0.4, 0.3])

        u_sorted = _dv_from_weighted_samples(sorted_pos, sorted_w)
        u_permuted = _dv_from_weighted_samples(permuted_pos, permuted_w)
        self.assertAlmostEqual(
            kolmogorov_smirnov_distance_uxhw_wrapper(u_sorted, gt),
            kolmogorov_smirnov_distance_uxhw_wrapper(u_permuted, gt),
            places=12,
        )

    def test_unnormalised_masses_normalise_internally(self) -> None:
        """Scaling all masses by a positive constant leaves KS unchanged."""
        u_pos = [-1.0, 0.5, 2.5]
        u_m = [0.1, 0.4, 0.5]
        v_pos = [-0.5, 1.0]
        v_m = [0.6, 0.4]

        canonical = kolmogorov_smirnov_distance_uxhw_wrapper(
            _dv_from_weighted_samples(u_pos, u_m),
            _dv_from_weighted_samples(v_pos, v_m),
        )
        scaled = kolmogorov_smirnov_distance_uxhw_wrapper(
            _dv_from_weighted_samples(u_pos, [m * 13.0 for m in u_m]),
            _dv_from_weighted_samples(v_pos, [m * 0.7 for m in v_m]),
        )
        self.assertAlmostEqual(canonical, scaled, places=12)

    def test_non_dv_rejected_on_either_side(self) -> None:
        """Wrapper funnels through the shared `_require_distributional_pair`
        check; both arguments are validated. Message-format coverage
        lives in `_validators_test.py`; here we only assert the wrapper
        raises `ValueError` mentioning the expected type."""
        good = _dv_from_weighted_samples([0.0, 1.0], [0.5, 0.5])
        for side, dist_u, dist_v in (
            ("first", "not a DV", good),
            ("second", good, 42),
        ):
            with self.subTest(side=side):
                with self.assertRaises(ValueError) as ctx:
                    kolmogorov_smirnov_distance_uxhw_wrapper(dist_u, dist_v)  # type: ignore[arg-type]
                self.assertIn("DistributionalValue", str(ctx.exception))

    def test_helper_rejects_non_finite_positions(self) -> None:
        """`_ks_distance` reuses `_validate_wp_inputs`, so NaN/Inf
        positions and masses are rejected the same way as Wasserstein."""
        with self.assertRaises(ValueError) as ctx:
            _ks_distance(
                u_positions=np.array([0.0, np.nan]),
                u_masses=np.array([0.5, 0.5]),
                v_positions=np.array([0.0, 1.0]),
                v_masses=np.array([0.5, 0.5]),
            )
        self.assertIn("finite", str(ctx.exception))

    def test_helper_rejects_empty_positions(self) -> None:
        """Empty positions are rejected via `_validate_wp_inputs`."""
        with self.assertRaises(ValueError) as ctx:
            _ks_distance(
                u_positions=np.array([], dtype=np.float64),
                u_masses=np.array([], dtype=np.float64),
                v_positions=np.array([0.0]),
                v_masses=np.array([1.0]),
            )
        self.assertIn("non-empty", str(ctx.exception))

    def test_helper_rejects_mismatched_lengths(self) -> None:
        """Position/mass length mismatch is rejected at the validator."""
        with self.assertRaises(ValueError) as ctx:
            _ks_distance(
                u_positions=np.array([0.0, 1.0]),
                u_masses=np.array([1.0]),
                v_positions=np.array([0.0]),
                v_masses=np.array([1.0]),
            )
        self.assertIn("matching lengths", str(ctx.exception))

    def test_helper_rejects_negative_masses(self) -> None:
        """Negative masses are rejected at the validator."""
        with self.assertRaises(ValueError) as ctx:
            _ks_distance(
                u_positions=np.array([0.0, 1.0]),
                u_masses=np.array([0.5, -0.5]),
                v_positions=np.array([0.0, 1.0]),
                v_masses=np.array([0.5, 0.5]),
            )
        self.assertIn("non-negative", str(ctx.exception))

    def test_helper_rejects_zero_total_mass(self) -> None:
        """All-zero masses on a side give zero total — rejected."""
        with self.assertRaises(ValueError) as ctx:
            _ks_distance(
                u_positions=np.array([0.0, 1.0]),
                u_masses=np.array([0.0, 0.0]),
                v_positions=np.array([0.0, 1.0]),
                v_masses=np.array([0.5, 0.5]),
            )
        self.assertIn("positive total mass", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
