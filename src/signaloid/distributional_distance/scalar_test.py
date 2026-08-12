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

"""Tests for the three scalar-comparison wrappers in
signaloid.distributional_distance.scalar."""

import math
import unittest

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance.scalar import (
    absolute_error_uxhw_wrapper,
    relative_error_uxhw_wrapper,
    signed_error_uxhw_wrapper,
)


def _single_dirac(position: float) -> DistributionalValue:
    """Build a DistributionalValue with a single Dirac at `position`."""
    return DistributionalValue(dirac_deltas=[DiracDelta(position=position, mass=1.0)])


class TestScalarDistance(unittest.TestCase):
    """Tests for scalar_distance_wrapper — relative error
    |test_dist[0] - ground_truth_dist[0]| / |ground_truth_dist[0]|."""

    def test_zero_when_identical(self) -> None:
        """Relative error is 0 when the two scalars match."""
        for value in (1.0, -5.0, 1e6, 0.001):
            distance = relative_error_uxhw_wrapper(
                _single_dirac(value), _single_dirac(value)
            )
            self.assertAlmostEqual(distance, 0.0, places=12, msg=f"v={value}")

    def test_one_percent_relative_error(self) -> None:
        """ground_truth_dist=100, test_dist=99 → 0.01."""
        distance = relative_error_uxhw_wrapper(
            _single_dirac(99.0), _single_dirac(100.0)
        )
        self.assertAlmostEqual(distance, 0.01, places=12)

    def test_ten_percent_relative_error(self) -> None:
        """ground_truth_dist=10, test_dist=11 → 0.1."""
        distance = relative_error_uxhw_wrapper(_single_dirac(11.0), _single_dirac(10.0))
        self.assertAlmostEqual(distance, 0.1, places=12)

    def test_sign_irrelevant_in_numerator_via_abs(self) -> None:
        """ground_truth_dist=1, test_dist=-1 → |-1-1|/|1| = 2."""
        distance = relative_error_uxhw_wrapper(_single_dirac(-1.0), _single_dirac(1.0))
        self.assertAlmostEqual(distance, 2.0, places=12)

    def test_negative_ground_truth_dist_uses_abs(self) -> None:
        """ground_truth_dist=-100, test_dist=-99 → |-99-(-100)|/|-100| = 0.01."""
        distance = relative_error_uxhw_wrapper(
            _single_dirac(-99.0), _single_dirac(-100.0)
        )
        self.assertAlmostEqual(distance, 0.01, places=12)

    def test_zero_ground_truth_dist_returns_positive_infinity(self) -> None:
        """Division by zero is intentional. Verify +inf specifically,
        not just any inf (`-inf` would also satisfy math.isinf)."""
        distance = relative_error_uxhw_wrapper(_single_dirac(1.0), _single_dirac(0.0))
        self.assertEqual(distance, math.inf)

    def test_very_large_numbers(self) -> None:
        """Relative error is scale-invariant. Verify no overflow at 1e300."""
        distance = relative_error_uxhw_wrapper(
            _single_dirac(1.01e300), _single_dirac(1.0e300)
        )
        self.assertAlmostEqual(distance, 0.01, places=12)

    def test_zero_over_zero_returns_nan(self) -> None:
        """ground_truth_dist=0, test_dist=0 → 0/0 = NaN (numpy invalid op, warning suppressed)."""
        distance = relative_error_uxhw_wrapper(_single_dirac(0.0), _single_dirac(0.0))
        self.assertTrue(math.isnan(distance))

    def test_empty_rejected_on_either_side(self) -> None:
        """Empty DistributionalValue is rejected naming the bad argument,
        on whichever side it appears."""
        empty = DistributionalValue(dirac_deltas=[])
        valid = _single_dirac(1.0)
        cases = (
            ("test_dist", empty, valid),
            ("ground_truth_dist", valid, empty),
        )
        for name, test_dist, ground_truth_dist in cases:
            with self.subTest(side=name):
                with self.assertRaises(ValueError) as ctx:
                    relative_error_uxhw_wrapper(test_dist, ground_truth_dist)
                self.assertIn(name, str(ctx.exception))

    def test_multi_dirac_rejected_on_either_side(self) -> None:
        """Function is named *scalar*_distance_wrapper — silently
        discarding positions[1:] would be a footgun. Rejected on
        whichever side a multi-Dirac DV appears."""
        multi = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=2.0, mass=0.5),
            ]
        )
        valid = _single_dirac(1.0)
        for name, test_dist, ground_truth_dist in (
            ("test_dist", multi, valid),
            ("ground_truth_dist", valid, multi),
        ):
            with self.subTest(side=name):
                with self.assertRaises(ValueError) as ctx:
                    relative_error_uxhw_wrapper(test_dist, ground_truth_dist)
                self.assertIn("scalar distribution", str(ctx.exception))

    def test_non_finite_position_rejected(self) -> None:
        """NaN / +inf / -inf positions are rejected on either side."""
        valid = _single_dirac(1.0)
        for side, value in (
            ("test_dist", float("nan")),
            ("ground_truth_dist", float("nan")),
            ("test_dist", float("inf")),
            ("ground_truth_dist", float("inf")),
            ("test_dist", float("-inf")),
            ("ground_truth_dist", float("-inf")),
        ):
            with self.subTest(side=side, value=value):
                bad = _single_dirac(value)
                test_dist, ground_truth_dist = (
                    (bad, valid) if side == "test_dist" else (valid, bad)
                )
                with self.assertRaises(ValueError) as ctx:
                    relative_error_uxhw_wrapper(test_dist, ground_truth_dist)
                self.assertIn("finite", str(ctx.exception))

    def test_non_dv_rejected_on_either_side(self) -> None:
        """Non-DistributionalValue inputs previously raised AttributeError
        on `.positions`. Now raise a clear ValueError on either side."""
        valid = _single_dirac(1.0)
        cases = (
            ("test_dist", "not a DV", valid),
            ("ground_truth_dist", valid, 42.0),
        )
        for name, test_dist, ground_truth_dist in cases:
            with self.subTest(side=name):
                with self.assertRaises(ValueError) as ctx:
                    relative_error_uxhw_wrapper(test_dist, ground_truth_dist)  # type: ignore[arg-type]
                self.assertIn(name, str(ctx.exception))
                self.assertIn("DistributionalValue", str(ctx.exception))

    def test_invalid_mass_rejected(self) -> None:
        """Regression: DistributionalValue allows constructing a Dirac
        with NaN / zero mass, which would silently flow through
        scalar_distance_wrapper since the mass doesn't enter the
        relative-error formula. Now rejected upfront.

        Negative-mass rejection lives in
        TestDistributionalValueInitMassValidation — DistributionalValue
        rejects negative masses at construction, so they can't reach
        this wrapper through a DV-shaped input."""
        valid = _single_dirac(1.0)
        for side, bad_mass in (
            ("test_dist", float("nan")),
            ("test_dist", 0.0),
            ("ground_truth_dist", float("nan")),
            ("ground_truth_dist", 0.0),
        ):
            with self.subTest(side=side, mass=bad_mass):
                bad = DistributionalValue(
                    dirac_deltas=[DiracDelta(position=1.0, mass=bad_mass)]
                )
                test_dist, ground_truth_dist = (
                    (bad, valid) if side == "test_dist" else (valid, bad)
                )
                with self.assertRaises(ValueError) as ctx:
                    relative_error_uxhw_wrapper(test_dist, ground_truth_dist)
                self.assertIn(side, str(ctx.exception))
                self.assertIn("masses", str(ctx.exception))


class TestSignedError(unittest.TestCase):
    """Tests for signed_error_uxhw_wrapper —
    test_dist[0] - ground_truth_dist[0]. Preserves sign. Same units as
    inputs and well-defined for any finite inputs (no zero-division
    edge case)."""

    def test_zero_when_identical(self) -> None:
        """signed_error(x, x) == 0 for any x."""
        for value in (1.0, -5.0, 1e6, 0.0, 0.001):
            distance = signed_error_uxhw_wrapper(
                _single_dirac(value), _single_dirac(value)
            )
            self.assertAlmostEqual(distance, 0.0, places=12, msg=f"v={value}")

    def test_positive_when_test_greater_than_ground_truth(self) -> None:
        """ground_truth_dist=10, test_dist=11 → +1."""
        distance = signed_error_uxhw_wrapper(_single_dirac(11.0), _single_dirac(10.0))
        self.assertAlmostEqual(distance, 1.0, places=12)

    def test_negative_when_test_less_than_ground_truth(self) -> None:
        """ground_truth_dist=10, test_dist=9 → -1.
        Sign is the load-bearing property distinguishing this wrapper
        from absolute_error."""
        distance = signed_error_uxhw_wrapper(_single_dirac(9.0), _single_dirac(10.0))
        self.assertAlmostEqual(distance, -1.0, places=12)

    def test_zero_ground_truth_dist_does_not_diverge(self) -> None:
        """signed_error has no division: ground_truth=0 is fully fine."""
        distance = signed_error_uxhw_wrapper(_single_dirac(1.0), _single_dirac(0.0))
        self.assertAlmostEqual(distance, 1.0, places=12)
        distance = signed_error_uxhw_wrapper(_single_dirac(0.0), _single_dirac(0.0))
        self.assertAlmostEqual(distance, 0.0, places=12)

    def test_both_negative_inputs(self) -> None:
        """ground_truth=-100, test=-99 → +1 (test - gt = -99 - -100)."""
        distance = signed_error_uxhw_wrapper(
            _single_dirac(-99.0), _single_dirac(-100.0)
        )
        self.assertAlmostEqual(distance, 1.0, places=12)


class TestAbsoluteError(unittest.TestCase):
    """Tests for absolute_error_uxhw_wrapper —
    |test_dist[0] - ground_truth_dist[0]|. Magnitude only. Same units
    as inputs and well-defined for any finite inputs."""

    def test_zero_when_identical(self) -> None:
        """absolute_error(x, x) == 0 for any x."""
        for value in (1.0, -5.0, 1e6, 0.0, 0.001):
            distance = absolute_error_uxhw_wrapper(
                _single_dirac(value), _single_dirac(value)
            )
            self.assertAlmostEqual(distance, 0.0, places=12, msg=f"v={value}")

    def test_one_unit_difference(self) -> None:
        """ground_truth_dist=10, test_dist=11 → 1."""
        distance = absolute_error_uxhw_wrapper(_single_dirac(11.0), _single_dirac(10.0))
        self.assertAlmostEqual(distance, 1.0, places=12)

    def test_sign_collapsed_via_abs(self) -> None:
        """abs collapses the sign — both (11, 10) and (9, 10) yield 1.
        Property check distinguishing this wrapper from signed_error."""
        positive = absolute_error_uxhw_wrapper(_single_dirac(11.0), _single_dirac(10.0))
        negative = absolute_error_uxhw_wrapper(_single_dirac(9.0), _single_dirac(10.0))
        self.assertAlmostEqual(positive, negative, places=12)
        self.assertAlmostEqual(positive, 1.0, places=12)

    def test_always_non_negative(self) -> None:
        """absolute_error is a magnitude — must be >= 0 across sign
        permutations of the inputs."""
        for test, gt in ((1.0, -1.0), (-1.0, 1.0), (-100.0, -99.0), (1e10, -1e10)):
            with self.subTest(test=test, gt=gt):
                distance = absolute_error_uxhw_wrapper(
                    _single_dirac(test), _single_dirac(gt)
                )
                self.assertGreaterEqual(distance, 0.0)

    def test_zero_ground_truth_dist_does_not_diverge(self) -> None:
        """absolute_error has no division: ground_truth=0 is fully fine."""
        distance = absolute_error_uxhw_wrapper(_single_dirac(1.0), _single_dirac(0.0))
        self.assertAlmostEqual(distance, 1.0, places=12)
        distance = absolute_error_uxhw_wrapper(_single_dirac(0.0), _single_dirac(0.0))
        self.assertAlmostEqual(distance, 0.0, places=12)

    def test_matches_abs_of_signed_error(self) -> None:
        """|signed_error(a, b)| == absolute_error(a, b) by construction.
        Regression guard: absolute_error is implemented as
        ``float(np.abs(signed_error_uxhw_wrapper(...)))``."""
        for test, gt in ((1.0, 10.0), (-5.0, 3.0), (1e6, -2e6), (0.001, 0.0)):
            with self.subTest(test=test, gt=gt):
                t, g = _single_dirac(test), _single_dirac(gt)
                self.assertAlmostEqual(
                    absolute_error_uxhw_wrapper(t, g),
                    abs(signed_error_uxhw_wrapper(t, g)),
                    places=12,
                )


if __name__ == "__main__":
    unittest.main()
