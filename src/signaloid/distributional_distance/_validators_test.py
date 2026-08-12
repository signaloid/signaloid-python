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

"""Tests for the package-private validators in `_validators.py`.

Direct exercise of the shared validators so that wrapper-level tests can
focus on wrapper behavior (ValueError raised on bad input) without
coupling to the validator's error-message text.
"""

import unittest

import numpy as np

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance._validators import (
    _require_distributional_pair,
    _validate_wp_inputs,
)


def _single_dirac(position: float) -> DistributionalValue:
    return DistributionalValue(dirac_deltas=[DiracDelta(position=position, mass=1.0)])


BAD_STAND_INS = ("not a dv", 42, None, [1.0, 2.0], np.array([1.0]))


class TestRequireDistributionalPair(unittest.TestCase):
    """Direct exercise of `_require_distributional_pair`.

    Wrappers across the package funnel non-DV inputs through this helper,
    so the message format / type-error coverage lives here once instead
    of being re-asserted in every wrapper's test file."""

    def test_accepts_two_valid_distributional_values(self) -> None:
        """Two DistributionalValues: returns without raising."""
        valid = _single_dirac(1.0)
        _require_distributional_pair(valid, valid)

    def test_rejects_non_dv_in_first_position(self) -> None:
        """Bad first argument: error names `dist_u` and surfaces the
        actual type, so the caller can see which argument was wrong."""
        valid = _single_dirac(1.0)
        for bad in BAD_STAND_INS:
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError) as ctx:
                    _require_distributional_pair(bad, valid)  # type: ignore[arg-type]
                message = str(ctx.exception)
                self.assertIn("dist_u", message)
                self.assertIn("DistributionalValue", message)
                self.assertIn(type(bad).__name__, message)

    def test_rejects_non_dv_in_second_position(self) -> None:
        """Bad second argument: error names `dist_v` and the actual type."""
        valid = _single_dirac(1.0)
        for bad in BAD_STAND_INS:
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError) as ctx:
                    _require_distributional_pair(valid, bad)  # type: ignore[arg-type]
                message = str(ctx.exception)
                self.assertIn("dist_v", message)
                self.assertIn("DistributionalValue", message)
                self.assertIn(type(bad).__name__, message)

    def test_first_argument_checked_before_second(self) -> None:
        """Both arguments bad: the first one raises, so the message
        identifies `dist_u`. Check matters because callers will fix the
        bug the error names — if both errors are reported as `dist_v`
        the user can't tell that the first argument is also wrong."""
        with self.assertRaises(ValueError) as ctx:
            _require_distributional_pair("bad_u", "bad_v")  # type: ignore[arg-type]
        self.assertIn("dist_u", str(ctx.exception))


class TestValidateWpInputs(unittest.TestCase):
    """Direct exercise of `_validate_wp_inputs`, the shared array-level
    validator. Wrapper-level tests should rely on these — not duplicate
    the per-condition coverage."""

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u_values = np.array([1.0, 2.0, 3.0])
        u_weights = np.array([0.3, 0.4, 0.3])
        v_values = np.array([0.5, 1.5, 2.5])
        v_weights = np.array([0.25, 0.5, 0.25])
        return u_values, u_weights, v_values, v_weights

    def test_returns_totals_when_all_inputs_valid(self) -> None:
        """Happy path: returns the (u_total, v_total) the caller will
        divide by. Both equal to the sum of their weights."""
        u_values, u_weights, v_values, v_weights = self._arrays()
        u_total, v_total = _validate_wp_inputs(
            u_values, u_weights, v_values, v_weights, p=1
        )
        self.assertAlmostEqual(u_total, 1.0)
        self.assertAlmostEqual(v_total, 1.0)

    def test_rejects_non_integer_p(self) -> None:
        """`p` must be int and >= 1 — float, str, bool all rejected.
        `bool` matters specifically: it subclasses int in Python, so
        `p=True` would silently be treated as `p=1` without the explicit
        bool check in the validator."""
        u_values, u_weights, v_values, v_weights = self._arrays()
        for bad_p in (1.0, "1", True, False, 0, -1):
            with self.subTest(p=bad_p):
                with self.assertRaises(ValueError) as ctx:
                    _validate_wp_inputs(
                        u_values, u_weights, v_values, v_weights, p=bad_p  # type: ignore[arg-type]
                    )
                self.assertIn("p", str(ctx.exception))

    def test_rejects_non_1d_arrays(self) -> None:
        """2-D arrays would silently flow into the Wp kernel and produce
        a wrong distance. Rejected up-front."""
        u_values, u_weights, v_values, v_weights = self._arrays()
        bad_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        for label, args in (
            ("u_values_2d", (bad_2d, u_weights, v_values, v_weights)),
            ("u_weights_2d", (u_values, bad_2d, v_values, v_weights)),
            ("v_values_2d", (u_values, u_weights, bad_2d, v_weights)),
            ("v_weights_2d", (u_values, u_weights, v_values, bad_2d)),
        ):
            with self.subTest(case=label):
                u, uw, v, vw = args
                with self.assertRaises(ValueError) as ctx:
                    _validate_wp_inputs(u, uw, v, vw, p=1)
                self.assertIn("one-dimensional", str(ctx.exception))

    def test_rejects_empty_positions(self) -> None:
        """Empty positions are rejected with a non-empty message."""
        u_values, u_weights, v_values, v_weights = self._arrays()
        empty = np.array([], dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            _validate_wp_inputs(empty, empty, v_values, v_weights, p=1)
        self.assertIn("non-empty", str(ctx.exception))

    def test_rejects_mismatched_lengths(self) -> None:
        """Positions and weights must have matching lengths per side."""
        u_values, _, v_values, v_weights = self._arrays()
        wrong_length = np.array([0.5, 0.5])
        with self.assertRaises(ValueError) as ctx:
            _validate_wp_inputs(u_values, wrong_length, v_values, v_weights, p=1)
        self.assertIn("matching lengths", str(ctx.exception))

    def test_rejects_non_finite_entries(self) -> None:
        """NaN / ±Inf in any of the four arrays is rejected — the Wp
        integral is undefined for non-finite positions or weights."""
        u_values, u_weights, v_values, v_weights = self._arrays()
        for label, mutated in (
            ("nan_position", np.array([1.0, float("nan"), 3.0])),
            ("posinf_position", np.array([1.0, float("inf"), 3.0])),
            ("neginf_position", np.array([1.0, float("-inf"), 3.0])),
        ):
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    _validate_wp_inputs(mutated, u_weights, v_values, v_weights, p=1)
                self.assertIn("finite", str(ctx.exception))

    def test_rejects_negative_weights(self) -> None:
        """Negative weights break the empirical-CDF interpretation."""
        u_values, _, v_values, v_weights = self._arrays()
        negative_weights = np.array([0.5, -0.1, 0.6])
        with self.assertRaises(ValueError) as ctx:
            _validate_wp_inputs(u_values, negative_weights, v_values, v_weights, p=1)
        self.assertIn("non-negative", str(ctx.exception))

    def test_rejects_zero_total_mass(self) -> None:
        """All-zero weights mean the CDF can't be normalised. Rejected."""
        u_values, _, v_values, v_weights = self._arrays()
        zero_weights = np.zeros(3)
        with self.assertRaises(ValueError) as ctx:
            _validate_wp_inputs(u_values, zero_weights, v_values, v_weights, p=1)
        self.assertIn("strictly positive", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
