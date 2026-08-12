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

from signaloid.benchmarking.distribution_helpers.representation_health import (
    _representation_blow_up_reason,
)
from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue


class TestRepresentationBlowUpGuard(unittest.TestCase):
    """``_representation_blow_up_reason`` distinguishes benign special-value
    remnants from genuine representation blow-ups."""

    def test_benign_zero_mass_special_values_returns_none(self) -> None:
        """Finite deltas plus zero-mass NaN / +inf slots are benign: after
        ``drop_zero_mass_positions`` the helper returns ``None`` and the
        distribution reads as finite."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.4),
                DiracDelta(position=2.0, mass=0.6),
                DiracDelta(position=float("nan"), mass=0.0),
                DiracDelta(position=float("inf"), mass=0.0),
            ]
        )

        # Mirror the production call site: shed zero-mass slots first.
        dist.drop_zero_mass_positions()

        self.assertIsNone(_representation_blow_up_reason(dist))
        self.assertIs(dist.is_finite, True)

    def test_blow_up_non_finite_with_mass_returns_reason(self) -> None:
        """A non-finite position carrying non-zero mass is a genuine blow-up
        (mode 1): the helper returns a reason string."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("inf"), mass=0.5),
            ]
        )

        dist.drop_zero_mass_positions()

        reason = _representation_blow_up_reason(dist)
        self.assertIsNotNone(reason)
        assert reason is not None  # narrow for type checker
        self.assertIn("non-finite", reason)

    def test_blow_up_nan_position_with_mass_returns_reason(self) -> None:
        """A NaN position carrying mass is also flagged by mode 1."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=3.0, mass=0.7),
                DiracDelta(position=float("nan"), mass=0.3),
            ]
        )

        dist.drop_zero_mass_positions()

        reason = _representation_blow_up_reason(dist)
        self.assertIsNotNone(reason)

    def test_blow_up_absurd_magnitude_mass_returns_reason(self) -> None:
        """A finite distribution with a meaningful mass fraction parked at an
        overflow-scale ``|position| > BLOW_UP_POSITION_MAGNITUDE`` (~1.34e154)
        is a genuine blow-up (mode 2)."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.9),
                DiracDelta(position=1e300, mass=0.1),
            ]
        )

        dist.drop_zero_mass_positions()

        # The distribution is finite (1e300 is a finite float), so mode 1 does
        # not fire. Mode 2 must catch the overflow-scale mass.
        self.assertIs(dist.is_finite, True)
        reason = _representation_blow_up_reason(dist)
        self.assertIsNotNone(reason)
        assert reason is not None  # narrow for type checker
        self.assertIn("position", reason)

    def test_clean_finite_distribution_returns_none(self) -> None:
        """An ordinary finite distribution is not a blow-up."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=-1.0, mass=0.2),
                DiracDelta(position=0.0, mass=0.5),
                DiracDelta(position=1.0, mass=0.3),
            ]
        )

        self.assertIsNone(_representation_blow_up_reason(dist))

    def test_negligible_mass_at_absurd_magnitude_returns_none(self) -> None:
        """A truly negligible mass fraction at an overflow-scale magnitude is a
        benign remnant, not a blow-up (mode 2 mass fraction stays below
        threshold)."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=1.0 - 1e-9),
                DiracDelta(position=1e300, mass=1e-9),
            ]
        )

        self.assertIsNone(_representation_blow_up_reason(dist))

    def test_check_magnitude_false_ignores_large_finite_value(self) -> None:
        """With ``check_magnitude=False`` (the scalar path), even an
        overflow-scale finite position is not a blow-up: mode 2 is skipped and
        mode 1 does not fire on a finite position. The same value WOULD be
        flagged with ``check_magnitude=True``, which is why the gate matters."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1e300, mass=1.0),
            ]
        )

        self.assertIs(dist.is_finite, True)
        self.assertIsNone(_representation_blow_up_reason(dist, check_magnitude=False))
        self.assertIsNotNone(_representation_blow_up_reason(dist, check_magnitude=True))

    def test_check_magnitude_false_still_flags_non_finite(self) -> None:
        """With ``check_magnitude=False`` the non-finite mode 1 check still
        runs: a non-finite position carrying mass is flagged. (A non-finite
        scalar would crash ``relative_error_uxhw_wrapper``'s validator, so the
        scalar branch must still detect it.)"""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=2.0, mass=0.5),
                DiracDelta(position=float("inf"), mass=0.5),
            ]
        )

        dist.drop_zero_mass_positions()

        reason = _representation_blow_up_reason(dist, check_magnitude=False)
        self.assertIsNotNone(reason)
        assert reason is not None  # narrow for type checker
        self.assertIn("non-finite", reason)


if __name__ == "__main__":
    unittest.main()
