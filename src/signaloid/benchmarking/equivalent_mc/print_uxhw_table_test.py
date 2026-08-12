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

import io
import unittest
from contextlib import redirect_stdout

from signaloid.benchmarking.equivalent_mc.equivalent_mc_main import (
    BLOW_UP_TABLE_TOKEN,
    _print_uxhw_table,
)
from signaloid.benchmarking.types import TaggedDistributionalValue
from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue


def _tagged(
    dv: DistributionalValue,
    representation_type: str,
    correlation_tracking: str = "Disabled",
) -> TaggedDistributionalValue:
    """A UxHw carrier as produced by the tracing-table loader in load.py."""
    return TaggedDistributionalValue(
        dv=dv,
        representation_type=representation_type,
        representation_size=dv.UR_order,
        correlation_tracking=correlation_tracking,
    )


class TestPrintUxHwTableBlowUp(unittest.TestCase):
    """``_print_uxhw_table`` degrades a blown-up representation gracefully
    instead of overflowing on the variance read of the re-loaded dv."""

    def test_blown_config_does_not_raise_and_emits_excluded_row(self) -> None:
        """A config whose dv parks real mass at ~1e303 (e.g. EDA-ngspice Athens
        order 32) would overflow ``DistributionalValue.calculate_variance``
        ((position - mean) ** 2). ``_print_uxhw_table`` must not raise and
        must mark that row "blow-up / excluded" while still printing the
        healthy rows' statistics."""
        healthy = _tagged(
            DistributionalValue(
                dirac_deltas=[
                    DiracDelta(position=1.0, mass=0.5),
                    DiracDelta(position=2.0, mass=0.5),
                ]
            ),
            representation_type="Athens",
        )
        blown = _tagged(
            DistributionalValue(
                dirac_deltas=[
                    DiracDelta(position=1.0, mass=0.5),
                    DiracDelta(position=1e303, mass=0.5),
                ]
            ),
            representation_type="Jupiter",
        )

        # Guard the premise: reading the blown dv's variance overflows.
        with self.assertRaises(OverflowError):
            blown.dv.calculate_variance()

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            # The bug: this used to raise OverflowError and crash the sweep.
            _print_uxhw_table(uxhw=[healthy, blown])
        output = buffer.getvalue()

        # The blown row is rendered as excluded for both Mean and Variance.
        self.assertEqual(output.count(BLOW_UP_TABLE_TOKEN), 2)
        self.assertIn("Jupiter", output)
        # The healthy row still reports its real statistics (mean 1.5).
        self.assertIn("Athens", output)
        self.assertIn("1.5", output)

    def test_all_healthy_configs_print_statistics(self) -> None:
        """With no blow-up, the table prints the real mean/variance and the
        excluded token does not appear."""
        healthy = _tagged(
            DistributionalValue(
                dirac_deltas=[
                    DiracDelta(position=10.0, mass=0.5),
                    DiracDelta(position=20.0, mass=0.5),
                ]
            ),
            representation_type="Athens",
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _print_uxhw_table(uxhw=[healthy])
        output = buffer.getvalue()

        self.assertNotIn(BLOW_UP_TABLE_TOKEN, output)
        # tabulate renders the mean (15.0) and variance (25.0) of the two
        # deltas. It normalises trailing ".0" away, so assert on the integers.
        self.assertIn("15", output)
        self.assertIn("25", output)


if __name__ == "__main__":
    unittest.main()
