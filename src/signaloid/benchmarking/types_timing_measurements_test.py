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

from signaloid.benchmarking.types import TimingMeasurements


class TestTimingMeasurements(unittest.TestCase):
    """TimingMeasurements accumulates per-config timing records."""

    def test_timing_measurements_defaults(self) -> None:
        """Confirm TimingMeasurements initialises with an empty measurement_dict."""
        timing = TimingMeasurements()
        self.assertEqual(timing.measurement_dict, {})

    def test_timing_measurements_append_all_fields(self) -> None:
        """Confirm append stores the five measurement fields under the given config key."""
        timing = TimingMeasurements()
        timing.append(
            config="Athens-16",
            time=1.0,
            e2e_time=1.5,
            pin_dyn_inst_count=1000.0,
            db_time=0.1,
            db_dyn_inst_count=50.0,
        )
        self.assertIn("Athens-16", timing.measurement_dict)
        record = timing.measurement_dict["Athens-16"]
        self.assertEqual(record["In Application Time"], 1.0)
        self.assertEqual(record["End-to-End Time"], 1.5)
        self.assertEqual(record["PIN Dyn. Inst. Count"], 1000.0)
        self.assertEqual(record["Database Time"], 0.1)
        self.assertEqual(record["Database Dyn. Inst. Count"], 50.0)

    def test_timing_measurements_append_default_db_fields(self) -> None:
        """Confirm db_time and db_dyn_inst_count default to 0.0 when omitted."""
        timing = TimingMeasurements()
        timing.append(
            config="Native-MC-256",
            time=2.0,
            e2e_time=2.5,
            pin_dyn_inst_count=2000.0,
        )
        self.assertIn("Native-MC-256", timing.measurement_dict)
        record = timing.measurement_dict["Native-MC-256"]
        self.assertEqual(record["Database Time"], 0.0)
        self.assertEqual(record["Database Dyn. Inst. Count"], 0.0)

    def test_timing_measurements_append_multiple_configs(self) -> None:
        """Confirm multiple configs accumulate independently in measurement_dict."""
        timing = TimingMeasurements()
        timing.append(
            config="Athens-16",
            time=1.0,
            e2e_time=1.5,
            pin_dyn_inst_count=1000.0,
        )
        timing.append(
            config="Athens-32",
            time=2.0,
            e2e_time=2.5,
            pin_dyn_inst_count=2000.0,
        )
        self.assertEqual(len(timing.measurement_dict), 2)
        self.assertEqual(
            timing.measurement_dict["Athens-16"]["In Application Time"], 1.0
        )
        self.assertEqual(
            timing.measurement_dict["Athens-32"]["In Application Time"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
