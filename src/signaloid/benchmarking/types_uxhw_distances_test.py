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

from signaloid.benchmarking.types import (
    TaggedDistributionalValue,
    UxhwDistanceRecord,
    UxhwDistances,
)
from signaloid.distributional.distributional import DistributionalValue


class TestUxhwDistances(unittest.TestCase):
    """UxhwDistances and UxhwDistanceRecord carry per-config UxHw distances."""

    def test_uxhw_distances_defaults_empty_records(self) -> None:
        """A freshly-constructed instance has an empty ``records`` list."""
        distances = UxhwDistances()

        self.assertEqual(distances.records, [])

    def test_uxhw_distance_record_with_binned_distance(self) -> None:
        """A record constructed with all three fields keeps every value."""
        record = UxhwDistanceRecord(
            uxhw_conf="Athens-16",
            uxhw_distance=0.123,
            uxhw_binned_distance=0.234,
        )

        self.assertEqual(record.uxhw_conf, "Athens-16")
        self.assertEqual(record.uxhw_distance, 0.123)
        self.assertEqual(record.uxhw_binned_distance, 0.234)

    def test_uxhw_distance_record_default_binned_distance_is_none(self) -> None:
        """``uxhw_binned_distance`` defaults to ``None`` when omitted —
        some producers or CSV-loaded records may omit the binned distance."""
        record = UxhwDistanceRecord(
            uxhw_conf="Athens-16",
            uxhw_distance=0.5,
        )

        self.assertIsNone(record.uxhw_binned_distance)

    def test_uxhw_distance_record_accepts_non_string_conf(self) -> None:
        """``uxhw_conf`` accepts the ``TaggedDistributionalValue`` carrier stored by
        the producer pipeline (``analysis.compute_uxhw_distances``), in addition to
        the ``str`` form stored by the CSV loader path."""
        carrier = TaggedDistributionalValue(
            dv=DistributionalValue.from_samples([0.0, 1.0, 2.0])
        )
        record = UxhwDistanceRecord(
            uxhw_conf=carrier,
            uxhw_distance=0.0,
        )

        self.assertIs(record.uxhw_conf, carrier)

    def test_uxhw_distances_records_append(self) -> None:
        """Records can be appended and read back."""
        distances = UxhwDistances()
        distances.records.append(UxhwDistanceRecord(uxhw_conf="A", uxhw_distance=0.1))
        distances.records.append(
            UxhwDistanceRecord(
                uxhw_conf="B",
                uxhw_distance=0.2,
                uxhw_binned_distance=0.3,
            )
        )

        self.assertEqual(len(distances.records), 2)
        self.assertEqual(distances.records[0].uxhw_conf, "A")
        self.assertEqual(distances.records[1].uxhw_binned_distance, 0.3)

    def test_uxhw_distances_independent_instances(self) -> None:
        """Two UxhwDistances instances do not share their records list."""
        first = UxhwDistances()
        second = UxhwDistances()
        first.records.append(UxhwDistanceRecord(uxhw_conf="A", uxhw_distance=0.1))

        self.assertEqual(second.records, [])


if __name__ == "__main__":
    unittest.main()
