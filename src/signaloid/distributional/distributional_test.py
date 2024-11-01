#   Copyright (c) 2024, Signaloid.
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
import csv
from .distributional import DistributionalValue
from typing import Optional, List, Tuple


def read_string_bytes_pairs_from_csv(
    csv_filename: str
) -> Optional[List[Tuple[str, bytes]]]:
    """
    Reads pairs of Ux strings and Ux bytes(in hex format) from a csv file

    Args:
        csv_filename: The input csv file path
    Returns:
        pairs: list of (string_value, bytearray_value) tuples
    """
    pairs = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                string_value = row[0]
                bytearray_value = bytes.fromhex(row[1])
                pairs.append((string_value, bytearray_value))
    return pairs


class TestUxParsing(unittest.TestCase):
    def test_parse_ux_strings_values(
        self,
        input_filename="src/signaloid/distributional/test_ux_value_pairs.csv"
    ) -> None:
        """
        Test parsing Ux string values and converting them to Ux bytes
        """

        ux_pairs = read_string_bytes_pairs_from_csv(input_filename)
        self.assertIsNotNone(ux_pairs)

        if ux_pairs is not None:
            for pair in ux_pairs:
                distValueFromUxString = DistributionalValue.parse(pair[0])
                self.assertIsNotNone(distValueFromUxString)
                if distValueFromUxString is not None:
                    self.assertEqual(distValueFromUxString.bytes(), pair[1])

    def test_parse_ux_bytes_values(
        self,
        input_filename="src/signaloid/distributional/test_ux_value_pairs.csv"
    ) -> None:
        """
        Test parsing Ux bytes and converting them to Ux strings
        """

        ux_pairs = read_string_bytes_pairs_from_csv(input_filename)
        self.assertIsNotNone(ux_pairs)

        if ux_pairs is not None:
            for pair in ux_pairs:
                distValueFromBytes = DistributionalValue.parse(pair[1])
                self.assertEqual(str(distValueFromBytes), pair[0])


if __name__ == "__main__":
    unittest.main()
