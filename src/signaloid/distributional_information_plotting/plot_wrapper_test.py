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
from .plot_wrapper import plot
from signaloid.distributional import DistributionalValue
from typing import Optional, List
import tempfile
import shutil
import os


def read_ux_strings_from_csv(
    csv_filename: str
) -> Optional[List[str]]:
    """
    Reads Ux strings from a csv file

    Args:
        csv_filename: The input csv file path
    Returns:
        ux_strings: list of Ux strings
    """
    ux_strings = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ux_strings.append(row[0])

    return ux_strings


class TestPlotting(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory even if the test fails
        shutil.rmtree(self.test_dir)

    def test_plot_wrapper(
        self,
        input_filename="src/signaloid/distributional/test_ux_value_pairs.csv"
    ) -> None:
        """
        Test parsing and plotting Ux Strings
        """

        # Read Ux strings from test data csv
        ux_strings = read_ux_strings_from_csv(input_filename)
        self.assertIsNotNone(ux_strings)

        if ux_strings is not None:
            # Run the save function for each filepath
            for i in range(len(ux_strings)):
                distValue = DistributionalValue.parse(ux_strings[i])
                self.assertIsNotNone(distValue)
                filepath = os.path.join(self.test_dir, f'file{i}.png')
                if distValue is not None:
                    plot(distValue, path=filepath, save=True)
                self.assertTrue(os.path.exists(filepath), f"File was not created: {filepath}")


if __name__ == "__main__":
    unittest.main()
