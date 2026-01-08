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

import csv
import os
import shutil
import tempfile
from typing import Optional
import unittest

from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import PlotData

from .plot_wrapper import plot


def read_ux_strings_from_csv(
    csv_filename: str
) -> Optional[list[str]]:
    """
    Reads Ux strings from a csv file

    Args:
        csv_filename: The input csv file path
    Returns:
        ux_strings: list of Ux strings
    """
    ux_strings: list[str] = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ux_strings.append(row[0])

    return ux_strings


class TestPlotting(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        # Remove the temporary directory even if the test fails
        shutil.rmtree(self.test_dir)

    def test_plot_wrapper(
        self,
        input_filename: str = "src/signaloid/distributional/test_ux_value_pairs.csv"
    ) -> None:
        """
        Test parsing and plotting Ux Strings
        """

        # Read Ux strings from test data csv
        ux_strings = read_ux_strings_from_csv(input_filename)
        self.assertIsNotNone(ux_strings)

        if ux_strings is None:
            return

        # Run the save function for each filepath
        for i, ux_string in enumerate(ux_strings):
            distValue = DistributionalValue.parse(ux_string)
            self.assertIsNotNone(distValue)
            if distValue is None:
                return

            plot_data = PlotData(distValue)
            self.assertIsNotNone(plot_data, "Failed to parse data")

            filepath = os.path.join(self.test_dir, f'file{i}.png')
            plot(plot_data, path=filepath, save=True)
            self.assertTrue(os.path.exists(filepath), f"File was not created: {filepath}")


if __name__ == "__main__":
    unittest.main()
