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

import os
import tempfile
import unittest
from unittest.mock import patch

from signaloid.benchmarking.automation.benchmark import Benchmark


class TestOutputDirectoryPaths(unittest.TestCase):
    """Output directory paths derive from the process CWD at construction."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = tmp_dir.name
        with patch.object(os, "getcwd", return_value=self.tmp_path):
            self.benchmark = Benchmark(
                path_to_application="/tmp/fake-app",
                path_to_uxhw_sdk="/tmp/fake-uxhw-sdk",
            )

    def test_results_dir_is_under_cwd(self) -> None:
        self.assertEqual(
            self.benchmark.results_dir, os.path.join(self.tmp_path, "results")
        )

    def test_logs_dir_is_under_cwd(self) -> None:
        self.assertEqual(self.benchmark.logs_dir, os.path.join(self.tmp_path, "logs"))

    def test_plots_dir_is_under_results(self) -> None:
        self.assertEqual(
            self.benchmark.plots_dir,
            os.path.join(self.tmp_path, "results", "plots"),
        )

    def test_output_data_file_is_under_results(self) -> None:
        self.assertEqual(
            self.benchmark.output_data_file,
            os.path.join(self.benchmark.results_dir, "output_data.csv"),
        )

    def test_asymptotic_dist_file_is_under_results(self) -> None:
        self.assertEqual(
            self.benchmark.asymptotic_dist_file,
            os.path.join(self.benchmark.results_dir, "asymptotic_distances.csv"),
        )

    def test_uxhw_distance_file_is_under_results(self) -> None:
        self.assertEqual(
            self.benchmark.uxhw_distance_file,
            os.path.join(self.benchmark.results_dir, "uxhw_distances.csv"),
        )


if __name__ == "__main__":
    unittest.main()
