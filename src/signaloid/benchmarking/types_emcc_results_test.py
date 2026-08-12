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

from signaloid.benchmarking.types import EmccResults


class TestEmccResults(unittest.TestCase):
    """EmccResults defaults, mutation, and per-instance isolation."""

    def test_emcc_results_defaults(self) -> None:
        """Confirm EmccResults initialises with empty lists for both fields."""
        results = EmccResults()
        self.assertEqual(results.equiv_mc_list, [])
        self.assertEqual(results.emcc_data, [])

    def test_emcc_results_equiv_mc_list_mutation(self) -> None:
        """Confirm equiv_mc_list can be populated and read back."""
        results = EmccResults()
        results.equiv_mc_list = [64, 128, 256]
        self.assertEqual(results.equiv_mc_list, [64, 128, 256])

    def test_emcc_results_emcc_data_mutation(self) -> None:
        """Confirm emcc_data can be populated with dicts and read back."""
        results = EmccResults()
        record = {"EMCC": 128, "EMCC_Predicted": 130}
        results.emcc_data.append(record)
        self.assertEqual(len(results.emcc_data), 1)
        self.assertEqual(results.emcc_data[0]["EMCC"], 128)

    def test_emcc_results_independent_instances(self) -> None:
        """Confirm two EmccResults instances do not share list state."""
        results_a = EmccResults()
        results_b = EmccResults()
        results_a.equiv_mc_list.append(64)
        results_a.emcc_data.append({"EMCC": 64})
        self.assertEqual(results_b.equiv_mc_list, [])
        self.assertEqual(results_b.emcc_data, [])


if __name__ == "__main__":
    unittest.main()
