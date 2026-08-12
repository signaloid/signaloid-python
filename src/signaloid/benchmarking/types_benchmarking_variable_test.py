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

from signaloid.benchmarking.config import VariableTypes
from signaloid.benchmarking.types import BenchmarkingVariable


class TestBenchmarkingVariable(unittest.TestCase):
    """Tests for BenchmarkingVariable construction and attribute defaults."""

    def test_benchmarking_variable_lives_at_new_module_path(self) -> None:
        """
        Confirm BenchmarkingVariable is importable from its new neutral module
        and that an instance can be constructed with expected defaults.
        """
        variable = BenchmarkingVariable(
            name="test_expr",
            description="A test expression",
        )
        self.assertIsInstance(variable, BenchmarkingVariable)
        self.assertEqual(variable.name, "test_expr")
        self.assertEqual(variable.description, "A test expression")
        self.assertEqual(variable.type, VariableTypes.DISTRIBUTION)
        self.assertEqual(variable.distribution_samples.values, [])
        self.assertEqual(variable.distribution_samples.weights, [])
        self.assertEqual(variable.timing_measurements.measurement_dict, {})
        self.assertEqual(variable.formatted_description, "a-test-expression")

    def test_benchmarking_variable_distribution_samples_setters(self) -> None:
        """
        Confirm BenchmarkingVariable.distribution_samples.set_values,
        set_weighted_values, and empty_values update sample data correctly.
        """
        variable = BenchmarkingVariable(
            name="x",
            description="x variable",
        )
        variable.distribution_samples.set_values([1.0, 2.0, 3.0])
        self.assertEqual(variable.distribution_samples.values, [1.0, 2.0, 3.0])
        self.assertEqual(variable.distribution_samples.weights, [])

        variable.distribution_samples.set_weighted_values([0.5, 1.5], [0.3, 0.7])
        self.assertEqual(variable.distribution_samples.values, [0.5, 1.5])
        self.assertEqual(variable.distribution_samples.weights, [0.3, 0.7])

        variable.distribution_samples.empty_values()
        self.assertEqual(variable.distribution_samples.values, [])
        self.assertEqual(variable.distribution_samples.weights, [])
        self.assertEqual(variable.distribution_samples.scalar_output_dict, {})

    def test_benchmarking_variable_timing_measurements_append(self) -> None:
        """
        Confirm BenchmarkingVariable.timing_measurements.append records all
        timing fields correctly.
        """
        variable = BenchmarkingVariable(
            name="y",
            description="y variable",
        )
        variable.timing_measurements.append(
            config="Athens-16",
            time=1.0,
            e2e_time=1.5,
            pin_dyn_inst_count=1000.0,
            db_time=0.1,
            db_dyn_inst_count=50.0,
        )
        self.assertIn("Athens-16", variable.timing_measurements.measurement_dict)
        record = variable.timing_measurements.measurement_dict["Athens-16"]
        self.assertEqual(record["In Application Time"], 1.0)
        self.assertEqual(record["End-to-End Time"], 1.5)
        self.assertEqual(record["PIN Dyn. Inst. Count"], 1000.0)
        self.assertEqual(record["Database Time"], 0.1)
        self.assertEqual(record["Database Dyn. Inst. Count"], 50.0)

    def test_benchmarking_variable_str(self) -> None:
        """
        Confirm __str__ returns a non-empty string containing the variable name.
        """
        variable = BenchmarkingVariable(
            name="z",
            description="z variable",
        )
        result = str(variable)
        self.assertIn("z", result)
        self.assertIn("BenchmarkingVariable", result)


if __name__ == "__main__":
    unittest.main()
