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

"""Unit tests for the SQLite sample-database writers."""

import os
import sqlite3
import tempfile
import unittest

from signaloid.benchmarking.automation.database_generator import (
    _mc_sample_rows,
    _weighted_sample_rows,
    generate_database_from_mc_samples,
    generate_database_from_weighted_samples,
)
from signaloid.benchmarking.config import VariableTypes
from signaloid.benchmarking.types import BenchmarkingVariable


def _make_var(var_type: str) -> BenchmarkingVariable:
    return BenchmarkingVariable(
        name="outputDistributions[0]",
        description="Variable 0",
        type=var_type,
        cla="-S 0",
        value_id="vid-1",
        program="main",
        path="main.c",
        line_number="42",
    )


# Metadata columns prefixing every data row, in their stored order.
_META = ("vid-1", "outputDistributions[0]", "main", "main.c", "42")


class TestSampleRowBuilders(unittest.TestCase):
    """The row builders emit the exact value-column tuples per variable."""

    def test_weighted_distribution_rows(self) -> None:
        var = _make_var(VariableTypes.DISTRIBUTION)
        var.distribution_samples.set_weighted_values([1.0, 2.0], [0.25, 0.75])
        # (Position, Weight, MonteCarlo_Count=1)
        self.assertEqual(_weighted_sample_rows(var), [(1.0, 0.25, 1), (2.0, 0.75, 1)])

    def test_weighted_scalar_rows(self) -> None:
        var = _make_var(VariableTypes.SCALAR)
        var.distribution_samples.scalar_output_dict = {10: [5.0, 6.0]}
        # (value, Weight=1.0, MonteCarlo_Count=mc_count)
        self.assertEqual(_weighted_sample_rows(var), [(5.0, 1.0, 10), (6.0, 1.0, 10)])

    def test_mc_distribution_rows(self) -> None:
        var = _make_var(VariableTypes.DISTRIBUTION)
        var.distribution_samples.set_values([1.0, 2.0])
        # (Assignment_Index=1, Particle_Value, MonteCarlo_Count=1)
        self.assertEqual(_mc_sample_rows(var), [(1, 1.0, 1), (1, 2.0, 1)])

    def test_mc_scalar_rows(self) -> None:
        var = _make_var(VariableTypes.SCALAR)
        var.distribution_samples.scalar_output_dict = {10: [5.0, 6.0]}
        # (Assignment_Index=enumerate, Particle_Value, MonteCarlo_Count=mc_count)
        self.assertEqual(_mc_sample_rows(var), [(0, 5.0, 10), (1, 6.0, 10)])

    def test_unsupported_type_yields_no_rows(self) -> None:
        var = _make_var("NotARealVariableType")
        self.assertEqual(_weighted_sample_rows(var), [])
        self.assertEqual(_mc_sample_rows(var), [])


class TestGenerateWeightedSamplesDatabase(unittest.TestCase):
    def test_roundtrip_columns_and_printed_value_ids(self) -> None:
        var = _make_var(VariableTypes.DISTRIBUTION)
        var.distribution_samples.set_weighted_values([1.0, 2.0], [0.25, 0.75])
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "weighted.db")
            generate_database_from_weighted_samples(db_path, [var])
            with sqlite3.connect(db_path) as con:
                rows = con.execute(
                    "SELECT ValueId, Expression_Name, Expression_Subprogram, "
                    "Expression_DeclarationFileName, "
                    "Expression_DeclarationLineNumber, "
                    "Position, Weight, MonteCarlo_Count "
                    "FROM WeightedSamples ORDER BY Id"
                ).fetchall()
                printed = con.execute(
                    "SELECT ValueId, SampleType FROM Printed_ValueIds"
                ).fetchall()
        self.assertEqual(
            rows,
            [_META + (1.0, 0.25, 1), _META + (2.0, 0.75, 1)],
        )
        self.assertEqual(printed, [("vid-1", "WeightedSamples")])


class TestGenerateMonteCarloDatabase(unittest.TestCase):
    def test_roundtrip_columns_and_printed_value_ids(self) -> None:
        var = _make_var(VariableTypes.SCALAR)
        var.distribution_samples.scalar_output_dict = {10: [5.0, 6.0]}
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "mc.db")
            generate_database_from_mc_samples(db_path, [var])
            with sqlite3.connect(db_path) as con:
                rows = con.execute(
                    "SELECT ValueId, Expression_Name, Expression_Subprogram, "
                    "Expression_DeclarationFileName, "
                    "Expression_DeclarationLineNumber, "
                    "Assignment_Index, Particle_Value, MonteCarlo_Count "
                    "FROM MonteCarlo ORDER BY MC_Id"
                ).fetchall()
                printed = con.execute(
                    "SELECT ValueId, SampleType FROM Printed_ValueIds"
                ).fetchall()
        self.assertEqual(
            rows,
            [_META + (0, 5.0, 10), _META + (1, 6.0, 10)],
        )
        self.assertEqual(printed, [("vid-1", "MonteCarlo")])


if __name__ == "__main__":
    unittest.main()
