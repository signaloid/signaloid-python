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

import sqlite3
import tempfile
import unittest
from pathlib import Path

from signaloid.benchmarking.automation.database_generator import (
    generate_analytic_ground_truth_database,
)
from signaloid.benchmarking.types import BenchmarkingVariable
from signaloid.benchmarking.config import VariableTypes


def _make_variable(name: str) -> BenchmarkingVariable:
    """
    Create a BenchmarkingVariable fixture for use in ground-truth tests.

    Args:
        name: The variable name, matched against CSV rows.

    Returns:
        A BenchmarkingVariable of type DISTRIBUTION with default metadata.
    """
    return BenchmarkingVariable(
        name=name,
        description=name,
        value_id=f"id_{name}",
        program="main",
        path="",
        line_number="1",
        type=VariableTypes.DISTRIBUTION,
    )


def _write_ground_truth_script(
    script_path: Path,
    rows: list[tuple[str, str, str]],
) -> None:
    """
    Write a Python helper script that writes a fixed CSV when invoked.

    The script accepts two positional arguments (<count> <csv_path>) to
    match the invocation pattern used by generate_analytic_ground_truth_database,
    but ignores <count> and always writes the fixed rows.

    Args:
        script_path: Destination path for the generated script.
        rows: List of (name, position, weight) tuples to write as CSV rows.
    """
    row_literals = repr(rows)
    script_path.write_text(
        "import csv, sys\n"
        "rows = " + row_literals + "\n"
        "csv_path = sys.argv[2]\n"
        "with open(csv_path, 'w', newline='') as f:\n"
        "    writer = csv.writer(f)\n"
        "    writer.writerows(rows)\n",
        encoding="utf-8",
    )


class TestAnalyticGroundTruthDatabase(unittest.TestCase):
    """Tests for generate_analytic_ground_truth_database."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_happy_path_two_variables(self) -> None:
        """
        Happy path: two variables, three CSV rows each, correct DB rows produced.

        The resulting WeightedSamples table must contain exactly the expected
        (ValueId, Position, Weight) tuples for each variable.
        """
        app_dir = self.tmp_path / "app"
        src_dir = app_dir / "src"
        src_dir.mkdir(parents=True)

        script_path = self.tmp_path / "ground_truth.py"
        rows = [
            ("alpha", "1.0", "0.5"),
            ("alpha", "2.0", "0.3"),
            ("alpha", "3.0", "0.2"),
            ("beta", "10.0", "0.6"),
            ("beta", "20.0", "0.4"),
            ("beta", "30.0", "0.0"),
        ]
        _write_ground_truth_script(script_path, rows)

        var_alpha = _make_variable("alpha")
        var_beta = _make_variable("beta")
        db_path = str(self.tmp_path / "ground_truth.db")

        generate_analytic_ground_truth_database(
            benchmarking_variables=[var_alpha, var_beta],
            path_to_application=str(app_dir),
            path_to_ground_truth_file=str(script_path),
            ground_truth_size=3,
            ground_truth_db_path=db_path,
        )

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT ValueId, Position, Weight FROM WeightedSamples"
            " WHERE ValueId = ? ORDER BY Position",
            ("id_alpha",),
        )
        alpha_rows = cursor.fetchall()
        self.assertEqual(len(alpha_rows), 3)
        self.assertEqual(alpha_rows[0], ("id_alpha", 1.0, 0.5))
        self.assertEqual(alpha_rows[1], ("id_alpha", 2.0, 0.3))
        self.assertEqual(alpha_rows[2], ("id_alpha", 3.0, 0.2))

        cursor.execute(
            "SELECT ValueId, Position, Weight FROM WeightedSamples"
            " WHERE ValueId = ? ORDER BY Position",
            ("id_beta",),
        )
        beta_rows = cursor.fetchall()
        self.assertEqual(len(beta_rows), 3)
        self.assertEqual(beta_rows[0], ("id_beta", 10.0, 0.6))
        self.assertEqual(beta_rows[1], ("id_beta", 20.0, 0.4))
        self.assertEqual(beta_rows[2], ("id_beta", 30.0, 0.0))

        conn.close()

    def test_pre_existing_values_are_cleared(self) -> None:
        """
        Pre-existing values and weights on each variable are cleared before
        being repopulated from the CSV.

        This verifies that variable.empty_values() is called first so that
        stale data from a previous pipeline stage does not accumulate.
        """
        app_dir = self.tmp_path / "app"
        src_dir = app_dir / "src"
        src_dir.mkdir(parents=True)

        script_path = self.tmp_path / "ground_truth.py"
        rows = [("gamma", "5.0", "1.0")]
        _write_ground_truth_script(script_path, rows)

        var_gamma = _make_variable("gamma")
        # Seed with dummy pre-existing data that must be erased.
        var_gamma.values = [999.0, 888.0]  # type: ignore[attr-defined]
        var_gamma.weights = [0.9, 0.1]  # type: ignore[attr-defined]

        db_path = str(self.tmp_path / "ground_truth.db")

        generate_analytic_ground_truth_database(
            benchmarking_variables=[var_gamma],
            path_to_application=str(app_dir),
            path_to_ground_truth_file=str(script_path),
            ground_truth_size=1,
            ground_truth_db_path=db_path,
        )

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Position FROM WeightedSamples WHERE ValueId = ?",
            ("id_gamma",),
        )
        rows_in_db = cursor.fetchall()
        conn.close()

        # Only the CSV row should appear. Pre-existing dummy values must be gone.
        self.assertEqual(len(rows_in_db), 1)
        self.assertEqual(rows_in_db[0], (5.0,))

    def test_subprocess_error_raises_runtime_error(self) -> None:
        """
        A ground-truth script that exits with non-zero must raise RuntimeError
        with a message that includes the exit code.
        """
        app_dir = self.tmp_path / "app"
        src_dir = app_dir / "src"
        src_dir.mkdir(parents=True)

        failing_script = self.tmp_path / "failing_gt.py"
        failing_script.write_text(
            "import sys\n"
            "print('something went wrong', file=sys.stderr)\n"
            "sys.exit(42)\n",
            encoding="utf-8",
        )

        var_delta = _make_variable("delta")
        db_path = str(self.tmp_path / "ground_truth.db")

        with self.assertRaises(RuntimeError) as cm:
            generate_analytic_ground_truth_database(
                benchmarking_variables=[var_delta],
                path_to_application=str(app_dir),
                path_to_ground_truth_file=str(failing_script),
                ground_truth_size=1,
                ground_truth_db_path=db_path,
            )

        error_message = str(cm.exception)
        self.assertIn("42", error_message)
        self.assertIn("Ground truth generation failed", error_message)


if __name__ == "__main__":
    unittest.main()
