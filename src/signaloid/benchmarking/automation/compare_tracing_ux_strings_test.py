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
from collections.abc import Mapping, Sequence
from pathlib import Path

from signaloid.benchmarking.automation.compare_tracing_ux_strings import (
    compare_ux_strings,
    main,
)

_TABLE = "TracingTable"


def _make_tracing_db(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    """
    Write a minimal tracing DB with the columns the comparator reads.

    Each entry in *rows* becomes one ``TracingTable`` write plus its
    ``Emulator_Execution_Info`` row, sharing ``Execution_ID = 1`` so all
    writes belong to one emulator configuration. Rows are inserted in list
    order, so a later row overwrites an earlier one with the same identity
    (exercising the last-write-wins grouping).
    """
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE Emulator_Execution_Info ("
            "Execution_ID INTEGER PRIMARY KEY AUTOINCREMENT, "
            "UR_Type TEXT, UR_Order INTEGER, UR_Order_CoreLibrary INTEGER, "
            "CorrelationTracking_Status TEXT)"
        )
        conn.execute(
            "INSERT INTO Emulator_Execution_Info "
            "(Execution_ID, UR_Type, UR_Order, UR_Order_CoreLibrary, "
            "CorrelationTracking_Status) VALUES (1, 'Athens', 64, 64, 'OFF')"
        )
        conn.execute(
            f'CREATE TABLE "{_TABLE}" ('
            "Expression_DeclarationFileName TEXT, "
            "Expression_Subprogram TEXT, "
            "Expression_Name TEXT, "
            "Expression_DeclarationLineNumber INTEGER, "
            "Execution_Info_Table_ID INTEGER, "
            "Dist_Value TEXT)"
        )
        for row in rows:
            conn.execute(
                f'INSERT INTO "{_TABLE}" '
                "(Expression_DeclarationFileName, Expression_Subprogram, "
                "Expression_Name, Expression_DeclarationLineNumber, "
                "Execution_Info_Table_ID, Dist_Value) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (
                    row.get("file", "main.c"),
                    row.get("subprogram", "main"),
                    row["name"],
                    row.get("line", 10),
                    row["dist_value"],
                ),
            )
        conn.commit()


class TestCompareUxStrings(unittest.TestCase):
    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp = Path(tmp_dir.name)

    def _db(self, name: str, rows: Sequence[Mapping[str, object]]) -> str:
        path = str(self.tmp / name)
        _make_tracing_db(path, rows)
        return path

    def test_identical_ux_strings_match(self) -> None:
        rows = [
            {"name": "a", "dist_value": "Ux04ffff"},
            {"name": "b", "dist_value": "Ux04aaaa"},
        ]
        result = compare_ux_strings(
            self._db("o0.db", rows), self._db("o2.db", rows), _TABLE
        )
        self.assertTrue(result.is_identical)
        self.assertEqual(result.matched, 2)
        self.assertEqual(result.mismatches, [])

    def test_differing_ux_string_is_reported(self) -> None:
        o0 = self._db("o0.db", [{"name": "a", "dist_value": "Ux04ffff"}])
        o2 = self._db("o2.db", [{"name": "a", "dist_value": "Ux04fffe"}])
        result = compare_ux_strings(o0, o2, _TABLE)
        self.assertFalse(result.is_identical)
        self.assertEqual(result.matched, 0)
        self.assertEqual(len(result.mismatches), 1)
        _key, baseline_value, candidate_value = result.mismatches[0]
        self.assertEqual(baseline_value, "Ux04ffff")
        self.assertEqual(candidate_value, "Ux04fffe")

    def test_last_write_wins_per_expression(self) -> None:
        # Both DBs' final write for `a` is the same, even though an earlier
        # write differs. The comparison must use the last write only.
        o0 = self._db(
            "o0.db",
            [
                {"name": "a", "dist_value": "Ux04early"},
                {"name": "a", "dist_value": "Ux04final"},
            ],
        )
        o2 = self._db(
            "o2.db",
            [{"name": "a", "dist_value": "Ux04final"}],
        )
        result = compare_ux_strings(o0, o2, _TABLE)
        self.assertTrue(result.is_identical)
        self.assertEqual(result.matched, 1)

    def test_expression_only_in_one_build(self) -> None:
        o0 = self._db(
            "o0.db",
            [
                {"name": "a", "dist_value": "Ux04aaaa"},
                {"name": "b", "dist_value": "Ux04bbbb"},
            ],
        )
        o2 = self._db("o2.db", [{"name": "a", "dist_value": "Ux04aaaa"}])
        result = compare_ux_strings(o0, o2, _TABLE)
        self.assertFalse(result.is_identical)
        self.assertEqual(result.matched, 1)
        self.assertEqual(len(result.only_in_baseline), 1)
        self.assertEqual(result.only_in_baseline[0][2], "b")  # Expression_Name
        self.assertEqual(result.only_in_candidate, [])

    def test_main_returns_zero_when_identical(self) -> None:
        rows = [{"name": "a", "dist_value": "Ux04aaaa"}]
        argv = [self._db("o0.db", rows), self._db("o2.db", rows), "--table", _TABLE]
        self.assertEqual(_run_main(argv), 0)

    def test_main_returns_one_when_differing(self) -> None:
        o0 = self._db("o0.db", [{"name": "a", "dist_value": "Ux04aaaa"}])
        o2 = self._db("o2.db", [{"name": "a", "dist_value": "Ux04bbbb"}])
        self.assertEqual(_run_main([o0, o2, "--table", _TABLE]), 1)

    def test_main_returns_two_on_missing_db(self) -> None:
        o0 = self._db("o0.db", [{"name": "a", "dist_value": "Ux04aaaa"}])
        missing = str(self.tmp / "does-not-exist.db")
        self.assertEqual(_run_main([o0, missing, "--table", _TABLE]), 2)


def _run_main(argv: list[str]) -> int:
    """Invoke ``main`` with a patched ``sys.argv`` and return its exit code."""
    from unittest.mock import patch

    with patch("sys.argv", ["compare_tracing_ux_strings", *argv]):
        return main()


if __name__ == "__main__":
    unittest.main()
