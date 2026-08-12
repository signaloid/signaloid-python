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

import filecmp
import io
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from signaloid.benchmarking.automation.merge_tracing_dbs import (
    main,
    merge_tracing_dbs,
)


def _make_source_db(path: str, execution_data: str, fk_value: int) -> None:
    """Create a per-config tracing DB with the expected schema.

    Mirrors the production tracing schema as used by `merge_tracing_dbs`:
    one auto-incrementing `Emulator_Execution_Info` table with a single
    row at `Execution_ID = 1`, one FK-bearing table referencing it via
    `Execution_Info_Table_ID`, and one lookup table without the FK
    (exercises the `INSERT OR IGNORE` branch).
    """
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE Emulator_Execution_Info ("
            "Execution_ID INTEGER PRIMARY KEY, "
            "some_data TEXT)"
        )
        conn.execute(
            "INSERT INTO Emulator_Execution_Info (some_data) VALUES (?)",
            (execution_data,),
        )
        conn.execute(
            "CREATE TABLE Printed_ValueIds ("
            "Execution_Info_Table_ID INTEGER, "
            "value REAL)"
        )
        conn.execute(
            "INSERT INTO Printed_ValueIds VALUES (?, ?)",
            (fk_value, 1.5),
        )
        conn.execute("CREATE TABLE Lookup (name TEXT PRIMARY KEY, value INTEGER)")
        conn.execute("INSERT INTO Lookup VALUES ('alpha', 1)")
        conn.commit()


def _read_all_rows(path: str, table: str) -> list[tuple]:
    """Return every row of *table* in declaration order."""
    with sqlite3.connect(path) as conn:
        cursor = conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
        return cursor.fetchall()


class TestMergeTracingDbs(unittest.TestCase):
    """Coverage for merge_tracing_dbs helper and CLI entry point."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_single_source_is_copied_to_target(self) -> None:
        """First source DB is byte-copied into a non-existent target.

        After processing, the source is removed (mirrors bash `rm -f`).
        """
        src = str(self.tmp_path / "src.db")
        target = str(self.tmp_path / "target.db")
        _make_source_db(src, "run1", fk_value=1)
        src_copy = str(self.tmp_path / "src_snapshot.db")
        shutil.copy(src, src_copy)

        merge_tracing_dbs(target_db=target, source_dbs=[src])

        self.assertTrue(filecmp.cmp(target, src_copy, shallow=False))
        self.assertFalse(
            os.path.exists(src), "source DB should be removed after merging"
        )

    def test_two_sources_remap_foreign_keys(self) -> None:
        """Second source DB's FK column gets offset by the target's old_max."""
        src_a = str(self.tmp_path / "a.db")
        src_b = str(self.tmp_path / "b.db")
        target = str(self.tmp_path / "target.db")
        _make_source_db(src_a, "run_a", fk_value=1)
        _make_source_db(src_b, "run_b", fk_value=1)

        merge_tracing_dbs(target_db=target, source_dbs=[src_a, src_b])

        exec_rows = _read_all_rows(target, "Emulator_Execution_Info")
        self.assertEqual(exec_rows, [(1, "run_a"), (2, "run_b")])

        fk_rows = _read_all_rows(target, "Printed_ValueIds")
        # Source A's FK=1 was copied (no remap on first source).
        # Source B's FK=1 was remapped to 1 + old_max (1) = 2.
        self.assertEqual(fk_rows, [(1, 1.5), (2, 1.5)])

    def test_missing_source_emits_warning_and_continues(self) -> None:
        """A non-existent source DB prints a stderr warning and is skipped.
        Other sources still merge."""
        missing = str(self.tmp_path / "missing.db")
        src = str(self.tmp_path / "src.db")
        target = str(self.tmp_path / "target.db")
        _make_source_db(src, "run_present", fk_value=1)

        stderr_buf: io.StringIO = io.StringIO()
        with patch.object(sys, "stderr", stderr_buf):
            merge_tracing_dbs(target_db=target, source_dbs=[missing, src])

        self.assertIn("missing.db not found", stderr_buf.getvalue())
        exec_rows = _read_all_rows(target, "Emulator_Execution_Info")
        self.assertEqual(exec_rows, [(1, "run_present")])

    def test_insert_or_ignore_table_is_idempotent(self) -> None:
        """A table without `Execution_Info_Table_ID` uses INSERT OR IGNORE,
        so re-merging the same lookup row is a no-op."""
        src_a = str(self.tmp_path / "a.db")
        src_b = str(self.tmp_path / "b.db")
        target = str(self.tmp_path / "target.db")
        _make_source_db(src_a, "run_a", fk_value=1)
        _make_source_db(src_b, "run_b", fk_value=1)

        merge_tracing_dbs(target_db=target, source_dbs=[src_a, src_b])

        # Both sources had `Lookup` row ('alpha', 1). INSERT OR IGNORE
        # collapses to one row in the target.
        lookup_rows = _read_all_rows(target, "Lookup")
        self.assertEqual(lookup_rows, [("alpha", 1)])

    def test_main_invokes_merge_via_argv(self) -> None:
        """`main()` parses argv into target + source list and merges."""
        src = str(self.tmp_path / "src.db")
        target = str(self.tmp_path / "target.db")
        _make_source_db(src, "run_main", fk_value=1)

        with patch.object(sys, "argv", ["merge_tracing_dbs", target, src]):
            main()

        exec_rows = _read_all_rows(target, "Emulator_Execution_Info")
        self.assertEqual(exec_rows, [(1, "run_main")])

    def test_main_rejects_missing_source_list(self) -> None:
        """`main()` requires at least one source DB. Argparse raises
        SystemExit when only the target positional is given."""
        target = str(self.tmp_path / "target.db")

        with patch.object(sys, "argv", ["merge_tracing_dbs", target]):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
