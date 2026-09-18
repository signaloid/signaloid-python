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
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from signaloid.benchmarking.automation.check_traced_values_printed import (
    find_unprinted_values,
    main,
)

_TABLE = "TracingTable"

_UX_A = "Ux0400000000000000AAAA"
_UX_B = "Ux0400000000000000BBBB"


def _make_tracing_db(path: str, values: Sequence[str]) -> None:
    """Write a tracing DB holding one traced row per entry in *values*."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            f'CREATE TABLE "{_TABLE}" ('
            "Expression_DeclarationFileName TEXT, "
            "Expression_Name TEXT, "
            "Expression_DeclarationLineNumber INTEGER, "
            "Dist_Value TEXT)"
        )
        for index, value in enumerate(values):
            conn.execute(
                f'INSERT INTO "{_TABLE}" VALUES (?, ?, ?, ?)',
                ("main.c", f"outputVariables[{index}]", 48, value),
            )
        conn.commit()


class TestCheckTracedValuesPrinted(unittest.TestCase):
    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp = Path(tmp_dir.name)

    def _db(self, name: str, values: Sequence[str]) -> str:
        path = str(self.tmp / name)
        _make_tracing_db(path, values)
        return path

    def _stdout(self, name: str, *ux_strings: str) -> str:
        path = self.tmp / name
        body = "".join(f"  value: 1.5{ux}\n" for ux in ux_strings)
        path.write_text(f"seed: 1024\n{body}", encoding="utf-8")
        return str(path)

    def test_every_recorded_value_printed(self) -> None:
        unprinted, recorded = find_unprinted_values(
            self._db("t.db", [_UX_A, _UX_B]),
            self._stdout("run.out", _UX_A, _UX_B),
            _TABLE,
        )
        self.assertEqual(unprinted, [])
        self.assertEqual(recorded, 2)

    def test_recorded_value_missing_from_stdout(self) -> None:
        unprinted, recorded = find_unprinted_values(
            self._db("t.db", [_UX_A, _UX_B]),
            self._stdout("run.out", _UX_A),
            _TABLE,
        )
        self.assertEqual(recorded, 2)
        self.assertEqual(len(unprinted), 1)
        value, identity = unprinted[0]
        self.assertEqual(value, _UX_B)
        self.assertEqual(identity, ("main.c", "outputVariables[1]", 48))

    def test_print_order_and_extras_do_not_matter(self) -> None:
        # stdout prints the recorded values in the other order, plus a value
        # that was never traced. The check is containment, not equality.
        unprinted, recorded = find_unprinted_values(
            self._db("t.db", [_UX_A]),
            self._stdout("run.out", _UX_B, _UX_A),
            _TABLE,
        )
        self.assertEqual(unprinted, [])
        self.assertEqual(recorded, 1)

    def test_repeated_recordings_count_once(self) -> None:
        # The same value written twice is one distinct value to cross-check.
        unprinted, recorded = find_unprinted_values(
            self._db("t.db", [_UX_A, _UX_A]),
            self._stdout("run.out", _UX_A),
            _TABLE,
        )
        self.assertEqual(unprinted, [])
        self.assertEqual(recorded, 1)

    def test_main_returns_zero_when_all_printed(self) -> None:
        argv = [self._db("t.db", [_UX_A]), self._stdout("run.out", _UX_A)]
        self.assertEqual(_run_main(argv), 0)

    def test_main_returns_one_when_value_unprinted(self) -> None:
        argv = [self._db("t.db", [_UX_A]), self._stdout("run.out", _UX_B)]
        self.assertEqual(_run_main(argv), 1)

    def test_main_returns_one_when_nothing_recorded(self) -> None:
        argv = [self._db("t.db", []), self._stdout("run.out", _UX_A)]
        self.assertEqual(_run_main(argv), 1)

    def test_main_returns_two_on_missing_db(self) -> None:
        missing = str(self.tmp / "does-not-exist.db")
        argv = [missing, self._stdout("run.out", _UX_A)]
        self.assertEqual(_run_main(argv), 2)


def _run_main(argv: list[str]) -> int:
    """Invoke ``main`` with a patched ``sys.argv`` and return its exit code."""
    with patch("sys.argv", ["check_traced_values_printed", *argv]):
        return main()


if __name__ == "__main__":
    unittest.main()
