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

from pathlib import Path
from unittest.mock import MagicMock

from signaloid.benchmarking.automation.database_generator import (
    generate_uxhw_tracing_database,
)
from signaloid.benchmarking.types import BenchmarkingVariable


def _make_variables(n: int) -> list[BenchmarkingVariable]:
    """Create n dummy BenchmarkingVariable instances."""
    return [
        BenchmarkingVariable(
            name=f"outputVariables[{i}]",
            description=f"Variable {i}",
            cla=f"-S {i}",
        )
        for i in range(n)
    ]


class TestGenerateUxHwTracingDatabase(unittest.TestCase):
    """``generate_uxhw_tracing_database`` deletes any stale DB then
    invokes the timing script once per variable, in index order."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_deletes_existing_db_before_tracing(self) -> None:
        """Existing tracing DB should be removed before any tracing runs."""
        db_path = str(self.tmp_path / "tracing.db")
        with open(db_path, "w") as f:
            f.write("stale data")

        variables = _make_variables(2)
        removal_observed: list[bool] = []

        def assert_db_removed_before_run(**kwargs: object) -> None:
            removal_observed.append(not os.path.exists(db_path))

        mock_run = MagicMock(side_effect=assert_db_removed_before_run)
        generate_uxhw_tracing_database(
            benchmarking_variables=variables,
            tracing_db_path=db_path,
            run_timing_script=mock_run,
        )

        # The DB was deleted before the first invocation.
        self.assertTrue(all(removal_observed))
        self.assertEqual(mock_run.call_count, 2)
        self.assertFalse(os.path.exists(db_path))

    def test_calls_run_timing_script_per_variable(self) -> None:
        """run_timing_script should be called once per benchmarking variable
        with tracing=True and the correct variable_index."""
        variables = _make_variables(3)
        mock_run = MagicMock()

        generate_uxhw_tracing_database(
            benchmarking_variables=variables,
            tracing_db_path="/nonexistent/tracing.db",
            run_timing_script=mock_run,
        )

        self.assertEqual(mock_run.call_count, 3)
        for i, call_args in enumerate(mock_run.call_args_list):
            self.assertEqual(call_args.kwargs["variable_index"], i)
            self.assertIs(call_args.kwargs["tracing"], True)

    def test_single_variable(self) -> None:
        """With a single variable, run_timing_script is called exactly once."""
        variables = _make_variables(1)
        mock_run = MagicMock()

        generate_uxhw_tracing_database(
            benchmarking_variables=variables,
            tracing_db_path="/nonexistent/tracing.db",
            run_timing_script=mock_run,
        )

        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(mock_run.call_args.kwargs["variable_index"], 0)
        self.assertIs(mock_run.call_args.kwargs["tracing"], True)

    def test_no_variables_does_not_call_run_timing_script(self) -> None:
        """With no benchmarking variables, run_timing_script not called."""
        mock_run = MagicMock()

        generate_uxhw_tracing_database(
            benchmarking_variables=[],
            tracing_db_path="/nonexistent/tracing.db",
            run_timing_script=mock_run,
        )

        mock_run.assert_not_called()

    def test_no_error_when_db_does_not_exist(self) -> None:
        """Should not raise when the tracing DB does not exist yet."""
        db_path = str(self.tmp_path / "tracing.db")
        self.assertFalse(os.path.exists(db_path))

        mock_run = MagicMock()
        generate_uxhw_tracing_database(
            benchmarking_variables=_make_variables(1),
            tracing_db_path=db_path,
            run_timing_script=mock_run,
        )

        self.assertEqual(mock_run.call_count, 1)

    def test_invocation_order_is_sequential(self) -> None:
        """Variables must be traced in index order so that per-variable
        DBs merge into the final DB in a deterministic sequence."""
        variables = _make_variables(5)
        invocation_order: list[object] = []

        def record_call(**kwargs: object) -> None:
            invocation_order.append(kwargs["variable_index"])

        generate_uxhw_tracing_database(
            benchmarking_variables=variables,
            tracing_db_path="/nonexistent/tracing.db",
            run_timing_script=record_call,
        )

        self.assertEqual(invocation_order, [0, 1, 2, 3, 4])

    def test_run_timing_script_is_importable_from_database_generator(
        self,
    ) -> None:
        """Sanity: generate_uxhw_tracing_database must be importable from
        database_generator so orchestrators and tests bind to the right
        symbol."""
        from signaloid.benchmarking.automation import (
            database_generator as dg_module,
        )

        self.assertTrue(hasattr(dg_module, "generate_uxhw_tracing_database"))


if __name__ == "__main__":
    unittest.main()
