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

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from signaloid.benchmarking.automation.parse_cpu_time import (
    _extract_cpu_time,
    main,
    _sum_cpu_times,
    read_cpu_time,
)

_VALID_STDOUT = (
    "Some preamble.\n" "Doing things...\n" "CPU time used: 0.001853 seconds\n" "Done.\n"
)

_MALFORMED_STDOUT = "Some preamble.\nNo CPU time line here.\n"


class TestParseCpuTime(unittest.TestCase):
    """Coverage for parse_cpu_time helper functions and CLI entry point."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_extract_cpu_time_parses_seconds_value(self) -> None:
        """The first ``CPU time used:`` float is returned."""
        self.assertEqual(_extract_cpu_time(_VALID_STDOUT), 0.001853)

    def test_extract_cpu_time_handles_integer_seconds(self) -> None:
        """A bare integer (no decimal) is accepted."""
        self.assertEqual(_extract_cpu_time("CPU time used: 5 seconds\n"), 5.0)

    def test_extract_cpu_time_raises_on_missing_line(self) -> None:
        """A file with no ``CPU time used:`` line raises ``ValueError``."""
        with self.assertRaisesRegex(ValueError, "no 'CPU time used:"):
            _extract_cpu_time(_MALFORMED_STDOUT)

    def test_extract_cpu_time_accepts_no_seconds_suffix(self) -> None:
        """``CPU time used: N`` with no trailing token still parses."""
        uxhw_format = "Some preamble.\nCPU time used: 0.000198\nDone.\n"
        self.assertEqual(_extract_cpu_time(uxhw_format), 0.000198)

    def test_extract_cpu_time_handles_ux_distributional_value(self) -> None:
        """The UxHw binary emits a Ux-encoded distributional value
        directly against the decimal time, e.g.
        ``CPU time used: 0.000163Ux<hex> seconds``. The regex must capture
        the leading decimal float and stop at the ``U`` rather than
        requiring whitespace-separated ``seconds``. Regression for the
        Step 10 timing-loop crash."""
        uxhw_with_ux = (
            "CPU time used: 0.000163Ux0400000000000000003F255D5F56A7AC82"
            "000000013F255D5F56A7AC828000000000000000 seconds\n"
        )
        self.assertEqual(_extract_cpu_time(uxhw_with_ux), 0.000163)

    def test_sum_cpu_times_adds_every_match_in_content(self) -> None:
        """``_sum_cpu_times`` finds every ``CPU time used:`` line, not just the
        first — preserves the original bash ``grep | sum`` semantics for
        multi-section stdouts."""
        multi = (
            "Run 1\nCPU time used: 0.1 seconds\n"
            "Run 2\nCPU time used: 0.2 seconds\n"
            "Run 3\nCPU time used: 0.3 seconds\n"
        )
        self.assertAlmostEqual(_sum_cpu_times(multi), 0.6)

    def test_sum_cpu_times_returns_zero_for_no_matches(self) -> None:
        """No ``CPU time used:`` lines → 0, not an exception."""
        self.assertEqual(_sum_cpu_times("Nothing relevant here.\n"), 0.0)

    def test_read_cpu_time_reads_from_file(self) -> None:
        """``read_cpu_time`` wraps ``_extract_cpu_time`` over a file."""
        stdout_path = self.tmp_path / "stdout.txt"
        stdout_path.write_text(_VALID_STDOUT)

        self.assertEqual(read_cpu_time(str(stdout_path)), 0.001853)

    def test_read_cpu_time_raises_on_missing_file(self) -> None:
        """Opening a non-existent file raises ``OSError`` (FileNotFoundError)."""
        with self.assertRaises(OSError):
            read_cpu_time(str(self.tmp_path / "missing.txt"))

    def test_read_cpu_time_survives_non_utf8_bytes(self) -> None:
        """Non-UTF-8 bytes in the stdout don't crash the read.

        Mirrors the ``grep -a`` flag the bash sites used to force-text
        interpretation even when the binary's stdout contains stray
        non-text bytes.
        """
        path = self.tmp_path / "binary.txt"
        payload = b"junk\xff\xfe preamble\nCPU time used: 0.42 seconds\n"
        path.write_bytes(payload)

        self.assertEqual(read_cpu_time(str(path)), 0.42)

    def test_main_single_path_prints_value(self) -> None:
        """``main`` with a single positional path prints the seconds value."""
        stdout_path = self.tmp_path / "stdout.txt"
        stdout_path.write_text(_VALID_STDOUT)

        buf: io.StringIO = io.StringIO()
        with patch.object(sys, "argv", ["parse_cpu_time", str(stdout_path)]):
            with redirect_stdout(buf):
                main()

        self.assertEqual(float(buf.getvalue().strip()), 0.001853)

    def test_main_sum_adds_across_paths(self) -> None:
        """``--sum`` adds the values across every positional path."""
        a = self.tmp_path / "a.txt"
        b = self.tmp_path / "b.txt"
        a.write_text("CPU time used: 0.1 seconds\n")
        b.write_text("CPU time used: 0.25 seconds\n")

        buf: io.StringIO = io.StringIO()
        with patch.object(sys, "argv", ["parse_cpu_time", "--sum", str(a), str(b)]):
            with redirect_stdout(buf):
                main()

        self.assertAlmostEqual(float(buf.getvalue().strip()), 0.35)

    def test_main_sum_adds_multiple_matches_within_a_file(self) -> None:
        """``--sum`` over a single multi-section stdout sums every matching
        line within that file (mirrors the original L506 ``grep ... | sum``)."""
        path = self.tmp_path / "multi.txt"
        path.write_text(
            "CPU time used: 0.1 seconds\n"
            "CPU time used: 0.2 seconds\n"
            "CPU time used: 0.3 seconds\n"
        )

        buf: io.StringIO = io.StringIO()
        with patch.object(sys, "argv", ["parse_cpu_time", "--sum", str(path)]):
            with redirect_stdout(buf):
                main()

        self.assertAlmostEqual(float(buf.getvalue().strip()), 0.6)

    def test_main_rejects_multiple_paths_without_sum(self) -> None:
        """Two positional paths without ``--sum`` is a usage error."""
        a = self.tmp_path / "a.txt"
        b = self.tmp_path / "b.txt"
        a.write_text(_VALID_STDOUT)
        b.write_text(_VALID_STDOUT)

        with patch.object(sys, "argv", ["parse_cpu_time", str(a), str(b)]):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
