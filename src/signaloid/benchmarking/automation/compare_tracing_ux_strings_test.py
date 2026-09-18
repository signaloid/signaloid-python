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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from signaloid.benchmarking.automation.compare_tracing_ux_strings import (
    compare_ux_strings,
    main,
)

# Shortened stand-ins for real Ux strings, which run to several hundred hex
# characters. Only the `Ux` prefix and the hex body matter to the comparator.
_UX_A = "Ux0400000000000000AAAA"
_UX_B = "Ux0400000000000000BBBB"


def _stdout_with(*ux_strings: str) -> str:
    """
    Render application stdout that prints *ux_strings* among ordinary output.

    Mirrors the real shape: a printed uncertain value appears inline, directly
    after its decimal rendering, surrounded by plain text that the comparator
    must ignore.
    """
    lines = ["Core library random seed: 1024"]
    for index, ux_string in enumerate(ux_strings):
        lines.append(f"  output {index}: 3.14159{ux_string} units")
    lines.append("CPU time used: 0.123 seconds")
    return "\n".join(lines) + "\n"


class TestCompareUxStrings(unittest.TestCase):
    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp = Path(tmp_dir.name)

    def _stdout(self, name: str, *ux_strings: str) -> str:
        path = self.tmp / name
        path.write_text(_stdout_with(*ux_strings), encoding="utf-8")
        return str(path)

    def test_identical_ux_strings_match(self) -> None:
        result = compare_ux_strings(
            self._stdout("o0.out", _UX_A, _UX_B),
            self._stdout("o2.out", _UX_A, _UX_B),
        )
        self.assertTrue(result.is_identical)
        self.assertEqual(result.matched, 2)
        self.assertEqual(result.mismatches, [])

    def test_differing_ux_string_is_reported(self) -> None:
        result = compare_ux_strings(
            self._stdout("o0.out", _UX_A),
            self._stdout("o2.out", _UX_B),
        )
        self.assertFalse(result.is_identical)
        self.assertEqual(result.matched, 0)
        self.assertEqual(len(result.mismatches), 1)
        index, baseline_value, candidate_value = result.mismatches[0]
        self.assertEqual(index, 0)
        self.assertEqual(baseline_value, _UX_A)
        self.assertEqual(candidate_value, _UX_B)

    def test_print_order_is_significant(self) -> None:
        # Same multiset of values, printed in the other order. The comparison
        # is positional, so this must be reported rather than matched.
        result = compare_ux_strings(
            self._stdout("o0.out", _UX_A, _UX_B),
            self._stdout("o2.out", _UX_B, _UX_A),
        )
        self.assertFalse(result.is_identical)
        self.assertEqual(len(result.mismatches), 2)

    def test_surrounding_output_is_ignored(self) -> None:
        # The two runs write different stats-DB paths and timings, which must
        # not register as a difference.
        baseline = self.tmp / "o0.out"
        baseline.write_text(
            f"ExecutionStatistics DB Name is /tmp/a-O0.db\nv: 1.0{_UX_A}\n",
            encoding="utf-8",
        )
        candidate = self.tmp / "o2.out"
        candidate.write_text(
            f"ExecutionStatistics DB Name is /tmp/b-O2.db\nv: 1.0{_UX_A}\n",
            encoding="utf-8",
        )
        result = compare_ux_strings(str(baseline), str(candidate))
        self.assertTrue(result.is_identical)
        self.assertEqual(result.matched, 1)

    def test_extra_value_printed_by_one_build(self) -> None:
        result = compare_ux_strings(
            self._stdout("o0.out", _UX_A, _UX_B),
            self._stdout("o2.out", _UX_A),
        )
        self.assertFalse(result.is_identical)
        self.assertEqual(result.matched, 1)
        self.assertEqual(result.baseline_count, 2)
        self.assertEqual(result.candidate_count, 1)

    def test_both_builds_empty_is_not_a_pass(self) -> None:
        """A run that printed no Ux string must not report as verified."""
        result = compare_ux_strings(self._stdout("o0.out"), self._stdout("o2.out"))
        # Vacuously identical: nothing mismatched and both counts are zero.
        self.assertTrue(result.is_identical)
        self.assertEqual(result.matched, 0)
        self.assertTrue(result.is_empty)

    def test_populated_comparison_is_not_empty(self) -> None:
        result = compare_ux_strings(
            self._stdout("o0.out", _UX_A), self._stdout("o2.out", _UX_A)
        )
        self.assertFalse(result.is_empty)
        self.assertTrue(result.is_identical)

    def test_main_returns_zero_when_identical(self) -> None:
        argv = [self._stdout("o0.out", _UX_A), self._stdout("o2.out", _UX_A)]
        self.assertEqual(_run_main(argv), 0)

    def test_main_returns_one_when_differing(self) -> None:
        argv = [self._stdout("o0.out", _UX_A), self._stdout("o2.out", _UX_B)]
        self.assertEqual(_run_main(argv), 1)

    def test_main_returns_one_when_neither_build_printed(self) -> None:
        argv = [self._stdout("o0.out"), self._stdout("o2.out")]
        self.assertEqual(_run_main(argv), 1)

    def test_main_returns_two_on_missing_stdout(self) -> None:
        baseline = self._stdout("o0.out", _UX_A)
        missing = str(self.tmp / "does-not-exist.out")
        self.assertEqual(_run_main([baseline, missing]), 2)


def _run_main(argv: list[str]) -> int:
    """Invoke ``main`` with a patched ``sys.argv`` and return its exit code."""
    with patch("sys.argv", ["compare_tracing_ux_strings", *argv]):
        return main()


if __name__ == "__main__":
    unittest.main()
