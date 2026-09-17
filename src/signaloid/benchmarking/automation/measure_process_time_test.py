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
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from signaloid.benchmarking.automation.measure_process_time import (
    measure_mean_time,
    scaled_repetitions,
    time_once,
)

# A command that exits 0 and does nothing else. It is spelled via the running
# interpreter so the tests do not depend on which coreutils are present.
_NOOP_COMMAND = f"{shlex.quote(sys.executable)} -c pass"
_FAILING_COMMAND = f"{shlex.quote(sys.executable)} -c 'raise SystemExit(3)'"

# A command whose text carries a shell redirection. Under the shell the
# redirection creates the target file. Executed directly the two tokens
# reach the interpreter as ignored extra arguments and no file appears.
_REDIRECT_TEMPLATE = f"{shlex.quote(sys.executable)} -c pass > {{target}}"


def _noisy_command(*, byte_count: int) -> str:
    """
    Build a command that writes many bytes to stderr and then fails.

    Args:
        byte_count: Number of bytes the command writes to stderr.

    Returns:
        A command line suitable for time_once.
    """
    script = "\n".join(
        [
            "import sys",
            f"sys.stderr.write('x' * {byte_count})",
            "raise SystemExit(4)",
        ]
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


class TestScaledRepetitions(unittest.TestCase):
    """scaled_repetitions mirrors the compute_time_scaled_repetitions shell
    helper. It divides, rounds up, then clamps."""

    def test_divides_target_by_single_run_cost(self) -> None:
        """A 1s run against a 10s target gives 10 repetitions."""
        self.assertEqual(
            scaled_repetitions(
                single_time=1.0,
                target_total_time=10.0,
                min_repetitions=2,
                max_repetitions=20,
            ),
            10,
        )

    def test_rounds_the_quotient_up(self) -> None:
        """
        A non-integer quotient rounds up. The target total time is then
        met rather than undershot.
        """
        self.assertEqual(
            scaled_repetitions(
                single_time=3.0,
                target_total_time=10.0,
                min_repetitions=1,
                max_repetitions=20,
            ),
            4,
        )

    def test_clamps_to_the_maximum_for_a_fast_command(self) -> None:
        """A very fast run would demand a huge count, so the max applies."""
        self.assertEqual(
            scaled_repetitions(
                single_time=0.001,
                target_total_time=30.0,
                min_repetitions=2,
                max_repetitions=20,
            ),
            20,
        )

    def test_clamps_to_the_minimum_for_a_slow_command(self) -> None:
        """
        A run already longer than the target still repeats
        min_repetitions times. A mean is then always taken over more
        than one sample.
        """
        self.assertEqual(
            scaled_repetitions(
                single_time=120.0,
                target_total_time=30.0,
                min_repetitions=2,
                max_repetitions=20,
            ),
            2,
        )

    def test_non_positive_single_time_falls_back_to_the_maximum(self) -> None:
        """
        A run too fast to measure yields a non-positive time. The shell
        helper returns max_reps in that case and so does this one.
        """
        for single_time in (0.0, -1.0):
            with self.subTest(single_time=single_time):
                self.assertEqual(
                    scaled_repetitions(
                        single_time=single_time,
                        target_total_time=30.0,
                        min_repetitions=2,
                        max_repetitions=20,
                    ),
                    20,
                )


class TestExecutionMode(unittest.TestCase):
    """time_once picks direct execution or the shell as asked."""

    def test_no_shell_passes_a_split_argument_list(self) -> None:
        """
        Without a shell the command is split by shlex and handed to
        subprocess as a list. Quoted whitespace survives as one argument.
        """
        with patch("subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess([], 0)
            time_once(command="prog --flag 'a b'", use_shell=False)
        positional, keyword = run_mock.call_args
        self.assertEqual(positional[0], ["prog", "--flag", "a b"])
        self.assertFalse(keyword["shell"])

    def test_shell_passes_the_command_unsplit(self) -> None:
        """With a shell the command string is handed over verbatim."""
        with patch("subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess([], 0)
            time_once(command="prog --flag 'a b'", use_shell=True)
        positional, keyword = run_mock.call_args
        self.assertEqual(positional[0], "prog --flag 'a b'")
        self.assertTrue(keyword["shell"])

    def test_no_shell_leaves_a_redirection_uninterpreted(self) -> None:
        """
        Direct execution must not honour shell metacharacters. The UxHw
        call site relies on this.
        """
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "redirected.txt")
            command = _REDIRECT_TEMPLATE.format(target=shlex.quote(target))
            time_once(command=command, use_shell=False)
            self.assertFalse(os.path.exists(target))

    def test_shell_honours_a_redirection(self) -> None:
        """
        Shell execution does honour metacharacters. This is the
        counterpart that proves the two modes really differ.
        """
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "redirected.txt")
            command = _REDIRECT_TEMPLATE.format(target=shlex.quote(target))
            time_once(command=command, use_shell=True)
            self.assertTrue(os.path.exists(target))


class TestTimeOnce(unittest.TestCase):
    """time_once measures a single run and rejects failed commands."""

    def test_returns_a_positive_duration(self) -> None:
        """A successful command yields a positive elapsed time."""
        self.assertGreater(
            time_once(command=_NOOP_COMMAND, use_shell=False),
            0.0,
        )

    def test_non_zero_exit_raises(self) -> None:
        """
        A failed command raises rather than returning a duration. Timing
        a run that did not complete would feed a meaningless number into
        the report.
        """
        with self.assertRaises(RuntimeError) as raised:
            time_once(command=_FAILING_COMMAND, use_shell=True)
        self.assertIn("exited 3", str(raised.exception))

    def test_failure_message_keeps_only_the_stderr_tail(self) -> None:
        """
        A chatty failing command must not paste its whole stderr into the
        message. Only the tail is reported, so the message stays bounded.
        """
        noisy_bytes = 50_000
        with self.assertRaises(RuntimeError) as raised:
            time_once(command=_noisy_command(byte_count=noisy_bytes), use_shell=False)
        message = str(raised.exception)
        self.assertIn("exited 4", message)
        self.assertLess(len(message), noisy_bytes)


class TestMeasureMeanTime(unittest.TestCase):
    """measure_mean_time discards a warmup run and averages the rest."""

    def test_returns_a_positive_mean(self) -> None:
        """The mean over the scaled repetitions is a positive duration."""
        mean = measure_mean_time(
            command=_NOOP_COMMAND,
            use_shell=False,
            target_total_time=0.05,
            min_repetitions=2,
            max_repetitions=3,
        )
        self.assertGreater(mean, 0.0)

    def test_runs_warmup_plus_the_scaled_repetitions(self) -> None:
        """
        The warmup run is timed and then discarded. With the clamp
        pinning the count at 2, that means three executions in total.
        """
        with patch(
            "signaloid.benchmarking.automation.measure_process_time.time_once",
            return_value=1.0,
        ) as time_once_mock:
            mean = measure_mean_time(
                command=_NOOP_COMMAND,
                use_shell=False,
                target_total_time=1.0,
                min_repetitions=2,
                max_repetitions=2,
            )
        self.assertEqual(time_once_mock.call_count, 3)
        self.assertEqual(mean, 1.0)

    def test_propagates_a_failing_command(self) -> None:
        """A command that fails on its warmup run raises immediately."""
        with self.assertRaises(RuntimeError):
            measure_mean_time(
                command=_FAILING_COMMAND,
                use_shell=True,
                target_total_time=0.05,
                min_repetitions=2,
                max_repetitions=2,
            )


class TestCommandLineInterface(unittest.TestCase):
    """The module entry point prints a single float for the shell layer."""

    def test_prints_a_parseable_float(self) -> None:
        """
        Stdout carries only the mean. The bash layer can then capture it
        with a plain command substitution.
        """
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "signaloid.benchmarking.automation.measure_process_time",
                "--target-total-time",
                "0.05",
                "--min-repetitions",
                "2",
                "--max-repetitions",
                "2",
                "--no-shell",
                _NOOP_COMMAND,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreater(float(completed.stdout.strip()), 0.0)

    def test_reports_a_failing_command_on_stderr(self) -> None:
        """A failed benchmark exits non-zero with a named cause."""
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "signaloid.benchmarking.automation.measure_process_time",
                _FAILING_COMMAND,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("measure_process_time:", completed.stderr)


if __name__ == "__main__":
    unittest.main()
