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

import argparse
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from typing import IO

# Longest stderr excerpt kept for a failure message. Benchmark binaries can
# be chatty. The useful part is at the end.
_STDERR_EXCERPT_LIMIT = 2000


def scaled_repetitions(
    *,
    single_time: float,
    target_total_time: float,
    min_repetitions: int,
    max_repetitions: int,
) -> int:
    """
    Scale a repetition count so the total run time approaches a target.

    This mirrors the compute_time_scaled_repetitions shell function.
    run_uxhw_benchmarks already uses that helper. Both timing paths
    therefore size their repetition counts the same way. The rule is to
    divide the target by one run's cost, round up, then clamp.

    A non-positive single_time means the run was too fast to measure.
    That case falls back to max_repetitions, matching the shell helper.

    Args:
        single_time: Measured duration of one run, in seconds.
        target_total_time: Total measurement time aimed for, in seconds.
        min_repetitions: Lower clamp on the returned count.
        max_repetitions: Upper clamp on the returned count.

    Returns:
        The repetition count to use.
    """
    if single_time <= 0:
        return max_repetitions
    repetitions = math.ceil(target_total_time / single_time)
    return max(min_repetitions, min(repetitions, max_repetitions))


def _read_stderr_tail(stderr_sink: IO[bytes]) -> str:
    """
    Read the last few kilobytes written to a stderr sink.

    Only the tail is read back. A chatty benchmark can write far more
    than a failure message needs.

    Args:
        stderr_sink: The temporary file the child wrote its stderr to.

    Returns:
        The decoded tail of the file, at most _STDERR_EXCERPT_LIMIT bytes.
    """
    stderr_sink.seek(0, os.SEEK_END)
    written = stderr_sink.tell()
    stderr_sink.seek(max(0, written - _STDERR_EXCERPT_LIMIT))
    return stderr_sink.read().decode("utf-8", errors="replace")


def time_once(*, command: str, use_shell: bool) -> float:
    """
    Run a command once and return its wall-clock duration.

    The command's stdout is discarded. Benchmark binaries print their
    results there. This helper's own stdout carries the measurement back
    to the shell layer.

    Stderr goes to a temporary file rather than a pipe. The child writes
    straight to that file descriptor. Nothing accumulates in this
    process, so the measurement stays undisturbed. Only the tail is read
    back, and only when the command fails.

    Args:
        command: The command line to run.
        use_shell: Run via the system shell when True. Otherwise split
            the command with shlex and execute it directly.

    Returns:
        The elapsed wall-clock time in seconds.

    Raises:
        RuntimeError: If the command exits non-zero. Timing a failed run
            would feed a meaningless number into the report.
    """
    arguments: str | list[str] = command if use_shell else shlex.split(command)
    with tempfile.TemporaryFile() as stderr_sink:
        start = time.perf_counter()
        completed = subprocess.run(
            arguments,
            shell=use_shell,
            stdout=subprocess.DEVNULL,
            stderr=stderr_sink,
        )
        elapsed = time.perf_counter() - start
        if completed.returncode != 0:
            raise RuntimeError(
                f"command exited {completed.returncode}: {command}\n"
                f"stderr: {_read_stderr_tail(stderr_sink)}"
            )
    return elapsed


def measure_mean_time(
    *,
    command: str,
    use_shell: bool,
    target_total_time: float,
    min_repetitions: int,
    max_repetitions: int,
) -> float:
    """
    Time a command repeatedly and return the mean wall-clock duration.

    One warmup run is timed first and then discarded. It primes caches.
    It also sizes the measured repetition count via scaled_repetitions.

    Args:
        command: The command line to benchmark.
        use_shell: Whether to run the command through the system shell.
        target_total_time: Total measurement time aimed for, in seconds.
        min_repetitions: Lower clamp on the repetition count.
        max_repetitions: Upper clamp on the repetition count.

    Returns:
        The mean elapsed wall-clock time in seconds.

    Raises:
        RuntimeError: If any run of the command exits non-zero.
    """
    warmup_time = time_once(command=command, use_shell=use_shell)
    repetitions = scaled_repetitions(
        single_time=warmup_time,
        target_total_time=target_total_time,
        min_repetitions=min_repetitions,
        max_repetitions=max_repetitions,
    )
    total = 0.0
    for _ in range(repetitions):
        total += time_once(command=command, use_shell=use_shell)
    return total / repetitions


def main() -> None:
    """
    Parse command-line arguments and emit the mean wall-clock time.

    The bash layer calls this through python3 -m
    signaloid.benchmarking.automation.measure_process_time. Stdout
    carries a single float in seconds.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Time a command over a scaled number of repetitions and print "
            "the mean wall-clock duration in seconds."
        ),
    )
    parser.add_argument(
        "command",
        help="The command line to benchmark, as a single string.",
    )
    parser.add_argument(
        "--target-total-time",
        type=float,
        default=30.0,
        help=(
            "Total measurement time to aim for, in seconds. Matches the "
            "shell layer's TIMING_TARGET_TOTAL_TIME."
        ),
    )
    parser.add_argument(
        "--min-repetitions",
        type=int,
        default=2,
        help="Lower clamp on the repetition count.",
    )
    parser.add_argument(
        "--max-repetitions",
        type=int,
        default=20,
        help="Upper clamp on the repetition count.",
    )
    parser.add_argument(
        "--no-shell",
        dest="use_shell",
        action="store_false",
        default=True,
        help=(
            "Execute the command directly instead of through the system "
            "shell. The command is split with shlex."
        ),
    )
    args = parser.parse_args()

    if args.min_repetitions < 1:
        parser.error(f"--min-repetitions must be >= 1, got {args.min_repetitions}")
    if args.max_repetitions < args.min_repetitions:
        parser.error(
            f"--max-repetitions ({args.max_repetitions}) must be >= "
            f"--min-repetitions ({args.min_repetitions})"
        )

    print(
        measure_mean_time(
            command=args.command,
            use_shell=args.use_shell,
            target_total_time=args.target_total_time,
            min_repetitions=args.min_repetitions,
            max_repetitions=args.max_repetitions,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"measure_process_time: {exc}", file=sys.stderr)
        sys.exit(1)
