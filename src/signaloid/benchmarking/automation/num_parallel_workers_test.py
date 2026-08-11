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

"""
Coverage for the ``-j`` / ``--num-parallel-workers`` worker-count dial.

"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from signaloid.benchmarking.automation import benchmark as benchmark_module
from signaloid.benchmarking.automation.arguments import (
    create_argument_parser,
    validate_args,
)
from signaloid.benchmarking.automation.benchmark import Benchmark


def _lscpu_output(n_cpus: int) -> str:
    """A minimal ``lscpu`` stdout exposing a model name and CPU count."""
    return f"Model name: Fake CPU\nCPU(s): {n_cpus}\n"


def _make_benchmark(num_parallel_workers: int | None) -> Benchmark:
    return Benchmark(
        path_to_application="/tmp/fake-app",
        num_parallel_workers=num_parallel_workers,
    )


def _detect_with(benchmark: Benchmark, n_cpus: int) -> None:
    """Run ``get_machine_info`` with a stubbed ``lscpu`` reporting
    ``n_cpus`` cores."""
    with patch.object(
        benchmark_module.subprocess,  # type: ignore[attr-defined]
        "check_output",
        return_value=_lscpu_output(n_cpus),
    ):
        benchmark.get_machine_info()


class TestNumParallelWorkers(unittest.TestCase):
    """Coverage for the ``-j`` / ``--num-parallel-workers`` worker-count parameter."""

    def test_j_flag_default_is_none(self) -> None:
        """An unset ``-j`` parses to ``None`` so the constructor / machine
        info can distinguish it from an explicit value and resolve it to the
        detected core count."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path-to-application", "/tmp/fake-app"])
        self.assertIsNone(args.num_parallel_workers)

    def test_j_flag_explicit_value_is_parsed(self) -> None:
        for requested in [1, 4, 64]:
            with self.subTest(requested=requested):
                parser = create_argument_parser()
                args = parser.parse_args(
                    [
                        "--path-to-application",
                        "/tmp/fake-app",
                        "-j",
                        str(requested),
                    ]
                )
                self.assertEqual(args.num_parallel_workers, requested)

    def test_num_parallel_workers_alias_is_parsed(self) -> None:
        """The ``--num-parallel-workers`` long alias referenced by the help
        text and validation message must resolve to the same destination as
        ``-j`` / ``--jobs``."""
        parser = create_argument_parser()
        args = parser.parse_args(
            ["--path-to-application", "/tmp/fake-app", "--num-parallel-workers", "8"]
        )
        self.assertEqual(args.num_parallel_workers, 8)

    def test_j_flag_rejects_non_positive(self) -> None:
        """An explicit ``-j`` below 1 is rejected at validation time, rather
        than crashing deep in a worker pool with ``max_workers < 1``."""
        for bad in [0, -1]:
            with self.subTest(bad=bad):
                parser = create_argument_parser()
                args = parser.parse_args(
                    ["--path-to-application", "/tmp/fake-app", "-j", str(bad)]
                )
                with self.assertRaisesRegex(ValueError, "must be >= 1"):
                    validate_args(args)

    def test_unset_j_resolves_to_detected_cores(self) -> None:
        """Leaving ``-j`` unset must preserve current behaviour: the MC
        stages default to the detected core count, not 1."""
        benchmark = _make_benchmark(None)
        self.assertIsNone(benchmark.num_parallel_workers)

        _detect_with(benchmark, n_cpus=32)

        self.assertEqual(benchmark.n_processors, 32)
        self.assertEqual(benchmark.num_parallel_workers, 32)

    def test_explicit_j_is_left_untouched_by_machine_info(self) -> None:
        """An explicit ``-j`` value is never overridden by the detected core
        count."""
        for requested in [1, 8]:
            with self.subTest(requested=requested):
                benchmark = _make_benchmark(requested)
                _detect_with(benchmark, n_cpus=32)
                self.assertEqual(benchmark.num_parallel_workers, requested)

    def test_warning_fires_only_when_request_exceeds_cores(self) -> None:
        """The 'requested N but only M detected' warning must describe a
        constraint that actually holds, i.e. only fire when the explicit
        ``-j`` exceeds the detected core count."""
        buf: io.StringIO = io.StringIO()
        benchmark = _make_benchmark(64)
        with redirect_stdout(buf):
            _detect_with(benchmark, n_cpus=32)
        self.assertIn(
            "Requested 64 parallel workers but only 32 detected", buf.getvalue()
        )

        buf2: io.StringIO = io.StringIO()
        benchmark = _make_benchmark(8)
        with redirect_stdout(buf2):
            _detect_with(benchmark, n_cpus=32)
        self.assertNotIn("parallel workers but only", buf2.getvalue())

        # Unset (resolved to detected cores) must never warn.
        buf3: io.StringIO = io.StringIO()
        benchmark = _make_benchmark(None)
        with redirect_stdout(buf3):
            _detect_with(benchmark, n_cpus=32)
        self.assertNotIn("parallel workers but only", buf3.getvalue())

    def test_num_parallel_workers_resolved_and_clamped(self) -> None:
        """After ``get_machine_info``, ``num_parallel_workers`` is the single
        resolved value that bounds every worker pool: an unset ``-j`` resolves to
        the detected core count, and a request above the core count is clamped
        down to it.

        This is the value all five pools read — the native compile and the EMCC
        adversary-distance stage read ``num_parallel_workers`` directly, and the
        three MC sample-generation stages derive ``mc_worker_count`` from it in
        ``benchmark_application._run_pipeline``. The ``(64, 32, 32)`` case in
        particular pins that EMCC no longer oversubscribes for ``-j > cores``.
        """
        cases = [
            # (num_parallel_workers, n_processors, expected)
            # Unset -> resolved to detected cores.
            (None, 32, 32),
            # Explicit -j below detected cores -> left as-is.
            (8, 32, 8),
            # Explicit -j above detected cores -> clamped down to cores.
            (64, 32, 32),
            # Explicit -j == detected cores.
            (32, 32, 32),
            # Serial run.
            (1, 32, 1),
        ]
        for num_parallel_workers, n_processors, expected in cases:
            with self.subTest(
                num_parallel_workers=num_parallel_workers,
                n_processors=n_processors,
                expected=expected,
            ):
                benchmark = _make_benchmark(num_parallel_workers)
                _detect_with(benchmark, n_cpus=n_processors)
                self.assertEqual(benchmark.num_parallel_workers, expected)


if __name__ == "__main__":
    unittest.main()
