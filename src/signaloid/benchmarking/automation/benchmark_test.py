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

"""Unit tests for ``Benchmark.bind_timing_script``.

``bind_timing_script`` binds the eight ``run_timing_script`` kwargs shared by
every timing pass into a ``functools.partial``. Callers then supply only the
per-call ``variable_index`` and one mode flag. It replaced four hand-maintained
copies of the same kwarg bundle, so these tests pin exactly which kwargs are
bound (and that the per-call args are deliberately left unbound).
"""

import functools
import unittest

from signaloid.benchmarking.automation.benchmark import Benchmark
from signaloid.benchmarking.automation.build import run_timing_script


class TestBindTimingScript(unittest.TestCase):
    def _make_benchmark(self) -> Benchmark:
        benchmark = Benchmark(
            path_to_application="/tmp/app",
            representation_types=["Athens"],
            representation_sizes=[16, 32],
            demo_cli_args="--demo",
        )
        # Normally populated by get_application_info / get_machine_info.
        benchmark.all_outputs_cla = "-S 1"
        benchmark.benchmarking_variables = []
        benchmark.application_name = "app"
        benchmark.application_version = "v0"
        return benchmark

    def test_binds_shared_kwargs(self) -> None:
        benchmark = self._make_benchmark()

        bound = benchmark.bind_timing_script()

        self.assertIsInstance(bound, functools.partial)
        self.assertIs(bound.func, run_timing_script)
        self.assertEqual(
            bound.keywords,
            {
                "all_outputs_cla": "-S 1",
                "benchmarking_variables": [],
                "demo_cli_args": "--demo",
                "representation_types": benchmark.representation_types,
                "representation_sizes": benchmark.representation_sizes,
                "correlations": benchmark.correlations,
                "logs_dir": benchmark.logs_dir,
                "intermediate_timings_path": benchmark.intermediate_timings_path(),
            },
        )

    def test_per_call_args_not_prebound(self) -> None:
        bound = self._make_benchmark().bind_timing_script()
        for per_call_key in (
            "variable_index",
            "timing",
            "native_mc_timing",
            "tracing",
        ):
            self.assertNotIn(per_call_key, bound.keywords)


if __name__ == "__main__":
    unittest.main()
