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

"""Benchmarking tooling for distributional workloads.

Provides the timing harness and the equivalent-Monte-Carlo analysis that compares
Signaloid (UxHw) computations against Monte-Carlo references. The application-level
dependencies (``pandas``, ``tabulate``, ``tqdm``, ``POT``) install with the package.
The optional Google Sheets reporting backend is the ``sheets`` extra (install
``signaloid[sheets]`` to enable it).

The public API is the ``signaloid-benchmarking`` console script plus the symbols
re-exported here (the equivalent-Monte-Carlo analysis is also available as a library
API). Modules under ``automation`` are internal orchestration and not part of
the API.
"""

from signaloid.benchmarking.config import (
    Correlations,
    DistanceMetrics,
    ReportingMethods,
    RepresentationTypes,
    VariableTypes,
)
from signaloid.benchmarking.equivalent_mc import (
    LoadDataComputeEquivalentMCArgs,
    anderson_darling_test,
    load_data_and_compute_equivalent_mc,
)
from signaloid.benchmarking.types import BenchmarkingVariable

__all__ = [
    "load_data_and_compute_equivalent_mc",
    "LoadDataComputeEquivalentMCArgs",
    "BenchmarkingVariable",
    "DistanceMetrics",
    "RepresentationTypes",
    "Correlations",
    "ReportingMethods",
    "VariableTypes",
    "anderson_darling_test",
]
