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

"""Equivalent-Monte-Carlo analysis.

Computes the number of Monte-Carlo samples whose accuracy is equivalent to a
Signaloid (UxHw) computation, by comparing distributional distances.
"""

from signaloid.benchmarking.equivalent_mc.equivalent_mc_main import (
    LoadDataComputeEquivalentMCArgs,
    load_data_and_compute_equivalent_mc,
)
from signaloid.benchmarking.equivalent_mc.equivalent_mc_utils import (
    anderson_darling_test,
)

__all__ = [
    "load_data_and_compute_equivalent_mc",
    "LoadDataComputeEquivalentMCArgs",
    "anderson_darling_test",
]
