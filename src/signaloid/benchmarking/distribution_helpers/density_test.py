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

import unittest

import numpy as np

from signaloid.benchmarking.distribution_helpers.density import _histogram_pdf
from signaloid.distributional.distributional import DistributionalValue


def _equal_mass_distribution() -> DistributionalValue:
    """Four equally-weighted points at 0, 1, 2, 3 (deterministic)."""
    return DistributionalValue.from_weighted_samples(
        [0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]
    )


class TestHistogramPdf(unittest.TestCase):
    def test_pdf_evaluates_and_returns_array(self) -> None:
        """Weighted branch returns a non-negative pdf estimate array."""
        out = _histogram_pdf(_equal_mass_distribution(), 1.5)
        self.assertIsInstance(out, np.ndarray)
        self.assertTrue(np.all(out >= 0.0))

    def test_pdf_samples_branch_evaluates(self) -> None:
        """Samples branch (auto bins) returns a non-negative pdf estimate."""
        out = _histogram_pdf(_equal_mass_distribution(), 1.5, treat_as_samples=True)
        self.assertIsInstance(out, np.ndarray)
        self.assertTrue(np.all(out >= 0.0))


if __name__ == "__main__":
    unittest.main()
