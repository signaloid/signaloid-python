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

from signaloid.benchmarking.types import DistributionSamples


class TestDistributionSamples(unittest.TestCase):
    """Exercise the DistributionSamples value/weight container."""

    def test_distribution_samples_defaults(self) -> None:
        """Confirm DistributionSamples initialises with empty values,
        weights, and scalar_output_dict."""
        samples = DistributionSamples()
        self.assertEqual(samples.values, [])
        self.assertEqual(samples.weights, [])
        self.assertEqual(samples.scalar_output_dict, {})

    def test_distribution_samples_set_values(self) -> None:
        """Confirm set_values updates values while leaving weights
        unchanged."""
        samples = DistributionSamples()
        samples.set_values([1.0, 2.0, 3.0])
        self.assertEqual(samples.values, [1.0, 2.0, 3.0])
        self.assertEqual(samples.weights, [])

    def test_distribution_samples_set_weighted_values(self) -> None:
        """Confirm set_weighted_values updates both values and weights."""
        samples = DistributionSamples()
        samples.set_weighted_values([0.5, 1.5], [0.3, 0.7])
        self.assertEqual(samples.values, [0.5, 1.5])
        self.assertEqual(samples.weights, [0.3, 0.7])

    def test_distribution_samples_empty_values(self) -> None:
        """Confirm empty_values clears values, weights, and
        scalar_output_dict."""
        samples = DistributionSamples()
        samples.set_weighted_values([0.5, 1.5], [0.3, 0.7])
        samples.scalar_output_dict = {10: [1.0, 2.0]}

        samples.empty_values()
        self.assertEqual(samples.values, [])
        self.assertEqual(samples.weights, [])
        self.assertEqual(samples.scalar_output_dict, {})

    def test_distribution_samples_set_values_then_empty(self) -> None:
        """Confirm set_values followed by empty_values leaves clean
        state."""
        samples = DistributionSamples()
        samples.set_values([1.0, 2.0, 3.0])
        self.assertEqual(samples.values, [1.0, 2.0, 3.0])

        samples.empty_values()
        self.assertEqual(samples.values, [])
        self.assertEqual(samples.weights, [])
        self.assertEqual(samples.scalar_output_dict, {})


if __name__ == "__main__":
    unittest.main()
