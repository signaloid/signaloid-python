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

from signaloid.benchmarking.types import AsymptoticDistribution
from signaloid.benchmarking.config import ReportingMethods


class TestAsymptoticDistributionDefaults(unittest.TestCase):
    """A freshly-constructed AsymptoticDistribution has all fields None."""

    def test_asymptotic_distribution_defaults_all_none(self) -> None:
        """A freshly-constructed instance has every field set to ``None``."""
        asymptotic = AsymptoticDistribution()

        self.assertIsNone(asymptotic.mean)
        self.assertIsNone(asymptotic.quantile_95)
        self.assertIsNone(asymptotic.quantile_99)
        self.assertIsNone(asymptotic.mean_quantile)
        self.assertIsNone(asymptotic.samples)
        self.assertIsNone(asymptotic.is_normal)
        self.assertIsNone(asymptotic.scale)


class TestAsymptoticDistributionPopulation(unittest.TestCase):
    """AsymptoticDistribution fields can be populated and read back correctly."""

    def test_asymptotic_distribution_distribution_path_population(self) -> None:
        """Distribution-typed variables populate ``mean``, the quantiles,
        and ``samples``."""
        asymptotic = AsymptoticDistribution()
        asymptotic.mean = 0.42
        asymptotic.quantile_95 = 0.5
        asymptotic.quantile_99 = 0.9
        asymptotic.samples = np.array([0.1, 0.2, 0.3])

        self.assertEqual(asymptotic.mean, 0.42)
        self.assertEqual(asymptotic.quantile_95, 0.5)
        self.assertEqual(asymptotic.quantile_99, 0.9)
        np.testing.assert_array_equal(asymptotic.samples, [0.1, 0.2, 0.3])

    def test_asymptotic_distribution_scalar_path_population(self) -> None:
        """Scalar-typed variables populate ``mean``, the quantiles,
        ``is_normal``, and ``scale``."""
        asymptotic = AsymptoticDistribution()
        asymptotic.mean = 0.1
        asymptotic.quantile_95 = 0.5
        asymptotic.quantile_99 = 0.9
        asymptotic.is_normal = True
        asymptotic.scale = 0.05

        self.assertEqual(asymptotic.mean, 0.1)
        self.assertIs(asymptotic.is_normal, True)
        self.assertEqual(asymptotic.scale, 0.05)

    def test_asymptotic_distribution_independent_instances(self) -> None:
        """Two AsymptoticDistribution instances do not share field state."""
        a = AsymptoticDistribution()
        b = AsymptoticDistribution()
        a.mean = 0.9

        self.assertIsNone(b.mean)


class TestAsymptoticDistributionValueFor(unittest.TestCase):
    """AsymptoticDistribution.value_for dispatches methods to typed fields."""

    def test_value_for_returns_the_typed_field(self) -> None:
        """``value_for`` dispatches each ReportingMethods string to the
        corresponding typed attribute."""
        cases = [
            (ReportingMethods.MEAN, "mean"),
            (ReportingMethods.QUANTILE_95, "quantile_95"),
            (ReportingMethods.QUANTILE_99, "quantile_99"),
        ]
        for method, attr in cases:
            with self.subTest(method=method, attr=attr):
                asymptotic = AsymptoticDistribution()
                setattr(asymptotic, attr, 0.123)

                self.assertEqual(asymptotic.value_for(method), 0.123)

    def test_value_for_returns_none_when_field_unpopulated(self) -> None:
        """``value_for`` returns ``None`` for a known method when the
        underlying field has not yet been set — caller decides how to
        handle missing data."""
        asymptotic = AsymptoticDistribution()

        self.assertIsNone(asymptotic.value_for(ReportingMethods.MEAN))

    def test_value_for_raises_on_unknown_method(self) -> None:
        """An unknown reporting method raises ``KeyError`` rather than
        silently returning ``None`` — silent failure would hide typos."""
        asymptotic = AsymptoticDistribution()
        asymptotic.mean = 0.5

        with self.assertRaisesRegex(KeyError, "Unknown reporting method"):
            asymptotic.value_for("not-a-real-method")


if __name__ == "__main__":
    unittest.main()
