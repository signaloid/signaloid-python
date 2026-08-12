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

from signaloid.distributional.distributional import DistributionalValue
from signaloid.benchmarking.config import RepresentationTypes
from signaloid.benchmarking.types import TaggedDistributionalValue
from signaloid.benchmarking.equivalent_mc.equivalent_mc_utils import (
    _distribution_range,
    _compute_shared_xlim,
)


def _tagged(
    samples: np.ndarray, samples_type: bool = False
) -> TaggedDistributionalValue:
    """Wrap float samples in a carrier, optionally tagged as the ``Samples``
    representation so the inverse-CDF takes the (unweighted) samples branch."""
    dv = DistributionalValue.from_samples(samples)
    representation_type = RepresentationTypes.SAMPLES if samples_type else None
    if samples_type:
        setattr(dv, "representation_type", RepresentationTypes.SAMPLES)
    return TaggedDistributionalValue(dv=dv, representation_type=representation_type)


class TestDistributionRange(unittest.TestCase):
    """``_distribution_range`` measures a distribution's x-axis span."""

    def test_full_range_spans_support(self) -> None:
        samples = np.linspace(-5.0, 5.0, 1001)
        full_range = _distribution_range(_tagged(samples).dv, robust=False)
        assert full_range is not None
        low, high = full_range
        self.assertAlmostEqual(low, -5.0, places=6)
        self.assertAlmostEqual(high, 5.0, places=6)

    def test_robust_range_clips_outliers(self) -> None:
        # A clean body in [-1, 1] with a handful of extreme outliers that a
        # full-support range would be stretched by.
        body = np.random.default_rng(0).uniform(-1.0, 1.0, 10_000)
        samples = np.concatenate([body, [1e6, -1e6, 5e5]])
        tagged = _tagged(samples, samples_type=True)

        full_range = _distribution_range(tagged.dv, robust=False)
        robust_range = _distribution_range(tagged.dv, robust=True)
        assert full_range is not None
        assert robust_range is not None
        full_low, full_high = full_range
        robust_low, robust_high = robust_range

        # Full range is dominated by the outliers. The robust range is not.
        self.assertLess(full_low, -1e5)
        self.assertGreater(full_high, 1e5)
        self.assertGreater(robust_low, -2.0)
        self.assertLess(robust_high, 2.0)

    def test_empty_distribution_returns_none(self) -> None:
        empty = DistributionalValue()
        # inverse_cdf on an empty distribution yields NaN, which the helper maps to None.
        self.assertIsNone(_distribution_range(empty, robust=False))


class TestComputeSharedXLim(unittest.TestCase):
    """``_compute_shared_xlim`` unions the plotted distributions' domains."""

    def test_domain_spans_ground_truth_and_uxhw(self) -> None:
        ground_truth = _tagged(np.linspace(-2.0, 2.0, 1001))
        uxhw = [_tagged(np.linspace(0.0, 6.0, 1001))]

        xlim = _compute_shared_xlim(ground_truth, uxhw, adversaries=[])

        assert xlim is not None
        low, high = xlim
        # The union of [-2, 2] and [0, 6] is [-2, 6]. Padding widens it further.
        self.assertLess(low, -2.0)
        self.assertGreater(high, 6.0)

    def test_adversary_outliers_do_not_blow_up_domain(self) -> None:
        ground_truth = _tagged(np.random.default_rng(1).normal(0.0, 1.0, 5000))
        uxhw = [_tagged(np.random.default_rng(2).normal(0.5, 1.0, 5000))]

        adv_body = np.random.default_rng(3).normal(0.0, 1.0, 5000)
        adversaries = [
            _tagged(np.concatenate([adv_body, [1e6, -1e6]]), samples_type=True)
        ]

        xlim = _compute_shared_xlim(ground_truth, uxhw, adversaries)

        assert xlim is not None
        low, high = xlim
        # Despite the ±1e6 adversary outliers, the domain stays close to the
        # ground-truth / UxHw support.
        self.assertGreater(low, -20.0)
        self.assertLess(high, 20.0)

    def test_returns_none_when_no_ranges_available(self) -> None:
        empty = TaggedDistributionalValue(dv=DistributionalValue())
        self.assertIsNone(_compute_shared_xlim(empty, uxhw=[], adversaries=[]))


if __name__ == "__main__":
    unittest.main()
