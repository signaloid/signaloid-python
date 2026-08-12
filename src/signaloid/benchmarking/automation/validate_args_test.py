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

from argparse import Namespace

from signaloid.benchmarking.automation.arguments import validate_args
from signaloid.benchmarking.config import (
    Correlations,
    DistanceMetrics,
    ReportingMethods,
    RepresentationTypes,
)


class TestValidateArgs(unittest.TestCase):
    """The python-O-safe validation guards."""

    def test_validate_args_analytic_ground_truth_requires_weighted_samples(
        self,
    ) -> None:
        # has_analytic_ground_truth=True with MonteCarlo ground_truth_type
        # must raise ValueError (not AssertionError) so the guard survives
        # python -O.
        args = Namespace(
            representation_types=[RepresentationTypes.ATHENS],
            representation_sizes=[16],
            correlations=[Correlations.DISABLED],
            reporting_methods=[ReportingMethods.MEAN],
            has_analytic_ground_truth=True,
            ground_truth_type=RepresentationTypes.MONTE_CARLO,
            use_binned_uxhw=False,
            distance_type=DistanceMetrics.WASSERSTEIN_1,
            num_parallel_workers=None,
        )
        with self.assertRaisesRegex(ValueError, "weighted samples"):
            validate_args(args)

    def test_validate_args_binned_uxhw_requires_wasserstein1(self) -> None:
        # use_binned_uxhw=True with a non-W1 distance must raise ValueError
        # (not AssertionError) so the guard survives python -O.
        args = Namespace(
            representation_types=[RepresentationTypes.ATHENS],
            representation_sizes=[16],
            correlations=[Correlations.DISABLED],
            reporting_methods=[ReportingMethods.MEAN],
            has_analytic_ground_truth=False,
            ground_truth_type=RepresentationTypes.MONTE_CARLO,
            use_binned_uxhw=True,
            distance_type=DistanceMetrics.WASSERSTEIN_2,
            num_parallel_workers=None,
        )
        with self.assertRaisesRegex(ValueError, "Binned distance"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
