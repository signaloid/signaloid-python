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
Histogram-based density (pdf) estimate for a DistributionalValue.

Benchmarking-internal helper (not part of the public API). This is an
opinionated density *estimate* (Freedman-Diaconis binning for weighted
samples, ``numpy`` auto-binning for the samples branch, both via
``scipy.stats.rv_histogram``), deliberately kept out of the core
``DistributionalValue`` class, which exposes only exact distribution
operations (``cdf`` / ``inverse_cdf``).
"""

import numpy as np
from scipy.stats import rv_histogram  # type: ignore

from signaloid.distributional.distributional import DistributionalValue


def _histogram_pdf(
    dist: DistributionalValue,
    x: float | np.floating | np.ndarray,
    *,
    treat_as_samples: bool = False,
) -> np.ndarray:
    """
    Estimate the pdf of ``dist`` at ``x`` via a histogram density.

    Args:
        dist: The distributional value whose pdf is estimated.
        x: Evaluation point(s).
        treat_as_samples: When True, ignore masses and bin the positions
            as equally-weighted samples (``numpy`` auto bins). When False
            (default), build a Freedman-Diaconis-binned weighted histogram.

    Returns:
        The estimated pdf value(s) at ``x``.
    """
    sort_idx = np.argsort(dist.positions)
    sorted_positions = dist.positions[sort_idx]
    sorted_masses = dist.masses[sort_idx]

    if treat_as_samples:
        hist_dist = rv_histogram(np.histogram(sorted_positions, bins="auto"))
    else:
        # Freedman-Diaconis rule for the bin count, with fallbacks when the
        # IQR or the support width is degenerate.
        n = len(sorted_positions)
        q75 = float(np.percentile(sorted_positions, 75))
        q25 = float(np.percentile(sorted_positions, 25))
        iqr = q75 - q25
        support_width = float(sorted_positions.max() - sorted_positions.min())
        bin_width = 2 * iqr / (n ** (1 / 3)) if iqr > 0 else support_width / 50
        bins = int(support_width / bin_width) if bin_width > 0 else 50
        bins = max(10, min(bins, 1000))
        hist_dist = rv_histogram(
            np.histogram(
                sorted_positions, bins=bins, weights=sorted_masses, density=True
            )
        )

    return np.asarray(hist_dist.pdf(x))
