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
Collapse a distribution to a small asymptotically-optimal Dirac-delta set.

Benchmarking-internal helper (not part of the public API). Builds a reduced
Wasserstein-1-optimal representation via the local ``quantization`` routines.
"""

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.benchmarking.distribution_helpers.quantization import (
    _asymptotically_optimal_wasserstein_p_representation,
)


def _collapse_asymptotically_optimal_w1(
    dist: DistributionalValue, *, n_dirac_deltas: int
) -> DistributionalValue:
    """
    Collapse a distribution to ``n_dirac_deltas`` Dirac deltas, asymptotically
    optimally with respect to the Wasserstein-1 distance.

    It sorts the distribution and delegates to the
    ``_asymptotically_optimal_wasserstein_p_representation`` algorithm (with
    ``p=1``).

    Args:
        dist: The distribution to collapse. Mutated in place by ``sort()``.
        n_dirac_deltas: The number of Dirac deltas in the collapsed
            representation.

    Returns:
        A new ``DistributionalValue`` with ``n_dirac_deltas`` Dirac deltas
        approximating ``dist``.

    Raises:
        ValueError: If ``n_dirac_deltas`` is less than 2. The underlying
            algorithm indexes the second-and-later collapsed positions, so
            fewer than two points is undefined (it would otherwise raise an
            opaque ``IndexError``).
    """
    if n_dirac_deltas < 2:
        raise ValueError(
            "n_dirac_deltas must be >= 2 for asymptotically-optimal-W1 collapse"
        )
    dist.sort()
    positions, masses = _asymptotically_optimal_wasserstein_p_representation(
        dist.positions, dist.masses, n_dirac_deltas=n_dirac_deltas, p=1
    )
    dirac_deltas = [
        DiracDelta(position=pos, mass=mass) for pos, mass in zip(positions, masses)
    ]
    return DistributionalValue(dirac_deltas=dirac_deltas)
