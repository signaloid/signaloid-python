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
Quantization helpers used by the collapse routines.

These pure-numpy functions compute weighted quantiles and an asymptotically
optimal discrete (Wasserstein-p) representation of a distribution.
"""

import numpy as np


def _weighted_quantile(
    values: np.ndarray | list[float],
    quantiles: np.ndarray | list[float],
    weights: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """
    Compute quantiles for weighted samples by linearly interpolating the
    probability mass function built from ``values`` and ``weights``.

    Args:
        values: The sample values.
        quantiles: The quantiles to compute (e.g. [0.25, 0.5, 0.75]).
        weights: The sample weights. Equal weights when omitted.

    Returns:
        The computed quantile values.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    weights = np.ones_like(values) if weights is None else np.array(weights)

    # Sort values and associated weights
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative sum of weights and normalize to [0, 1]
    cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    # Use linear interpolation to find the quantile values
    return np.asarray(np.interp(quantiles, cumulative_weights, sorted_values))


def _asymptotically_optimal_wasserstein_p_representation(
    positions: np.ndarray, masses: np.ndarray, n_dirac_deltas: int, p: int | float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute an asymptotically optimal discrete representation of a distribution
    under the Wasserstein-p distance (method from Section 7.3 of 'Foundations
    of Quantization').

    Args:
        positions: Array of sorted sample positions.
        masses: Corresponding mass values (weights).
        n_dirac_deltas: Number of Dirac deltas in the approximation.
        p: Order of the Wasserstein distance to optimize for.

    Returns:
        The discrete positions and their corresponding masses.
    """
    if len(positions) != len(masses):
        raise ValueError("positions and masses must have the same length")

    # Compute the cumulative distribution function (CDF)
    cumulative_masses = np.cumsum(masses)
    cumulative_masses /= cumulative_masses[-1]

    def cdf(x: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.interp(x, positions, cumulative_masses, left=0.0, right=1.0)
        )

    probabilities = (2 * np.arange(1, n_dirac_deltas + 1) - 1) / (2 * n_dirac_deltas)

    # Calculate transformed masses
    transformed_masses = masses ** (1 / (1 + p))
    transformed_masses /= np.sum(transformed_masses)

    # Compute positions
    new_positions = _weighted_quantile(
        positions, quantiles=probabilities, weights=transformed_masses
    )
    midpoints = (new_positions[1:] + new_positions[:-1]) / 2

    # Compute CDF values
    cdf_midpoints = cdf(midpoints)

    # Compute new masses
    new_masses = np.zeros(n_dirac_deltas)
    new_masses[0] = cdf_midpoints[0]
    new_masses[-1] = 1 - cdf_midpoints[-1]
    new_masses[1:-1] = np.diff(cdf_midpoints)

    new_masses /= np.sum(new_masses)

    return new_positions, new_masses
