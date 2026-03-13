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


from __future__ import annotations
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
import sys

if sys.implementation.name != "circuitpython":
    # Use numpy for accelerated computing
    import numpy as np
else:
    # Use the extended version of ulab's numpy when running on CircuitPython
    from signaloid.circuitpython.extended_ulab_numpy import np  # type: ignore[no-redef]


def sample_generator(ux_data: str, n_samples: int) -> np.ndarray:
    """
    Parse a Ux-encoded string and generate samples from the resulting distribution.

    Args:
        ux_data: A Ux-string or Ux-bytes encoding a distributional value.
        n_samples: Number of samples to generate.

    Returns:
        Array of samples drawn from the distribution.

    Raises:
        ValueError: If the Ux-data cannot be parsed into a DistributionalValue.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

    dist_value = DistributionalValue.parse(ux_data)
    if dist_value is None:
        raise ValueError("Failed to parse Ux-data into DistributionalValue.")

    samples = sample_from_distributional_value(dist_value, n_samples)
    return samples


def sample_from_distributional_value(
    distributional_value: DistributionalValue, n_samples: int
) -> np.ndarray:
    """
    Generate samples from a DistributionalValue using inverse CDF sampling.

    For particle distributions (single Dirac delta), returns identical copies
    of the particle position. For distributions with non-finite Dirac deltas
    (NaN, -Inf, +Inf), samples are drawn from a mixture: with probability
    equal to the total finite mass, a finite sample is drawn via inverse CDF;
    otherwise a non-finite value is chosen based on its relative mass.

    Args:
        distributional_value: The parsed distributional value to sample from.
        n_samples: Number of samples to generate.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

    finite_deltas = distributional_value.finite_dirac_deltas

    # Collect non-finite (special) Dirac deltas with non-zero mass.
    special_positions: list[float] = []
    special_masses: list[float] = []
    if distributional_value.has_special_values:
        for dd in [
            distributional_value.nan_dirac_delta,
            distributional_value.neg_inf_dirac_delta,
            distributional_value.pos_inf_dirac_delta,
        ]:
            if dd.mass > 0:
                special_positions.append(dd.position)
                special_masses.append(dd.mass)

    total_finite_mass = sum(dd.mass for dd in finite_deltas)
    total_special_mass = sum(special_masses)
    total_mass = total_finite_mass + total_special_mass

    # If there are no finite deltas, sample entirely from special values.
    if len(finite_deltas) == 0:
        return _sample_special_values(
            np.array(special_positions),
            np.array(special_masses) / total_mass,
            n_samples,
        )

    # If there is exactly one finite delta and no special values, return
    # identical copies (particle distribution).
    if len(finite_deltas) == 1 and total_special_mass == 0:
        return np.full(n_samples, finite_deltas[0].position)

    # If there are no special values, sample entirely from finite deltas.
    if total_special_mass == 0:
        return _sample_finite(distributional_value, n_samples)

    # Mixed case: decide per-sample whether to draw from finite or special.
    finite_prob = total_finite_mass / total_mass
    us = np.random.uniform(0, 1, size=n_samples)
    is_finite = us < finite_prob

    n_finite = int(np.sum(is_finite))
    n_special = n_samples - n_finite

    samples = np.empty(n_samples)

    if n_finite > 0:
        if len(finite_deltas) == 1:
            samples[is_finite] = finite_deltas[0].position
        else:
            samples[is_finite] = _sample_finite(distributional_value, n_finite)

    if n_special > 0:
        special_probs = np.array(special_masses) / total_special_mass
        samples[~is_finite] = _sample_special_values(
            np.array(special_positions), special_probs, n_special
        )

    return samples


def _sample_finite(
    distributional_value: DistributionalValue, n_samples: int
) -> np.ndarray:
    """Sample from the finite part of a distributional value via inverse CDF."""
    distributional_value.drop_zero_mass_positions()
    distributional_value.combine_dirac_deltas()
    boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
        distributional_value.finite_dirac_deltas, 0, False
    )
    cdf_values = generate_cdf_values(boundary_positions, bin_widths, bin_heights)
    return generate_samples(cdf_values, boundary_positions, n_samples)


def _sample_special_values(
    positions: np.ndarray, probabilities: np.ndarray, n_samples: int
) -> np.ndarray:
    """Sample non-finite values (NaN, -Inf, +Inf) according to their masses."""
    cumulative = np.cumsum(probabilities)
    indices = np.searchsorted(cumulative, np.random.uniform(0, 1, size=n_samples))
    indices = np.clip(indices, 0, len(positions) - 1)
    return positions[indices]


def generate_samples(
    cdf_values: np.ndarray, boundary_positions: np.ndarray, n: int
) -> np.ndarray:
    """
    Generate samples via inverse CDF sampling with linear interpolation.

    Draws uniform random values and maps them through the inverse CDF by
    locating the corresponding bin and linearly interpolating within it.

    Args:
        cdf_values: Cumulative distribution function values at each bin boundary.
        boundary_positions: Positions of the bin boundaries.
        n: Number of samples to generate.

    Returns:
        Array of samples drawn from the distribution.
    """
    us = np.random.uniform(0, 1, size=n)
    indices = np.searchsorted(cdf_values, us, side="right") - 1
    indices = np.clip(indices, 0, len(cdf_values) - 2)
    samples = boundary_positions[indices] + (
        boundary_positions[indices + 1] - boundary_positions[indices]
    ) * (us - cdf_values[indices]) / (cdf_values[indices + 1] - cdf_values[indices])
    return np.asarray(samples)


def generate_cdf_values(
    boundary_positions: np.ndarray, bin_widths: np.ndarray, bin_heights: np.ndarray
) -> np.ndarray:
    """
    Compute CDF values at each bin boundary from a binned density histogram.

    The CDF is computed as the cumulative sum of bin areas (width * height),
    starting from zero at the first boundary.

    Args:
        boundary_positions: Positions of the bin boundaries.
        bin_widths: Width of each bin.
        bin_heights: Height (density) of each bin.

    Returns:
        Array of CDF values at each bin boundary, starting at 0.0.
    """
    cdf_values = np.zeros(len(boundary_positions))
    cdf_values[1:] = np.cumsum(bin_widths * bin_heights)
    total_area = cdf_values[-1]
    if total_area > 0.0:
        # Normalize to ensure the CDF spans [0, 1], avoiding sampling issues
        # when drawing u ~ U(0, 1) in generate_samples.
        cdf_values /= total_area
        cdf_values[-1] = 1.0
    return cdf_values
