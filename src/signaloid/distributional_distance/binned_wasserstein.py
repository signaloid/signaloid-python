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

import argparse
import sys
import warnings

import numpy as np

from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from .wasserstein import (
    wasserstein_1_uxhw_wrapper,
)
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance._validators import (
    _require_distributional_pair,
)


def integrate_trapezoid(x0: float, x1: float, y0: float, y1: float) -> float:
    """Area of a trapezoid / triangle / rectangle defined by two heights.

    Args:
        x0: left position
        x1: right position
        y0: left height
        y1: right height

    Returns:
        Area of the trapezoid / triangle / rectangle.
    """
    return 0.5 * (y0 + y1) * (x1 - x0)


def normalized_uxhw_cdf_heights(
    bin_widths: np.ndarray, bin_heights: np.ndarray
) -> np.ndarray:
    """CDF heights for the binned representation of a UxHw distribution.

    Returns the cumulative bin areas (heights * widths) after
    normalising the total mass to 1 if it has drifted from 1 by more
    than 1e-12.

    Args:
        bin_widths: Width of each bin.
        bin_heights: PDF height of each bin.

    Returns:
        Cumulative bin areas (length = len(bin_widths)).
    """
    single_bin_areas: np.ndarray = bin_heights * bin_widths

    uxhw_cdf_total_mass: float = np.sum(single_bin_areas)
    if uxhw_cdf_total_mass <= 0.0:
        raise ValueError(
            "Binned UxHw distribution has non-positive total mass "
            f"({uxhw_cdf_total_mass}); cannot normalise."
        )
    if abs(uxhw_cdf_total_mass - 1.0) >= 1e-12:
        warnings.warn(
            "Binned uxhw distribution was not normalized. Normalizing cdf.",
            stacklevel=2,
        )
        single_bin_areas = single_bin_areas / uxhw_cdf_total_mass

    return np.cumsum(single_bin_areas)


def normalized_sample_cdf_heights(
    sample_positions: np.ndarray, sample_weights: np.ndarray
) -> np.ndarray:
    """Cumulative heights of the empirical (step-wise) CDF.

    Normalises the total weight to 1 if it has drifted from 1 by more
    than 1e-12.

    Args:
        sample_positions: Sorted array of sample positions. Currently
            unused — kept for symmetry with `normalized_uxhw_cdf_heights`
            and to document the calling contract (positions and weights
            must already be sorted by position so that `cumsum(weights)`
            gives the empirical CDF heights).
        sample_weights: Weight for each sample (must be aligned with
            `sample_positions`).

    Returns:
        Cumulative weights, length = len(sample_weights).
    """
    single_step_areas: np.ndarray = sample_weights

    sample_cdf_total_mass: float = np.sum(single_step_areas)
    if sample_cdf_total_mass <= 0.0:
        raise ValueError(
            "Empirical sample distribution has non-positive total mass "
            f"({sample_cdf_total_mass}); cannot normalise."
        )
    if abs(sample_cdf_total_mass - 1.0) >= 1e-12:
        warnings.warn(
            "Sample empirical distribution was not normalized. Normalizing cdf.",
            stacklevel=2,
        )
        single_step_areas = single_step_areas / sample_cdf_total_mass

    return np.cumsum(single_step_areas)


def build_position_and_type_array(
    uxhw_cdf_boundaries: np.ndarray,
    sample_cdf_positions: np.ndarray,
    positions: np.ndarray,
    types: np.ndarray,
    num_boundaries: int,
    num_samples: int,
) -> None:
    """Build a sorted union of UxHw boundaries and sample positions.

    Each entry in `positions` is tagged in `types` with its origin:
    `0` for a UxHw bin boundary, `1` for a sample position. The two
    output arrays are filled in place and then sorted in tandem by
    position.

    Args:
        uxhw_cdf_boundaries: Boundaries of the UxHw distribution binning.
        sample_cdf_positions: Sample positions.
        positions: Output array (length num_boundaries + num_samples)
            to be filled with all points and then sorted in place.
        types: Output array (same length) tagging each position's origin
            (0 = boundary, 1 = sample), kept aligned with `positions`.
        num_boundaries: len(uxhw_cdf_boundaries).
        num_samples: len(sample_cdf_positions).
    """
    positions[:num_boundaries] = uxhw_cdf_boundaries
    types[:num_boundaries] = 0
    positions[num_boundaries:] = sample_cdf_positions
    types[num_boundaries:] = 1

    # Sort positions and types in tandem. Use lexsort with `types` as
    # the secondary key so that at tied positions (boundary coincident
    # with a sample) boundaries (type=0) come before samples (type=1)
    # — `wasserstein_1_core`'s segment/step bookkeeping assumes this
    # ordering. `np.argsort(positions)` alone would not be stable on
    # all numpy versions/platforms.
    indices = np.lexsort((types, positions))
    positions[:] = positions[indices]
    types[:] = types[indices]


def fill_cdf_values(
    uxhw_cdf_vals: np.ndarray,
    uxhw_cdf_idx: np.ndarray,
    uxhw_cdf_boundaries: np.ndarray,
    uxhw_cdf_heights: np.ndarray,
    uxhw_cdf_widths: np.ndarray,
    sample_cdf_vals: np.ndarray,
    sample_cdf_idx: np.ndarray,
    sample_cdf_heights: np.ndarray,
    positions: np.ndarray,
    num_points_total: int,
    num_boundaries: int,
) -> None:
    """Evaluate the UxHw and empirical CDFs at every merged position.

    Fills `uxhw_cdf_vals` and `sample_cdf_vals` in place. For the UxHw
    CDF, each `position[i]` lies within bin segment `uxhw_cdf_idx[i]`;
    the value is linearly interpolated using the segment's slope, with
    clamps below the first / above the last boundary. For the empirical
    CDF, each `position[i]` lies after step `sample_cdf_idx[i]`; the
    value is the cumulative weight at that step (left as 0.0 when the
    point precedes the first sample, courtesy of the caller's `zeros`
    initialisation of `sample_cdf_vals`).

    Args:
        uxhw_cdf_vals: Output — UxHw CDF values at each merged position.
        uxhw_cdf_idx: Per-position UxHw segment index.
        uxhw_cdf_boundaries: UxHw bin boundaries (length N+1).
        uxhw_cdf_heights: UxHw cumulative CDF heights (length N+1,
            starting at 0).
        uxhw_cdf_widths: UxHw bin widths (length N).
        sample_cdf_vals: Output — empirical CDF values. Caller must
            pre-initialise to zeros; entries with `sample_cdf_idx[i] == 0`
            are left untouched.
        sample_cdf_idx: Per-position sample step index.
        sample_cdf_heights: Empirical cumulative CDF heights.
        positions: Merged + sorted positions array.
        num_points_total: Length of `positions`.
        num_boundaries: Length of `uxhw_cdf_boundaries`.
    """
    for i in range(num_points_total):
        if sample_cdf_idx[i] > 0:
            sample_cdf_vals[i] = sample_cdf_heights[sample_cdf_idx[i]]

        idx = uxhw_cdf_idx[i]
        if idx < 0:
            uxhw_cdf_vals[i] = 0.0
        elif idx >= num_boundaries - 1:
            uxhw_cdf_vals[i] = 1.0
        else:
            uxhw_cdf_slope = (
                uxhw_cdf_heights[idx + 1] - uxhw_cdf_heights[idx]
            ) / uxhw_cdf_widths[idx]
            uxhw_cdf_vals[i] = uxhw_cdf_heights[idx] + uxhw_cdf_slope * (
                positions[i] - uxhw_cdf_boundaries[idx]
            )


def wasserstein_1_core(
    uxhw_cdf_boundaries: np.ndarray,
    uxhw_cdf_widths: np.ndarray,
    uxhw_cdf_heights: np.ndarray,
    sample_cdf_positions: np.ndarray,
    sample_cdf_heights: np.ndarray,
) -> float:
    """Wasserstein-1 distance between a binned UxHw CDF and an empirical
    sample CDF.

    Caller-validates contract: this is the inner algorithmic core and
    does not validate its inputs. The only intended caller is
    `wasserstein_1_between_distribution_and_samples`, which validates
    upstream via `_validate_binned_lengths` and
    `_validate_binned_semantics`. Callers must ensure shapes are
    length-consistent and values are finite.

    Args:
        uxhw_cdf_boundaries: Sorted bin boundaries for the UxHw CDF
            (length N+1).
        uxhw_cdf_widths: Bin widths (length N).
        uxhw_cdf_heights: Cumulative CDF heights at each boundary
            (length N+1, starting at 0).
        sample_cdf_positions: Sorted positions of the empirical samples
            (length M).
        sample_cdf_heights: Cumulative empirical-CDF heights at each
            sample position (length M+1, starting at 0).

    Returns:
        Wasserstein-1 distance between the two CDFs.
    """
    # Fit all positions in a single array that will be sorted afterwards
    # by build_position_and_type_array.
    num_boundaries = len(uxhw_cdf_boundaries)
    num_samples = len(sample_cdf_positions)
    num_points_total = num_boundaries + num_samples

    positions = np.empty(num_points_total, dtype=np.float64)
    types = np.empty(num_points_total, dtype=np.int8)

    build_position_and_type_array(
        uxhw_cdf_boundaries,
        sample_cdf_positions,
        positions,
        types,
        num_boundaries,
        num_samples,
    )

    # `uxhw_cdf_idx[j]` is the linear segment of the UxHw CDF that
    # position[j] lives in; `sample_cdf_idx[j]` is the constant segment
    # of the sample empirical CDF.
    uxhw_cdf_idx = np.empty(num_points_total, dtype=np.int32)
    sample_cdf_idx = np.empty(num_points_total, dtype=np.int32)
    current_seg = -1
    current_step = 0

    for j in range(num_points_total):
        if types[j] == 0:
            current_seg += 1
        else:
            current_step += 1

        uxhw_cdf_idx[j] = current_seg
        sample_cdf_idx[j] = current_step

    # Now that we know the type for each point, fill in the CDF values.
    uxhw_cdf_vals = np.empty(num_points_total, dtype=np.float64)
    sample_cdf_vals = np.zeros(num_points_total, dtype=np.float64)

    fill_cdf_values(
        uxhw_cdf_vals,
        uxhw_cdf_idx,
        uxhw_cdf_boundaries,
        uxhw_cdf_heights,
        uxhw_cdf_widths,
        sample_cdf_vals,
        sample_cdf_idx,
        sample_cdf_heights,
        positions,
        num_points_total,
        num_boundaries,
    )

    # Compute the trapezoid / triangle / rectangle segment sizes.
    # delta_left and delta_right are the relative CDF heights at the
    # left and right of each trapezoid; left_pos / right_pos are the
    # corresponding x-positions. We use `sample_cdf_vals[:-1]` for
    # delta_right because the step-wise CDF is right-continuous — we
    # compare to the left (previous) CDF height.
    delta_left = uxhw_cdf_vals[:-1] - sample_cdf_vals[:-1]
    delta_right = uxhw_cdf_vals[1:] - sample_cdf_vals[:-1]
    left_pos = positions[:-1]
    right_pos = positions[1:]

    total_distance = np.zeros(num_points_total - 1, dtype=np.float64)
    for i in range(num_points_total - 1):
        y0 = delta_left[i]
        y1 = delta_right[i]
        x0 = left_pos[i]
        x1 = right_pos[i]

        if y0 * y1 < 0.0:
            mid = x0 + y0 / (y0 - y1) * (x1 - x0)
            total_distance[i] = integrate_trapezoid(x0, mid, abs(y0), 0.0)
            total_distance[i] += integrate_trapezoid(mid, x1, 0.0, abs(y1))
        else:
            total_distance[i] = integrate_trapezoid(x0, x1, abs(y0), abs(y1))

    return float(np.sum(total_distance))


def _validate_binned_lengths(
    bin_boundaries: list[float],
    bin_heights: list[float],
    bin_widths: list[float],
    sample_positions: list[float],
    sample_weights: list[float] | None,
) -> None:
    """Structural length / non-empty checks for the binned wrapper."""
    if len(bin_widths) == 0:
        raise ValueError("bin_widths must be non-empty.")
    if len(bin_heights) != len(bin_widths):
        raise ValueError("bin_widths and bin_heights must have the same length.")
    if len(bin_boundaries) != len(bin_widths) + 1:
        raise ValueError(
            "bin_boundaries must contain exactly one more entry than bin_widths."
        )
    if len(sample_positions) == 0:
        raise ValueError("sample_positions must not be empty.")
    if sample_weights is not None and len(sample_weights) != len(sample_positions):
        raise ValueError(
            "sample_positions and sample_weights must have the same length."
        )


def _validate_binned_semantics(
    bin_boundaries: list[float],
    bin_heights: list[float],
    bin_widths: list[float],
    sample_positions: list[float],
    sample_weights: list[float] | None,
) -> None:
    """Finite + non-negativity / positivity checks on the raw inputs.

    Checks raw inputs (not post-cumsum CDF arrays). The cumulative
    arrays can hide negative entries — e.g. weights=[0.5, -0.1, 0.6]
    cumsums to [0.5, 0.4, 1.0], all non-negative, so a downstream
    `np.any(cdf < 0)` check would miss the negative.
    """
    bin_boundaries_arr = np.asarray(bin_boundaries, dtype=np.float64)
    bin_heights_arr = np.asarray(bin_heights, dtype=np.float64)
    bin_widths_arr = np.asarray(bin_widths, dtype=np.float64)
    finite_checks = (
        ("bin_boundaries", bin_boundaries_arr),
        ("bin_heights", bin_heights_arr),
        ("bin_widths", bin_widths_arr),
        ("sample_positions", np.asarray(sample_positions, dtype=np.float64)),
    )
    for name, arr in finite_checks:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")

    if np.any(bin_widths_arr <= 0.0):
        raise ValueError("All bin_widths must be strictly positive.")
    if np.any(bin_heights_arr < 0.0) or not np.any(bin_heights_arr > 0.0):
        raise ValueError(
            "bin_heights must be non-negative with at least one "
            "strictly positive entry."
        )

    # `wasserstein_1_core` assumes bin_boundaries is strictly
    # increasing and that bin_widths[i] == boundaries[i+1] - boundaries[i].
    # Non-monotonic boundaries would produce negative segment lengths
    # and silently-wrong distances; width/boundary disagreement breaks
    # the CDF-merge integration.
    boundary_gaps = np.diff(bin_boundaries_arr)
    if np.any(boundary_gaps <= 0.0):
        raise ValueError("bin_boundaries must be strictly increasing.")
    if not np.allclose(boundary_gaps, bin_widths_arr, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "bin_widths must match the implied widths "
            "np.diff(bin_boundaries) within tolerance."
        )

    if sample_weights is None:
        return
    sample_weights_arr = np.asarray(sample_weights, dtype=np.float64)
    if not np.all(np.isfinite(sample_weights_arr)):
        raise ValueError("sample_weights must contain only finite values.")
    if np.any(sample_weights_arr < 0.0) or not np.any(sample_weights_arr > 0.0):
        raise ValueError(
            "sample_weights must be non-negative with at least one "
            "strictly positive entry."
        )


def _validate_binned_inputs(
    bin_boundaries: list[float],
    bin_heights: list[float],
    bin_widths: list[float],
    sample_positions: list[float],
    sample_weights: list[float] | None,
) -> None:
    """Full validation for `wasserstein_1_between_distribution_and_samples`."""
    _validate_binned_lengths(
        bin_boundaries, bin_heights, bin_widths, sample_positions, sample_weights
    )
    _validate_binned_semantics(
        bin_boundaries, bin_heights, bin_widths, sample_positions, sample_weights
    )


def wasserstein_1_between_distribution_and_samples(
    bin_boundaries: list[float],
    bin_heights: list[float],
    bin_widths: list[float],
    sample_positions: list[float],
    sample_weights: list[float] | None = None,
) -> float:
    """Wasserstein-1 distance between a binned distribution and samples.

    Wraps the core CDF-merge algorithm: validates the raw inputs,
    builds normalised cumulative CDFs for both sides, sorts the sample
    side by position so the empirical CDF is well-defined, and delegates
    to `wasserstein_1_core`.

    Args:
        bin_boundaries: Distribution bin boundaries, length N+1, sorted.
        bin_heights: PDF heights of each bin (length N, non-negative).
        bin_widths: Width of each bin (length N, strictly positive).
        sample_positions: Sample positions, any order.
        sample_weights: Optional sample weights aligned with
            `sample_positions`. Defaults to uniform `1 / len(samples)`.

    Returns:
        Wasserstein-1 distance between the binned distribution and the
        empirical sample distribution.

    Raises:
        ValueError: On any structural mismatch (lengths, empty inputs),
            non-finite values, non-positive widths, or zero total mass
            on either side. See `_validate_binned_lengths` and
            `_validate_binned_semantics` for the full check list.
    """
    _validate_binned_inputs(
        bin_boundaries, bin_heights, bin_widths, sample_positions, sample_weights
    )

    if sample_weights is None:
        sample_weights = [1.0 / len(sample_positions)] * len(sample_positions)

    # Convert to np arrays.
    uxhw_cdf_boundaries: np.ndarray = np.array(bin_boundaries, dtype=np.float64)
    uxhw_cdf_widths: np.ndarray = np.array(bin_widths, dtype=np.float64)
    uxhw_cdf_heights: np.ndarray = normalized_uxhw_cdf_heights(
        uxhw_cdf_widths, np.array(bin_heights, dtype=np.float64)
    )
    uxhw_cdf_heights = np.append([0.0], uxhw_cdf_heights)

    # The empirical CDF is cumulative-in-position-order, so positions
    # (and their matching weights) must be sorted by position before
    # cumsum. Callers may pass any ordering; we sort here.
    sample_cdf_positions: np.ndarray = np.array(sample_positions, dtype=np.float64)
    sample_cdf_weights: np.ndarray = np.array(sample_weights, dtype=np.float64)
    sample_sort_order: np.ndarray = np.argsort(sample_cdf_positions)
    sample_cdf_positions = sample_cdf_positions[sample_sort_order]
    sample_cdf_weights = sample_cdf_weights[sample_sort_order]
    sample_cdf_heights: np.ndarray = normalized_sample_cdf_heights(
        sample_cdf_positions, sample_cdf_weights
    )
    sample_cdf_heights = np.append([0.0], sample_cdf_heights)

    return wasserstein_1_core(
        uxhw_cdf_boundaries,
        uxhw_cdf_widths,
        uxhw_cdf_heights,
        sample_cdf_positions,
        sample_cdf_heights,
    )


def binned_wasserstein_1_uxhw_wrapper(
    binned_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> float:
    """Binned Wasserstein-1 distance between a UxHw distribution and a
    ground-truth distribution.

    Bins the UxHw distribution via `PlotData` (PDF binning, then TTR
    rebinning), then compares the resulting bins to the ground truth's
    weighted samples. Short-circuits to the non-binned W1 wrapper when
    `binned_dist` collapses to a single Dirac delta (binning is ill-defined
    for a single point).

    Args:
        binned_dist: Distribution under test. Sorted in place as a side effect.
        ground_truth_dist: Reference distribution treated as weighted samples
            (positions + masses).

    Returns:
        Binned Wasserstein-1 distance.

    Raises:
        ValueError: If either argument is not a `DistributionalValue`,
            or if `PlotData` fails to resolve a `plotting_ttr_order`
            for the input UxHw distribution.
    """
    _require_distributional_pair(binned_dist, ground_truth_dist)

    # Binning is ill-defined for distributions with ≤1 *finite* Dirac
    # delta — delegate to the non-binned W1 wrapper. Checking
    # `finite_dirac_deltas` (rather than `positions`) is important
    # because special-value Diracs (NaN / ±Inf) with non-zero mass
    # appear in `positions` after sorting but cannot be binned.
    # `finite_dirac_deltas` internally calls `binned_dist.sort()` as a side
    # effect (mutates the caller's distribution in place).
    if len(binned_dist.finite_dirac_deltas) <= 1:
        return wasserstein_1_uxhw_wrapper(dist_u=binned_dist, dist_v=ground_truth_dist)

    # Use PlotData to compute the binning (binned_dist is already sorted via
    # the finite_dirac_deltas access above).
    plot_interface = PlotData(binned_dist)

    boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
        binned_dist.finite_dirac_deltas, 0, False
    )
    if plot_interface.plotting_ttr_order is None:
        raise ValueError(
            "PlotData failed to resolve plotting_ttr_order for the input "
            "UxHw distribution (got None); cannot compute binned "
            "Wasserstein-1."
        )

    # Find the TTR of the created binning; this is always a valid TTR.
    ttr = PlotData.bin_pdf_to_ttr(
        boundary_positions,
        bin_widths,
        bin_heights,
        plot_interface.plotting_ttr_order,
    )
    # Rebuild the binning from the valid TTR via the TTR binning method.
    bin_boundaries, bin_widths, bin_heights = PlotData.create_binning(
        ttr, plot_interface.plotting_ttr_order, True
    )

    return wasserstein_1_between_distribution_and_samples(
        bin_boundaries=list(bin_boundaries),
        bin_heights=list(bin_heights),
        bin_widths=list(bin_widths),
        sample_positions=list(ground_truth_dist.positions),
        sample_weights=list(ground_truth_dist.masses),
    )


def binned_wasserstein_1_ux_string_wrapper(
    distribution_ux: str, ground_truth_ux: str
) -> float:
    """Binned Wasserstein-1 distance from Ux-string inputs.

    Parses both arguments into `DistributionalValue` instances and
    delegates to `binned_wasserstein_1_uxhw_wrapper`.

    Args:
        distribution_ux: Ux-string for the distribution under test.
        ground_truth_ux: Ux-string for the reference distribution.

    Returns:
        Binned Wasserstein-1 distance.

    Raises:
        ValueError: If either Ux-string fails to parse, or if the
            downstream binned-W1 computation rejects its inputs.
    """
    distribution = DistributionalValue.parse(distribution_ux)
    if distribution is None:
        raise ValueError(
            f"Could not parse distribution from ux string {distribution_ux}"
        )

    ground_truth = DistributionalValue.parse(ground_truth_ux)
    if ground_truth is None:
        raise ValueError(
            f"Could not parse ground truth from ux string {ground_truth_ux}"
        )

    return binned_wasserstein_1_uxhw_wrapper(distribution, ground_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m signaloid.distributional_distance.binned_wasserstein",
        description=(
            "Compute binned Wasserstein-1 distance between two ux strings. "
            "Asymmetric: the first argument is binned, the second is "
            "treated as raw weighted samples."
        ),
    )
    parser.add_argument("distribution_ux", help="ux string to be binned")
    parser.add_argument("ground_truth_ux", help="ux string treated as weighted samples")
    parser.add_argument("tolerance", type=float)
    args = parser.parse_args()

    distance: float = binned_wasserstein_1_ux_string_wrapper(
        args.distribution_ux, args.ground_truth_ux
    )
    if distance <= args.tolerance:
        print(
            f"[SUCCESS] Binned Wasserstein-1 distance within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    else:
        print(
            f"[FAILURE] Binned Wasserstein-1 distance NOT within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    sys.exit(0 if distance <= args.tolerance else 1)
