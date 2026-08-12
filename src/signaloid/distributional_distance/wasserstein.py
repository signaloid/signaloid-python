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

import numpy as np
import numpy.typing as npt

from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance._validators import (
    _require_distributional_pair,
    _validate_wp_inputs,
)


def _wp_1d_weighted_pair(
    u_positions: np.ndarray,
    u_masses: np.ndarray,
    v_positions: np.ndarray,
    v_masses: np.ndarray,
    p: int = 1,
) -> float:
    """1D Wasserstein-p distance between two weighted empirical distributions.

    Pure-numpy replacement for ``scipy.stats.wasserstein_distance`` (p=1)
    and ``ot.wasserstein_1d`` (any p≥1, then take the p-th root).

    Uses the quantile-based formulation
    ``W_p(u, v)^p = ∫_0^1 |F_u^{-1}(t) - F_v^{-1}(t)|^p dt``,
    which is the canonical 1D Wasserstein-p. Note: integrating
    ``∫|F_u(x) - F_v(x)|^p dx`` along x (the CDF formula) only equals
    W_p^p when p = 1. For p ≥ 2 that integral is the energy distance,
    not W_p.

    Args:
        u_positions: Sample positions for distribution u (any order).
        u_masses: Non-negative masses for each u position. May be
            unnormalised — internally divided by their sum.
        v_positions: Sample positions for distribution v (any order).
        v_masses: Non-negative masses for each v position. May be
            unnormalised — internally divided by their sum.
        p: Wasserstein order (1 or 2 are exercised. Any positive
            integer works).

    Returns:
        Wasserstein-p distance between u and v.

    Raises:
        ValueError: see `_validate_wp_inputs`.
    """
    u_values = np.asarray(u_positions, dtype=np.float64)
    v_values = np.asarray(v_positions, dtype=np.float64)
    u_weights = np.asarray(u_masses, dtype=np.float64)
    v_weights = np.asarray(v_masses, dtype=np.float64)

    u_total_mass, v_total_mass = _validate_wp_inputs(
        u_values, u_weights, v_values, v_weights, p
    )

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)
    u_sorted = u_values[u_sorter]
    v_sorted = v_values[v_sorter]

    u_cum = np.cumsum(u_weights[u_sorter]) / u_total_mass
    v_cum = np.cumsum(v_weights[v_sorter]) / v_total_mass
    # Pin endpoints exactly to avoid float-precision overshoot in searchsorted.
    u_cum[-1] = 1.0
    v_cum[-1] = 1.0

    # Merged quantile breakpoints in [0, 1]. 0 prepended so the first
    # interval is covered.
    qs = np.unique(np.concatenate(([0.0], u_cum, v_cum)))

    # F^{-1}(t) is constant on each interval (qs[i], qs[i+1]]. Evaluate
    # at the midpoint and pick the smallest index k with cum[k] ≥ mid.
    mid_t = 0.5 * (qs[:-1] + qs[1:])
    u_idx = np.clip(np.searchsorted(u_cum, mid_t, side="left"), 0, len(u_sorted) - 1)
    v_idx = np.clip(np.searchsorted(v_cum, mid_t, side="left"), 0, len(v_sorted) - 1)

    diff = np.abs(u_sorted[u_idx] - v_sorted[v_idx])
    dt = qs[1:] - qs[:-1]

    if p == 1:
        return float(np.sum(diff * dt))
    return float(np.power(np.sum(np.power(diff, p) * dt), 1.0 / p))


def wasserstein_p_distance(
    u_positions: npt.ArrayLike,
    u_masses: npt.ArrayLike | None,
    v_positions: npt.ArrayLike,
    v_masses: npt.ArrayLike | None,
    *,
    p: int,
) -> float:
    """Wasserstein-p distance between two weighted empirical distributions.

    Public raw-array entry point over `_wp_1d_weighted_pair`, for callers
    that hold `(positions, masses)` arrays rather than
    `DistributionalValue` objects. It is the drop-in replacement for
    ``scipy.stats.wasserstein_distance`` (p=1) and
    ``sqrt(ot.wasserstein_1d(..., p=2))`` (p=2): the p-th root is taken
    internally, so callers do not apply their own ``np.sqrt``.

    Masses may be `None` to request uniform weighting (matching the
    `None`-means-uniform semantics of scipy / POT). Otherwise they are
    treated as non-negative, possibly unnormalised masses.

    Args:
        u_positions: Sample positions for distribution u (any order).
        u_masses: Non-negative masses for each u position, or `None`
            for uniform weighting.
        v_positions: Sample positions for distribution v (any order).
        v_masses: Non-negative masses for each v position, or `None`
            for uniform weighting.
        p: Wasserstein order (any integer >= 1; 1 and 2 are exercised).

    Returns:
        Wasserstein-p distance between u and v.

    Raises:
        ValueError: see `_validate_wp_inputs`.
    """
    u_pos = np.asarray(u_positions, dtype=np.float64)
    v_pos = np.asarray(v_positions, dtype=np.float64)
    u_wts = (
        np.ones_like(u_pos)
        if u_masses is None
        else np.asarray(u_masses, dtype=np.float64)
    )
    v_wts = (
        np.ones_like(v_pos)
        if v_masses is None
        else np.asarray(v_masses, dtype=np.float64)
    )
    return _wp_1d_weighted_pair(u_pos, u_wts, v_pos, v_wts, p=p)


def wasserstein_1_distance(
    u_values: np.ndarray,
    v_values: np.ndarray,
    all_values: np.ndarray,
) -> float:
    """Wasserstein-1 distance between two equally-weighted sample sets.

    Pure-numpy implementation that produces the same output as
    ``scipy.stats.wasserstein_distance`` but requires all input arrays
    to be pre-sorted and does not accept weights — both u and v are
    treated as equal-mass samples.

    Args:
        u_values: Sorted unweighted samples from distribution A.
        v_values: Sorted unweighted samples from distribution B.
        all_values: Sorted concatenation of u_values and v_values.

    Returns:
        Wasserstein-1 distance between distribution A and distribution B.

    Raises:
        ValueError: if u_values or v_values is empty (the empirical CDF
            denominators `len(u_values)` and `len(v_values)` would
            ZeroDivision otherwise).
    """
    if len(u_values) == 0:
        raise ValueError("u_values must be non-empty.")
    if len(v_values) == 0:
        raise ValueError("v_values must be non-empty.")
    deltas = np.diff(all_values)

    u_cdf = np.zeros(len(all_values) - 1)
    v_cdf = np.zeros(len(all_values) - 1)

    u_idx, v_idx = 0, 0
    for i in range(len(all_values) - 1):
        while u_idx < len(u_values) and u_values[u_idx] <= all_values[i]:
            u_idx += 1
        while v_idx < len(v_values) and v_values[v_idx] <= all_values[i]:
            v_idx += 1

        u_cdf[i] = u_idx / len(u_values)
        v_cdf[i] = v_idx / len(v_values)

    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def wasserstein_1_distance_with_weights(
    u_values: np.ndarray,
    v_values: np.ndarray,
    v_cum_weights: np.ndarray,
    all_values: np.ndarray,
) -> float:
    """Wasserstein-1 distance between an unweighted u and a weighted v.

    Args:
        u_values: Sorted unweighted samples from distribution A.
        v_values: Sorted positions of weighted samples from distribution B.
        v_cum_weights: Cumulative sum of the sorted weights of
            distribution B (the natural ``np.cumsum(masses)``). No
            leading zero is required.
        all_values: Sorted concatenation of u_values and v_values.

    Returns:
        Wasserstein-1 distance between distribution A and distribution B.
    """
    if len(u_values) == 0:
        raise ValueError("u_values must be non-empty.")
    if len(v_cum_weights) == 0:
        raise ValueError("v_cum_weights must be non-empty.")
    if len(v_values) != len(v_cum_weights):
        raise ValueError(
            "v_values and v_cum_weights must have the same length "
            f"(got {len(v_values)} and {len(v_cum_weights)})."
        )
    deltas = np.diff(all_values)
    v_total_weight = v_cum_weights[-1]
    if v_total_weight <= 0.0:
        raise ValueError(
            "v_cum_weights[-1] (total mass) must be strictly positive; "
            f"got {v_total_weight}."
        )
    # Prepend 0 so v_cdf_lookup[k] is the mass of the first k v_values —
    # in particular [0]=0 and [len(v)]=total. The original port indexed
    # v_cum_weights[v_idx] directly, which was off by one and crashed
    # when v_idx reached len(v_values). Equivalent to scipy's
    # ``concatenate([0], cumsum)`` pattern.
    v_cdf_lookup = np.concatenate(([0.0], v_cum_weights))

    u_cdf = np.zeros(len(all_values) - 1)
    v_cdf = np.zeros(len(all_values) - 1)

    u_idx, v_idx = 0, 0
    for i in range(len(all_values) - 1):
        while u_idx < len(u_values) and u_values[u_idx] <= all_values[i]:
            u_idx += 1
        u_cdf[i] = u_idx / len(u_values)

        while v_idx < len(v_values) and v_values[v_idx] <= all_values[i]:
            v_idx += 1
        v_cdf[i] = v_cdf_lookup[v_idx] / v_total_weight

    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def wasserstein_p_uxhw_wrapper(
    dist_u: DistributionalValue,
    dist_v: DistributionalValue,
    *,
    p: int,
) -> float:
    """Wasserstein-p distance between two DistributionalValues.

    Args:
        dist_u: First DistributionalValue.
        dist_v: Second DistributionalValue.
        p: Wasserstein order (any integer >= 1). p=1 and p=2 have
            ergonomic shortcuts `wasserstein_1_uxhw_wrapper` and
            `wasserstein_2_uxhw_wrapper`.

    Returns:
        Wasserstein-p distance.

    Raises:
        ValueError: see `_require_distributional_pair` and
            `_validate_wp_inputs` (downstream of `_wp_1d_weighted_pair`).
    """
    _require_distributional_pair(dist_u, dist_v)
    return _wp_1d_weighted_pair(
        u_positions=dist_u.positions,
        u_masses=dist_u.masses,
        v_positions=dist_v.positions,
        v_masses=dist_v.masses,
        p=p,
    )


def wasserstein_1_uxhw_wrapper(
    dist_u: DistributionalValue,
    dist_v: DistributionalValue,
) -> float:
    """W1 between two DistributionalValues.

    Shortcut for `wasserstein_p_uxhw_wrapper(..., p=1)`.

    Args:
        dist_u: First DistributionalValue.
        dist_v: Second DistributionalValue.

    Returns:
        Wasserstein-1 distance.

    Raises:
        ValueError: see `wasserstein_p_uxhw_wrapper`.
    """
    return wasserstein_p_uxhw_wrapper(dist_u, dist_v, p=1)


def wasserstein_2_uxhw_wrapper(
    dist_u: DistributionalValue,
    dist_v: DistributionalValue,
) -> float:
    """W2 between two DistributionalValues.

    Shortcut for `wasserstein_p_uxhw_wrapper(..., p=2)`.

    Args:
        dist_u: First DistributionalValue.
        dist_v: Second DistributionalValue.

    Returns:
        Wasserstein-2 distance.

    Raises:
        ValueError: see `wasserstein_p_uxhw_wrapper`.
    """
    return wasserstein_p_uxhw_wrapper(dist_u, dist_v, p=2)


def _rescale_to_gt_support(
    dist_to_scale: DistributionalValue, reference_dist: DistributionalValue
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale both position arrays into [0, 1] using the ground-truth support.

    Validates non-emptiness explicitly before calling `.min()` / `.max()`
    so callers see a clear ValueError on empty inputs rather than the
    opaque "zero-size array to reduction operation" raised by numpy.
    Position / mass finite + non-negative checks run downstream inside
    `_wp_1d_weighted_pair` via `_validate_wp_inputs`.

    Args:
        dist_to_scale: DistributionalValue whose positions are
            rescaled into the [0, 1] frame defined by `reference_dist`.
        reference_dist: DistributionalValue whose [min, max] support
            defines the rescaling range. Also rescaled (its positions
            land in [0, 1] by construction).

    Returns:
        Tuple (rescaled_dist_to_scale, rescaled_reference_dist).

    Raises:
        ValueError: if either DistributionalValue is empty.
    """
    if len(dist_to_scale.positions) == 0 or len(reference_dist.positions) == 0:
        raise ValueError("u_positions and v_positions must be non-empty.")
    gt_min = reference_dist.positions.min()
    normalizing_factor = reference_dist.positions.max() - gt_min
    if normalizing_factor == 0.0:
        normalizing_factor = 1.0
    return (
        (dist_to_scale.positions - gt_min) / normalizing_factor,
        (reference_dist.positions - gt_min) / normalizing_factor,
    )


def normalized_wasserstein_p_uxhw_wrapper(
    test_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
    *,
    p: int,
) -> float:
    """Wasserstein-p over positions rescaled to the ground-truth support [0, 1].

    Args:
        test_dist: Test DistributionalValue.
        ground_truth_dist: Ground-truth DistributionalValue whose
            support [min, max] defines the rescaling range.
        p: Wasserstein order (any integer >= 1). p=1 and p=2 have
            ergonomic shortcuts `normalized_wasserstein_1_uxhw_wrapper`
            and `normalized_wasserstein_2_uxhw_wrapper`.

    Returns:
        Wasserstein-p distance over rescaled positions.

    Raises:
        ValueError: see `_require_distributional_pair`,
            `_rescale_to_gt_support`, and downstream
            `_validate_wp_inputs`.
    """
    _require_distributional_pair(test_dist, ground_truth_dist)
    u_rescaled, v_rescaled = _rescale_to_gt_support(test_dist, ground_truth_dist)
    return _wp_1d_weighted_pair(
        u_positions=u_rescaled,
        u_masses=test_dist.masses,
        v_positions=v_rescaled,
        v_masses=ground_truth_dist.masses,
        p=p,
    )


def normalized_wasserstein_1_uxhw_wrapper(
    test_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> float:
    """W1 over positions rescaled to the ground-truth support [0, 1].

    Shortcut for `normalized_wasserstein_p_uxhw_wrapper(..., p=1)`.

    Args:
        test_dist: Test DistributionalValue.
        ground_truth_dist: Ground-truth DistributionalValue whose
            support [min, max] defines the rescaling range.

    Returns:
        Wasserstein-1 distance over rescaled positions.

    Raises:
        ValueError: see `normalized_wasserstein_p_uxhw_wrapper`.
    """
    return normalized_wasserstein_p_uxhw_wrapper(test_dist, ground_truth_dist, p=1)


def normalized_wasserstein_2_uxhw_wrapper(
    test_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> float:
    """W2 over positions rescaled to the ground-truth support [0, 1].

    Shortcut for `normalized_wasserstein_p_uxhw_wrapper(..., p=2)`.

    Args:
        test_dist: Test DistributionalValue.
        ground_truth_dist: Ground-truth DistributionalValue whose
            support [min, max] defines the rescaling range.

    Returns:
        Wasserstein-2 distance over rescaled positions.

    Raises:
        ValueError: see `normalized_wasserstein_p_uxhw_wrapper`.
    """
    return normalized_wasserstein_p_uxhw_wrapper(test_dist, ground_truth_dist, p=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m signaloid.distributional_distance.wasserstein",
        description="Compute Wasserstein-p distance between two ux strings.",
    )
    parser.add_argument("dist_u_ux", help="ux string for dist_u")
    parser.add_argument("dist_v_ux", help="ux string for dist_v")
    parser.add_argument("tolerance", type=float)
    parser.add_argument(
        "--p", type=int, default=1, help="Wasserstein order (default: 1)"
    )
    args = parser.parse_args()

    dist_u = DistributionalValue.parse(args.dist_u_ux)
    if dist_u is None:
        raise ValueError(f"Could not parse dist_u from ux string {args.dist_u_ux}")
    dist_v = DistributionalValue.parse(args.dist_v_ux)
    if dist_v is None:
        raise ValueError(f"Could not parse dist_v from ux string {args.dist_v_ux}")

    distance: float = wasserstein_p_uxhw_wrapper(dist_u, dist_v, p=args.p)
    if distance <= args.tolerance:
        print(
            f"[SUCCESS] Wasserstein-{args.p} distance within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    else:
        print(
            f"[FAILURE] Wasserstein-{args.p} distance NOT within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    sys.exit(0 if distance <= args.tolerance else 1)
