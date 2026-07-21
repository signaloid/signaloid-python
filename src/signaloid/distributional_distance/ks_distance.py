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


import numpy as np

from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance._validators import (
    _require_distributional_pair,
    _validate_wp_inputs,
)


def _ks_distance(
    u_positions: np.ndarray,
    u_masses: np.ndarray,
    v_positions: np.ndarray,
    v_masses: np.ndarray,
) -> float:
    """Sup-norm distance between two empirical step-function CDFs.

    The maximum can only occur at one of the merged sample positions
    (where at least one of the two CDFs has a jump), so we evaluate
    both CDFs on the union of positions and take the max-abs-diff.

    Right-continuous step CDF: ``F(x) = Σ_i m_i · 1[x_i ≤ x]``.

    Args:
        u_positions: Sample positions for distribution u (any order).
        u_masses: Non-negative masses for u; unnormalised/raw values allowed.
        v_positions: Sample positions for distribution v (any order).
        v_masses: Non-negative masses for v; unnormalised/raw values allowed.

    Returns:
        KS distance in [0, 1].

    Raises:
        ValueError: see `_validate_wp_inputs` (shared validator).
    """
    u_values = np.asarray(u_positions, dtype=np.float64)
    v_values = np.asarray(v_positions, dtype=np.float64)
    u_weights = np.asarray(u_masses, dtype=np.float64)
    v_weights = np.asarray(v_masses, dtype=np.float64)

    # KS has no `p` parameter; pass p=1 purely to satisfy the shared
    # validator (which guards against bool/<1). The validator's checks
    # on shapes, finiteness, non-negative masses, and positive totals
    # are independent of `p`.
    u_total, v_total = _validate_wp_inputs(
        u_values, u_weights, v_values, v_weights, p=1
    )

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)
    u_sorted = u_values[u_sorter]
    v_sorted = v_values[v_sorter]

    u_cum = np.cumsum(u_weights[u_sorter]) / u_total
    v_cum = np.cumsum(v_weights[v_sorter]) / v_total
    # Pin endpoints exactly to avoid float-precision overshoot.
    u_cum[-1] = 1.0
    v_cum[-1] = 1.0

    # Evaluate F_u and F_v on the union of positions. For a right-
    # continuous step CDF, F(x) is the cumulative mass over all jumps
    # at positions ≤ x, so `searchsorted(side="right")` returns the
    # count of sorted positions ≤ x — exactly the index we want into a
    # 0-prepended cumulative-mass table (index 0 → 0, index k → first
    # k masses). This naturally handles positions below the support
    # (index 0 → F=0) without a separate clip.
    joint = np.sort(np.concatenate([u_sorted, v_sorted]))
    u_lookup = np.concatenate(([0.0], u_cum))
    v_lookup = np.concatenate(([0.0], v_cum))
    u_idx = np.searchsorted(u_sorted, joint, side="right")
    v_idx = np.searchsorted(v_sorted, joint, side="right")

    return float(np.max(np.abs(u_lookup[u_idx] - v_lookup[v_idx])))


def kolmogorov_smirnov_distance_uxhw_wrapper(
    dist_u: DistributionalValue,
    dist_v: DistributionalValue,
) -> float:
    """Kolmogorov-Smirnov distance between two DistributionalValues."""
    _require_distributional_pair(dist_u, dist_v)
    return _ks_distance(
        u_positions=dist_u.positions,
        u_masses=dist_u.masses,
        v_positions=dist_v.positions,
        v_masses=dist_v.masses,
    )
