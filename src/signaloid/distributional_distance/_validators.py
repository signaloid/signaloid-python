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


def _require_distributional_pair(
    dist_u: DistributionalValue, dist_v: DistributionalValue
) -> None:
    """Raise ValueError if either argument is not a DistributionalValue.

    Shared by the uxhw wrappers across the package so they all surface
    the same error type and message format at the public-API boundary.
    """
    if not isinstance(dist_u, DistributionalValue):
        raise ValueError(
            f"dist_u is of type {type(dist_u).__name__} and not an "
            "instance of DistributionalValue."
        )
    if not isinstance(dist_v, DistributionalValue):
        raise ValueError(
            f"dist_v is of type {type(dist_v).__name__} and "
            "not an instance of DistributionalValue."
        )


def _validate_wp_inputs(
    u_values: np.ndarray,
    u_weights: np.ndarray,
    v_values: np.ndarray,
    v_weights: np.ndarray,
    p: int,
) -> tuple[float, float]:
    """Validate inputs to `_wp_1d_weighted_pair` and return (u_total, v_total).

    `DistributionalValue` does not itself enforce these invariants
    (special values like NaN/Inf are first-class via nan_dirac_delta
    etc.), so we validate at the algorithmic boundary instead of
    duplicating the check in every uxhw wrapper.
    """
    # `bool` is a subclass of `int` in Python; reject it explicitly so
    # `p=True` isn't silently treated as `p=1`.
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError("p must be an integer greater than or equal to 1.")
    if (
        u_values.ndim != 1
        or v_values.ndim != 1
        or u_weights.ndim != 1
        or v_weights.ndim != 1
    ):
        raise ValueError(
            "u_positions, v_positions, u_masses, and v_masses must be "
            "one-dimensional arrays."
        )
    if len(u_values) == 0 or len(v_values) == 0:
        raise ValueError("u_positions and v_positions must be non-empty.")
    if len(u_values) != len(u_weights) or len(v_values) != len(v_weights):
        raise ValueError("positions and masses arrays must have matching lengths.")
    if not (
        np.all(np.isfinite(u_values))
        and np.all(np.isfinite(v_values))
        and np.all(np.isfinite(u_weights))
        and np.all(np.isfinite(v_weights))
    ):
        raise ValueError("positions and masses must contain only finite values.")
    if np.any(u_weights < 0.0) or np.any(v_weights < 0.0):
        raise ValueError("u_masses and v_masses must be non-negative.")

    u_total = float(np.sum(u_weights))
    v_total = float(np.sum(v_weights))
    if u_total <= 0.0 or v_total <= 0.0:
        raise ValueError(
            "u_masses and v_masses must each have a strictly positive total mass."
        )
    return u_total, v_total
