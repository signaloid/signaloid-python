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

from signaloid.distributional.distributional import DistributionalValue


def _validate_scalar_inputs(
    test_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> None:
    """Validate inputs for scalar_distance_wrapper."""
    for name, dv in (
        ("test_dist", test_dist),
        ("ground_truth_dist", ground_truth_dist),
    ):
        if not isinstance(dv, DistributionalValue):
            raise ValueError(
                f"{name} must be a DistributionalValue; " f"got {type(dv).__name__}."
            )
        if len(dv.positions) == 0:
            raise ValueError(f"{name} must contain at least one Dirac delta.")
        if len(dv.positions) > 1:
            raise ValueError(
                f"{name} must be a scalar distribution (exactly one Dirac "
                f"delta); got {len(dv.positions)}."
            )
        if not np.isfinite(dv.positions[0]):
            raise ValueError(
                f"{name}.positions[0] must be finite; got {dv.positions[0]}."
            )
        mass = dv.masses[0]
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                f"{name}.masses[0] must be finite and strictly "
                f"positive; got {mass}."
            )


def relative_error_uxhw_wrapper(
    test_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> float:
    """Relative error between two scalar (single-Dirac) distributions.

    Computes ``|test_dist[0] - ground_truth_dist[0]| / |ground_truth_dist[0]|``.
    The function is intended for scalar comparisons.
    Each input must hold exactly one finite Dirac delta.
    Non-scalar inputs are rejected to avoid silently using only ``positions[0]``.

    Args:
        test_dist: The single-Dirac distribution produced by the UxHw
            configuration.
        ground_truth_dist: The single-Dirac reference distribution to
            compare against.

    Returns:
        The relative error. When
        ``ground_truth_dist.positions[0] == 0`` and
        ``test_dist.positions[0] != 0`` the result is ``+inf``. When
        both are zero the result is ``NaN`` (the 0/0 indeterminate
        form). The numpy RuntimeWarning for both cases is suppressed.

    Raises:
        ValueError: if either input is not a DistributionalValue, is
            empty, holds more than one Dirac delta, contains a
            non-finite position, or carries a non-finite or
            non-positive mass.
    """
    _validate_scalar_inputs(test_dist, ground_truth_dist)

    # ground_truth.positions[0] == 0 is allowed: callers see +inf.
    # Suppress numpy's RuntimeWarning so it doesn't leak through.
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.abs(
            test_dist.positions[0] - ground_truth_dist.positions[0]
        ) / np.abs(ground_truth_dist.positions[0])
    return float(distance)


def signed_error_uxhw_wrapper(
    test_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> float:
    """Signed error between two scalar (single-Dirac) distributions.

    Computes ``test_dist[0] - ground_truth_dist[0]``. Preserves sign.
    Same units as the inputs. The function is intended for scalar
    comparisons. Each input must hold exactly one finite Dirac delta.
    Non-scalar inputs are rejected to avoid silently using only
    ``positions[0]``.

    Args:
        test_dist: The single-Dirac distribution produced by the UxHw
            configuration.
        ground_truth_dist: The single-Dirac reference distribution to
            compare against.

    Returns:
        The signed error. Finite for any finite inputs. May be
        positive, negative, or zero.

    Raises:
        ValueError: if either input is not a DistributionalValue, is
            empty, holds more than one Dirac delta, contains a
            non-finite position, or carries a non-finite or
            non-positive mass.
    """
    _validate_scalar_inputs(test_dist, ground_truth_dist)

    return float(test_dist.positions[0] - ground_truth_dist.positions[0])


def absolute_error_uxhw_wrapper(
    test_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> float:
    """Absolute error between two scalar (single-Dirac) distributions.

    Computes ``|test_dist[0] - ground_truth_dist[0]|``. Magnitude only.
    Same units as the inputs. The function is intended for scalar
    comparisons. Each input must hold exactly one finite Dirac delta.
    Non-scalar inputs are rejected to avoid silently using only
    ``positions[0]``.

    Args:
        test_dist: The single-Dirac distribution produced by the UxHw
            configuration.
        ground_truth_dist: The single-Dirac reference distribution to
            compare against.

    Returns:
        The absolute error. Non-negative and finite for any finite
        inputs.

    Raises:
        ValueError: if either input is not a DistributionalValue, is
            empty, holds more than one Dirac delta, contains a
            non-finite position, or carries a non-finite or
            non-positive mass.
    """
    return float(np.abs(signed_error_uxhw_wrapper(test_dist, ground_truth_dist)))


if __name__ == "__main__":
    METRICS = {
        "relative": relative_error_uxhw_wrapper,
        "absolute": absolute_error_uxhw_wrapper,
        "signed": signed_error_uxhw_wrapper,
    }

    parser = argparse.ArgumentParser(
        prog="python -m signaloid.distributional_distance.scalar",
        description=(
            "Compute scalar error between two ux strings (single-Dirac " "inputs)."
        ),
    )
    parser.add_argument("test_dist_ux", help="ux string for test_dist")
    parser.add_argument("ground_truth_dist_ux", help="ux string for ground_truth_dist")
    parser.add_argument("tolerance", type=float)
    parser.add_argument(
        "--metric",
        choices=list(METRICS),
        default="relative",
        help="error metric (default: relative)",
    )
    args = parser.parse_args()

    test_dist = DistributionalValue.parse(args.test_dist_ux)
    if test_dist is None:
        raise ValueError(
            f"Could not parse test_dist from ux string {args.test_dist_ux}"
        )
    ground_truth_dist = DistributionalValue.parse(args.ground_truth_dist_ux)
    if ground_truth_dist is None:
        raise ValueError(
            f"Could not parse ground_truth_dist from ux string "
            f"{args.ground_truth_dist_ux}"
        )

    distance: float = METRICS[args.metric](test_dist, ground_truth_dist)
    # signed_error can be negative. Compare |distance| to tolerance so the
    # SUCCESS / FAILURE semantics are the same magnitude check across all
    # three metrics. abs() is a no-op for the non-negative ones.
    within = abs(distance) <= args.tolerance
    label = f"{args.metric.capitalize()} error"
    if within:
        print(
            f"[SUCCESS] {label} within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    else:
        print(
            f"[FAILURE] {label} NOT within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    sys.exit(0 if within else 1)
