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
Shared representation-health guard for traced UxHw distributions.

Traced representations can blow up into non-finite or absurd-magnitude
positions, which the ``signaloid.distributional_distance`` wrappers reject.
Callers use :func:`_representation_blow_up_reason` to detect a genuine blow-up
and surface it as the worst-case distance instead of crashing.
"""

import numpy as np

from signaloid.distributional.distributional import DistributionalValue

# Positions above `sqrt(float max)` (~1.34e154) overflow float64 when the
# variance squares them, so that is the blow-up cutoff: corruption (~1e300) is
# caught while legitimate large values (finance ~1e15) pass. A float-derived
# bound avoids the false flags an arbitrary cutoff (e.g. 1e10) would cause.
BLOW_UP_POSITION_MAGNITUDE = float(np.sqrt(np.finfo(np.float64).max))

# Max fraction of total mass allowed at an overflow-scale position before the
# representation counts as a blow-up. Below it the mass is a benign remnant. At
# or above it the distribution is genuinely corrupt.
BLOW_UP_MASS_FRACTION = 1e-6


def _representation_blow_up_reason(
    dv: DistributionalValue,
    check_magnitude: bool = True,
) -> str | None:
    """
    Decide whether a distribution (after ``drop_zero_mass_positions``) is a
    genuine representation blow-up rather than a benign one that merely carried
    zero-mass special-value slots.

    Two modes: (1) a non-finite position still carries mass (always checked),
    and (2) a non-trivial mass fraction sits at an overflow-scale magnitude
    (``|position| > BLOW_UP_POSITION_MAGNITUDE``). Mode 2 is gated on
    ``check_magnitude`` because a large-but-finite scalar (e.g. ~1e15) is not
    corruption.

    Args:
        dv: The distribution to inspect. Callers should drop zero-mass deltas
            first so benign special-value slots do not trip mode 1.
        check_magnitude: Apply the overflow-magnitude check (mode 2). ``True``
            for distribution outputs. ``False`` for scalar outputs, where a
            large finite value is legitimate.

    Returns:
        A human-readable reason string when ``dv`` is a genuine blow-up, else
        ``None``.
    """
    # Mode 1: a non-finite position still carrying mass. `is_finite` is
    # tri-state. Only the explicit `False` is a blow-up. `None` (no deltas /
    # indeterminate) deliberately falls through. Always runs.
    if dv.is_finite is False:
        return (
            "non-finite position carries non-zero mass after dropping zero-mass deltas"
        )

    # Mode 2: a meaningful mass fraction parked at an absurd magnitude. Gated on
    # `check_magnitude` so a large-but-finite scalar output is not mis-flagged.
    if check_magnitude:
        positions = np.asarray(dv.positions, dtype=np.float64)
        masses = np.asarray(dv.masses, dtype=np.float64)

        total_mass = float(masses.sum())
        if total_mass > 0:
            absurd = np.abs(positions) > BLOW_UP_POSITION_MAGNITUDE
            absurd_fraction = float(masses[absurd].sum()) / total_mass
            if absurd_fraction > BLOW_UP_MASS_FRACTION:
                return (
                    f"{absurd_fraction:.3g} of mass sits at |position| > "
                    f"{BLOW_UP_POSITION_MAGNITUDE:.0e}"
                )

    return None
