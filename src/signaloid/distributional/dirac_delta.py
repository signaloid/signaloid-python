#   Copyright (c) 2025, Signaloid.
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

import sys
import math

if sys.implementation.name != 'circuitpython':
    from typing import Optional

    # Use numpy for accelerated computing
    import numpy as np
else:
    # Use the extended version of ulab's numpy when running on CircuitPython
    from signaloid.circuitpython.extended_ulab_numpy import np  # type: ignore[no-redef]

# Use this multiplier to convert to/from floating point to fixed point masses
FIXED_POINT_ONE = 0x8000000000000000


class DiracDelta:
    def __init__(
        self,
        position: float,
        raw_mass: Optional[int] = None,
        mass: Optional[float] = None
    ) -> None:
        """Initializes the Dirac Delta, given a position, and a mass in either
        a fixed-point or a floating point representation. If both representations
        are given, the fixed-point one will be used, and the floating-point one
        will be ignored.

        :param position: The Dirac Delta position.
        :type position: float
        :param raw_mass: The Dirac Delta mass in 64bit fixed-point representation,
        defaults to None.
        :type raw_mass: Optional[int]
        :param mass: The Dirac Delta mass in floating-point representation,
        defaults to None.
        :type mass: Optional[float]
        """
        self.position: float = position
        self._raw_mass: int = 0
        self._mass: float = 0

        if raw_mass is not None:
            self.raw_mass = raw_mass
        elif mass is not None:
            self.mass = mass

    @property
    def raw_mass(self) -> int:
        """The Dirac Delta mass in 64bit fixed-point representation.

        :return: The mass
        :rtype: int
        """
        return self._raw_mass

    @raw_mass.setter
    def raw_mass(self, value: int) -> None:
        """Sets the Dirac Delta mass given a 64bit fixed-point mass.

        :param value: The 64bit fixed-point mass to use.
        :type value: int
        """
        self._raw_mass = value

        # The probability mass is a fixed-point format with FIXED_POINT_ONE
        # representing 1.0. Dividing by FIXED_POINT_ONE gets the float it
        # represents.
        self._mass = value / FIXED_POINT_ONE

    @property
    def mass(self) -> float:
        """The Dirac Delta mass in floating-point representation.

        :return: The mass
        :rtype: float
        """
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        """Sets the Dirac Delta mass given a floating-point mass.

        :param value: The floating-point mass to use.
        :type value: float
        """
        self._mass = value

        if np.isnan(self._mass):
            self._raw_mass = 0
        else:
            # The probability mass is a fixed-point format with FIXED_POINT_ONE
            # representing 1.0. Multiplying by FIXED_POINT_ONE gets the fixed point
            # it represents.
            self._raw_mass = int(value * FIXED_POINT_ONE)

    def __add__(self, other: "DiracDelta") -> "DiracDelta":
        """Adds two Dirac Delta generating a new one, combining the two positions,
        and masses.

        :param other: The second Dirac Delta to use for adding.
        :type other: DiracDelta
        :return: A new Dirac Delta, with combined position, and mass.
        :rtype: DiracDelta
        """
        combined_mass: float = self.mass + other.mass
        combined_position: float = (
            self.position * self.mass
            + other.position * other.mass
        ) / combined_mass

        return DiracDelta(position=combined_position, mass=combined_mass)

    def __lt__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is less than the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is less than the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        return self.position < other.position

    def __le__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is less than or equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is less than or equal to the given
        Dirac Delta, `False` otherwise.
        :rtype: bool
        """
        return self.position <= other.position

    def __eq__(self, other: object) -> bool:
        """Checks if this Dirac Delta is equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is equal to the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        if not isinstance(other, DiracDelta):
            return NotImplemented
        return self.position == other.position

    def __ne__(self, other: object) -> bool:
        """Checks if this Dirac Delta is not equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is not equal to the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        if not isinstance(other, DiracDelta):
            return NotImplemented
        return self.position != other.position

    def __ge__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is greater than the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is greater than the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        return self.position >= other.position

    def __gt__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is greater than or equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is greater than or equal to the given
        Dirac Delta, `False` otherwise.
        :rtype: bool
        """
        return self.position > other.position

    def __str__(self) -> str:
        """Creates a string representation of the Dirac Delta.

        :return: The string representation.
        :rtype: str
        """
        return f"[{self.position}, {self.mass}]"

    def similar(
        self: "DiracDelta",
        other: "DiracDelta",
        threshold: float,
    ) -> bool:
        """Checks if this Dirac Delta and the given one are similar to each other,
        based on the given threshold.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :param threshold: The threshold to use for the comparison
        :type threshold: float
        :return: `True` if similar, `False` otherwise.
        :rtype: bool
        """
        return abs(self.position - other.position) <= threshold

    @property
    def isFinite(self) -> bool:
        """Checks if the Dirac Delta is has a finite position.

        :return: `True` is the Dirac Delta position is finite, `False` otherwise.
        :rtype: bool
        """
        return math.isfinite(self.position)
