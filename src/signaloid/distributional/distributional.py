#   Copyright (c) 2024, Signaloid.
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

import math
import sys
import re
import struct

if sys.implementation.name != 'circuitpython':
    from typing import Optional, Union

    # Use numpy for accelerated computing
    import numpy as np
    from numpy.typing import NDArray
else:
    # Use the extended version of ulab's numpy when running on CircuitPython
    from signaloid.circuitpython.extended_ulab_numpy import np  # type: ignore[no-redef]

from signaloid.distributional.dirac_delta import DiracDelta

# The format strings used with struct.pack & struct.unpack for parsing and
# dumping `DistributionalValue` data.
STRUCT_FORMATS: dict[str, dict[str, str]] = {
    "str": {
        "particle": "",
        "UR_type": ">B",
        "sample_count": ">Q",
        "mean": ">d",
        "UR_order": ">I",
        "position_single": ">f",
        "position_double": ">d",
        "mass": ">Q",
    },
    "bytes": {
        "particle": "<d",
        "UR_type": "<B",
        "sample_count": "<Q",
        "mean": "<d",
        "UR_order": "<I",
        "position_single": "<f",
        "position_double": "<d",
        "mass": "<Q",
    }
}


class DistributionalValue:
    def __init__(
        self,
        particle_value: Optional[float] = None,
        UR_type: Optional[int] = None,
        dirac_deltas: Optional[list[DiracDelta]] = None,
        double_precision: bool = True
    ) -> None:
        self.particle_value: Optional[float] = particle_value
        self.UR_type: Optional[int] = UR_type
        self._dirac_deltas: list[DiracDelta] = dirac_deltas if dirac_deltas is not None else []
        self.double_precision = double_precision
        """
        properties
        """
        self._mean: Optional[float] = None
        self._variance: Optional[float] = None

        self.nan_dirac_delta: DiracDelta = DiracDelta(position=np.nan, mass=np.nan)
        self.neg_inf_dirac_delta: DiracDelta = DiracDelta(position=-np.inf, mass=np.nan)
        self.pos_inf_dirac_delta: DiracDelta = DiracDelta(position=np.inf, mass=np.nan)

        self._has_no_zero_mass: Optional[bool] = None
        self._is_finite: Optional[bool] = None
        self._is_sorted: Optional[bool] = None
        self._is_cured: Optional[bool] = None
        self._is_full_valid_TTR: Optional[bool] = None

    @property
    def dirac_deltas(self) -> list[DiracDelta]:
        """The list of all the Dirac Deltas of the DistributionalValue.

        :return: The list of the Dirac Deltas.
        :rtype: list[DiracDelta]
        """
        return self._dirac_deltas

    @property
    def finite_dirac_deltas(self) -> list[DiracDelta]:
        """The list of all the finite Dirac Deltas of the DistributionalValue.

        :return: The list of the finite Dirac Deltas.
        :rtype: list[DiracDelta]
        """
        self.sort()
        if self.has_special_values:
            return self.dirac_deltas[:-3]

        return self.dirac_deltas

    @property
    def positions(self) -> NDArray[np.float_]:
        """The list af all the Dirac Delta positions.

        :return: The list of all the Dirac Delta positions.
        :rtype: NDArray[np.float_]
        """
        arr: NDArray[np.float_] = np.array([dd.position for dd in self.dirac_deltas], dtype=np.float_)
        return arr

    @property
    def masses(self) -> NDArray[np.float_]:
        """The list af all the Dirac Delta floating-point masses.

        :return: The list of all the Dirac Delta floating-point masses.
        :rtype: NDArray[np.float_]
        """
        arr: NDArray[np.float_] = np.array([dd.mass for dd in self.dirac_deltas], dtype=np.float_)
        return arr

    @property
    def raw_masses(self) -> NDArray[np.uint64]:
        """The list af all the Dirac Delta fixed-point masses.

        :return: The list of all the Dirac Delta fixed-point masses.
        :rtype: NDArray[np.uint64]
        """
        arr: NDArray[np.uint64] = np.array([dd.raw_mass for dd in self.dirac_deltas], dtype=np.uint64)
        return arr

    @property
    def mean(self) -> Optional[float]:
        """The mean position of all the Dirac Deltas.

        :return: The mean position of all the Dirac Deltas.
        :rtype: Optional[float]
        """
        if self._mean is None:
            self.calculate_mean()

        return self._mean

    @mean.setter
    def mean(self, mean: Optional[float]):
        """Sets the mean position of all the Dirac Deltas explicitly (it does
        not interfere with the Dirac Deltas list).

        :param mean: The mean value to use.
        :type mean: Optional[float]
        """
        self._mean = mean

    def calculate_mean(self) -> Optional[float]:
        """Calculated the mean position of all the Dirac Deltas based on the
        Dirac Deltas list.

        :return: The calculated mean position of all the Dirac Deltas.
        :rtype: float
        """
        if self.nan_dirac_delta.mass > 0:
            self._mean = np.nan
        elif self.neg_inf_dirac_delta.mass > 0 and self.pos_inf_dirac_delta.mass > 0:
            self._mean = np.nan
        elif self.neg_inf_dirac_delta.mass > 0:
            self._mean = -np.inf
        elif self.pos_inf_dirac_delta.mass > 0:
            self._mean = np.inf
        else:
            total_mass: float = 0
            total_weighted_position: float = 0
            for dd in self.dirac_deltas:
                total_weighted_position += dd.position * dd.mass
                total_mass += dd.mass

            self._mean = total_weighted_position / total_mass

        return self._mean

    @property
    def UR_order(self) -> int:
        """The number of non-zero mass Dirac Deltas.

        :return: The number of non-zero mass Dirac Deltas.
        :rtype: int
        """
        return len(self.dirac_deltas)

    @property
    def variance(self) -> Optional[float]:
        """The positional variance of all the Dirac Deltas.

        :return: The positional variance of all the Dirac Deltas.
        :rtype: Optional[float]
        """
        if self._variance is None:
            self.calculate_variance()

        return self._variance

    def calculate_variance(self) -> Optional[float]:
        """Calculates the positional variance of all the Dirac Deltas.

        :return: The calculated positional variance of all the Dirac Deltas.
        :rtype: Optional[float]
        """
        # Calculate weighted sample variance
        if (
            self.mean is None
            or self.UR_order == 0
            or not np.isfinite(self.mean)
        ):
            self._variance = None
        else:
            total_mass: float = 0
            total_weighted_squared_diffs: float = 0
            for dd in self.dirac_deltas:
                total_weighted_squared_diffs += (
                    ((dd.position - self.mean) ** 2)
                    * dd.mass
                )
                total_mass += dd.mass

            self._variance = total_weighted_squared_diffs / total_mass

        return self._variance

    @property
    def has_special_values(self) -> bool:
        """Property that identifies if there are non-zero mass Dirac Deltas,
        with non-finite position, i.e. `NaN`, `-Inf`, `Inf`.

        :return: `True` if there are non-zero mass, non-finite position Dirac Deltas,
        `False` otherwise.
        :rtype: bool
        """
        return bool(
            self.nan_dirac_delta.mass > 0
            or self.neg_inf_dirac_delta.mass > 0
            or self.pos_inf_dirac_delta.mass > 0
        )

    def __repr__(self) -> str:
        """Constructs the representation type for the `DistributionalValue`.

        :return: The representation type for the `DistributionalValue`.
        :rtype: str
        """
        return f"{id(self)}:{self.UR_type}-{self.UR_order}"

    def __str__(self) -> str:
        """Constructs the Ux-string representation of the `DistributionalValue`.

        :return: The `DistributionalValue` Ux-string.
        :rtype: str
        """
        result = self.export(to_str=True)
        return result if isinstance(result, str) else ""

    def __bytes__(self) -> bytes:
        """Constructs the byte array for the `DistributionalValue`. Uses either
        single or double precision for support positions based on `self.double_precision`.

        :return: The byte array for the `DistributionalValue`.
        :rtype: bytes
        """

        result = self.export(to_str=False)
        return result if isinstance(result, bytes) else bytes()

    def export(self, to_str: bool = True) -> Union[str, bytes]:
        """
        Constructs the Ux string/Ux bytes with particle value for the `DistributionalValue`.

        Args:
            to_str: Weather to export a Ux string or a Ux bytes
                    - True: Export a Ux string
                    - False: Export a Ux bytes

        Returns:
            The Ux string or the Ux bytes with particle value for the `DistributionalValue`.

        Ux-string format specification:
            - Particle value (double in string format)
            - "Ux"                                              (   2 chars)
            - Representation type (uint8_t)                     (   2 chars)
            - Number of samples (uint64_t)                      (  16 chars) (unused)
            - Mean value of distribution (double)               (  16 chars)
            - Number of non-zero mass Dirac deltas (uint32_t)   (   8 chars)
            - Pairs of:
                - Support position (float/double)               (8/16 chars)
                - Probability mass (uint64_t)                   (  16 chars)

        Ux-bytes specification:
            - Particle value (double)                           (  8 bytes)
            - Representation type (uint8_t)                     (  1 byte )
            - Number of samples (uint64_t)                      (  8 bytes) (unused)
            - Mean value of distribution (double)               (  8 bytes)
            - Number of non-zero mass Dirac deltas (uint32_t)   (  4 bytes)
            - Pairs of:
                - Support position (float/double)               (4/8 bytes)
                - Probability mass (uint64_t)                   (  8 bytes)
        """

        # Create byte representation
        buffer: bytes = bytes()
        UxString: str = ""

        if to_str:
            # Particle value (double in string format)
            UxString += str(self.particle_value) if self.particle_value is not None else ""
            UxString += "Ux"

            fmt = STRUCT_FORMATS["str"]
        else:
            # Particle value (double)                           (8 bytes)
            particle_value = self.particle_value if self.particle_value is not None else 0
            fmt = STRUCT_FORMATS["bytes"]
            buffer += struct.pack(fmt["particle"], particle_value)

        # Representation type (uint8_t)                     (1 byte)
        buffer += struct.pack(fmt["UR_type"], self.UR_type)

        # Number of samples (uint64_t)                      (8 bytes)
        buffer += struct.pack(fmt["sample_count"], self.UR_order)

        # Mean value of distribution (double)               (8 bytes)
        # Mean is always double precision regardless of self.double_precision
        buffer += struct.pack(fmt["mean"], self.mean)

        # Number of non-zero mass Dirac deltas (uint32_t)   (4 bytes)
        buffer += struct.pack(fmt["UR_order"], self.UR_order)

        # Choose the format based on double_precision flag
        position_format = fmt["position_double" if self.double_precision else "position_single"]

        # Pairs of:
        # - Support position (double or float)              (8 or 4 bytes)
        # - Probability mass (uint64_t)                     (8 bytes)
        for dd in self.dirac_deltas:
            # Pack the position using either double or float precision
            buffer += struct.pack(position_format, dd.position)

            # Probability mass is always uint64_t
            buffer += struct.pack(fmt["mass"], dd.raw_mass)

        if to_str:
            return UxString + buffer.hex().upper()

        return buffer

    @staticmethod
    def parse(
        dist: Union[str, bytes],
        double_precision: bool = True
    ) -> "Optional[DistributionalValue]":
        """
        Constructs a `DistributionalValue` after parsing an input that can be
        a UxString or a byte array.

        Args:
            dist: The input UxString or byte array.
            double_precision: The floating point representation precision.
                - true: double precision
                - false: single precision
        Returns:
            The constructed `DistributionalValue` or None if parsing fails.

        Ux-string format specification:
            - Particle value (double in string format)
            - "Ux"                                              (   2 chars)
            - Representation type (uint8_t)                     (   2 chars)
            - Number of samples (uint64_t)                      (  16 chars) (unused)
            - Mean value of distribution (double)               (  16 chars)
            - Number of non-zero mass Dirac deltas (uint32_t)   (   8 chars)
            - Pairs of:
                - Support position (float/double)               (8/16 chars)
                - Probability mass (uint64_t)                   (  16 chars)

        Ux-bytes specification:
            - Particle value (double)                           (  8 bytes)
            - Representation type (uint8_t)                     (  1 byte )
            - Number of samples (uint64_t)                      (  8 bytes) (unused)
            - Mean value of distribution (double)               (  8 bytes)
            - Number of non-zero mass Dirac deltas (uint32_t)   (  4 bytes)
            - Pairs of:
                - Support position (float/double)               (4/8 bytes)
                - Probability mass (uint64_t)                   (  8 bytes)
        """

        if not dist:
            return None

        buffer: bytes = bytes()
        offset = 0
        dist_value = DistributionalValue(double_precision=double_precision)

        if isinstance(dist, str):
            # Parse Hex string
            fmt = STRUCT_FORMATS["str"]

            # Match an optional floating-point or integer number, followed by 'Ux',
            # and then hexadecimal characters
            if sys.implementation.name == 'circuitpython':
                parts = dist.split("Ux")
                index = 0
                if len(parts) == 2:
                    dist_value.particle_value = float(parts[index])
                    index += 1

                buffer = bytes.fromhex(parts[index])
            else:
                pattern = r"^([-+]?\d*\.?\d+(?:e[-+]\d+)?|nan|[-+]?inf)?Ux([0-9A-Fa-f]+)$"
                match = re.match(pattern, dist)
                if not match:
                    return None

                dist_value.particle_value = float(match.group(1)) if match.group(1) else None

                buffer = bytes.fromhex(match.group(2))
        elif isinstance(dist, (bytes, bytearray)):
            # Parse byte array
            fmt = STRUCT_FORMATS["bytes"]

            buffer = dist
            dist_value.particle_value = struct.unpack(fmt["particle"], buffer[offset:offset + 8])[0]
            offset += 8
        else:
            print("Error: ", type(dist))
            return None

        # Check if the buffer has the minimum required length (in bytes)
        # Minimum length = 1 (repr) + 8 (samples) + 8 (mean) + 4 (count) = 21
        min_length = offset + 21
        if len(buffer) < min_length:
            return None

        dist_value.UR_type = struct.unpack(fmt["UR_type"], buffer[offset:offset + 1])[0]
        offset += 1

        # Not used, uncomment if needed
        # number_of_samples = struct.unpack(fmt["sample_count"], buffer[offset:offset + 8])[0]
        offset += 8

        dist_value.mean = struct.unpack(fmt["mean"], buffer[offset:offset + 8])[0]
        offset += 8

        UR_order = struct.unpack(fmt["UR_order"], buffer[offset:offset + 4])[0]
        offset += 4

        # Validate UR_order - reasonable upper limit to prevent processing
        # extremely large inputs that might be malicious
        if (
            UR_order is None
            or UR_order < 0
            or UR_order > 10000
        ):
            return None

        # Calculate expected length based on UR_order
        # 4 or 8 for position + 8 for mass
        bytes_per_position = 8 if double_precision else 4
        bytes_per_dirac_delta = bytes_per_position + 8
        expected_length = min_length + (UR_order * bytes_per_dirac_delta)
        if len(buffer) < expected_length:
            return None

        position_format = fmt["position_double" if double_precision else "position_single"]

        for _ in range(UR_order):
            support_position_bytes = buffer[offset:offset + bytes_per_position]
            offset += bytes_per_position
            position = struct.unpack(position_format, support_position_bytes)[0]

            mass_bytes = buffer[offset:offset + 8]
            offset += 8
            raw_mass = struct.unpack(fmt["mass"], mass_bytes)[0]

            dist_value.dirac_deltas.append(DiracDelta(position, raw_mass=raw_mass))

        return dist_value

    def mean_distance(self, other: "DistributionalValue") -> float:
        """
        Calculates the distance between the mean values of this `DistributionalValue`
        and another `DistributionalValue`.

        Args:
            other: The other `DistributionalValue`.
        Returns:
            The distance between the mean values.
        """

        if self.mean is None or other.mean is None:
            return math.nan

        return abs(self.mean - other.mean)

    def mean_relative_diff(self, other: "DistributionalValue") -> float:
        """
        Calculates the relative difference between the mean values of
        this `DistributionalValue` and another `DistributionalValue`.

        Args:
            other: The other `DistributionalValue`.
        Returns:
            The relative difference between the mean values (normalized by
            the mean value of the other `DistributionalValue`).
        """
        if self.mean is None or other.mean is None:
            return math.nan

        return abs((self.mean - other.mean) / other.mean)

    @property
    def has_no_zero_mass(self) -> Optional[bool]:
        """
        The property that no Dirac delta of the `DistributionalValue` has a zero mass.

        Returns:
            `True` if `DistributionalValue` has a zero mass Dirac delta, `False` else.
        """
        if self._has_no_zero_mass is None:
            self._has_no_zero_mass = self.check_has_no_zero_mass()

        return self._has_no_zero_mass

    def check_has_no_zero_mass(self) -> Optional[bool]:
        """
        Checks the property that no Dirac delta of the `DistributionalValue` has a zero
        mass.

        Returns:
            `True` if `DistributionalValue` has a zero mass Dirac delta, `False` else.
        """
        if self.UR_order == 0:
            return None

        for dd in self.dirac_deltas:
            if dd.mass == 0:
                return False

        return True

    def drop_zero_mass_positions(self) -> None:
        """
        Drops (removes) the Dirac deltas of the `DistributionalValue` that has a zero
        mass.
        """
        if self._has_no_zero_mass:
            return

        if self.UR_order == 0:
            return

        self._dirac_deltas = [dd for dd in self._dirac_deltas if dd.mass > 0]
        self._has_no_zero_mass = True

    @property
    def is_finite(self) -> Optional[bool]:
        """
        The property that all Dirac deltas of the `DistributionalValue` have finite
        positions, i.e., no NaN, -Inf, or Inf values.

        Returns:
            `True` if `DistributionalValue` has a Dirac delta with non-finite position,
            `False` else.
        """
        if self._is_finite is None:
            self._is_finite = self.check_is_finite()

        return self._is_finite

    def check_is_finite(self) -> Optional[bool]:
        """
        Checks the property that all Dirac deltas of the `DistributionalValue` have
        finite positions, i.e., no NaN, -Inf, or Inf values.

        Returns:
            `None` if `DistributionalValue` has no Dirac deltas. Else, `True` if
            `DistributionalValue` has a Dirac delta with non-finite position, `False` else.
        """
        if self.UR_order == 0:
            return None

        for dd in self.dirac_deltas:
            if not dd.isFinite:
                return False

        return True

    @property
    def is_sorted(self) -> Optional[bool]:
        """
        The property that the Dirac deltas of the `DistributionalValue` are sorted
        according to their positions. The NaN, -Inf, and Inf positional values
        are cured and sorted to the end in the order [NaN, -Inf, Inf].

        Returns:
            `True` if the Dirac deltas of the `DistributionalValue` are sorted according
            to their positions, `False` else.
        """
        if self._is_sorted is None:
            self._is_sorted = self.check_is_sorted()

        return self._is_sorted

    def check_is_sorted(self) -> Optional[bool]:
        """
        Checks The property that the Dirac deltas of the `DistributionalValue` are
        sorted according to their positions. The NaN, -Inf, and Inf positional values
        are cured and sorted to the end in the order [NaN, -Inf, Inf].

        Returns:
            `None` if the `DistributionalValue` has no Dirac deltas. Else, `True` if
            the Dirac deltas of the `DistributionalValue` are sorted according to
            their positions, `False` else.
        """
        if self.UR_order == 0:
            return None

        if self.UR_order == 1:
            return True

        for i in range(0, self.UR_order - 1):
            if not self.dirac_deltas[i].isFinite:
                return False

            if self.dirac_deltas[i] > self.dirac_deltas[i + 1]:
                return False

        return True

    def sort(self) -> None:
        """
        Sorts the positions and the masses of a `DistributionalValue` according to
        the positions. Also, cures multiple entries for NaN, -Inf, and Inf and
        places them at the end of positions and masses with the order [NaN, -Inf, Inf].
        """
        if self._is_sorted:
            return

        self.nan_dirac_delta.mass = 0
        self.neg_inf_dirac_delta.mass = 0
        self.pos_inf_dirac_delta.mass = 0
        finite_dirac_deltas: list[DiracDelta] = []
        for dd in self.dirac_deltas:
            if np.isfinite(dd.position):
                finite_dirac_deltas.append(dd)
            elif np.isnan(dd.position):
                self.nan_dirac_delta.mass += dd.mass
            elif np.isneginf(dd.position):
                self.neg_inf_dirac_delta.mass += dd.mass
            elif np.isposinf(dd.position):
                self.pos_inf_dirac_delta.mass += dd.mass

        self._dirac_deltas = finite_dirac_deltas
        self._dirac_deltas.sort()

        if self.has_special_values:
            self.dirac_deltas.append(self.nan_dirac_delta)
            self.dirac_deltas.append(self.neg_inf_dirac_delta)
            self.dirac_deltas.append(self.pos_inf_dirac_delta)

        self._is_sorted = True

    @property
    def is_cured(self) -> Optional[bool]:
        """
        The property that no two Dirac deltas of the `DistributionalValue` have
        the same positional value, including NaN, -Inf, and Inf.

        Returns:
            `True` if no two Dirac deltas of the `DistributionalValue` have the same
            positional value, including NaN, -Inf, and Inf, `False` else.
        """
        if self._is_cured is None:
            self._is_cured = self.check_is_cured()

        return self._is_cured

    def check_is_cured(self) -> Optional[bool]:
        """
        Checks the property that no two Dirac deltas of the `DistributionalValue` have
        the same positional value, including NaN, -Inf, and Inf.

        Returns:
            `None` if the `DistributionalValue` has no Dirac deltas. Else, `True` if
            no two Dirac deltas of the `DistributionalValue` have the same positional
            value, including NaN, -Inf, and Inf, `False` else.
        """

        if self.UR_order == 0:
            return None

        return self.UR_order == len(set(self.positions))

    def cure(self) -> None:
        """
        Cures the positions and masses of the `DistributionalVariable` from multiple
        entries of the same positional value, including NaN, -Inf, and Inf.
        """
        if self._is_cured:
            return

        self.combine_dirac_deltas(0, 0)

    def combine_dirac_deltas(
        self,
        relative_mean_threshold: float = 1e-14,
        relative_range_threshold: float = 1e-12,
    ) -> None:
        """
        Combine Dirac deltas with same, very-close-relative-to-range,
        very-close-relative-to-mean-value positions.

        :param relative_mean_threshold: The threshold multiplier for the relative
        mean.
        :type relative_mean_threshold: float
        :param relative_range_threshold: The threshold multiplier for the relative
        range.
        :type relative_range_threshold: float
        """
        self.sort()

        if len(self.finite_dirac_deltas) <= 1:
            self._is_cured = True
            return

        threshold: float = 0.0
        if relative_mean_threshold > 0 and relative_range_threshold > 0:
            # Calculate the mean value of finite part
            finite_mass = 0.0
            finite_mean = 0.0

            # Last three positions are for non-finite values
            for dd in self.finite_dirac_deltas:
                finite_mass += dd.mass
                finite_mean += dd.position * dd.mass
            finite_mean /= finite_mass
            mean_threshold = finite_mean * relative_mean_threshold

            range_threshold = (
                (self.finite_dirac_deltas[-1].position - self.dirac_deltas[0].position)
                * relative_range_threshold
            )

            threshold = max(mean_threshold, range_threshold)

        dirac_deltas = [self.dirac_deltas[0]]
        for dd in self.finite_dirac_deltas[1:]:
            if dirac_deltas[-1].similar(dd, threshold):
                dirac_deltas[-1] += dd
                continue

            dirac_deltas.append(DiracDelta(dd.position, raw_mass=dd.raw_mass))

        self._dirac_deltas = dirac_deltas

        if self.has_special_values:
            self.dirac_deltas.append(self.nan_dirac_delta)
            self.dirac_deltas.append(self.neg_inf_dirac_delta)
            self.dirac_deltas.append(self.pos_inf_dirac_delta)

        self._is_cured = True

    @property
    def is_full_valid_TTR(self) -> Optional[bool]:
        """
        The property that the Dirac deltas of the `DistributionalValue` form
        a full valid TTR. "Full" means that there are 2^n Dirac deltas (after
        dropping zero mass Dirac deltas and curing to combine same position
        Dirac deltas). "Valid" means that there is a distribution whose TTR
        exactly contains the Dirac deltas of the `DistributionalValue`.

        Returns:
            `True` if the Dirac deltas of the `DistributionalValue` form a full
            and valid TTR, `False` else.
        """
        if self._is_full_valid_TTR is None:
            self._is_full_valid_TTR = self.check_is_full_valid_TTR()

        return self._is_full_valid_TTR

    def check_is_full_valid_TTR(self) -> Optional[bool]:
        """
        Checks the property that the Dirac deltas of the `DistributionalValue`
        form a full and valid TTR. "Full" means that there are 2^n Dirac deltas
        (after dropping zero mass Dirac deltas and curing to combine same position
        Dirac deltas). "Valid" means that there is a distribution whose TTR
        exactly contains the Dirac deltas of the `DistributionalValue`.

        Returns:
            `None` if the `DistributionalValue` has no Dirac deltas. Else, `True` if
            the Dirac deltas of the `DistributionalValue` form a full and valid TTR,
            `False` else.
        """

        self.drop_zero_mass_positions()
        self.cure()

        if not self.is_finite:
            return False

        if self.UR_order == 0:
            return None

        if self.UR_order == 1:
            return True

        # Check UR_order is power of 2
        if self.UR_order & (self.UR_order - 1) != 0:
            return False

        ttr_order = int(np.log2(self.UR_order))
        number_of_boundaries = 2 * self.UR_order - 1
        boundary_positions = np.array([np.nan] * number_of_boundaries)
        boundary_probabilities = np.array([np.nan] * number_of_boundaries)
        boundary_positions[::2] = self.positions
        boundary_probabilities[::2] = self.masses

        for n in range(ttr_order):
            step = 2**n
            for i in range(2 ** (n + 1) - 1, number_of_boundaries, 2 ** (n + 2)):
                boundary_probabilities[i] = (
                    boundary_probabilities[i - step] + boundary_probabilities[i + step]
                )
                boundary_positions[i] = (
                    boundary_probabilities[i - step] * boundary_positions[i - step]
                    + boundary_probabilities[i + step] * boundary_positions[i + step]
                ) / boundary_probabilities[i]

        return bool(np.all(boundary_positions[:-1] < boundary_positions[1:]))
