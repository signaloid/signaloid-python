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

import re
import struct
from typing import List, Optional, Union, Tuple
import numpy as np
from numpy.typing import NDArray


class DistributionalValue:
    def __init__(self) -> None:
        self.positions: NDArray[np.float_] = np.array([], dtype=np.float_)
        self.masses: NDArray[np.float_] = np.array([], dtype=np.float_)
        self.raw_masses: List[int] = []
        self.adjusted_masses: NDArray[np.float_] = np.array([], dtype=np.float_)
        self.widths: NDArray[np.float_] = np.array([], dtype=np.float_)
        self.mean: Union[None, float] = None
        self.particle_value: Union[None, float] = None
        self.variance: Union[None, float] = None
        self.particle = False
        self.UR_type: Union[None, int, str] = None
        self.UR_order: Union[None, int] = None
        """
        properties
        """
        self._has_no_zero_mass: Union[None, bool] = None
        self._is_finite: Union[None, bool] = None
        self._is_sorted: Union[None, bool] = None
        self._is_cured: Union[None, bool] = None
        self._is_full_valid_TTR: Union[None, bool] = None

    def __str__(self):
        """
        Constructs the Ux string with particle value for the `DistributionalValue`.

        Returns:
            UxString: The Ux string with particle value for the `DistributionalValue`.
        """
        particleStr = ""
        if self.particle_value is not None:
            particleStr = str(self.particle_value)
        return particleStr + self._UxString()

    def _UxString(self):
        """
        Constructs the distributional part of the Ux string value for the `DistributionalValue`.

        Returns:
            UxString: The Ux string for the `DistributionalValue`.
        """
        UxString = "Ux"

        # Representation type (uint8_t)                     (1 byte)
        UxString += struct.pack("B", self.UR_type).hex().upper()

        # Number of samples (uint64_t)                      (8 bytes)
        UxString += struct.pack(">Q", len(self.positions)).hex().upper()

        # Mean value of distribution (double)               (8 bytes)
        UxString += struct.pack(">1d", self.mean).hex().upper()

        # Number of non-zero mass Dirac deltas (uint32_t)   (4 bytes)
        UxString += struct.pack(">I", self.UR_order).hex().upper()

        # Pairs of:
        # - Support position (double)                         (8 bytes)
        # - Probability mass (uint64_t)                       (8 bytes)
        for i in range(len(self.positions)):
            UxString += struct.pack(">1d", self.positions[i]).hex().upper()
            UxString += struct.pack(">Q", self.raw_masses[i]).hex().upper()

        return UxString

    def __repr__(self):
        """
        Constructs the representation type for the `DistributionalValue`.

        Returns:
            The representation type for the `DistributionalValue`.
        """
        return f"{id(self)}:{self.UR_type}-{self.UR_order}"

    @staticmethod
    def parse(dist: Union[str, bytes]) -> Optional["DistributionalValue"]:
        """
        Constructs a `DistributionalValue` after parsing an input that can be
        a UxString or a byte array.

        Args:
            dist: The input UxString or byte array.
        Returns:
            The constructed `DistributionalValue`.
        """
        # Parse Hex string
        if isinstance(dist, str):
            return DistributionalValue._parse_ux_string(dist)
        # Parse byte array
        elif isinstance(dist, (bytes, bytearray)):
            return DistributionalValue._parse_bytes(dist)
        else:
            print("Error: ", type(dist))
            return None

    @staticmethod
    def _parse_ux_string(text: str) -> Optional["DistributionalValue"]:
        """
        Constructs a `DistributionalValue` after parsing an input UxString.

        Here is the specification of the format:
            - "Ux"                                              ( 2 characters)
            - Representation type (uint8_t)                     ( 2 characters)
            - Number of samples (uint64_t)                      (16 characters) (unused)
            - Mean value of distribution (double)               (16 characters)
            - Number of non-zero mass Dirac deltas (uint32_t)   ( 8 characters)
            Pairs of:
            - Support position (double)                         (16 characters)
            - Probability mass (uint64_t)                       (16 characters)

        Args:
            The input UxString.
        Returns:
            The constructed `DistributionalValue`.
        """

        # Define the regex pattern to match an optional floating-point or integer number,
        # followed by 'Ux', and then hexadecimal characters
        pattern = r"^([-+]?\d*\.?\d+)?Ux([0-9A-Fa-f]+)$"

        # Match the pattern
        match = re.match(pattern, text)

        if not match:
            return None

        particleValue = match.group(1)
        if particleValue is not None:
            particleValue = float(particleValue)
        UxString = "Ux" + match.group(2)

        # Indices for UxString[] below are based on the specification in the
        # docstring above.

        # Parse metadata
        representation_type = int(UxString[2:4], 16)
        mean_value = struct.unpack("!d", bytes.fromhex(UxString[20:36]))[0]
        dirac_delta_count = int(UxString[36:44], 16)

        # Parse data
        # Offset value 44 is the index at which the actual data starts (44 == 2+2+16+16+8).
        offset = 44
        support_position_list = []
        probability_mass_list = []
        raw_probability_mass_list = []
        for _ in range(dirac_delta_count):
            support_position_list.append(
                struct.unpack("!d", bytes.fromhex(UxString[offset : (offset + 16)]))[0]
            )
            # The probability mass is a fixed-point format with 0x8000000000000000 representing 1.0.
            # Divide by 0x8000000000000000 to get the float it represents.
            mass = int(UxString[(offset + 16) : (offset + 32)], 16)
            raw_probability_mass_list.append(mass)
            probability_mass_list.append(mass / 0x8000000000000000)
            # Set offset for next data pair.
            offset += 32

        # Initialize an instance of the class to return
        dist_value = DistributionalValue()
        dist_value.mean = mean_value
        dist_value.UR_type = representation_type
        dist_value.UR_order = dirac_delta_count
        dist_value.positions = np.array(support_position_list, dtype=np.float_)
        dist_value.masses = np.array(probability_mass_list, dtype=np.float_)
        dist_value.raw_masses = raw_probability_mass_list
        dist_value.particle_value = particleValue

        # Calculate weighted sample variance
        if dist_value.mean is not None:
            dist_value.variance = np.average(
                np.power((np.subtract(dist_value.positions, dist_value.mean)), 2),
                weights=dist_value.masses,
            )

        return dist_value

    @staticmethod
    def _parse_bytes(buffer: Union[bytes, bytearray]) -> Optional["DistributionalValue"]:
        """
        Constructs a `DistributionalValue` after parsing an input byte array.

        Here is the specification of the format:
            - Particle value (double)                           (8 bytes)
            - Representation type (uint8_t)                     (1 byte)
            - Number of samples (uint64_t)                      (8 bytes) (unused)
            - Mean value of distribution (double)               (8 bytes)
            - Number of non-zero mass Dirac deltas (uint32_t)   (4 bytes)
            Pairs of:
            - Support position (double)                         (8 bytes)
            - Probability mass (uint64_t)                       (8 bytes)

        Args:
            The input byte array.
        Returns:
            The constructed `DistributionalValue`.
        """
        # Interpret the particle value and remove from buffer
        particle_value = struct.unpack("1d", buffer[:8])[0]
        buffer = buffer[8:]

        representation_type = buffer[0]

        mean_value = struct.unpack("1d", buffer[9:17])[0]
        dirac_delta_count = struct.unpack("I", buffer[17:21])[0]

        # clean buffer
        buffer = buffer[21:]

        support_position_list = []
        probability_mass_list = []
        raw_probability_mass_list = []
        # Ensure buffer length is divisible by 16
        if len(buffer) % 16 != 0:
            raise ValueError("Buffer length must be divisible by 16")

        # Iterate through the buffer in sections of 16 bytes
        for i in range(0, len(buffer), 16):
            support_position_hex = buffer[i : i + 8]
            mass_hex = buffer[i + 8 : i + 16]

            support_position = struct.unpack("1d", support_position_hex)[0]
            mass = struct.unpack("<Q", mass_hex)[0]

            support_position_list.append(support_position)
            raw_probability_mass_list.append(mass)

            # The probability mass is a fixed-point format with 0x8000000000000000 representing 1.0.
            # Divide by 0x8000000000000000 to get the float it represents.
            probability_mass_list.append(mass / 0x8000000000000000)

        # Initialize an instance of the class to return
        dist_value = DistributionalValue()
        dist_value.particle_value = particle_value
        dist_value.mean = mean_value
        dist_value.UR_type = representation_type
        dist_value.UR_order = dirac_delta_count
        dist_value.positions = np.array(support_position_list, dtype=np.float_)
        dist_value.masses = np.array(probability_mass_list, dtype=np.float_)
        dist_value.raw_masses = raw_probability_mass_list

        # Calculate weighted sample variance
        if dist_value.mean is not None:
            dist_value.variance = np.average(
                np.power((np.subtract(dist_value.positions, dist_value.mean)), 2),
                weights=dist_value.masses,
            )

        return dist_value

    def bytes(self) -> bytes:
        """
        Constructs the byte array for the `DistributionalValue`.

        Returns:
            The byte array for the `DistributionalValue`.
        """
        particle_value = 0.0
        if self.particle_value is not None:
            particle_value = self.particle_value

        # Create byte representation
        byte_representation = bytearray()

        # - Particle value (double)                           (8 bytes)
        # - Representation type (uint8_t)                     (1 byte)
        # - Number of samples (uint64_t)                      (8 bytes)
        # - Mean value of distribution (double)               (8 bytes)
        # - Number of non-zero mass Dirac deltas (uint32_t)   (4 bytes)
        # Pairs of:
        # - Support position (double)                         (8 bytes)
        # - Probability mass (uint64_t)                       (8 bytes)

        # - Particle value (double)                           (8 bytes)
        byte_representation += struct.pack("1d", particle_value)

        # - Representation type (uint8_t)                     (1 byte)
        byte_representation += struct.pack("B", self.UR_type)

        # - Number of samples (uint64_t)                      (8 bytes)
        byte_representation += struct.pack("<Q", len(self.positions))

        # - Mean value of distribution (double)               (8 bytes)
        byte_representation += struct.pack("1d", self.mean)

        # - Number of non-zero mass Dirac deltas (uint32_t)   (4 bytes)
        byte_representation += struct.pack("<I", self.UR_order)

        # Pairs of:
        # - Support position (double)                         (8 bytes)
        # - Probability mass (uint64_t)                       (8 bytes)
        for i in range(len(self.positions)):
            byte_representation += struct.pack("1d", self.positions[i])
            byte_representation += struct.pack("<Q", self.raw_masses[i])

        return bytes(byte_representation)

    def calculate_steps(self) -> Optional[Tuple[List[float], List[float]]]:
        """
        Calculates the locations of x-axis and  y-axis step locations. These are
        the locations that drawstyle='steps-mid' plots the steps on.

        Returns:
            (stepsX, stepsY): The x-axis and  y-axis step locations.
        """
        if self.UR_order is None:
            return None
        else:
            DD_count = self.UR_order

        stepsX = [round(self.positions[0] - self.widths[0] / 2, 2)]
        stepsY = [0.0]

        for i in range(0, DD_count):
            newStepX = round(self.positions[i] - self.widths[i] / 2, 2)
            # print 'newStepX ' + str(newStepX) + ' stepsX[i-1] ' + str(stepsX[i-1])
            if newStepX - stepsX[-1] > 1.1:
                stepsX.append(stepsX[-1])
                stepsY.append(0)
                stepsX.append(newStepX)
                stepsY.append(0)
                stepsX.append(newStepX)
            else:
                stepsX.append(newStepX)

            stepsY.append(self.adjusted_masses[i])
            stepsX.append(round(self.positions[i] + self.widths[i] / 2, 2))
            stepsY.append(self.adjusted_masses[i])

        stepsX.append(round(self.positions[DD_count - 1] + self.widths[i] / 2, 2))
        stepsY.append(0)

        # print('Steps X: ' + str(stepsX))
        # print('Steps Y: ' + str(stepsY))
        # print('len Steps X: ' + str(len(stepsX)))
        # print('len Steps Y: ' + str(len(stepsY)))

        return (stepsX, stepsY)

    def mean_relative_diff(self, other=None) -> float:
        """
        Calculates the relative difference between the mean values of
        this `DistributionalValue` and another `DistributionalValue`.

        Args:
            other: The other `DistributionalValue`.
        Returns:
            The relative difference between the mean values (normalized by
            the mean value of the other `DistributionalValue`).
        """
        assert len(self.positions) > 0
        assert len(other.positions) > 0
        assert type(self) is type(other)

        if self.mean is None:
            print(type(self.positions[0]))
            if np.isnan(np.asarray(self.positions)).any():
                return np.nan
            dist_value_a_mean = float(np.average(
                self.positions,
                weights=self.masses if len(self.masses) > 0 else None
            ))
        else:
            dist_value_a_mean = self.mean

        if other.mean is None:
            if len(other.masses) == 0:
                dist_value_2_mean = np.nanmean(
                    np.array(other.positions).astype("float"))
            else:
                dist_value_2_mean = float(np.average(
                    other.positions,
                    weights=other.masses if len(other.masses) > 0 else None,
                ))
        else:
            dist_value_2_mean = other.mean

        return np.abs(
                (dist_value_a_mean - dist_value_2_mean) / dist_value_2_mean
            )

    def mean_distance(self, other=None) -> float:
        """
        Calculates the distance between the mean values of this `DistributionalValue`
        and another `DistributionalValue`.

        Args:
            other: The other `DistributionalValue`.
        Returns:
            The distance between the mean values.
        """
        assert len(self.positions) > 0
        assert len(other.positions) > 0
        assert type(self) is type(other)

        if self.mean is None:
            if np.isnan(np.asarray(self.positions)).any():
                return np.nan
            dist_value_a_mean = float(np.average(
                self.positions,
                weights=self.masses if len(self.masses) > 0 else None
            ))
        else:
            dist_value_a_mean = self.mean

        if other.mean is None:
            if len(other.masses) == 0:
                dist_value_2_mean = np.nanmean(
                    np.array(other.positions).astype("float"))
            else:
                dist_value_2_mean = float(np.average(
                    other.positions,
                    weights=other.masses if len(other.masses) > 0 else None,
                ))
        else:
            dist_value_2_mean = other.mean

        return np.abs(dist_value_a_mean - dist_value_2_mean)

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
        if len(self.positions) == 0:
            return None

        return not np.any(self.masses == 0)

    def drop_zero_mass_positions(self) -> None:
        """
        Drops (removes) the Dirac deltas of the `DistributionalValue` that has a zero
        mass.
        """
        if len(self.masses) == 0:
            return

        # zip -> filter -> unzip
        filtered_positions, filtered_masses = zip(
            *[(x, y) for x, y in zip(self.positions, self.masses) if y != 0]
        )
        # zip() returns tuple
        self.positions = np.array(list(filtered_positions), dtype=np.float_)
        self.masses = np.array(list(filtered_masses), dtype=np.float_)
        self._has_no_zero_mass = True

        return

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
        if len(self.positions) == 0:
            return None

        return bool(np.all(np.isfinite(self.positions)))

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
        if len(self.positions) == 0:
            return None
        elif len(self.positions) == 1:
            return True

        return bool(np.all(self.positions[:-1] < self.positions[1:]))

    def sort(self) -> None:
        """
        Sorts the positions and the masses of a `DistributionalValue` according to
        the positions. Also, cures multiple entries for NaN, -Inf, and Inf and
        places them at the end of positions and masses with the order [NaN, -Inf, Inf].
        """
        if self._is_sorted:
            return

        positions = np.array(self.positions)
        masses = np.array(self.masses)
        n = len(positions)
        finite_indices = [pos for pos in positions if np.isfinite(pos)]
        finite_positions = positions[finite_indices]
        finite_masses = masses[finite_indices]
        finite_dirac_deltas = list(zip(finite_positions, finite_masses))
        sorted_finite_dirac_deltas = np.array(sorted(finite_dirac_deltas))

        nan_indices = np.array(list(range(n)))[np.isnan(positions)]
        nan_mass = sum(masses[nan_indices])
        neg_inf_indices = np.where(positions == float("-inf"))[0]
        neg_inf_mass = sum(masses[neg_inf_indices])
        inf_indices = np.where(positions == float("inf"))[0]
        inf_mass = sum(masses[inf_indices])

        self.positions = np.concatenate(
            (
                sorted_finite_dirac_deltas[:, 0],
                np.array([float("nan"), float("-inf"), float("inf")]),
            )
        )
        self.masses = np.concatenate(
            (
                sorted_finite_dirac_deltas[:, 1],
                np.array([nan_mass, neg_inf_mass, inf_mass])
            )
        )
        self._is_sorted = True

        return

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
        if len(self.positions) == 0:
            return None
        if sum(np.isnan(self.positions)) > 1:
            return False

        return len(self.positions) == len(set(self.positions))

    def cure(self) -> None:
        """
        Cures the positions and masses of the `DistributionalVariable` from multiple
        entries of the same positional value, including NaN, -Inf, and Inf.
        """
        if self._is_cured:
            return

        positions = np.array(self.positions)
        masses = np.array(self.masses)
        n = len(positions)
        if n == 0:
            return
        elif n == 1:
            self._is_cured = True
            return
        else:
            i = 0
            while i < n - 1:
                if np.isnan(positions[i]):
                    indices = np.array(list(range(n)))[np.isnan(positions)][1:]
                else:
                    indices = np.where(positions == positions[i])[0][1:]
                masses[i] += sum(masses[indices])
                positions = np.delete(positions, indices)
                masses = np.delete(masses, indices)
                i += 1
                n = len(positions)

        self.positions = positions
        self.masses = masses
        self._is_cured = True

        return

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
        (afterdropping zero mass Dirac deltas and curing to combine same position
        Dirac deltas). "Valid" means that there is a distribution whose TTR
        exactly contains the Dirac deltas of the `DistributionalValue`.

        Returns:
            `None` if the `DistributionalValue` has no Dirac deltas. Else, `True` if
            the Dirac deltas of the `DistributionalValue` form a full and valid TTR,
            `False` else.
        """
        if self._has_no_zero_mass is not True:
            self.drop_zero_mass_positions()

        if self._is_cured is not True:
            self.cure()

        if self._is_finite is False:
            return False
        elif self._is_finite is None:
            if self.check_is_finite() is False:
                return False

        if self._is_sorted is not True:
            self.sort()

        number_of_samples = len(self.positions)

        if number_of_samples == 0:
            return None
        elif number_of_samples == 1:
            return True
        else:
            number = number_of_samples

        while number > 1:
            if number % 2 == 0:
                number //= 2
            else:
                return False

        ttr_order = int(np.log2(number_of_samples))
        number_of_boundaries = 2 * number_of_samples + 1
        boundary_positions = np.array([np.nan] * number_of_boundaries)
        boundary_probabilities = np.array([np.nan] * number_of_boundaries)
        boundary_positions[1::2] = self.positions
        boundary_probabilities[1::2] = self.masses

        for n in range(ttr_order):
            step = 2**n
            for i in range(2 ** (n + 1), number_of_boundaries - 1, 2 ** (n + 2)):
                boundary_probabilities[i] = (
                    boundary_probabilities[i - step] + boundary_probabilities[i + step]
                )
                boundary_positions[i] = (
                    boundary_probabilities[i - step] * boundary_positions[i - step]
                    + boundary_probabilities[i + step] * boundary_positions[i + step]
                ) / boundary_probabilities[i]

        return bool(np.all(boundary_positions[:-1] < boundary_positions[1:]))
