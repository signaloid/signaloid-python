# fmt: off

# 	Copyright (c) 2021, Signaloid.
#
# 	Permission is hereby granted, free of charge, to any person obtaining a copy
# 	of this software and associated documentation files (the "Software"), to
# 	deal in the Software without restriction, including without limitation the
# 	rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# 	sell copies of the Software, and to permit persons to whom the Software is
# 	furnished to do so, subject to the following conditions:
#
# 	The above copyright notice and this permission notice shall be included in
# 	all copies or substantial portions of the Software.
#
# 	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# 	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# 	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# 	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# 	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# 	FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# 	DEALINGS IN THE SOFTWARE.

import struct
from typing import List, Optional, Union
import numpy as np


class DistributionalValue:
    def __init__(self):
        self.positions: List[float] = []
        self.masses: List[float] = []
        self.adjusted_masses = []
        self.graphical = []
        self.widths = []
        self.mean: Union[None, float] = None
        self.particle_value: Union[None, float] = None
        self.variance: Union[None, float] = None
        self.initialized = False
        self.particle = False
        self.UR_type: Union[None, int, str] = None
        self.UR_order: Union[None, int] = None
        self.correlation_tracking: Union[str, None] = None
        self.UXString: Union[str, None] = None
        """
        metadata
        """
        self.expression_name: Union[None, str] = None
        self.expression_subprogram: Union[None, str] = None
        self.value_id: Union[None, str] = None
        self.source_db: Union[None, str] = None
        self.source_db_table: Union[None, str] = None
        self.label_prefix = ""
        self.label = ""
        self.plot_color: Union[None, str] = None
        self.plot_hatch: Union[None, str] = None
        self.plot_z_order = None
        self.wasserstein_distance_function = None
        """
        properties
        """
        self._has_no_zero_mass: Union[None, bool] = None
        self._is_finite: Union[None, bool] = None
        self._is_sorted: Union[None, bool] = None
        self._is_cured: Union[None, bool] = None
        self._is_full_valid_TTR: Union[None, bool] = None

    def __str__(self):
        strVal = f"{id(self)}:{self.UR_type}-{self.UR_order}"
        strVal += "\nPosition\t Mass"
        for pos, mass, graph in zip(self.positions, self.masses, self.graphical):
            strVal += f"\n{pos}\t {mass}\t {graph}"
        return strVal

    def __repr__(self):
        return f"{id(self)}:{self.UR_type}-{self.UR_order}"

    @staticmethod
    def parse(dist: Union[str, bytes]) -> Optional["DistributionalValue"]:
        """
        Parse a distributional value. The input can be a HexString or a bytes buffer.
        """
        # Parse Hex string
        if (isinstance(dist, str)):
            return DistributionalValue._parse_hex(dist)
        # Parse bytes buffer
        elif (isinstance(dist, bytes)):
            return DistributionalValue._parse_bytes(dist)
        else:
            print("Error: ", type(dist))
            return None

    @staticmethod
    def _parse_hex(text: str) -> Optional["DistributionalValue"]:
        """
        Parses a HexString representation of a Distributional Value.

        Here is the specification of the format:
            - "Ux"                                              ( 2 characters)
            - Representation type (uint8_t)                     ( 2 characters)
            - Number of samples (uint64_t)                      (16 characters)
            - Mean value of distribution (double)               (16 characters)
            - Number of non-zero mass Dirac deltas (uint32_t)   ( 8 characters)
            Pairs of:
            - Support position (double)                         (16 characters)
            - Probability mass (uint64_t)                       (16 characters)
        """
        if not text.startswith("Ux"):
            return None

        # Indices for text[] below are based on the specification in the
        # docstring above.

        # Parse metadata
        representation_type = int(text[2:4], 16)
        # sample_count = int(text[4:20], 16)
        mean_value = struct.unpack("!d", bytes.fromhex(text[20:36]))[0]
        dirac_delta_count = int(text[36:44], 16)

        # Parse data
        # Offset value 44 is the index at which the actual data starts (44 == 2+2+16+16+8).
        offset = 44
        support_position_list = []
        probability_mass_list = []
        for _ in range(dirac_delta_count):
            support_position_list.append(
                struct.unpack("!d", bytes.fromhex(text[offset: (offset + 16)]))[0]
            )
            # The probability mass is a fixed-point format with 0x8000000000000000 representing 1.0.
            # Divide by 0x8000000000000000 to get the float it represents.
            probability_mass_list.append(
                int(text[(offset + 16): (offset + 32)], 16) / 0x8000000000000000
            )
            # Set offset for next data pair.
            offset += 32

        # Initialize an instance of the class to return
        dist_value = DistributionalValue()
        dist_value.mean = mean_value
        dist_value.UR_type = representation_type
        dist_value.UR_order = dirac_delta_count
        dist_value.positions = support_position_list
        dist_value.masses = probability_mass_list

        # Calculate weighted sample variance
        if dist_value.mean is not None:
            dist_value.variance = np.average(
                np.power((np.subtract(dist_value.positions, dist_value.mean)), 2),
                weights=dist_value.masses,
            )

        return dist_value

    @staticmethod
    def _parse_bytes(buffer: bytes) -> Optional["DistributionalValue"]:
        """
        Parses a bytes representation of a Distributional Value.

        Here is the specification of the format:
            - Particle value (double)                           (8 bytes)
            - Representation type (uint8_t)                     (1 byte)
            - Number of samples (uint64_t)                      (8 bytes)
            - Mean value of distribution (double)               (8 bytes)
            - Number of non-zero mass Dirac deltas (uint32_t)   (4 bytes)
            Pairs of:
            - Support position (double)                         (8 bytes)
            - Probability mass (uint64_t)                       (8 bytes)

        :param buffer: Byte buffer containing the distributional data

        :return: DistributionalValue object.
        """
        # Interpret the particle value and remove from buffer
        particle_value = struct.unpack("1d", buffer[:8])[0]
        buffer = buffer[8:]

        representation_type = buffer[0]
        # number_of_samples = struct.unpack("<Q", buffer[1:9])[0]
        mean_value = struct.unpack("1d", buffer[9:17])[0]
        dirac_delta_count = struct.unpack("I", buffer[17:21])[0]

        # clean buffer
        buffer = buffer[21:]

        support_position_list = []
        probability_mass_list = []
        # Ensure buffer length is divisible by 16
        if len(buffer) % 16 != 0:
            raise ValueError("Buffer length must be divisible by 16")

        # Iterate through the buffer in sections of 16 bytes
        for i in range(0, len(buffer), 16):
            support_position_hex = buffer[i: i + 8]
            mass_hex = buffer[i + 8: i + 16]

            support_position = struct.unpack("1d", support_position_hex)[0]
            mass = struct.unpack("<Q", mass_hex)[0]

            support_position_list.append(support_position)

            # The probability mass is a fixed-point format with 0x8000000000000000 representing 1.0.
            # Divide by 0x8000000000000000 to get the float it represents.
            probability_mass_list.append(mass / 0x8000000000000000)

        # Initialize an instance of the class to return
        dist_value = DistributionalValue()
        dist_value.particle_value = particle_value
        dist_value.mean = mean_value
        dist_value.UR_type = representation_type
        dist_value.UR_order = dirac_delta_count
        dist_value.positions = support_position_list
        dist_value.masses = probability_mass_list

        # Calculate weighted sample variance
        if dist_value.mean is not None:
            dist_value.variance = np.average(
                np.power((np.subtract(dist_value.positions, dist_value.mean)), 2),
                weights=dist_value.masses,
            )

        return dist_value

    def calculate_steps(self):
        """
        Calculates the locations of X axis and Y axis step locations. These are
        the locations that drawstyle='steps-mid' plots the steps on.
        """
        DD_count = self.UR_order
        stepsX = [round(self.positions[0] - self.widths[0] / 2, 2)]
        stepsY = [0]

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

    def wasserstein(self, other=None):
        """
        Calculate Wasserstein distance between DistributionalValue instances.
        """
        assert len(self.positions) > 0
        assert len(other.positions) > 0
        assert type(self) is type(other)

        if self.wasserstein_distance_function is None:
            print("Need to initialize wasserstein_distance_function")
            assert self.wasserstein_distance_function is not None

        weights_dict = dict()
        if len(self.masses) > 0:
            weights_dict["u_weights"] = self.masses
        if len(other.masses) > 0:
            weights_dict["v_weights"] = other.masses

        return self.wasserstein_distance_function(
            self.positions, other.positions, **weights_dict
        )

    def mean_relative_diff(self, other=None):
        """
        Calculate relative difference.
        ```
        np.abs((mean_1-mean_2)/mean_2)
        ```
        """
        assert len(self.positions) > 0
        assert len(other.positions) > 0
        assert type(self) is type(other)

        if self.mean is None:
            print(type(self.positions[0]))
            if np.isnan(np.asarray(self.positions)).any():
                return np.nan
            mean_1 = np.average(
                self.positions, weights=self.masses if len(self.masses) > 0 else None
            )
        else:
            mean_1 = float(self.mean)

        if other.mean is None:
            if len(other.masses) == 0:
                mean_2 = np.nanmean(np.array(other.positions).astype("float"))
                # np.array(other.positions).astype('float')
            else:
                mean_2 = np.average(
                    other.positions,
                    weights=other.masses if len(other.masses) > 0 else None,
                )
        else:
            mean_2 = float(other.mean)

        return np.abs((mean_1 - mean_2) / mean_2)

    def mean_distance(self, other=None):
        """
        Calculate mean distance.
        ```
        np.abs(mean_1 - mean_2)
        ```
        """
        assert len(self.positions) > 0
        assert len(other.positions) > 0
        assert type(self) is type(other)

        if self.mean is None:
            if np.isnan(np.asarray(self.positions)).any():
                return np.nan
            mean_1 = np.average(
                self.positions, weights=self.masses if len(self.masses) > 0 else None
            )
        else:
            mean_1 = float(self.mean)

        if other.mean is None:
            if len(other.masses) == 0:
                mean_2 = np.nanmean(np.array(other.positions).astype("float"))
                # other.positions = np.array(other.positions).astype('float')
            else:
                mean_2 = np.average(
                    other.positions,
                    weights=other.masses if len(other.masses) > 0 else None,
                )
        else:
            mean_2 = float(other.mean)

        return np.abs(mean_1 - mean_2)

    @property
    def has_no_zero_mass(self):
        """
        The property that no Dirac delta of the DistributionalValue has a zero
        mass.
        """
        if self._has_no_zero_mass is None:
            self._has_no_zero_mass = self.check_has_no_zero_mass()

        return self._has_no_zero_mass

    def check_has_no_zero_mass(self):
        if len(self.positions) == 0:
            return None

        return not np.any(self.masses == 0)

    def drop_zero_mass_positions(self):
        if len(self.masses) == 0:
            return

        # zip -> filter -> unzip
        filtered_positions, filtered_masses = zip(
            *[(x, y) for x, y in zip(self.positions, self.masses) if y != 0]
        )
        # zip() returns tuple
        self.positions = list(filtered_positions)
        self.masses = list(filtered_masses)
        self._has_no_zero_mass = True

    @property
    def is_finite(self):
        """
        The property that all Dirac deltas of the DistributionalValue have
        finite positions, i.e., no NaN, -Inf, or Inf values.
        """
        if self._is_finite is None:
            self._is_finite = self.check_is_finite()

        return self._is_finite

    def check_is_finite(self):
        if len(self.positions) == 0:
            return None

        return np.all(np.isfinite(self.positions))

    @property
    def is_sorted(self):
        """
        The property that the Dirac deltas of the DistributionalValue are sorted
        according to their positions. The NaN, -Inf, and Inf positional values
        are cured and sorted to the end in this order.
        """
        if self._is_sorted is None:
            self._is_sorted = self.check_is_sorted()

        return self._is_sorted

    def check_is_sorted(self):
        if len(self.positions) == 0:
            return None
        elif len(self.positions) == 1:
            return True

        return np.all(self.positions[:-1] < self.positions[1:])

    def sort(self):
        """
        Sorts the positions and the masses of a DistributionalValue according to
        the positions. Also, cures multiple entries for NaN, -Inf, and Inf and
        places them at the end of positions and masses with the order [NaN,
        -Inf, Inf].
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
                [float("nan"), float("-inf"), float("inf")],
            )
        )
        self.masses = np.concatenate(
            (sorted_finite_dirac_deltas[:, 1], [nan_mass, neg_inf_mass, inf_mass])
        )
        self._is_sorted = True

    @property
    def is_cured(self):
        """
        The property that no two Dirac deltas of the DistributionalValue have
        the same positional value, including NaN, -Inf, and Inf.
        """
        if self._is_cured is None:
            self._is_cured = self.check_is_cured()

        return self._is_cured

    def check_is_cured(self):
        if len(self.positions) == 0:
            return None
        if sum(np.isnan(self.positions)) > 1:
            return False

        return len(self.positions) == len(set(self.positions))

    def cure(self):
        """
        Cures the positions and masses of the DistributionalVariable from
        multiple entires of the same positional value, including NaN, -Inf, and
        Inf.
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

    @property
    def is_full_valid_TTR(self):
        """
        The property that the Dirac deltas of the DistributionalValue form a
        full valid TTR.
        """
        if self._is_full_valid_TTR is None:
            self._is_full_valid_TTR = self.check_is_full_valid_TTR()

        return self._is_full_valid_TTR

    def check_is_full_valid_TTR(self):
        """
        Check whether the Dirac deltas of the DistributionalValue form a full
        and valid TTR. "Full" means that there are 2^n Dirac deltas (after
        dropping zero mass Dirac deltas and curing to combine same position
        Dirac deltas). "Valid" means that there is a distribution whose TTR
        exactly contains the Dirac deltas of the DistributionalValue.
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
        boundary_positions = np.array([None] * number_of_boundaries)
        boundary_probabilities = np.array([None] * number_of_boundaries)
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

        return np.all(boundary_positions[:-1] < boundary_positions[1:])
