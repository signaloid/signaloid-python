#   Copyright (c) 2021, Signaloid.
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
    from numpy.typing import NDArray
else:
    # Use the extended version of ulab's numpy when running on CircuitPython
    from signaloid.circuitpython.extended_ulab_numpy import np  # type: ignore[no-redef]

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue


class PlotData:
    def __init__(
        self,
        dist: DistributionalValue,
        plotting_resolution: Optional[int] = None
    ) -> None:
        if (
            dist.mean is None
            or dist.UR_order == 0
        ):
            raise ValueError("Failed to load data")

        self.dist = dist
        self.plotting_resolution: Optional[int] = plotting_resolution
        self.plotting_ttr_order: Optional[int] = None

        self._positions: NDArray[np.float_] = np.array([], dtype=np.float_)
        self._masses: NDArray[np.float_] = np.array([], dtype=np.float_)
        self._widths: NDArray[np.float_] = np.array([], dtype=np.float_)
        self._max_value: Optional[float] = None

        self._construct_plot_data()

    @property
    def positions(self) -> NDArray[np.float_]:
        """The boundary positions list.

        :return: The boundary positions list.
        :rtype: NDArray[np.float_]
        """
        return self._positions

    @positions.setter
    def positions(self, positions: NDArray[np.float_]) -> None:
        """Sets the boundary positions list, resetting the widths to avoid faulty
        values.

        :param positions: The boundary positions list to use
        :type positions: NDArray[np.float_]
        """
        self._positions = positions
        self._widths = np.array([], dtype=np.float_)

    @property
    def masses(self) -> NDArray[np.float_]:
        """The bin heights list.

        :return: The bin heights list.
        :rtype: NDArray[np.float_]
        """
        return self._masses

    @masses.setter
    def masses(self, masses: NDArray[np.float_]) -> None:
        """Sets the bin heights list, resetting the max value to avoid faulty value.

        :param masses: The bin heights list to use.
        :type masses: NDArray[np.float_]
        """
        self._masses = masses
        self._max_value = None

    @property
    def min_range(self) -> float:
        """The minimum position.

        :return: The minimum position.
        :rtype: float
        """
        if len(self.positions) == 1:
            return float(self.positions[0] - 0.5)
        return float(self.positions[0])

    @property
    def max_range(self) -> float:
        """The maximum position.

        :return: The maximum position.
        :rtype: float
        """
        if len(self.positions) == 1:
            return float(self.positions[-1] + 0.5)
        return float(self.positions[-1])

    @property
    def total_range(self) -> float:
        """The total range of positions, i.e. the width between the minimum and
        maximum position.

        :return: The total range of positions.
        :rtype: float
        """
        if len(self.positions) == 1:
            return 1.0
        return float(self.positions[-1] - self.positions[0])

    @property
    def max_value(self) -> float:
        """The maximum bin height.

        :return: The maximum bin height.
        :rtype: float
        """
        if self._max_value is None:
            self._max_value = float(max(self._masses))

        return self._max_value

    @property
    def widths(self) -> NDArray[np.float_]:
        """The widths list between each pair of positions.

        :return: The widths list.
        :rtype: NDArray[np.float_]
        """
        if not self._widths.size > 0:
            self._widths = self.positions[1:] - self.positions[:-1]

        return self._widths

    @staticmethod
    def _determine_boundary_positions(
        finite_sorted_dirac_deltas: list[DiracDelta],
        exponent: int,
        use_ttr_binning: bool
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        If `use_ttr_binning` is true:
        Determines the internal boundary positions (and probabilities) using the
        TTR binning method.
        If `use_ttr_binning` is false:
        Determines the internal boundary positions (and probabilities) by only
        looking at the adjacent Dirac deltas.

        Args:
            finite_sorted_dirac_deltas: The input Dirac deltas with finite and
                sorted positions.
            exponent: The TTR order, i.e., the base-2 logarithm of the number of
                Dirac deltas in the TTR. The number of bins in the output binning
                is twice the number of Dirac deltas in the TTR.
            use_ttr_binning: Flag specifying whether to use the TTR binning method.
        Returns:
            (boundary_positions, boundary_probabilities): The internal boundary positions
                and boundary probabilities that are intermediaries to get a binning.
        """

        number_of_finite_dirac_deltas = len(finite_sorted_dirac_deltas)
        number_of_boundaries = 2 * number_of_finite_dirac_deltas + 1
        boundary_positions = np.array([np.nan] * number_of_boundaries)
        boundary_probabilities = np.array([np.nan] * number_of_boundaries)
        boundary_positions[1::2] = [dd.position for dd in finite_sorted_dirac_deltas]
        boundary_probabilities[1::2] = [dd.mass for dd in finite_sorted_dirac_deltas]

        if not use_ttr_binning:
            # Determine the 'NaN'-valued boundary points from adjacent Dirac deltas.
            for i in range(2, number_of_boundaries - 1, 2):
                if np.isnan(boundary_positions[i]):
                    boundary_positions[i] = (
                        boundary_probabilities[i - 1] * boundary_positions[i - 1]
                        + boundary_probabilities[i + 1] * boundary_positions[i + 1]
                    ) / (boundary_probabilities[i - 1] + boundary_probabilities[i + 1])

            return (boundary_positions, boundary_probabilities)

        # First handle internal boundary positions.
        for n in range(exponent):
            step = 2**n
            for i in range(2 ** (n + 1), number_of_boundaries - 1, 2 ** (n + 2)):
                boundary_probabilities[i] = (
                    boundary_probabilities[i - step]
                    + boundary_probabilities[i + step]
                )
                boundary_positions[i] = (
                    boundary_probabilities[i - step] * boundary_positions[i - step]
                    + boundary_probabilities[i + step] * boundary_positions[i + step]
                ) / boundary_probabilities[i]

        # Above process might not produce a strictly increasing sequence of
        # positions if not a valid TTR, and it will leave 'NaN'-valued
        # boundary points if the number of Dirac deltas is not a power of 2.
        # Handle both cases by sweeping over the boundary positions.
        for i in range(2, number_of_boundaries - 1, 2):
            if (
                np.isnan(boundary_positions[i])
                or boundary_positions[i] <= boundary_positions[i - 1]
                or boundary_positions[i] >= boundary_positions[i + 1]
            ):
                boundary_positions[i] = (
                    boundary_probabilities[i - 1] * boundary_positions[i - 1]
                    + boundary_probabilities[i + 1] * boundary_positions[i + 1]
                ) / (boundary_probabilities[i - 1] + boundary_probabilities[i + 1])

        return (boundary_positions, boundary_probabilities)

    @staticmethod
    def _handle_extremal_bins(
        finite_sorted_dirac_deltas: list[DiracDelta],
        boundary_positions: NDArray[np.float_],
        boundary_probabilities: NDArray[np.float_],
        bin_widths: NDArray[np.float_],
        bin_heights: NDArray[np.float_],
        left: bool = True,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
        """
        Checking if (d/dx)^2 = 0 boundary condition has a solution.
        If not, falling back to the boundary condition d/dx = 0.

        Args:
            finite_sorted_dirac_deltas: The input Dirac deltas with finite and
                sorted positions.
            boundary_positions: The internal boundary positions that are
                intermediaries to get a binning.
            boundary_probabilities: The internal boundary probabilities that are
                intermediaries to get a binning.
            bin_widths: The internal bin widths that are intermediaries to get
                a binning.
            bin_heights:The internal bin heights that are intermediaries to get
                a binning.
            left: The position to which to do the handling.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        """

        w0 = None
        det: float = np.nan
        if len(finite_sorted_dirac_deltas) >= 6:
            p0 = boundary_probabilities[1 if left else -2]
            w1 = bin_widths[1 if left else -2]
            w2 = bin_widths[2 if left else -3]
            d2 = bin_heights[2 if left else -3]
            a = d2 * w1 - p0
            b = a * w1 - p0 * w2
            c = p0 * w1 * (w1 + w2)
            det = b * b - 4 * a * c

            if det >= 0:
                # There are real roots. Pick the smallest positive root if there is one.
                root1 = (-b + math.sqrt(det)) / (2 * a)
                root2 = (-b - math.sqrt(det)) / (2 * a)
                roots_positive = [root1 > 0, root2 > 0]

                if all(roots_positive):
                    w0 = min(root1, root2)
                elif any(roots_positive):
                    w0 = max(root1, root2)

        if (
            w0 is None
            or math.isinf(det)
            or math.isnan(det)
        ):
            # The boundary condition d/dx = 0.
            boundary_positions[0 if left else -1] = (
                boundary_positions[1 if left else -2]
                + (-1 if left else 1) * (
                    boundary_positions[2 if left else -2]
                    - boundary_positions[1 if left else -3]
                )
            )
        else:
            # The boundary condition (d/dx)^2 = 0.
            boundary_positions[0 if left else -1] = (
                boundary_positions[1 if left else -2]
                + (-1 if left else 1) * w0
            )

        bin_widths[0 if left else -1] = (
            boundary_positions[1 if left else -1]
            - boundary_positions[0 if left else -2]
        )
        averageHeight = (
            finite_sorted_dirac_deltas[0 if left else -1].mass
            / (bin_widths[0 if left else -1] + bin_widths[1 if left else -2])
        )
        bin_heights[0 if left else -1] = (
            averageHeight * bin_widths[1 if left else -2]
            / bin_widths[0 if left else -1]
        )
        bin_heights[1 if left else -2] = (
            averageHeight * bin_widths[0 if left else -1]
            / bin_widths[1 if left else -2]
        )

        return (boundary_positions, bin_widths, bin_heights)

    @staticmethod
    def _get_binning(
        finite_sorted_dirac_deltas: list[DiracDelta],
        boundary_positions: NDArray[np.float_],
        boundary_probabilities: NDArray[np.float_]
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
        """
        Finds the binning for the given finite and sorted Dirac deltas and the
        calculated internal boundary positions and probabilities.

        Args:
            finite_sorted_dirac_deltas: The input Dirac deltas with finite and
                sorted positions.
            boundary_positions: The internal boundary positions that are
                intermediaries to get a binning.
            boundary_probabilities: The internal boundary probabilities that are
                intermediaries to get a binning.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        """

        number_of_finite_dirac_deltas = len(finite_sorted_dirac_deltas)

        # Initialize the binning and populate it for the internal bins.
        numberOfBins = 2 * number_of_finite_dirac_deltas
        bin_widths = np.array([np.nan] * numberOfBins)
        bin_widths[1:-1] = boundary_positions[2:-1] - boundary_positions[1:-2]
        bin_heights = np.array([np.nan] * numberOfBins)

        for i in range(1, number_of_finite_dirac_deltas - 1):
            averageHeight = finite_sorted_dirac_deltas[i].mass / (
                bin_widths[2 * i] + bin_widths[2 * i + 1]
            )
            bin_heights[2 * i] = (
                averageHeight * bin_widths[2 * i + 1] / bin_widths[2 * i]
            )
            bin_heights[2 * i + 1] = (
                averageHeight * bin_widths[2 * i] / bin_widths[2 * i + 1]
            )

        boundary_positions, bin_widths, bin_heights = PlotData._handle_extremal_bins(
            finite_sorted_dirac_deltas,
            boundary_positions,
            boundary_probabilities,
            bin_widths,
            bin_heights,
            left=True,
        )
        boundary_positions, bin_widths, bin_heights = PlotData._handle_extremal_bins(
            finite_sorted_dirac_deltas,
            boundary_positions,
            boundary_probabilities,
            bin_widths,
            bin_heights,
            left=False,
        )

        return (boundary_positions, bin_widths, bin_heights)

    @staticmethod
    def _create_binning(
            finite_sorted_dirac_deltas: list[DiracDelta],
            exponent: int,
            use_ttr_binning: bool
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
        """
        - If `use_ttr_binning` is true:
            Creates a binning using the TTR binning method. The TTR binning method
            creates the unique binning (up to extremal bins determined by the imposed
            boundary conditions) such that the TTR of the binning exactly coincides
            with the input Dirac deltas. Requires the input Dirac deltas to form
            a valid TTR.
        - If `use_ttr_binning` is false:
            Creates a binning without requiring the valid TTR property, where the
            internal bin boundaries are determined only by adjacent Dirac deltas and
            the average of two bins surrounding a Dirac delta is the Dirac delta itself.

        Args:
            finite_sorted_dirac_deltas: The input Dirac deltas with finite
                and sorted positions.
            exponent: The TTR order, i.e., the base-2 logarithm of the number of
                Dirac deltas in the TTR. The number of bins in the output binning
                is twice the number of Dirac deltas in the TTR.
            use_ttr_binning: Flag specifying whether to use the TTR binning method.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        """

        boundary_positions, boundary_probabilities = PlotData._determine_boundary_positions(
            finite_sorted_dirac_deltas, exponent, use_ttr_binning
        )

        boundary_positions, bin_widths, bin_heights = PlotData._get_binning(
            finite_sorted_dirac_deltas, boundary_positions, boundary_probabilities
        )

        return (boundary_positions, bin_widths, bin_heights)

    @staticmethod
    def _bin_pdf_expected_dirac_delta(
            boundary_positions: NDArray[np.float_],
            bin_widths: NDArray[np.float_],
            bin_heights: NDArray[np.float_]
    ) -> DiracDelta:
        """
        Computes the expected Dirac delta of an input bin PDF.

        Args:
            boundary_positions: Positions of bin boundaries of the input bin PDF.
            bin_widths: Widths of the bins of the input bin PDF.
            bin_heights: Heights of the bins of the input bin PDF.
        Returns:
            expected_dirac_delta: The expected Dirac delta in the format np.array([position, mass]).
        """

        moment_sum = 0.0
        probability_sum = 0.0

        for i, (bin_width, bin_height) in enumerate(zip(bin_widths, bin_heights)):
            probability = bin_width * bin_height
            probability_sum += probability
            moment_sum += (
                probability * (boundary_positions[i + 1] + boundary_positions[i]) / 2
            )

        expected_dirac_delta = DiracDelta(moment_sum / probability_sum, mass=probability_sum)

        return expected_dirac_delta

    @staticmethod
    def _bin_pdf_to_ttr(
            boundary_positions: NDArray[np.float_],
            bin_widths: NDArray[np.float_],
            bin_heights: NDArray[np.float_],
            order: int
    ) -> list[DiracDelta]:
        """
        Computes TTR for an input bin PDF.

        Args:
            boundary_positions: Positions of the bin boundaries of the input bin PDF.
            bin_widths: Widths of the bins of the input bin PDF.
            bin_heights: Heights of the bins of the input bin PDF.
            order: TTR order.
        Returns:
            ttr: The TTR of the input bin PDF, a (2 ** `order`)-length array of Dirac deltas
                with each Dirac delta of the form np.array([position, mass]).
        """

        expected_dirac_delta = PlotData._bin_pdf_expected_dirac_delta(
            boundary_positions, bin_widths, bin_heights
        )
        ttr: list[DiracDelta] = []

        if order == 0:
            return [expected_dirac_delta]

        low_boundary_positions: NDArray[np.float_] = np.array([], dtype=np.float_)
        low_bin_widths: NDArray[np.float_] = np.array([], dtype=np.float_)
        low_bin_heights: NDArray[np.float_] = np.array([], dtype=np.float_)
        high_boundary_positions: NDArray[np.float_] = np.array([], dtype=np.float_)
        high_bin_widths: NDArray[np.float_] = np.array([], dtype=np.float_)
        high_bin_heights: NDArray[np.float_] = np.array([], dtype=np.float_)

        for i, boundary_position in enumerate(boundary_positions):
            if boundary_position == expected_dirac_delta.position:
                low_boundary_positions = boundary_positions[: i + 1]
                low_bin_widths = bin_widths[:i]
                low_bin_heights = bin_heights[:i]
                high_boundary_positions = boundary_positions[i:]
                high_bin_widths = bin_widths[i:]
                high_bin_heights = bin_heights[i:]
                break

            if boundary_position > expected_dirac_delta.position:
                low_boundary_positions = np.append(
                    boundary_positions[:i], expected_dirac_delta.position
                )
                low_bin_widths = np.append(
                    bin_widths[: i - 1],
                    expected_dirac_delta.position - boundary_positions[i - 1],
                )
                low_bin_heights = bin_heights[:i]
                high_boundary_positions = np.insert(
                    boundary_positions[i:], 0, expected_dirac_delta.position
                )
                high_bin_widths = np.insert(
                    bin_widths[i:], 0, boundary_position - expected_dirac_delta.position
                )
                high_bin_heights = bin_heights[i - 1 :]
                break

        ttr += PlotData._bin_pdf_to_ttr(
            low_boundary_positions, low_bin_widths, low_bin_heights, order - 1
        )
        ttr += PlotData._bin_pdf_to_ttr(
            high_boundary_positions, high_bin_widths, high_bin_heights, order - 1
        )

        return ttr

    def _construct_plot_data(self) -> None:
        """Constructs the `PlotData`, after parsing the given `DistributionalValue`.
        Generates the boundary positions and bin heights, ready for plotting.

        :raises ValueError: When the plotting_resolution is not a power of 2.
        """
        # Create the list of finite Dirac deltas.
        self.dist.drop_zero_mass_positions()

        self.dist.combine_dirac_deltas()

        # Create the list of finite sorted Dirac deltas.
        # Last three positions are for non-finite values
        finite_dirac_deltas: list[DiracDelta] = self.dist.finite_dirac_deltas

        # If no finite Dirac deltas found, then return.
        if len(finite_dirac_deltas) == 0:
            return

        if len(finite_dirac_deltas) == 1:
            self.positions = np.array([finite_dirac_deltas[0].position], dtype=np.float_)
            self.masses = np.array([finite_dirac_deltas[0].mass], dtype=np.float_)
            return

        # Set plot resolution to (N*2) where N is machine representation
        machine_representation = 2 ** math.floor(math.log(self.dist.UR_order, 2))
        self.plotting_resolution = int(
            machine_representation * 2
            if self.plotting_resolution is None
            else min((machine_representation * 2), self.plotting_resolution)
        )
        log2_of_plotting_resolution = self.plotting_resolution.bit_length() - 1
        self.plotting_ttr_order = log2_of_plotting_resolution - 1

        if (
            self.plotting_resolution > 2
            and self.plotting_resolution > 2 ** (self.plotting_ttr_order + 1)
        ):
            raise ValueError(
                "plot_histogram_dirac_deltas: plotting_resolution must be a power of 2!"
            )

        # Create the binning such that the average of two bins surrounding a Dirac delta
        # is the Dirac delta itself.
        boundary_positions, bin_widths, bin_heights = PlotData._create_binning(
            finite_dirac_deltas, 0, False
        )

        # Find the TTR of the created binning. This is always a valid TTR.
        ttr = PlotData._bin_pdf_to_ttr(
            boundary_positions, bin_widths, bin_heights, self.plotting_ttr_order
        )

        # Create the binning from the obtained (valid) TTR using the TTR binning method.
        boundary_positions, bin_widths, bin_heights = PlotData._create_binning(
            ttr, self.plotting_ttr_order, True
        )

        self.positions = boundary_positions
        self.masses = bin_heights
