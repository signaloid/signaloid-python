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

import math
from typing import List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from signaloid.distributional import DistributionalValue


class PlotHistogramDiracDeltas:

    def __init__(self):
        pass

    def plot_histogram_dirac_deltas(
        self,
        dist_list: List[DistributionalValue],
        plotting_resolution: int = 64,
        dist_samples: List[List[float]] = [],
        hatches: List[str] = [],
        colors: List[str] = [],
    ) -> Tuple[float, float, float]:
        """
        Takes a DistributionalValue list and plots each DistributionalValue as a histogram.

        Args:
            dist_list: The input DistributionalValue list.
            plotting_resolution: The number of bins in the plot. Should always be a power
                of 2.
            dist_samples: A list of lists of samples. Plots each list of samples on the
                same plot as the `dist_list`.
            hatches: Plotting hatches to be used.
            colors: Plotting colors to be used.
        Returns:
            (min_range, max_range, max_value): Respectively the minimum x-value, maximum
                x-value, and maximum y-value of the plotted histograms.
        """

        log2_of_plotting_resolution = plotting_resolution.bit_length() - 1
        plotting_ttr_order = log2_of_plotting_resolution - 1

        if (isinstance(plotting_resolution, int)) and (plotting_resolution > 2):
            if plotting_resolution > 2 ** (plotting_ttr_order + 1):
                raise Exception(
                    "plot_histogram_dirac_deltas: plotting_resolution must be a power of 2!"
                )

        hatches = ["\\", "|", "+", "-"] if len(hatches) == 0 else hatches
        colors = (
            ["#2ECC71", "#A569BD", "#F1C40F", "#33A333"] if len(colors) == 0 else colors
        )
        hatch_iter = cycle(hatches)
        color_iter = cycle(colors)

        min_range = float("inf")
        max_range = float("-inf")
        max_value = 0

        for dist in dist_list + dist_samples:
            if isinstance(dist, DistributionalValue):
                if dist.UR_type != "MonteCarlo":
                    """
                    Create the list of finite Dirac deltas.
                    """
                    dist.drop_zero_mass_positions()
                    all_dirac_deltas = list(zip(dist.positions, dist.masses))
                else:
                    dd_masses = len(dist.positions) * [1 / len(dist.positions)]
                    all_dirac_deltas = list(zip(dist.positions, dd_masses))

                    finite_dirac_deltas = [
                        dd for dd in all_dirac_deltas if np.isfinite(dd[0])
                    ]
                    finite_dirac_deltas = [
                        dd for dd in all_dirac_deltas if np.isfinite(dd[0])
                    ]

                finite_dirac_deltas = [
                    dd for dd in all_dirac_deltas if np.isfinite(dd[0])
                ]

                """
                Create the list of finite sorted Dirac deltas.
                """
                finite_sorted_dirac_deltas = np.array(sorted(finite_dirac_deltas))

                """
                If no finite Dirac deltas found, then continue to the next dist.
                """
                if len(finite_sorted_dirac_deltas) == 0:
                    continue

                """
                Calculate the mean value of finite part
                """
                finite_mass = 0.0
                finite_mean = 0.0
                for dd in finite_sorted_dirac_deltas:
                    finite_mass += dd[1]
                    finite_mean += dd[0] * dd[1]
                finite_mean /= finite_mass

                """
                Combine Dirac deltas with same/very-close-relative-to-range/very-close-relative-to-mean-value positions.
                """
                realtive_range_threshold = 1e-12
                total_range = (
                    finite_sorted_dirac_deltas[-1][0] - finite_sorted_dirac_deltas[0][0]
                )
                range_threshold = total_range * realtive_range_threshold
                realtive_mean_threshold = 1e-14
                mean_threshold = finite_mean * realtive_mean_threshold
                cured_finite_sorted_dirac_deltas = [finite_sorted_dirac_deltas[0]]

                for dd in finite_sorted_dirac_deltas[1:]:
                    gap = abs(dd[0] - cured_finite_sorted_dirac_deltas[-1][0])

                    if (gap < range_threshold) or (gap < mean_threshold):
                        combined_mass = cured_finite_sorted_dirac_deltas[-1][1] + dd[1]
                        combined_position = (
                            cured_finite_sorted_dirac_deltas[-1][0]
                            * cured_finite_sorted_dirac_deltas[-1][1]
                            + dd[0] * dd[1]
                        ) / combined_mass
                        cured_finite_sorted_dirac_deltas[-1][1] = combined_mass
                        cured_finite_sorted_dirac_deltas[-1][0] = combined_position
                    else:
                        cured_finite_sorted_dirac_deltas.append(dd)

                finite_sorted_dirac_deltas = np.array(cured_finite_sorted_dirac_deltas)
                number_of_finite_dirac_deltas = len(finite_sorted_dirac_deltas)

                """
                If there is only one finite Dirac delta, then plot just an arrow representing a Dirac delta.
                """
                if number_of_finite_dirac_deltas == 1:
                    candidate_edgecolor = (
                        next(color_iter)
                    )
                    candidate_facecolor = candidate_edgecolor + "40"
                    plt.annotate(
                        "",
                        xy=(
                            finite_sorted_dirac_deltas[0][0],
                            finite_sorted_dirac_deltas[0][1],
                        ),
                        xytext=(finite_sorted_dirac_deltas[0][0], 0),
                        arrowprops=dict(arrowstyle="->", facecolor="black", lw=3),
                    )
                    min_range = finite_sorted_dirac_deltas[0][0] - 0.5
                    max_range = finite_sorted_dirac_deltas[0][0] + 0.5
                    max_value = max(max_value, finite_sorted_dirac_deltas[0][1])
                    continue

                """
                Create the binning such that the average of two bins surrounding a Dirac delta
                is the Dirac delta itself.
                """
                (boundary_positions, bin_widths, bin_heights) = list(
                    self.create_binning(finite_sorted_dirac_deltas, 0, False)
                )
                """
                Find the TTR of the created binning. This is always a valid TTR.
                """
                ttr = np.array(
                    self.bin_pdf_to_ttr(
                        boundary_positions, bin_widths, bin_heights, plotting_ttr_order
                    )
                )
                """
                Create the binning from the obtained (valid) TTR using the TTR binning method.
                """
                (boundary_positions, bin_widths, bin_heights) = self.create_binning(
                    ttr, plotting_ttr_order, True
                )

                """
                Plot the binning.
                """
                candidate_edgecolor = next(color_iter)
                candidate_facecolor = candidate_edgecolor + "40"
                candidate_hatch = next(hatch_iter)
                plt.bar(
                    boundary_positions[:-1],
                    bin_heights,
                    width=bin_widths,
                    align="edge",
                    edgecolor=candidate_edgecolor,
                    facecolor=candidate_facecolor,
                    hatch=candidate_hatch
                )
                min_range = min(min_range, boundary_positions[0])
                max_range = max(max_range, boundary_positions[-1])
                max_value = max(max_value, max(bin_heights))

            else:
                raise Exception("plot_histogram_dirac_deltas: input is not recognized!")

        return (min_range, max_range, max_value)

    def create_binning(
            self,
            finite_sorted_dirac_deltas: np.ndarray,
            exponent: int,
            use_ttr_binning: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        (boundary_positions, boundary_probabilities) = self.determine_boundary_positions(
            finite_sorted_dirac_deltas, exponent, use_ttr_binning
        )

        (boundary_positions, bin_widths, bin_heights) = self.get_binning(
            finite_sorted_dirac_deltas, boundary_positions, boundary_probabilities
        )

        return (boundary_positions, bin_widths, bin_heights)

    def determine_boundary_positions(
        self,
        finite_sorted_dirac_deltas: np.ndarray,
        exponent: int,
        use_ttr_binning: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        boundary_positions[1::2] = finite_sorted_dirac_deltas[:, 0]
        boundary_probabilities[1::2] = finite_sorted_dirac_deltas[:, 1]

        if use_ttr_binning:
            """
            First handle internal boundary positions.
            """
            for n in range(exponent):
                step = 2**n
                for i in range(2 ** (n + 1), number_of_boundaries - 1, 2 ** (n + 2)):
                    boundary_probabilities[i] = (
                        boundary_probabilities[i - step]
                        + boundary_probabilities[i + step]
                    )
                    boundary_positions[i] = (
                        boundary_probabilities[i - step] * boundary_positions[i - step]
                        + boundary_probabilities[i + step]
                        * boundary_positions[i + step]
                    ) / boundary_probabilities[i]

            """
            Above process might not produce a strictly increasing sequence of positions if not a valid TTR,
            and it will leave 'NaN'-valued boundary points if the number of Dirac deltas is not a power of 2.
            Handle both cases by sweeping over the boundary positions.
            """
            for i in range(2, number_of_boundaries - 1, 2):
                if (
                    (np.isnan(boundary_positions[i]))
                    or (boundary_positions[i] <= boundary_positions[i - 1])
                    or (boundary_positions[i] >= boundary_positions[i + 1])
                ):
                    boundary_positions[i] = (
                        boundary_probabilities[i - 1] * boundary_positions[i - 1]
                        + boundary_probabilities[i + 1] * boundary_positions[i + 1]
                    ) / (boundary_probabilities[i - 1] + boundary_probabilities[i + 1])
        else:
            """
            Determine the 'NaN'-valued boundary points from adjacent Dirac deltas.
            """
            for i in range(2, number_of_boundaries - 1, 2):
                if np.isnan(boundary_positions[i]):
                    boundary_positions[i] = (
                        boundary_probabilities[i - 1] * boundary_positions[i - 1]
                        + boundary_probabilities[i + 1] * boundary_positions[i + 1]
                    ) / (boundary_probabilities[i - 1] + boundary_probabilities[i + 1])

        return (boundary_positions, boundary_probabilities)

    def get_binning(
        self,
        finite_sorted_dirac_deltas: np.ndarray,
        boundary_positions: np.ndarray,
        boundary_probabilities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        """
        Initialize the binning and populate it for the internal bins.
        """
        numberOfBins = 2 * number_of_finite_dirac_deltas
        bin_widths = np.array([np.nan] * numberOfBins)
        bin_widths[1:-1] = boundary_positions[2:-1] - boundary_positions[1:-2]
        bin_heights = np.array([np.nan] * numberOfBins)

        for i in range(1, number_of_finite_dirac_deltas - 1):
            averageHeight = finite_sorted_dirac_deltas[i][1] / (
                bin_widths[2 * i] + bin_widths[2 * i + 1]
            )
            bin_heights[2 * i] = (
                averageHeight * bin_widths[2 * i + 1] / bin_widths[2 * i]
            )
            bin_heights[2 * i + 1] = (
                averageHeight * bin_widths[2 * i] / bin_widths[2 * i + 1]
            )

        """
        Now, handle the extremal bins.
        First checking if (d/dx)^2 = 0 boundary condition has a solution.
        If not, falling back to the boundary condition d/dx = 0.
        """
        """
        First handling the left extereme bin.
        """
        w0 = None
        if number_of_finite_dirac_deltas >= 6:
            p0 = boundary_probabilities[1]
            w1 = bin_widths[1]
            w2 = bin_widths[2]
            d2 = bin_heights[2]
            a = d2 * w1 - p0
            b = a * w1 - p0 * w2
            c = p0 * w1 * (w1 + w2)
            det = b * b - 4 * a * c

            if det >= 0:
                """
                There are real roots. Pick the smallest positive root if there is one.
                """
                root1 = (-b + math.sqrt(det)) / (2 * a)
                root2 = (-b - math.sqrt(det)) / (2 * a)
                roots_positive = [root1 > 0, root2 > 0]

                if all(roots_positive):
                    w0 = min(root1, root2)
                elif any(roots_positive):
                    w0 = max(root1, root2)

        if (w0 is None) or (math.isinf(det)) or (math.isnan(det)):
            """
            The boundary condition d/dx = 0.
            """
            boundary_positions[0] = boundary_positions[1] - (
                boundary_positions[2] - boundary_positions[1]
            )
        else:
            """
            The boundary condition (d/dx)^2 = 0.
            """
            boundary_positions[0] = boundary_positions[1] - w0

        bin_widths[0] = boundary_positions[1] - boundary_positions[0]
        averageHeight = finite_sorted_dirac_deltas[0][1] / (
            bin_widths[0] + bin_widths[1]
        )
        bin_heights[0] = averageHeight * bin_widths[1] / bin_widths[0]
        bin_heights[1] = averageHeight * bin_widths[0] / bin_widths[1]

        """
        Now handling the right extereme bin.
        """
        w0 = None
        if number_of_finite_dirac_deltas >= 6:
            p0 = boundary_probabilities[-2]
            w1 = bin_widths[-2]
            w2 = bin_widths[-3]
            d2 = bin_heights[-3]
            a = d2 * w1 - p0
            b = a * w1 - p0 * w2
            c = p0 * w1 * (w1 + w2)
            det = b * b - 4 * a * c

            if det >= 0:
                """
                There are real roots. Pick the smallest positive root if there is one.
                """
                root1 = (-b + math.sqrt(det)) / (2 * a)
                root2 = (-b - math.sqrt(det)) / (2 * a)
                roots_positive = [root1 > 0, root2 > 0]

                if all(roots_positive):
                    w0 = min(root1, root2)
                elif any(roots_positive):
                    w0 = max(root1, root2)

        if (w0 is None) or (math.isinf(det)) or (math.isnan(det)):
            """
            The boundary condition d/dx = 0.
            """
            boundary_positions[-1] = boundary_positions[-2] + (
                boundary_positions[-2] - boundary_positions[-3]
            )
        else:
            """
            The boundary condition (d/dx)^2 = 0.
            """
            boundary_positions[-1] = boundary_positions[-2] + w0

        bin_widths[-1] = boundary_positions[-1] - boundary_positions[-2]
        averageHeight = finite_sorted_dirac_deltas[-1][1] / (
            bin_widths[-1] + bin_widths[-2]
        )
        bin_heights[-1] = averageHeight * bin_widths[-2] / bin_widths[-1]
        bin_heights[-2] = averageHeight * bin_widths[-1] / bin_widths[-2]

        return (boundary_positions, bin_widths, bin_heights)

    def plot_special_values_barplot(
        self,
        dist_value: DistributionalValue,
        extra_special_values: List[str] = []
    ) -> None:
        """
        Checks a `DistributionalValue` for Dirac deltas with positions equal to special
        values `NaN`, `INF`, and `-INF`, and potentially other `extra_special_values`,
        and creates a bar plot for these special values.
        """

        barplot_categories: List[Union[str, float]] = ["NaN", "-Inf", "Inf"]
        barplot_frequencies = [0.0, 0.0, 0.0]

        if len(dist_value.positions) == 0:
            raise ValueError("Input DistributionalValue has empty positions field!")

        default_mass = 1 / len(dist_value.positions)

        for index, value in enumerate(dist_value.positions):
            mass = (
                dist_value.masses[index]
                if len(dist_value.masses) != 0
                else default_mass
            )

            if math.isnan(value):
                barplot_frequencies[0] += mass
            elif (math.isinf(value)) and (value < 0):
                barplot_frequencies[1] += mass
            elif (math.isinf(value)) and (value > 0):
                barplot_frequencies[2] += mass
            elif value in extra_special_values:
                barplot_categories.append(value)
                barplot_frequencies.append(mass)
            else:
                continue

        plt.bar(
            barplot_categories,
            barplot_frequencies,
            width=0.55,
            facecolor="#75757540",
            edgecolor="#33A333",
            hatch="\\",
        )

        return

    def bin_pdf_expected_dirac_delta(
            self,
            boundary_positions: np.ndarray,
            bin_widths: np.ndarray,
            bin_heights: np.ndarray
    ) -> np.ndarray:
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

        for i in range(len(bin_widths)):
            probability = bin_widths[i] * bin_heights[i]
            probability_sum += probability
            moment_sum += (
                probability * (boundary_positions[i + 1] + boundary_positions[i]) / 2
            )

        expected_dirac_delta = np.array([moment_sum / probability_sum, probability_sum])

        return expected_dirac_delta

    def bin_pdf_to_ttr(
            self,
            boundary_positions: np.ndarray,
            bin_widths: np.ndarray,
            bin_heights: np.ndarray,
            order: int
    ) -> List[np.ndarray]:
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

        expected_dirac_delta = self.bin_pdf_expected_dirac_delta(
            boundary_positions, bin_widths, bin_heights
        )
        ttr = []

        if order == 0:
            return [expected_dirac_delta]
        else:
            for i, boundary_position in enumerate(boundary_positions):
                if boundary_position == expected_dirac_delta[0]:
                    low_boundary_positions = boundary_positions[: i + 1]
                    low_bin_widths = bin_widths[:i]
                    low_bin_heights = bin_heights[:i]
                    high_boundary_positions = boundary_positions[i:]
                    high_bin_widths = bin_widths[i:]
                    high_bin_heights = bin_heights[i:]
                    break
                elif boundary_position > expected_dirac_delta[0]:
                    low_boundary_positions = np.append(
                        boundary_positions[:i], expected_dirac_delta[0]
                    )
                    low_bin_widths = np.append(
                        bin_widths[: i - 1],
                        expected_dirac_delta[0] - boundary_positions[i - 1],
                    )
                    low_bin_heights = bin_heights[:i]
                    high_boundary_positions = np.insert(
                        boundary_positions[i:], 0, expected_dirac_delta[0]
                    )
                    high_bin_widths = np.insert(
                        bin_widths[i:], 0, boundary_position - expected_dirac_delta[0]
                    )
                    high_bin_heights = bin_heights[i - 1 :]
                    break

            ttr += self.bin_pdf_to_ttr(
                low_boundary_positions, low_bin_widths, low_bin_heights, order - 1
            )
            ttr += self.bin_pdf_to_ttr(
                high_boundary_positions, high_bin_widths, high_bin_heights, order - 1
            )

            return ttr
