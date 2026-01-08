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

import random
import unittest

import numpy as np

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import PlotData


class TestCreateBinning(unittest.TestCase):
    def test_create_binning_property_dirac_deltas_average_of_bins(self) -> None:
        """
        Tests that the binning created by the method `PlotData.create_binning()`
        satisfies for each input Dirac delta the property that the average of the two bins that
        surround the input Dirac delta is equal to the Dirac delta itself.
        """
        probability_threshold: float = 1e-12
        position_threshold: float = 1e-12
        number_of_testcases: int = 1000
        low_range: tuple[int, int] = (-100, 0)
        high_range: tuple[int, int] = (0, 100)
        input_dirac_delta_ensembles: list[list[DiracDelta]] = []
        number_of_dirac_deltas: list[int] = []

        for i in range(number_of_testcases):
            current_number_of_dirac_deltas = random.sample(range(2, 1_000 + 1), 1)[0]
            number_of_dirac_deltas.append(current_number_of_dirac_deltas)
            low_value = np.random.uniform(*low_range)
            high_value = np.random.uniform(*high_range)
            dirac_delta_positions = np.random.uniform(
                low_value, high_value, current_number_of_dirac_deltas
            )
            dirac_delta_masses = np.random.uniform(0, 1, current_number_of_dirac_deltas)
            dirac_delta_masses /= sum(dirac_delta_masses)

            dirac_deltas = [
                DiracDelta(position, mass=mass)
                for position, mass in
                zip(dirac_delta_positions, dirac_delta_masses)
            ]
            dirac_deltas.sort()

            input_dirac_delta_ensembles.append(dirac_deltas)

        for i, input_ensemble in enumerate(input_dirac_delta_ensembles):
            boundary_positions, bin_widths, bin_heights = PlotData._create_binning(
                input_ensemble, 0, False
            )

            for j in range(number_of_dirac_deltas[i]):
                probability_under_first_bin = bin_widths[2 * j] * bin_heights[2 * j]
                probability_under_second_bin = bin_widths[2 * j + 1] * bin_heights[2 * j + 1]
                probability_under_bins = probability_under_first_bin + probability_under_second_bin
                mean_of_first_bin = boundary_positions[2 * j] + bin_widths[2 * j] / 2
                mean_of_second_bin = boundary_positions[2 * j + 1] + bin_widths[2 * j + 1] / 2
                mean_of_bins = (
                    mean_of_first_bin * probability_under_first_bin +
                    mean_of_second_bin * probability_under_second_bin
                ) / probability_under_bins
                self.assertLess(
                    abs(input_ensemble[j].mass - probability_under_bins),
                    probability_threshold
                )
                self.assertLess(
                    abs(input_ensemble[j].position - mean_of_bins),
                    position_threshold
                )

    def test_create_binning_property_preserve_ttr(self) -> None:
        """
        Tests that the binning created by the method `PlotData.create_binning()`
        preserves TTRs, that is, if the input Dirac deltas form a valid TTR, then the Dirac deltas
        of the TTR of the created binning exactly coincide with the input Dirac deltas.
        """
        probability_threshold: float = 1e-12
        position_threshold: float = 1e-12
        number_of_testcases: int = 1000
        number_of_samples: int = 10000
        gaussian_mean_range: tuple[int, int] = (-100, 100)
        gaussian_standard_deviation_range: tuple[int, int] = (1, 100)
        ttr_orders: list[int] = []
        input_ttrs: list[list[DiracDelta]] = []

        for i in range(number_of_testcases):
            gaussian_mean = np.random.uniform(*gaussian_mean_range)
            gaussian_standard_deviation = np.random.uniform(*gaussian_standard_deviation_range)
            dirac_delta_positions = np.random.normal(
                gaussian_mean, gaussian_standard_deviation, number_of_samples
            )
            dirac_delta_masses = [1/number_of_samples] * number_of_samples
            ttr_order = random.sample(range(4, 11), 1)[0]

            input_dirac_deltas = [
                DiracDelta(position, mass=mass)
                for position, mass in
                zip(dirac_delta_positions, dirac_delta_masses)
            ]

            input_ttr_dirac_deltas = dirac_deltas_to_ttr(
                input_dirac_deltas, ttr_order
            )

            if len(input_ttr_dirac_deltas) != 2 ** ttr_order:
                continue
            ttr_orders.append(ttr_order)
            input_ttrs.append(input_ttr_dirac_deltas)

        for i, input_ttr in enumerate(input_ttrs):
            exponent = ttr_orders[i]

            boundary_positions, bin_widths, bin_heights = PlotData._create_binning(
                input_ttr, exponent, True
            )

            binning_dirac_deltas: list[DiracDelta] = []
            for position, width, height in zip(boundary_positions[:-1], bin_widths, bin_heights):
                binning_dirac_deltas.append(DiracDelta(
                    position=position + width / 2,
                    mass=width * height
                ))

            binning_ttr = dirac_deltas_to_ttr(
                binning_dirac_deltas, exponent
            )

            self.assertEqual(len(input_ttr), len(binning_ttr))

            for input_ttr_dd, binning_ttr_dd in zip(input_ttr, binning_ttr):
                self.assertLess(
                    abs(input_ttr_dd.mass - binning_ttr_dd.mass),
                    probability_threshold
                )
                self.assertLess(
                    abs(input_ttr_dd.position - binning_ttr_dd.position),
                    position_threshold
                )


def dirac_deltas_to_ttr(
    dirac_deltas: list[DiracDelta],
    order: int,
    count: int = 0
) -> list[DiracDelta]:
    """
    Computes the TTR for an input ensemble of Dirac deltas.

    Args:
        dirac_deltas: Input ensemble of n Dirac deltas specified as [position, probability mass].
            Numpy array with shape (n, 2).
        order: TTR order.
        count: Counts recursion level. Always use 0.
    Returns:
        ttr: The TTR of the input bin PDF, a (2 ** `order`)-length array of Dirac deltas
            with each Dirac delta of the form np.array([position, mass]).
    """
    if count == 0:
        # Normalize mass
        normalizer_total_mass: float = 0
        for dd in dirac_deltas:
            normalizer_total_mass += dd.mass

        for dd in dirac_deltas:
            dd.mass /= normalizer_total_mass

    count += 1

    current_dirac_delta: list[DiracDelta] = []
    low_dirac_deltas: list[DiracDelta] = []
    high_dirac_deltas: list[DiracDelta] = []

    if len(dirac_deltas) > 0:
        total_mass: float = 0
        average_position: float = 0
        for dd in dirac_deltas:
            total_mass += dd.mass
            average_position += dd.position * dd.mass
        average_position /= total_mass

        current_dirac_delta = [DiracDelta(average_position, mass=total_mass)]
        low_dirac_deltas = [dd for dd in dirac_deltas if dd.position < average_position]
        high_dirac_deltas = [dd for dd in dirac_deltas if dd.position >= average_position]

    ttr: list[DiracDelta] = []
    if order > 0:
        order -= 1

        ttr.extend(dirac_deltas_to_ttr(low_dirac_deltas, order, count))
        ttr.extend(dirac_deltas_to_ttr(high_dirac_deltas, order, count))

        return ttr

    ttr.extend(current_dirac_delta)

    return ttr


if __name__ == "__main__":
    unittest.main()
