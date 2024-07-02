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

import unittest
import random
import numpy as np

from .plot_histogram_dirac_deltas import PlotHistogramDiracDeltas


class TestCreateBinning(unittest.TestCase):
    def test_create_binning_property_dirac_deltas_average_of_bins(self):
        probability_threshold = 1e-12
        position_threshold = 1e-12
        number_of_testcases = 1000
        low_range = [-100, 0]
        high_range = [0, 100]
        input_dirac_delta_ensembles = []
        number_of_dirac_deltas = []
        plotter = PlotHistogramDiracDeltas()

        for i in range(number_of_testcases):
            current_number_of_dirac_deltas = random.sample(range(2, 1_000 + 1), 1)[0]
            number_of_dirac_deltas.append(current_number_of_dirac_deltas)
            low_value = np.random.uniform(*low_range)
            high_value = np.random.uniform(*high_range)
            dirac_delta_positions = np.random.uniform(low_value, high_value, current_number_of_dirac_deltas)
            dirac_delta_masses = np.random.uniform(0, 1, current_number_of_dirac_deltas)
            dirac_delta_masses /= sum(dirac_delta_masses)
            input_dirac_delta_ensembles.append(
                np.array(sorted(list(zip(dirac_delta_positions, dirac_delta_masses))))
            )

        for i, input_ensemble in enumerate(input_dirac_delta_ensembles):
            boundary_positions, bin_widths, bin_heights = plotter.create_binning(input_ensemble, 0, False)
            input_dirac_delta_positions = input_ensemble[:, 0]
            input_dirac_delta_masses = input_ensemble[:, 1]

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
                self.assertLess(abs(input_dirac_delta_masses[j] - probability_under_bins), probability_threshold)
                self.assertLess(abs(input_dirac_delta_positions[j] - mean_of_bins), position_threshold)

    def test_create_binning_property_preserve_ttr(self):
        probability_threshold = 1e-12
        position_threshold = 1e-12
        number_of_testcases = 1000
        number_of_samples = 10000
        gaussian_mean_range = [-100, 100]
        gaussian_standard_deviation_range = [1, 100]
        ttr_orders = []
        input_ttrs = []
        plotter = PlotHistogramDiracDeltas()

        for i in range(number_of_testcases):
            gaussian_mean = np.random.uniform(*gaussian_mean_range)
            gaussian_standard_deviation = np.random.uniform(*gaussian_standard_deviation_range)
            dirac_delta_positions = np.random.normal(gaussian_mean, gaussian_standard_deviation, number_of_samples)
            dirac_delta_masses = [1/number_of_samples] * number_of_samples
            ttr_order = random.sample(range(4, 11), 1)[0]
            input_dirac_deltas = np.array(list(zip(dirac_delta_positions, dirac_delta_masses)))
            input_ttr_positions, input_ttr_masses = dirac_deltas_to_ttr(input_dirac_deltas, ttr_order)
            if len(input_ttr_positions) != 2 ** ttr_order:
                continue
            ttr_orders.append(ttr_order)
            input_ttrs.append(np.array(list(zip(input_ttr_positions, input_ttr_masses))))

        for i, input_ttr in enumerate(input_ttrs):
            exponent = ttr_orders[i]
            boundary_positions, bin_widths, bin_heights = plotter.create_binning(input_ttr, exponent, True)
            binning_dirac_deltas = np.array(
                list(zip(boundary_positions[:-1] + bin_widths[::] / 2, np.multiply(bin_heights[::], bin_widths[::])))
            )
            binning_ttr_positions, binning_ttr_masses = dirac_deltas_to_ttr(binning_dirac_deltas, exponent)
            input_ttr_positions = input_ttr[:, 0]
            input_ttr_masses = input_ttr[:, 1]
            self.assertEqual(len(input_ttr_positions), len(binning_ttr_positions))

            for j in range(len(input_ttr_positions)):
                self.assertLess(abs(input_ttr_masses[j] - binning_ttr_masses[j]), probability_threshold)
                self.assertLess(abs(input_ttr_positions[j] - binning_ttr_positions[j]), position_threshold)


def dirac_deltas_to_ttr(dirac_deltas, order, count=0):
    """
    Computes TTR for an input ensemble of Dirac deltas.
    Args:
        dirac_deltas: Input ensemble of n Dirac deltas specified as [position, probability mass].
                      Numpy array with shape (n, 2).
        order: TTR order.
        count: Counts recursion level. Always use 0.
    Returns:
        (ttr_positions, ttr_masses): Positions and probability masses of Dirac deltas in the output TTR.
    """
    if count == 0:
        dirac_deltas[:, 1] /= sum(dirac_deltas[:, 1])
    count += 1
    ttr = np.array([])
    if len(dirac_deltas) == 0:
        current_dirac_delta = np.array([])
        low_dirac_deltas = np.array([])
        high_dirac_deltas = np.array([])
    else:
        p = sum(dirac_deltas[:, 1])
        average = sum(np.multiply(dirac_deltas[:, 1], dirac_deltas[:, 0])) / p
        current_dirac_delta = np.array([average, p])
        low_dirac_deltas = np.array([dd for dd in dirac_deltas if dd[0] < average])
        high_dirac_deltas = np.array([dd for dd in dirac_deltas if dd[0] >= average])

    if order > 0:
        order -= 1
        ttr = np.concatenate((ttr, dirac_deltas_to_ttr(low_dirac_deltas, order, count)))
        ttr = np.concatenate((ttr, dirac_deltas_to_ttr(high_dirac_deltas, order, count)))
        if count == 1:
            ttr_positions = ttr[0::2]
            ttr_masses = ttr[1::2]
            return ttr_positions, ttr_masses
        return ttr

    ttr = np.append(ttr, current_dirac_delta)
    return ttr


if __name__ == "__main__":
    unittest.main()
