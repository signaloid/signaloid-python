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
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)


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
                for position, mass in zip(dirac_delta_positions, dirac_delta_masses)
            ]
            dirac_deltas.sort()

            input_dirac_delta_ensembles.append(dirac_deltas)

        for i, input_ensemble in enumerate(input_dirac_delta_ensembles):
            boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
                input_ensemble, 0, False
            )

            for j in range(number_of_dirac_deltas[i]):
                probability_under_first_bin = bin_widths[2 * j] * bin_heights[2 * j]
                probability_under_second_bin = (
                    bin_widths[2 * j + 1] * bin_heights[2 * j + 1]
                )
                probability_under_bins = (
                    probability_under_first_bin + probability_under_second_bin
                )
                mean_of_first_bin = boundary_positions[2 * j] + bin_widths[2 * j] / 2
                mean_of_second_bin = (
                    boundary_positions[2 * j + 1] + bin_widths[2 * j + 1] / 2
                )
                mean_of_bins = (
                    mean_of_first_bin * probability_under_first_bin
                    + mean_of_second_bin * probability_under_second_bin
                ) / probability_under_bins
                self.assertLess(
                    abs(input_ensemble[j].mass - probability_under_bins),
                    probability_threshold,
                )
                self.assertLess(
                    abs(input_ensemble[j].position - mean_of_bins), position_threshold
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
            gaussian_standard_deviation = np.random.uniform(
                *gaussian_standard_deviation_range
            )
            dirac_delta_positions = np.random.normal(
                gaussian_mean, gaussian_standard_deviation, number_of_samples
            )
            dirac_delta_masses = [1 / number_of_samples] * number_of_samples
            ttr_order = random.sample(range(4, 11), 1)[0]

            input_dirac_deltas = [
                DiracDelta(position, mass=mass)
                for position, mass in zip(dirac_delta_positions, dirac_delta_masses)
            ]

            input_ttr_dirac_deltas = dirac_deltas_to_ttr(input_dirac_deltas, ttr_order)

            if len(input_ttr_dirac_deltas) != 2**ttr_order:
                continue
            ttr_orders.append(ttr_order)
            input_ttrs.append(input_ttr_dirac_deltas)

        for i, input_ttr in enumerate(input_ttrs):
            exponent = ttr_orders[i]

            boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
                input_ttr, exponent, True
            )

            binning_dirac_deltas: list[DiracDelta] = []
            for position, width, height in zip(
                boundary_positions[:-1], bin_widths, bin_heights
            ):
                binning_dirac_deltas.append(
                    DiracDelta(position=position + width / 2, mass=width * height)
                )

            binning_ttr = dirac_deltas_to_ttr(binning_dirac_deltas, exponent)

            self.assertEqual(len(input_ttr), len(binning_ttr))

            for input_ttr_dd, binning_ttr_dd in zip(input_ttr, binning_ttr):
                self.assertLess(
                    abs(input_ttr_dd.mass - binning_ttr_dd.mass), probability_threshold
                )
                self.assertLess(
                    abs(input_ttr_dd.position - binning_ttr_dd.position),
                    position_threshold,
                )


class TestPlotDataFromSamples(unittest.TestCase):
    """Tests for PlotData.from_samples() (delegates to DistributionalValue)."""

    def test_basic_finite_samples(self) -> None:
        """from_samples should produce valid PlotData from finite floats."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        pd = PlotData.from_samples(samples)

        self.assertGreater(len(pd.positions), 0)
        self.assertGreater(len(pd.masses), 0)
        self.assertFalse(pd.dist.has_special_values)
        self.assertAlmostEqual(pd.dist.nan_dirac_delta.mass, 0.0)
        self.assertAlmostEqual(pd.dist.neg_inf_dirac_delta.mass, 0.0)
        self.assertAlmostEqual(pd.dist.pos_inf_dirac_delta.mass, 0.0)
        self.assertIsNotNone(pd.dist.mean)

    def test_density_integrates_to_approximately_one(self) -> None:
        """Histogram density should integrate to ~1 when all samples are finite."""
        np.random.seed(42)
        samples = np.random.normal(5, 2, 10_000)
        pd = PlotData.from_samples(samples)

        bin_widths = pd.positions[1:] - pd.positions[:-1]
        total_area = float(np.sum(bin_widths * pd.masses))
        self.assertAlmostEqual(total_area, 1.0, places=1)

    def test_special_value_masses(self) -> None:
        """NaN, -Inf, +Inf masses should match their proportions."""
        samples = np.array(
            [1.0, 2.0, 3.0, np.nan, np.nan, -np.inf, np.inf, np.inf, np.inf, 4.0]
        )
        pd = PlotData.from_samples(samples)

        self.assertTrue(pd.dist.has_special_values)
        self.assertAlmostEqual(pd.dist.nan_dirac_delta.mass, 2 / 10)
        self.assertAlmostEqual(pd.dist.neg_inf_dirac_delta.mass, 1 / 10)
        self.assertAlmostEqual(pd.dist.pos_inf_dirac_delta.mass, 3 / 10)

    def test_all_identical_samples(self) -> None:
        """All-identical finite samples should produce a single Dirac delta."""
        samples = np.full(100, 3.14)
        pd = PlotData.from_samples(samples)

        self.assertEqual(len(pd.positions), 1)
        self.assertAlmostEqual(pd.positions[0], 3.14)
        self.assertAlmostEqual(pd.masses[0], 1.0)

    def test_empty_samples_raises(self) -> None:
        """An empty array should raise ValueError."""
        with self.assertRaises(ValueError):
            PlotData.from_samples(np.array([]))

    def test_plotting_resolution_is_power_of_two(self) -> None:
        """The plotting_resolution should be a power of 2."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        pd = PlotData.from_samples(samples)

        self.assertIsNotNone(pd.plotting_resolution)
        assert pd.plotting_resolution is not None
        self.assertEqual(pd.plotting_resolution & (pd.plotting_resolution - 1), 0)

    def test_custom_plotting_resolution(self) -> None:
        """plotting_resolution parameter should control the number of bins."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        pd = PlotData.from_samples(samples, plotting_resolution=32)

        self.assertIsNotNone(pd.plotting_resolution)
        assert pd.plotting_resolution is not None
        self.assertLessEqual(pd.plotting_resolution, 32)

    def test_mean_value_close_to_sample_mean(self) -> None:
        """mean_value should be close to the mean of the finite samples."""
        np.random.seed(42)
        samples = np.random.normal(5, 1, 500)
        pd = PlotData.from_samples(samples)

        finite = samples[np.isfinite(samples)]
        self.assertIsNotNone(pd.dist.mean)
        assert pd.dist.mean is not None
        self.assertAlmostEqual(pd.dist.mean, float(np.mean(finite)), places=5)

    def test_plot_from_samples_succeeds(self) -> None:
        """plot() should succeed when given PlotData built from samples."""
        import matplotlib

        matplotlib.use("Agg")
        from signaloid.distributional_information_plotting.plot_wrapper import (
            plot,
        )

        np.random.seed(42)
        samples = np.random.normal(0, 1, 500)
        pd = PlotData.from_samples(samples)

        result = plot(pd, path="/dev/null", save=True)
        self.assertTrue(result)

    def test_plot_from_samples_with_special_values_succeeds(self) -> None:
        """plot() should succeed for samples containing NaN/Inf."""
        import matplotlib

        matplotlib.use("Agg")
        from signaloid.distributional_information_plotting.plot_wrapper import (
            plot,
        )

        np.random.seed(42)
        samples = np.concatenate(
            [
                np.random.normal(0, 1, 400),
                np.full(50, np.nan),
                np.full(25, np.inf),
                np.full(25, -np.inf),
            ]
        )
        pd = PlotData.from_samples(samples)

        result = plot(pd, path="/dev/null", save=True)
        self.assertTrue(result)


def dirac_deltas_to_ttr(
    dirac_deltas: list[DiracDelta], order: int, count: int = 0
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
        high_dirac_deltas = [
            dd for dd in dirac_deltas if dd.position >= average_position
        ]

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
