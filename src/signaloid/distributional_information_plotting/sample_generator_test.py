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


import unittest

import numpy as np
from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.sample_generator import (
    sample_from_distributional_value,
    sample_generator,
)

SAMPLE_UX_STRING = "-0.000000Ux040000000000000001BCB03C52D58D3CE400000020C0048D2279B8AFF701B1807E6239F600BFFFF1F7A03E82B602A7EB881EDAA6C0BFFAFF0EC92B7D6E0321FCB58BB7F440BFF763B747227C880386DDE1E09EAEC0BFF47A086FFA91B003BA2C11EF08E580BFF1FDCC06ED8E9703F77BE72797F440BFEF88C5501503BA0429DAF9A7420E80BFEB75E0A582D1610454636C25A09D40BFE7B77D572D585D0456D709533085C0BFE43A6FD58615450472E978D476CD40BFE0E4BE4A8177310489FC4176BD8E80BFDB5AB694DC2F6D049CBBFB39689F80BFD51A3EFE52F0A004AAD48A9FB27AC0BFCDF7B07C22983104B59D8153294540BFC1E9102D4ACC1304BCB0EDEE5C1900BFA7D59C0E0AD59404C0374A760BE0C03FA7D59C0E0AD5A204C0374A760BE0C03FC1E9102D4ACC0504BCB0EDEE5C1C803FCDF7B07C22983104B59D81532945403FD51A3EFE52F0A904AAD48A9FB277003FDB5AB694DC2F6D049CBBFB39689F803FE0E4BE4A8177310489FC4176BD8E803FE43A6FD586153C0472E978D476D0C03FE7B77D572D58680456D709533082403FEB75E0A582D1560454636C25A0A1003FEF88C5501503D10429DAF9A7420AC03FF1FDCC06ED8EA203F77BE72797F7E03FF47A086FFA91AE03BA2C11EF08DE603FF763B747227C810386DDE1E09EAB203FFAFF0EC92B7D710321FCB58BB7F4403FFFF1F7A03E82AE02A7EB881EDAA32040048D2279B8AFD801B1807E6239FD40"
SAMPLE_UX_STRING_WITH_SPECIAL_VALUES = "nanUx0400000000000000017FF8000000000000000000063FF0000000000000155555555555550040080000000000001555555555555500000000000000000015555555555555007FF80000000000001555555555555500FFF000000000000015555555555555007FF00000000000001555555555555500"
# A well-formed Ux String whose 32 Dirac deltas all share the position 0.0, so
# the support collapses to a single point once coincident deltas are combined.
SAMPLE_UX_STRING_COINCIDENT_POSITIONS = "Ux040000000000000001000000000000000000000020000000000000000001B1807E6239F600000000000000000002A7EB881EDAA6C000000000000000000321FCB58BB7F44000000000000000000386DDE1E09EAEC0000000000000000003BA2C11EF08E580000000000000000003F77BE72797F44000000000000000000429DAF9A7420E8000000000000000000454636C25A09D4000000000000000000456D709533085C000000000000000000472E978D476CD4000000000000000000489FC4176BD8E800000000000000000049CBBFB39689F80000000000000000004AAD48A9FB27AC0000000000000000004B59D8153294540000000000000000004BCB0EDEE5C1900000000000000000004C0374A760BE0C0000000000000000004C0374A760BE0C0000000000000000004BCB0EDEE5C1C80000000000000000004B59D8153294540000000000000000004AAD48A9FB277000000000000000000049CBBFB39689F8000000000000000000489FC4176BD8E8000000000000000000472E978D476D0C000000000000000000456D7095330824000000000000000000454636C25A0A10000000000000000000429DAF9A7420AC0000000000000000003F77BE72797F7E0000000000000000003BA2C11EF08DE6000000000000000000386DDE1E09EAB2000000000000000000321FCB58BB7F440000000000000000002A7EB881EDAA320000000000000000001B1807E6239FD40"


# ============================================================================
# Unit Tests: sample_generator
# ============================================================================
class TestSampleGenerator(unittest.TestCase):
    """Tests for the sample_generator function."""

    def test_particle_returns_identical_samples(self) -> None:
        """A particle (single value) should return n identical samples."""
        particle_ux = "0.785398Ux0400000000000000003FE921FB54442D18000000013FE921FB54442D188000000000000000"
        n = 10
        samples = sample_generator(particle_ux, n)

        self.assertEqual(len(samples), n)
        self.assertTrue(np.all(samples == samples[0]))

    def test_distribution_returns_correct_number_of_samples(self) -> None:
        """Should return exactly n_samples samples."""
        n = 100
        samples = sample_generator(SAMPLE_UX_STRING, n)

        self.assertEqual(len(samples), n)

    def test_samples_are_finite(self) -> None:
        """All samples should be finite float values."""
        n = 1000
        samples = sample_generator(SAMPLE_UX_STRING, n)

        self.assertIsInstance(samples, np.ndarray)
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_coincident_positions_returns_identical_samples(self) -> None:
        """Many Dirac deltas at one position should return n identical samples."""
        n = 10
        samples = sample_generator(SAMPLE_UX_STRING_COINCIDENT_POSITIONS, n)

        self.assertEqual(len(samples), n)
        self.assertTrue(np.all(samples == 0.0))

    def test_coincident_positions_with_special_values(self) -> None:
        """A collapsed finite support mixed with NaN mass should still sample."""
        # Dirac deltas that all share one position, plus mass on NaN, so both
        # branches of the mixed case run against a support that collapses to a
        # single position.
        dirac_deltas = [DiracDelta(0.0, mass=0.9 / 32) for _ in range(32)]
        dirac_deltas.append(DiracDelta(np.nan, mass=0.1))
        dist_value = DistributionalValue(dirac_deltas=dirac_deltas)

        n = 1_000
        np.random.seed(42)
        samples = sample_from_distributional_value(dist_value, n)

        self.assertEqual(len(samples), n)
        self.assertGreater(np.sum(np.isnan(samples)), 0)
        self.assertTrue(np.all(samples[~np.isnan(samples)] == 0.0))

    def test_zero_finite_mass_raises(self) -> None:
        """A distributional value with no mass at all should raise ValueError."""
        dist_value = DistributionalValue.parse(SAMPLE_UX_STRING)
        if dist_value is None:
            raise ValueError("Failed to parse Ux-data into DistributionalValue.")

        for dd in dist_value.dirac_deltas:
            dd.mass = 0.0

        with self.assertRaises(ValueError):
            sample_from_distributional_value(dist_value, 10)

    def test_invalid_ux_data_raises(self) -> None:
        """Should raise ValueError for invalid Ux-data."""
        with self.assertRaises(ValueError):
            sample_generator("invalid_data", 10)

    def test_zero_samples_raises(self) -> None:
        """Should raise ValueError when n_samples is 0."""
        with self.assertRaises(ValueError):
            sample_generator(SAMPLE_UX_STRING, 0)

    def test_negative_samples_raises(self) -> None:
        """Should raise ValueError when n_samples is negative."""
        with self.assertRaises(ValueError):
            sample_generator(SAMPLE_UX_STRING, -1)

    def test_special_values_returns_correct_count(self) -> None:
        """Sampling from a distribution with NaN/Inf should return n samples."""
        n = 1000
        samples = sample_generator(SAMPLE_UX_STRING_WITH_SPECIAL_VALUES, n)
        self.assertEqual(len(samples), n)

    def test_special_values_contains_non_finite(self) -> None:
        """Samples from a distribution with NaN/Inf should include non-finite values."""
        n = 10_000
        np.random.seed(42)
        samples = sample_generator(SAMPLE_UX_STRING_WITH_SPECIAL_VALUES, n)
        n_non_finite = np.sum(~np.isfinite(samples))
        self.assertGreater(n_non_finite, 0)

    def test_special_values_contains_nan(self) -> None:
        """Samples should include NaN when the distribution has NaN mass."""
        n = 10_000
        np.random.seed(42)
        samples = sample_generator(SAMPLE_UX_STRING_WITH_SPECIAL_VALUES, n)
        self.assertGreater(np.sum(np.isnan(samples)), 0)

    def test_statistical_sanity(self) -> None:
        """Verify sample mean and std are close to distribution parameters."""
        n = 10_000
        np.random.seed(0)
        samples = sample_generator(SAMPLE_UX_STRING, n)

        dist_value = DistributionalValue.parse(SAMPLE_UX_STRING)
        if dist_value is None:
            raise ValueError("Failed to parse Ux-data into DistributionalValue.")

        if dist_value.mean is not None and dist_value.variance is not None:
            std = np.sqrt(dist_value.variance)

            threshold_mean = 5 * std / np.sqrt(n)
            self.assertLess(np.abs(np.mean(samples) - dist_value.mean), threshold_mean)

            threshold_std = 5 * std / np.sqrt(2 * n)
            self.assertLess(np.abs(np.std(samples) - std), threshold_std)


if __name__ == "__main__":
    unittest.main()
