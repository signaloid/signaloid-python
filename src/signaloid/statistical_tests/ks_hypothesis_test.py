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
from scipy.stats import uniform  # type: ignore[import-untyped]

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.statistical_tests.ks_hypothesis import (
    bootstrapped_kolmogorov_smirnov,
    bootstrapped_kolmogorov_smirnov_wrapper,
    kolmogorov_smirnov_wrapper,
)


class TestBootstrappedKolmogorovSmirnov(unittest.TestCase):
    def test_bootstrapped_ks_returns_valid_pvalue(self) -> None:
        """Combined p-value lies in [0, 1] across a TTR-order sweep."""
        min_ttr_order = 5
        max_ttr_order = 11
        uniform_ttr_positions = [
            [(2 * i - 1) / (2 ** (n + 1)) for i in range(1, 2**n + 1)]
            for n in range(min_ttr_order, max_ttr_order)
        ]
        distributions: list[DistributionalValue] = []
        for ttr_positions in uniform_ttr_positions:
            uniform_mass = 1 / len(ttr_positions)
            d = DistributionalValue(
                dirac_deltas=[
                    DiracDelta(position=float(p), mass=uniform_mass)
                    for p in ttr_positions
                ]
            )
            distributions.append(d)

        bootstrap_sample_sizes = [
            2 ** (i - 1) for i in range(min_ttr_order, max_ttr_order)
        ]
        number_of_bootstrap_samples = 10
        significance_levels = [1 / i for i in range(min_ttr_order, max_ttr_order)]

        rng = np.random.default_rng(seed=42)
        for i in range(max_ttr_order - min_ttr_order):
            with self.subTest(ttr_order=min_ttr_order + i):
                _, pvalue = bootstrapped_kolmogorov_smirnov(
                    discrete_representation=distributions[i],
                    true_cdf=uniform.cdf,
                    bootstrap_sample_size=bootstrap_sample_sizes[i],
                    number_of_bootstraps=number_of_bootstrap_samples,
                    significance_level=significance_levels[i],
                    rng=rng,
                )
                self.assertGreaterEqual(pvalue, 0.0)
                self.assertLessEqual(pvalue, 1.0)

    def test_bootstrapped_ks_is_reproducible_with_seeded_rng(self) -> None:
        """Same seed → identical p-values."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=(2 * i - 1) / 16, mass=1 / 8) for i in range(1, 9)
            ]
        )
        rng_a = np.random.default_rng(seed=12345)
        rng_b = np.random.default_rng(seed=12345)
        _, pvalue_a = bootstrapped_kolmogorov_smirnov(
            discrete_representation=dist,
            true_cdf=uniform.cdf,
            bootstrap_sample_size=64,
            number_of_bootstraps=5,
            significance_level=0.05,
            rng=rng_a,
        )
        _, pvalue_b = bootstrapped_kolmogorov_smirnov(
            discrete_representation=dist,
            true_cdf=uniform.cdf,
            bootstrap_sample_size=64,
            number_of_bootstraps=5,
            significance_level=0.05,
            rng=rng_b,
        )
        self.assertAlmostEqual(pvalue_a, pvalue_b, places=12)


class TestBootstrappedKolmogorovSmirnovValidation(unittest.TestCase):
    """`bootstrapped_kolmogorov_smirnov` rejects unusable representations
    with a clear ValueError instead of producing NaN p-values."""

    def _run(self, dist: DistributionalValue) -> tuple[bool, float]:
        return bootstrapped_kolmogorov_smirnov(
            discrete_representation=dist,
            true_cdf=uniform.cdf,
            bootstrap_sample_size=8,
            number_of_bootstraps=3,
            significance_level=0.05,
            rng=np.random.default_rng(seed=0),
        )

    def test_rejects_empty_distribution(self) -> None:
        """No Dirac deltas → zero total mass → ValueError."""
        with self.assertRaises(ValueError):
            self._run(DistributionalValue(dirac_deltas=[]))

    def test_rejects_all_zero_mass(self) -> None:
        """Zero total mass → ValueError."""
        with self.assertRaises(ValueError):
            self._run(
                DistributionalValue(dirac_deltas=[DiracDelta(position=1.0, mass=0.0)])
            )

    def test_rejects_non_finite_positive_mass_position(self) -> None:
        """A positive-mass Inf position is not sampleable → ValueError."""
        with self.assertRaises(ValueError):
            self._run(
                DistributionalValue(
                    dirac_deltas=[
                        DiracDelta(position=0.5, mass=0.5),
                        DiracDelta(position=float("inf"), mass=0.5),
                    ]
                )
            )

    def test_ignores_zero_mass_placeholder(self) -> None:
        """A zero-mass NaN placeholder is dropped, not rejected."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=0.25, mass=0.5),
                DiracDelta(position=0.75, mass=0.5),
                DiracDelta(position=float("nan"), mass=0.0),
            ]
        )
        accept, p_value = self._run(dist)
        self.assertIsInstance(accept, bool)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)


class TestBootstrappedKolmogorovSmirnovWrapper(unittest.TestCase):
    def test_wrapper_matches_core_when_called_with_same_inputs(self) -> None:
        """Wrapper(d, d_true) == core(d, d_true.cdf) for the same RNG."""
        d = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=p, mass=0.2) for p in (0.1, 0.3, 0.5, 0.7, 0.9)
            ]
        )
        d_true = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=(2 * i - 1) / 16, mass=1 / 8) for i in range(1, 9)
            ]
        )
        rng_a = np.random.default_rng(seed=99)
        rng_b = np.random.default_rng(seed=99)
        _, p_wrapper = bootstrapped_kolmogorov_smirnov_wrapper(
            discrete_representation=d,
            true_distribution=d_true,
            bootstrap_sample_size=32,
            number_of_bootstraps=5,
            significance_level=0.05,
            rng=rng_a,
        )
        _, p_core = bootstrapped_kolmogorov_smirnov(
            discrete_representation=d,
            true_cdf=d_true.cdf,
            bootstrap_sample_size=32,
            number_of_bootstraps=5,
            significance_level=0.05,
            rng=rng_b,
        )
        self.assertAlmostEqual(p_wrapper, p_core, places=12)

    def test_wrapper_rejects_non_distributional_value_inputs(self) -> None:
        """Non-DistributionalValue on either side raises ValueError."""
        good = DistributionalValue(dirac_deltas=[DiracDelta(position=0.5, mass=1.0)])
        with self.assertRaises(ValueError) as ctx:
            bootstrapped_kolmogorov_smirnov_wrapper(
                discrete_representation="not a DV",  # type: ignore[arg-type]
                true_distribution=good,
                bootstrap_sample_size=10,
                number_of_bootstraps=3,
                significance_level=0.05,
            )
        self.assertIn("discrete_representation", str(ctx.exception))
        self.assertIn("DistributionalValue", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            bootstrapped_kolmogorov_smirnov_wrapper(
                discrete_representation=good,
                true_distribution=42,  # type: ignore[arg-type]
                bootstrap_sample_size=10,
                number_of_bootstraps=3,
                significance_level=0.05,
            )
        self.assertIn("true_distribution", str(ctx.exception))
        self.assertIn("DistributionalValue", str(ctx.exception))


class TestKolmogorovSmirnovWrapper(unittest.TestCase):
    def test_accepts_list_float_sample_input(self) -> None:
        """Accepts list[float] sample, not just np.ndarray."""
        d_true = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=(2 * i - 1) / 16, mass=1 / 8) for i in range(1, 9)
            ]
        )
        result = kolmogorov_smirnov_wrapper(
            sample=[0.1, 0.3, 0.5, 0.7, 0.9],
            true_distribution=d_true,
            significance_level=0.05,
        )
        self.assertIsInstance(result, bool)

    def test_returns_bool(self) -> None:
        """Return type is a plain bool, not np.bool_."""
        d_true = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=(2 * i - 1) / 16, mass=1 / 8) for i in range(1, 9)
            ]
        )
        rng = np.random.default_rng(seed=7)
        sample = rng.uniform(size=100)
        result = kolmogorov_smirnov_wrapper(
            sample=sample, true_distribution=d_true, significance_level=0.05
        )
        self.assertIsInstance(result, bool)

    def test_rejects_non_distributional_value_true_distribution(self) -> None:
        """true_distribution must be a DistributionalValue."""
        with self.assertRaises(ValueError) as ctx:
            kolmogorov_smirnov_wrapper(
                sample=[0.1, 0.2, 0.3],
                true_distribution="not a DV",  # type: ignore[arg-type]
                significance_level=0.05,
            )
        self.assertIn("true_distribution", str(ctx.exception))
        self.assertIn("DistributionalValue", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
