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

import argparse
import sys
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2, ks_1samp  # type: ignore[import-untyped]

from signaloid.distributional.distributional import DistributionalValue

CdfFunction = Callable[[Any], Any]


def bootstrapped_kolmogorov_smirnov(
    discrete_representation: DistributionalValue,
    true_cdf: CdfFunction,
    bootstrap_sample_size: int,
    number_of_bootstraps: int,
    significance_level: float,
    rng: np.random.Generator | None = None,
) -> tuple[bool, float]:
    """Bootstrapped Kolmogorov-Smirnov goodness-of-fit test.

    Args:
        discrete_representation: Discrete approximation of the target
            distribution.
        true_cdf: Callable CDF of the true continuous distribution.
        bootstrap_sample_size: Number of samples drawn per bootstrap.
        number_of_bootstraps: How many bootstrap iterations to run.
        significance_level: Threshold below which the hypothesis is
            rejected.
        rng: Optional generator for reproducibility.

    Returns:
        Tuple ``(accept_hypothesis, combined_p_value)``.
    """
    if bootstrap_sample_size <= 0:
        raise ValueError("bootstrap_sample_size must be greater than 0.")
    if number_of_bootstraps <= 0:
        raise ValueError("number_of_bootstraps must be greater than 0.")
    if significance_level < 0.0 or significance_level > 1.0:
        raise ValueError("significance_level must be in [0, 1].")
    if rng is None:
        rng = np.random.default_rng()

    positions = discrete_representation.positions
    masses = discrete_representation.masses
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError(
            "discrete_representation must have a strictly positive total "
            "mass to sample from."
        )
    # Zero-mass deltas (including NaN/Inf placeholders) carry no
    # probability; sample only from the positive-mass support and require
    # those positions to be finite so rng.choice / ks_1samp get usable
    # values instead of producing NaN probabilities or NaN p-values.
    positive_mass = masses > 0.0
    positions = positions[positive_mass]
    masses = masses[positive_mass]
    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "discrete_representation must have finite positions for all "
            "positive-mass Dirac deltas."
        )
    normalised_masses = masses / total_mass

    p_values = np.empty(number_of_bootstraps)
    for i in range(number_of_bootstraps):
        sample = rng.choice(positions, size=bootstrap_sample_size, p=normalised_masses)
        result = ks_1samp(sample, true_cdf)
        p_values[i] = result.pvalue

    # Fisher's method: combine p-values via -2 Σ log(p_i) ~ chi²(2k).
    clipped_p_values = np.maximum(p_values, np.finfo(float).tiny)
    fisher_statistic = -2.0 * float(np.sum(np.log(clipped_p_values)))
    degrees_of_freedom = 2 * number_of_bootstraps
    combined_p_value = 1.0 - float(chi2.cdf(fisher_statistic, df=degrees_of_freedom))
    return combined_p_value >= significance_level, combined_p_value


def bootstrapped_kolmogorov_smirnov_wrapper(
    discrete_representation: DistributionalValue,
    true_distribution: DistributionalValue,
    bootstrap_sample_size: int,
    number_of_bootstraps: int,
    significance_level: float,
    rng: np.random.Generator | None = None,
) -> tuple[bool, float]:
    """Bootstrapped KS test between two DistributionalValues.

    Args:
        discrete_representation: Discrete approximation under test.
        true_distribution: Reference DistributionalValue.
        bootstrap_sample_size: Number of samples drawn per bootstrap.
        number_of_bootstraps: How many bootstrap iterations to run.
        significance_level: Threshold below which the hypothesis is
            rejected.
        rng: Optional generator for reproducibility.

    Returns:
        Tuple ``(accept_hypothesis, combined_p_value)``.

    Raises:
        ValueError: if either argument is not a DistributionalValue.
    """
    if not isinstance(discrete_representation, DistributionalValue):
        raise ValueError(
            f"discrete_representation is of type "
            f"{type(discrete_representation).__name__} and not an "
            "instance of DistributionalValue."
        )
    if not isinstance(true_distribution, DistributionalValue):
        raise ValueError(
            f"true_distribution is of type "
            f"{type(true_distribution).__name__} and not an "
            "instance of DistributionalValue."
        )
    return bootstrapped_kolmogorov_smirnov(
        discrete_representation=discrete_representation,
        true_cdf=true_distribution.cdf,
        bootstrap_sample_size=bootstrap_sample_size,
        number_of_bootstraps=number_of_bootstraps,
        significance_level=significance_level,
        rng=rng,
    )


def kolmogorov_smirnov_wrapper(
    sample: list[float] | np.ndarray,
    true_distribution: DistributionalValue,
    significance_level: float,
) -> bool:
    """One-sample Kolmogorov-Smirnov test against a DistributionalValue.

    Args:
        sample: Observed sample.
        true_distribution: Reference DistributionalValue whose empirical
            CDF defines the null hypothesis.
        significance_level: Threshold below which the hypothesis is
            rejected.

    Returns:
        ``True`` when the p-value is at least the significance level.

    Raises:
        ValueError: if ``true_distribution`` is not a DistributionalValue.
    """
    if not isinstance(true_distribution, DistributionalValue):
        raise ValueError(
            f"true_distribution is of type "
            f"{type(true_distribution).__name__} and not an "
            "instance of DistributionalValue."
        )
    if significance_level < 0.0 or significance_level > 1.0:
        raise ValueError("significance_level must be in [0, 1].")
    p_value = float(ks_1samp(sample, true_distribution.cdf).pvalue)
    return p_value >= significance_level


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m signaloid.statistical_tests.ks_hypothesis",
        description=(
            "Run a Kolmogorov-Smirnov hypothesis test between two ux "
            "strings at a given significance level."
        ),
    )
    parser.add_argument("dist_ux", help="ux string for the distribution under test")
    parser.add_argument("reference_ux", help="ux string for the reference distribution")
    parser.add_argument("significance_level", type=float)
    parser.add_argument(
        "--test",
        choices=["bootstrapped", "one-sample"],
        default="bootstrapped",
        help="which KS hypothesis test to run (default: bootstrapped)",
    )
    parser.add_argument(
        "--bootstrap-sample-size",
        type=int,
        default=128,
        help="samples drawn per bootstrap (bootstrapped only; default: 128)",
    )
    parser.add_argument(
        "--number-of-bootstraps",
        type=int,
        default=50,
        help="number of bootstrap iterations (bootstrapped only; default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility (bootstrapped only)",
    )
    args = parser.parse_args()

    dist = DistributionalValue.parse(args.dist_ux)
    if dist is None:
        raise ValueError(f"Could not parse distribution from ux string {args.dist_ux}")
    reference = DistributionalValue.parse(args.reference_ux)
    if reference is None:
        raise ValueError(
            f"Could not parse reference from ux string {args.reference_ux}"
        )

    if args.test == "bootstrapped":
        accept, p_value = bootstrapped_kolmogorov_smirnov_wrapper(
            discrete_representation=dist,
            true_distribution=reference,
            bootstrap_sample_size=args.bootstrap_sample_size,
            number_of_bootstraps=args.number_of_bootstraps,
            significance_level=args.significance_level,
            rng=np.random.default_rng(args.seed),
        )
        detail = f"Combined p-value: {p_value}"
    else:
        # The one-sample test takes a raw sample; use the support positions
        # of the first distribution as the observed sample.
        accept = kolmogorov_smirnov_wrapper(
            sample=dist.positions,
            true_distribution=reference,
            significance_level=args.significance_level,
        )
        detail = "One-sample test on the first distribution's positions."

    label = f"KS {args.test} test"
    if accept:
        print(
            f"[SUCCESS] {label}: null hypothesis accepted at significance "
            f"{args.significance_level}. {detail}"
        )
    else:
        print(
            f"[FAILURE] {label}: null hypothesis rejected at significance "
            f"{args.significance_level}. {detail}"
        )
    sys.exit(0 if accept else 1)
