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

import os
from typing import TYPE_CHECKING
import numpy as np
from scipy.stats import halfnorm, anderson  # type: ignore
from scipy.integrate import trapezoid  # type: ignore
from signaloid.benchmarking.types import TaggedDistributionalValue
from signaloid.distributional.distributional import DistributionalValue
from signaloid.benchmarking.distribution_helpers.density import _histogram_pdf
from signaloid.benchmarking.config import (
    DistanceMetrics,
    VariableTypes,
    EquivMC,
    BenchmarkingVariables,
    RepresentationTypes,
)
import signaloid.distributional_information_plotting.plot_wrapper as plot_wrapper
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from signaloid.distributional_distance.wasserstein import wasserstein_p_distance
from numpy.typing import ArrayLike
import pandas as pd
import re
import math
from heapq import merge
from signaloid.benchmarking.equivalent_mc.load import (
    _load_mc_scalar,
    _load_mc,
)
import numpy.typing as npt

if TYPE_CHECKING:
    from signaloid.benchmarking.equivalent_mc.equivalent_monte_carlo import (
        EquivalentMonteCarlo,
    )
    from signaloid.benchmarking.equivalent_mc.equivalent_mc_main import (
        LoadDataComputeEquivalentMCArgs,
    )
import matplotlib.pyplot as plt

# Quantile bounds used to derive an outlier-robust plotting range for
# sample-based distributions (e.g. Monte Carlo adversaries), so that a few
# extreme samples in the tails do not stretch the shared x-axis domain.
_ROBUST_RANGE_LOWER_QUANTILE = 0.005
_ROBUST_RANGE_UPPER_QUANTILE = 0.995


def _distribution_range(
    dv: DistributionalValue, robust: bool
) -> tuple[float, float] | None:
    """
    Compute the x-axis range spanned by a distribution.

    Args:
        dv: The distribution whose support range to measure.
        robust: When True, clip the range to the
            [``_ROBUST_RANGE_LOWER_QUANTILE``, ``_ROBUST_RANGE_UPPER_QUANTILE``]
            quantiles so extreme outliers (common in Monte Carlo samples) do not
            stretch the domain. When False, use the full support.

    Returns:
        The ``(low, high)`` range, or ``None`` if it cannot be determined
        (e.g. empty distribution or non-finite bounds).
    """
    treat_as_samples = (
        getattr(dv, "representation_type", None) == RepresentationTypes.SAMPLES
    )
    lower, upper = (
        (_ROBUST_RANGE_LOWER_QUANTILE, _ROBUST_RANGE_UPPER_QUANTILE)
        if robust
        else (0.0, 1.0)
    )
    try:
        low, high = dv.inverse_cdf(
            np.array([lower, upper]), treat_as_samples=treat_as_samples
        )
    except Exception:
        return None
    if not (math.isfinite(low) and math.isfinite(high)) or high < low:
        return None
    return float(low), float(high)


def _compute_shared_xlim(
    ground_truth: TaggedDistributionalValue,
    uxhw: list[TaggedDistributionalValue],
    adversaries: list[TaggedDistributionalValue],
    padding_fraction: float = 0.05,
) -> tuple[float, float] | None:
    """
    Compute a single x-axis domain shared by the ground-truth, UxHw, and
    adversary/Monte-Carlo distribution plots so they are directly comparable.

    The ground-truth and UxHw distributions contribute their full support (they
    are trustworthy binned representations), while the adversary distributions
    contribute an outlier-robust range so that Monte Carlo tails do not blow up
    the domain. The union of these ranges, padded by ``padding_fraction`` on
    each side (matching the padding ``plot_wrapper.plot`` applies when it
    auto-scales), is returned.

    Args:
        ground_truth: The ground-truth distribution.
        uxhw: The UxHw configurations to be plotted.
        adversaries: The adversary distributions the Monte Carlo plots are
            drawn from.
        padding_fraction: Fraction of the total range to pad on each side.

    Returns:
        The shared ``(min, max)`` x-limits, or ``None`` if no range could be
        determined (callers then fall back to per-plot auto-scaling).
    """
    ranges: list[tuple[float, float]] = []

    gt_range = _distribution_range(ground_truth.dv, robust=False)
    if gt_range is not None:
        ranges.append(gt_range)

    for dist in uxhw:
        dist_range = _distribution_range(dist.dv, robust=False)
        if dist_range is not None:
            ranges.append(dist_range)

    for adversary in adversaries:
        adversary_range = _distribution_range(adversary.dv, robust=True)
        if adversary_range is not None:
            ranges.append(adversary_range)

    if not ranges:
        return None

    min_x = min(low for low, _ in ranges)
    max_x = max(high for _, high in ranges)
    range_spacing = padding_fraction * (max_x - min_x)
    return (min_x - range_spacing, max_x + range_spacing)


def _compute_asymptotic_distribution_brownian_bridge(
    ground_truth: TaggedDistributionalValue,
    num_points: int,
    num_samples: int,
    distance_type: str,
) -> DistributionalValue:
    """
    Compute the asymptotic distance distribution for the given metric.

    Dispatches to the Wasserstein-1, Wasserstein-2, or absolute-mean-deviation
    variant based on ``distance_type``.

    Args:
        ground_truth: The ground-truth distribution.
        num_points: Number of discretization points for the integral.
        num_samples: Number of Monte Carlo samples.
        distance_type: Which asymptotic distribution to compute.

    Returns:
        The asymptotic distance distribution.

    Raises:
        ValueError: If ``distance_type`` is not supported.
    """

    if distance_type == DistanceMetrics.WASSERSTEIN_1:
        dist = _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
            ground_truth, num_points=num_points, num_samples=num_samples
        )
    elif distance_type == DistanceMetrics.WASSERSTEIN_2:
        dist = _compute_wasserstein_2_asymptotic_distribution_brownian_bridge(
            ground_truth, num_points=num_points, num_samples=num_samples
        )
    elif distance_type == "AbsoluteMeanDeviation":
        dist = _compute_absolute_mean_deviation_asymptotic_distribution(
            ground_truth, num_points
        )
    else:
        raise ValueError(f"Unsupported distance_type: {distance_type!r}")

    return dist


def _compute_wasserstein_1_asymptotic_distribution_brownian_bridge(
    ground_truth: TaggedDistributionalValue, num_points: int, num_samples: int
) -> DistributionalValue:
    """
    Compute the distribution of the integral of |B(t)| dQ(t) over t in [0, 1],
    where B(t) is a Brownian bridge and Q(t) is the ground truth's quantile
    function.

    Args:
        ground_truth: The ground-truth distribution.
        num_points: Number of discretization points for the integral.
        num_samples: Number of Monte Carlo samples.

    Returns:
        The asymptotic Wasserstein-1 distance distribution.
    """
    small_number = 1e-5  # Prevents quantile function blowing up

    # The carried DistributionalValue has no `representation_type` attribute, so
    # `treat_as_samples` is False and we take the mass-weighted branch. This is
    # identical to the prior behaviour for MonteCarlo / WeightedSamples ground
    # truths (neither is the "Samples" type that selects the other branch).
    treat_as_samples = (
        getattr(ground_truth.dv, "representation_type", None)
        == RepresentationTypes.SAMPLES
    )
    t = np.linspace(small_number, 1 - small_number, num_points)
    # inverse_cdf is the vectorised inverse-CDF, so it accepts an array
    # (unlike the scalar-only ``quantile``).
    quantile_vals = ground_truth.dv.inverse_cdf(t, treat_as_samples=treat_as_samples)

    pdf_vals = np.asarray(
        _histogram_pdf(
            ground_truth.dv, quantile_vals, treat_as_samples=treat_as_samples
        )
    )
    pdf_vals = pdf_vals[pdf_vals > 0]

    values = 1 / pdf_vals

    # Call specific function for performing integral
    brownian_bridge_integrals = _compute_integral_of_brownian_bridge(
        len(values), num_samples, values
    )
    dist = DistributionalValue.from_samples(np.array(brownian_bridge_integrals))
    # Tag as SAMPLES so a downstream consumer deriving `treat_as_samples` from
    # `representation_type` takes the samples branch (`np.quantile` inverse- CDF
    # / auto-binned pdf). preserving the prior behaviour when this returned a
    # `Distribution` built via `from_samples`.
    setattr(dist, "representation_type", RepresentationTypes.SAMPLES)

    return dist


def _compute_wasserstein_2_asymptotic_distribution_brownian_bridge(
    ground_truth: TaggedDistributionalValue, num_points: int, num_samples: int
) -> DistributionalValue:
    """
    Compute the distribution of sqrt(integral of |B(t)|^2 dQ(t)) over t in
    [0, 1], where B(t) is a Brownian bridge and Q(t) is the ground truth's
    quantile function.

    Args:
        ground_truth: The ground-truth distribution.
        num_points: Number of discretization points for the integral.
        num_samples: Number of Monte Carlo samples.

    Returns:
        The asymptotic Wasserstein-2 distance distribution.
    """
    small_number = 1e-5  # Prevents quantile function blowing up

    # See the W1 variant for why the carried DistributionalValue (no
    # ``representation_type``) takes the mass-weighted branch.
    treat_as_samples = (
        getattr(ground_truth.dv, "representation_type", None)
        == RepresentationTypes.SAMPLES
    )
    t = np.linspace(small_number, 1 - small_number, num_points)
    # inverse_cdf is the vectorised inverse-CDF, so it accepts an array
    # (unlike the scalar-only ``quantile``).
    quantile_vals = ground_truth.dv.inverse_cdf(t, treat_as_samples=treat_as_samples)
    values = 1 / _histogram_pdf(
        ground_truth.dv, quantile_vals, treat_as_samples=treat_as_samples
    )

    # Call specific function for performing integral
    brownian_bridge_integrals = _compute_integral_of_brownian_bridge_squared(
        num_points, num_samples, np.asarray(values)
    )
    brownian_bridge_integrals = np.sqrt(np.array(brownian_bridge_integrals))

    dist = DistributionalValue.from_samples(brownian_bridge_integrals)
    # Tag as SAMPLES so a downstream consumer deriving `treat_as_samples` from
    # `representation_type` takes the samples branch (`np.quantile` inverse- CDF
    # / auto-binned pdf). This preserves the prior behaviour when this returned a
    # `Distribution` built via `from_samples`.
    setattr(dist, "representation_type", RepresentationTypes.SAMPLES)

    return dist


def _compute_absolute_mean_deviation_asymptotic_distribution(
    ground_truth: TaggedDistributionalValue, num_points: int
) -> DistributionalValue:
    """
    Compute the distribution of the absolute mean deviation |E(X) - E(Xₙ)|,
    where Xₙ is the empirical process simulating X with n samples. This quantity
    is distributed as HalfNormal(sigma * √2 / √π).

    Args:
        ground_truth: The ground-truth distribution whose mean and standard
            deviation parameterize the asymptotic half-normal distribution.
        num_points: Number of discretization points for the integral.

    Returns:
        The asymptotic half-normal distance distribution.
    """
    positions = ground_truth.dv.positions
    masses = ground_truth.dv.masses
    mean = np.dot(positions, masses)
    std = np.sqrt(np.dot(positions**2, masses) - mean**2)

    positions = np.linspace(0, 12 * std, num_points)
    weights = halfnorm.pdf(x=positions, scale=std)

    dist = DistributionalValue.from_weighted_samples(positions, weights)

    return dist


def _simulate_brownian_bridge(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a standard Brownian bridge path over t in [0, 1].

    Args:
        N: Number of discretization points.

    Returns:
        The time points and the simulated Brownian bridge values.
    """
    # Generate discrete time values betwen 0 and 1
    t = np.linspace(0, 1, N + 1)

    # Simulate regular Wiener process
    W = np.random.normal(0, np.sqrt(np.diff(np.insert(t, 0, 0))), N + 1)

    # Compute Browian Bridge (pinned Wiener process)
    B = np.cumsum(W)
    B = B - t * B[-1]

    return t, B


def _compute_integral_of_brownian_bridge(
    num_points: int, num_samples: int, quantile_derivative_values: np.ndarray
) -> list[float]:
    """
    Compute the distribution of the integral of |B(t)| dQ(t) over t in [0, 1],
    where B(t) is a Brownian bridge and Q'(t) (the quantile function's
    derivative) is supplied via ``quantile_derivative_values``.

    Args:
        num_points: Number of discretization points.
        num_samples: Number of Monte Carlo paths to simulate.
        quantile_derivative_values: Quantile-derivative values Q'(t).

    Returns:
        One integral value per simulated path.
    """
    integrals = []

    for _ in range(num_samples):
        # Simulate Brownian bridge B(u) over u in [0, 1]
        u, B_u = _simulate_brownian_bridge(num_points)

        # Compute the integral numerically
        integral = trapezoid(
            y=np.abs(B_u[1:]) * quantile_derivative_values, x=u[1:], dx=1 / num_points
        )
        integrals.append(integral)

    return integrals


def _compute_integral_of_brownian_bridge_squared(
    num_points: int, num_samples: int, quantile_derivative_values: np.ndarray
) -> list[float]:
    """
    Compute the distribution of the integral of |B(t)|^2 dQ(t) over t in
    [0, 1], where B(t) is a Brownian bridge and Q'(t) (the quantile function's
    derivative) is supplied via ``quantile_derivative_values``.

    Args:
        num_points: Number of discretization points.
        num_samples: Number of Monte Carlo paths to simulate.
        quantile_derivative_values: Quantile-derivative values Q'(t).

    Returns:
        One integral value per simulated path.
    """
    integrals = []

    for _ in range(num_samples):
        # Simulate Brownian bridge B(u) over u in [0, 1]
        u, B_u = _simulate_brownian_bridge(num_points)

        # Compute the integral numerically
        integral = trapezoid(
            y=B_u[1:] ** 2 * quantile_derivative_values**2, x=u[1:], dx=1 / num_points
        )
        integrals.append(integral)

    return integrals


def _compute_asymptotic_wasserstein_distribution_mean(
    ground_truth: TaggedDistributionalValue,
) -> float:
    """
    Compute the integral of sqrt(2 / pi) * sqrt(F * (1 - F)), where F is the
    ground truth's CDF. Divided by sqrt(N), this equals the mean Wasserstein-1
    distance between the ground truth and an N-sample empirical MC simulation
    of it.

    Args:
        ground_truth: The ground-truth distribution.

    Returns:
        The value of the integral.
    """

    def integrand(F: np.ndarray) -> np.ndarray:
        return np.asarray(np.sqrt(F * (1 - F)))

    # Calculate the support, CDF and integrand values. The carried
    # DistributionalValue carries no `representation_type`, so use its
    # cumulative-mass CDF directly (the prior
    # `Distribution.calculate_distribution_values` weighted-samples branch).
    support = ground_truth.dv.positions
    cdf_values = np.cumsum(ground_truth.dv.masses)
    y = integrand(cdf_values)

    # Remove numerical errors, all values should be in (0, 0.5)
    y = np.where((np.isreal(y)) & (y >= 0) & (y < 0.5), y, 0)

    # Compute the integral and multiply by prefactor
    mean = trapezoid(y, support)
    mean *= np.sqrt(2 / np.pi)

    return float(mean)


def _generate_representative_mc_plots(
    emcc: "EquivalentMonteCarlo",
    variable_name: str,
    distance_type: str,
    plots_dir: str = "",
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Generate MC plots from representative samples that approximate the UxHw
    distribution's distance to the ground truth.

    Builds up Monte Carlo samples progressively in chunks until the distance
    between the MC samples and the ground truth approximates the UxHw-to-ground
    truth distance:

    1. For each row in the results DataFrame, extract the target EMCC count.
    2. Compute the target distance (UxHw vs ground truth).
    3. Build MC samples progressively in chunks to match that target distance.
    4. Fall back to the best attempt if convergence fails within the attempt
       limit.

    Args:
        emcc: EquivalentMonteCarlo object holding the results DataFrame,
            adversary array, and ground truth.
        variable_name: Name of the variable being analysed.
        distance_type: Distance metric to use.
        plots_dir: Directory to write the plots into. Defaults to the current working directory when empty.
        xlim: Limits applied to the x-axis so the MC plots use the same domain
            as the ground-truth and UxHw plots. ``None`` falls back to per-plot
            auto-scaling.
    """
    # Convert variable name to filesystem-safe string
    expr_string = re.sub(r"\s+", "-", variable_name.lower())

    # Maximum MC size we generate
    max_mc_size = 1_000_000

    count_list = []
    # Main loop over UxHw configurations
    df = pd.DataFrame(emcc.variable.emcc_results.emcc_data)
    for _, row in df.iterrows():
        try:
            # Skip if the EMCC column does not exist.
            if EquivMC.EMCC not in df.columns:
                continue

            # Get equivalent Monte Carlo count if it exists
            equiv_mc_count = row[EquivMC.EMCC]
            if pd.isna(equiv_mc_count) or not isinstance(equiv_mc_count, (int, float)):
                continue
            equiv_mc_count = int(equiv_mc_count)

            # Some UxHw results will have same equivMC counts so we can skip repeats
            if equiv_mc_count in count_list:
                continue
            # Set upper limit on MC size to avoid memory related issues
            if equiv_mc_count > max_mc_size:
                equiv_mc_count = max_mc_size
            count_list.append(equiv_mc_count)

            # Get target distance based on whether we're using binned UxHw or not
            target_distance = (
                row[BenchmarkingVariables.UXHW_BINNED_DISTANCE]
                if emcc.use_binned_uxhw
                else row[BenchmarkingVariables.UXHW_DISTANCE]
            )
            selected_samples = _generate_progressive_samples(
                emcc, equiv_mc_count, target_distance, distance_type
            )
            adversary_dist = DistributionalValue.from_samples(selected_samples)
            adv_plot_path = f"{expr_string}-adversary-{equiv_mc_count}.png"
            if plots_dir:
                adv_plot_path = os.path.join(plots_dir, adv_plot_path)
            plot_wrapper.plot(
                plot_data=PlotData(adversary_dist),
                path=adv_plot_path,
                save=True,
                xlim=xlim,
            )
        except Exception as e:
            print(
                f"Failed to generate representative MC plot for {expr_string} with an equivalent MC count of {equiv_mc_count}: {e}"
            )
            continue


def _distance_func(
    distance_type: str,
    u_values: ArrayLike,
    v_values: ArrayLike,
    u_weights: ArrayLike | None = None,
    v_weights: ArrayLike | None = None,
) -> float:
    """
    Compute the distance between two weighted distributions.

    Only used for computing the equivalent Monte Carlo counts.

    Args:
        distance_type: Distance metric (Wasserstein-1 or Wasserstein-2).
        u_values: Support points for the first distribution.
        v_values: Support points for the second distribution.
        u_weights: Weights for the first distribution.
        v_weights: Weights for the second distribution.

    Returns:
        The distance between the two distributions.

    Raises:
        ValueError: If ``distance_type`` is not a supported metric.
    """

    if distance_type == DistanceMetrics.WASSERSTEIN_1:
        return wasserstein_p_distance(u_values, u_weights, v_values, v_weights, p=1)
    elif distance_type == DistanceMetrics.WASSERSTEIN_2:
        return wasserstein_p_distance(u_values, u_weights, v_values, v_weights, p=2)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def _generate_adversary_list(
    args: "LoadDataComputeEquivalentMCArgs", target_expr: str, expr_type: str
) -> list[TaggedDistributionalValue]:
    """
    Load the adversary distributions for Monte Carlo comparison.

    Distribution variables load a single distribution (wrapped in a list).
    Scalar variables load several directly.

    Args:
        args: Config holding the adversary database path and table name.
        target_expr: Variable name to load from the database.
        expr_type: Expression type (distribution or scalar).

    Returns:
        The adversary distributions loaded from the database.
    """
    adversary_list: list[TaggedDistributionalValue] = []
    if expr_type == VariableTypes.DISTRIBUTION:
        adversary_list.append(
            _load_mc(
                db_path=args["adversary_database_path"],
                table=args["adversary_table_name"],
                target_expression=target_expr,
            )
        )
    else:
        adversary_list = _load_mc_scalar(
            db_path=args["adversary_database_path"],
            table=args["adversary_table_name"],
            target_expression=target_expr,
        )
    return adversary_list


def _generate_comparison_plots(emcc: "EquivalentMonteCarlo", prefix: str) -> None:
    """
    Plot the distance distribution (scaled by sqrt(MC_count)) against the
    asymptotic distribution (e.g. from a Brownian-bridge simulation).

    Args:
        emcc: The Equivalent Monte Carlo object.
        prefix: Prefix for the figure name.
    """

    for mc_count in emcc.variable.emcc_results.equiv_mc_list:
        if mc_count > 0 and mc_count < emcc.adversary_size_max:
            if emcc.variable.type == VariableTypes.DISTRIBUTION:
                samples = emcc.variable.asymptotic_distribution.samples
                assert samples is not None, (
                    "asymptotic_distribution.samples must be populated "
                    "for distribution-typed variables before plotting"
                )
                emcc._plot_equiv_mc_vs_brownian_bridge(
                    mc_count,
                    samples,
                    prefix=prefix,
                )
            else:
                emcc._plot_equiv_mc_vs_half_norm(mc_count, prefix=prefix)


def _plot_adversary_distances(emcc: "EquivalentMonteCarlo", prefix: str) -> None:
    """
    Plot the distance distribution for each adversary.

    Args:
        emcc: The Equivalent Monte Carlo object.
        prefix: Prefix for the figure name.
    """
    for i, (k, lst) in enumerate(emcc._adversary_distances.items()):
        plt.scatter(
            np.log(k),
            np.log(np.mean(lst)),
            marker="x",
            color="r",
            label="Measured Adversary Distances" if i == 0 else None,
        )

    plt.xlabel("Log of Adversary Size")
    plt.ylabel("Log of Mean Wasserstein Distance")
    plt.legend()
    # Tag the ground truth in the filename by its representation type (from the
    # carrier metadata, never off a Distribution). Only WeightedSamples counts
    # as "weighted". MonteCarlo / Samples are "unweighted".
    string = (
        "unweighted"
        if emcc.ground_truth.representation_type in ("Samples", "MonteCarlo")
        else "weighted"
    )
    plt.savefig(
        f"{prefix}-adversary_distances-{emcc.ground_truth.dv.UR_order}_{string}_ground_truth_samples-{emcc.n_adversaries}_adversaries.png",
        dpi=500,
    )
    plt.close()


def _generate_equivalent_mc_plots(
    emcc: "EquivalentMonteCarlo",
    args: "LoadDataComputeEquivalentMCArgs",
    target_expr: str,
    expr_description: str,
    distance_type: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Generate the configured equivalent Monte Carlo analysis plots.

    Args:
        emcc: The Equivalent Monte Carlo object.
        args: Config specifying which plots to generate.
        target_expr: Variable name used for plot file naming.
        expr_description: Human-readable description for plot titles.
        distance_type: Distance metric used (e.g. Wasserstein-1).
        xlim: Limits applied to the x-axis for the representative MC plots, so they use
            the same domain as the ground-truth and UxHw plots. ``None`` falls
            back to per-plot auto-scaling.
    """
    plots_dir = args["plots_dir"]
    prefixed_expr = os.path.join(plots_dir, target_expr) if plots_dir else target_expr

    if args["plot_adversary_distances"]:
        _plot_adversary_distances(emcc=emcc, prefix=prefixed_expr)

    if args["plot_comparison_distributions"]:
        _generate_comparison_plots(emcc=emcc, prefix=prefixed_expr)
    # Plot equivalent MC run with "similar" distance to UxHw
    if args["plot_distributions"] and emcc.variable.type == VariableTypes.DISTRIBUTION:
        print("Generating MC plots...")
        _generate_representative_mc_plots(
            emcc, expr_description, distance_type, plots_dir=plots_dir, xlim=xlim
        )


def _attempt_convergence(
    emcc: "EquivalentMonteCarlo",
    selected_samples: np.ndarray,
    sample_size_step: int,
    sample_size: int,
    target_constant: float,
    distance_type: str,
    threshold: float = 0.05,
) -> tuple[bool, np.ndarray, float]:
    """
    Draw one more chunk of samples and check convergence to the target distance.

    Args:
        emcc: The Equivalent Monte Carlo object.
        selected_samples: Samples accumulated so far (sorted).
        sample_size_step: Number of new samples to draw this attempt.
        sample_size: Total sample size the distance is scaled against.
        target_constant: Target value of distance * sqrt(sample_size).
        distance_type: Distance metric to use.
        threshold: Max proportional difference from the target to count as
            converged.

    Returns:
        A tuple of (converged, merged samples, proportional difference from
        the target).
    """
    new_samples = np.random.choice(emcc._adversary_array, size=sample_size_step)
    new_samples.sort()

    # Merge sort with previously selected samples in O(n) time
    samples = np.array(list(merge(selected_samples, new_samples)))

    mc_distance = _distance_func(
        distance_type,
        u_values=samples,
        v_values=emcc.ground_truth.dv.positions,
        v_weights=emcc.ground_truth.dv.masses,
    )

    # Check convergence using the target constant
    attempt_constant = mc_distance * np.sqrt(sample_size)
    proportion_diff = np.abs(attempt_constant - target_constant) / target_constant

    converged = proportion_diff < threshold
    return converged, samples, proportion_diff


def _generate_mc_samples_for_plotting(
    emcc: "EquivalentMonteCarlo",
    selected_samples: np.ndarray,
    sample_size: int,
    sample_size_step: int,
    target_constant: float,
    distance_type: str,
    equiv_mc_count: int,
) -> np.ndarray:
    """
    Generate samples for one target size, retrying until convergence.

    Repeatedly calls :func:`_attempt_convergence`, keeping the best attempt as
    a fallback if none converge within the attempt limit.

    Args:
        emcc: The Equivalent Monte Carlo object.
        selected_samples: Samples accumulated so far (sorted).
        sample_size: Total sample size the distance is scaled against.
        sample_size_step: Number of new samples to draw per attempt.
        target_constant: Target value of distance * sqrt(sample_size).
        distance_type: Distance metric to use.
        equiv_mc_count: Equivalent MC count, used to bound the attempt count.

    Returns:
        The converged samples, or the best-effort fallback samples.
    """
    max_attempts = max(5, min(1000, math.ceil(10_000_000 / equiv_mc_count)))
    num_attempts = 0
    smallest_diff = np.inf
    fallback_samples = np.array([])

    while num_attempts < max_attempts:
        num_attempts += 1

        converged, samples, proportion_diff = _attempt_convergence(
            emcc,
            selected_samples,
            sample_size_step,
            sample_size,
            target_constant,
            distance_type,
        )

        # Update fallback option with best result so far
        if proportion_diff < smallest_diff:
            smallest_diff = proportion_diff
            fallback_samples = samples.copy()

        if converged:
            return samples

    # Return fallback if no convergence
    return fallback_samples


def _generate_progressive_samples(
    emcc: "EquivalentMonteCarlo",
    equiv_mc_count: int,
    target_distance: float,
    distance_type: str,
) -> np.ndarray:
    """
    Build samples up to ``equiv_mc_count`` in chunks, converging each chunk.

    Args:
        emcc: The Equivalent Monte Carlo object.
        equiv_mc_count: Target total number of samples.
        target_distance: UxHw-to-ground-truth distance to match.
        distance_type: Distance metric to use.

    Returns:
        The accumulated samples.
    """
    target_constant = target_distance * np.sqrt(equiv_mc_count)
    sample_sizes, sample_size_steps = _calculate_sample_sizes(equiv_mc_count)

    selected_samples = np.array([])

    # Progressive sampling: build up samples in chunks
    for sample_size, sample_size_step in zip(sample_sizes, sample_size_steps):
        selected_samples = _generate_mc_samples_for_plotting(
            emcc,
            selected_samples,
            sample_size,
            sample_size_step,
            target_constant,
            distance_type,
            equiv_mc_count,
        )

    return selected_samples


def _calculate_sample_sizes(
    equiv_mc_count: int, sample_size_step: int = 100_000
) -> tuple[list[int], list[int]]:
    """
    Calculate cumulative sample sizes and step sizes for progressive sampling.

    Args:
        equiv_mc_count: Target total number of samples.
        sample_size_step: Chunk size for each progressive step.

    Returns:
        A tuple of (cumulative sample sizes, per-step increments).
    """
    if equiv_mc_count <= sample_size_step:
        sample_sizes = [equiv_mc_count]
    else:
        sample_sizes = list(
            range(sample_size_step, equiv_mc_count, sample_size_step)
        ) + [equiv_mc_count]

    sample_size_steps = [sample_sizes[0]] + [
        sample_sizes[i] - sample_sizes[i - 1] for i in range(1, len(sample_sizes))
    ]

    return sample_sizes, sample_size_steps


def _compute_asymptotic_distribution_scalar_empirical(
    samples: npt.ArrayLike, sample_size: int, ground_truth_value: float
) -> tuple[float, float, bool]:
    """
    Compute the asymptotic distribution of N samples, fitting a Gaussian to
    ``samples * sqrt(N)`` centred on the ground truth.

    Args:
        samples: Input samples.
        sample_size: Number of samples.
        ground_truth_value: Ground-truth value to centre samples around.

    Returns:
        A tuple of (mean, standard deviation, is_normal) for the centred and
        scaled samples, where ``is_normal`` is the Anderson-Darling verdict.
    """
    samples_array = (np.asarray(samples, dtype=float) - ground_truth_value) * np.sqrt(
        sample_size
    )

    mean = np.mean(samples_array)
    std = np.std(samples_array, ddof=1)

    is_normal = anderson_darling_test(samples_array)

    return mean.item(), std.item(), is_normal


def anderson_darling_test(samples: npt.ArrayLike) -> bool:
    """
    Anderson-Darling test for normality.

    Args:
        samples: Samples to test.

    Returns:
        ``True`` if the samples appear normal at the 15% significance level.
    """
    result = anderson(samples, dist="norm")

    if result.statistic < result.critical_values[0]:
        print("✓ Data appears normally distributed at 15% level")
        return True
    else:
        print("✗ Data does NOT appear normally distributed at 15% level")
        return False
