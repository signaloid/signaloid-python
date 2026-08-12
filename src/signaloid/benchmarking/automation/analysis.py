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

import csv
import math
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import halfnorm  # type: ignore

from signaloid.benchmarking.automation.benchmarking_utils import (
    compute_emcc_prediction,
    write_uxhw_distance_file,
)
from signaloid.benchmarking.automation.measurement_loader import (
    load_asymptotic_dist,
    load_uxhw_distances,
)
from signaloid.benchmarking.automation.sample_generator import (
    generate_scalar_samples,
)
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
    UxhwDistanceRecord,
)
from signaloid.benchmarking.config import (
    AsymptoticDistanceDistribution,
    BenchmarkingVariables,
    DistanceMetrics,
    EquivMC,
    Measurements,
    ReportingMethods,
    ReportingNumbers,
    RepresentationTypes,
    VariableTypes,
)
from signaloid.benchmarking.distribution_helpers.representation_health import (
    _representation_blow_up_reason,
)
from signaloid.benchmarking.equivalent_mc.equivalent_mc_main import (
    load_data_and_compute_equivalent_mc,
)
from signaloid.benchmarking.equivalent_mc.equivalent_mc_utils import (
    _compute_asymptotic_distribution_brownian_bridge,
    _compute_asymptotic_distribution_scalar_empirical,
)
from signaloid.benchmarking.equivalent_mc.load import (
    _load_ground_truth,
    _load_uxhw_distributions,
)
from signaloid.distributional_distance.wasserstein import (
    wasserstein_1_uxhw_wrapper,
    wasserstein_2_uxhw_wrapper,
)
from signaloid.distributional_distance.binned_wasserstein import (
    binned_wasserstein_1_uxhw_wrapper,
)
from signaloid.distributional_distance.scalar import relative_error_uxhw_wrapper


def _get_equiv_mc_list(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
) -> None:
    """
    Populate ``equiv_mc_list`` from each variable's ``emcc_data``.

    Uses the measured ``EMCC`` values, falling back to the predicted
    ``EMCC_PREDICTED`` values when no measured ones are present.

    Args:
        benchmarking_variables: Variables whose ``equiv_mc_list`` is populated.
    """
    for variable in benchmarking_variables:
        variable.emcc_results.equiv_mc_list = sorted(
            set(
                [
                    d[EquivMC.EMCC]
                    for d in variable.emcc_results.emcc_data
                    if EquivMC.EMCC in d
                ]
            )
        )
        if len(variable.emcc_results.equiv_mc_list) == 0:
            variable.emcc_results.equiv_mc_list = sorted(
                set(
                    [
                        d[EquivMC.EMCC_PREDICTED]
                        for d in variable.emcc_results.emcc_data
                        if EquivMC.EMCC_PREDICTED in d
                    ]
                )
            )


def load_emcc_data(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    output_data_file: str,
) -> None:
    """
    Load EMCC data into each variable's `emcc_results.emcc_data` list
    if the list has not already been populated.

    Args:
        benchmarking_variables: Variables to populate.
        output_data_file: CSV file written by
            :func:`compute_emcc_predictions` that backs the EMCC data
            when the in-memory list is empty.
    """
    if not benchmarking_variables:
        return

    if not benchmarking_variables[0].emcc_results.emcc_data:
        print(
            f"In-memory EMCC data not found. "
            f"Loading EMCC data from {output_data_file}"
        )

        try:
            # Load and group data by variable name
            emcc_df = pd.read_csv(output_data_file)
            emcc_dict = emcc_df.to_dict("records")

            # Create a lookup dict for faster access
            data_by_variable: dict = {}
            for record in emcc_dict:
                var_name = record[BenchmarkingVariables.VARIABLE_DESCRIPTION]
                data_by_variable.setdefault(var_name, []).append(record)

            # Populate each variable's emcc_results.emcc_data
            for variable in benchmarking_variables:
                variable.emcc_results.emcc_data.extend(
                    data_by_variable.get(variable.description, [])
                )

        except FileNotFoundError:
            print(f"Error: File not found at {output_data_file}")
            print("Cannot continue without EMCC data. Terminating.")
            raise

    _get_equiv_mc_list(benchmarking_variables=benchmarking_variables)


def compute_speedups(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
) -> None:
    """
    Compute the ``SPEEDUP`` field for each EMCC record.

    Sets ``SPEEDUP`` to native-time / database-time where both are available and
    the database time is non-zero, else ``None``.

    Args:
        benchmarking_variables: Variables whose ``EMCC`` records are updated.
    """
    # Calculate speedup for each record in each variable
    for variable in benchmarking_variables:
        for dic in variable.emcc_results.emcc_data:
            native_time = dic.get(Measurements.NATIVE_IN_APP_TIME)
            db_time = dic.get(Measurements.DB_TIME)

            # Calculate speedup if both values exist and db_time is non-zero
            if native_time is not None and db_time is not None and db_time != 0:
                dic[Measurements.SPEEDUP] = native_time / db_time
            else:
                dic[Measurements.SPEEDUP] = None


def compute_uxhw_distances(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    distance_type: str,
    ground_truth_db_path: str,
    ground_truth_type: str,
    tracing_db_path: str,
    representation_types: list[str],
    representation_sizes: list[int],
    uxhw_distance_file: str,
) -> None:
    """
    Compute UxHw distances for every variable / representation pair.

    Loads each variable's ground-truth and UxHw distributions
    from the timing/tracing databases and records distance metrics on
    the corresponding :class:`UxhwDistanceRecord` entries.

    Args:
        benchmarking_variables: Variables to process.
        distance_type: Which distance metric to use (e.g.
            ``DistanceMetrics.WASSERSTEIN_1``).
        ground_truth_db_path: SQLite database containing ground-truth
            samples.
        ground_truth_type: Representation type string (e.g.
            ``RepresentationTypes.MONTE_CARLO``).
        tracing_db_path: SQLite database containing the traced UxHw
            distributions.
        representation_types: UxHw UR-type strings to evaluate.
        representation_sizes: UxHw UR-size integers to evaluate.
        uxhw_distance_file: CSV file to write the resulting distance
            records to.
    """
    # binned_uxhw_distance_fn stays `None` for the W2 / Binned-W1 metrics so
    # the per-variable `is not None` check below falls through correctly.
    uxhw_distance_fn: Callable | None = None
    binned_uxhw_distance_fn: Callable | None = None
    if distance_type == DistanceMetrics.WASSERSTEIN_1:
        uxhw_distance_fn = wasserstein_1_uxhw_wrapper
        binned_uxhw_distance_fn = binned_wasserstein_1_uxhw_wrapper
    elif distance_type == DistanceMetrics.WASSERSTEIN_2:
        uxhw_distance_fn = wasserstein_2_uxhw_wrapper
    elif distance_type == DistanceMetrics.BINNED_WASSERSTEIN_1:
        uxhw_distance_fn = binned_wasserstein_1_uxhw_wrapper
    else:
        raise RuntimeError(
            f"Invalid distance type configuration "
            f"{distance_type}. Check py for valid options"
        )

    for variable in benchmarking_variables:
        ground_truth = _load_ground_truth(
            db_path=ground_truth_db_path,
            table=ground_truth_type,
            target_expression=variable.name,
            expression_type=variable.type,
            monte_carlo=ground_truth_type == RepresentationTypes.MONTE_CARLO,
        )

        # Get all distributions from databases
        variable.emcc_results.emcc_data = []
        uxhw_data: list[TaggedDistributionalValue] = _load_uxhw_distributions(
            db_path=tracing_db_path,
            tables=[EquivMC.TRACING_TABLE],
            target_expr=variable.name,
            ur_types=representation_types,
            ur_sizes=representation_sizes,
        )

        for uxhw_conf in uxhw_data:

            # Marks a degraded representation. If not None, the distances are
            # set to `inf` and the row is reported as "blow-up / excluded".
            # It's not dropped (see BenchmarkingVariables.BLOW_UP_REASON).
            blow_up_reason: str | None = None
            if variable.type == VariableTypes.DISTRIBUTION:
                assert uxhw_distance_fn is not None
                # Traced representations may carry benign special-value
                # (NaN / +inf / -inf) Dirac deltas at exactly zero mass. Dropping
                # them lets the finite-only distance validator accept the
                # otherwise-valid distribution.
                uxhw_conf.dv.drop_zero_mass_positions()

                blow_up_reason = _representation_blow_up_reason(uxhw_conf.dv)
                if blow_up_reason is not None:
                    # Warn and set both distances to +inf
                    # (reported as the worst config) rather than masking it with
                    # a silent 0 or crashing the pipeline.
                    warnings.warn(
                        f"Representation blow-up for variable "
                        f"{variable.description!r} config {uxhw_conf!r}: "
                        f"{blow_up_reason}. Setting UxHw distances to inf."
                    )
                    uxhw_distance = float("inf")
                    binned_uxhw_distance = float("inf")
                else:
                    uxhw_distance = uxhw_distance_fn(
                        uxhw_conf.dv,
                        ground_truth.dv,
                    )
                    if binned_uxhw_distance_fn is not None:
                        try:
                            binned_uxhw_distance = binned_uxhw_distance_fn(
                                uxhw_conf.dv,
                                ground_truth.dv,
                            )
                        except Exception as e:
                            # Treat failed binned-distance computation
                            # as inf, matching the blow-up path above.
                            warnings.warn(
                                f"Binned UxHw distance failed for config "
                                f"{uxhw_conf!r}: {e}. Setting it to inf."
                            )
                            binned_uxhw_distance = float("inf")
                    else:
                        binned_uxhw_distance = uxhw_distance
            else:
                # Scalar branch: a large finite scalar is legitimate, so only
                # the non-finite check applies (check_magnitude=False). Guard it
                # because a non-finite scalar would make
                # relative_error_uxhw_wrapper's validator raise.
                blow_up_reason = _representation_blow_up_reason(
                    uxhw_conf.dv, check_magnitude=False
                )
                if blow_up_reason is not None:
                    warnings.warn(
                        f"Representation blow-up for scalar variable "
                        f"{variable.description!r} config {uxhw_conf!r}: "
                        f"{blow_up_reason}. Setting UxHw distances to inf."
                    )
                    uxhw_distance = float("inf")
                    binned_uxhw_distance = float("inf")
                else:
                    uxhw_distance = (
                        relative_error_uxhw_wrapper(uxhw_conf.dv, ground_truth.dv)
                        * EquivMC.BASIS_POINT_CONVERSION_FACTOR
                    )
                    binned_uxhw_distance = uxhw_distance

            variable.uxhw_distances.records.append(
                UxhwDistanceRecord(
                    uxhw_conf=uxhw_conf,
                    uxhw_distance=uxhw_distance,
                    uxhw_binned_distance=binned_uxhw_distance,
                    blow_up_reason=blow_up_reason,
                )
            )

    write_uxhw_distance_file(uxhw_distance_file, benchmarking_variables)


def compute_emcc_predictions(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    use_binned_uxhw: bool,
    reporting_methods: list[str],
    output_data_file: str,
    uxhw_distance_file: str,
    asymptotic_dist_file: str,
    distance_type: str,
) -> None:
    """
    Compute EMCC (equivalent Monte Carlo count) predictions per
    reporting method and persist them to ``output_data_file``.

    Args:
        benchmarking_variables: Variables to process.
        use_binned_uxhw: When ``True``, prefer the binned
            Wasserstein-1 distance and fall back to the plain
            Wasserstein distance when the binned variant is missing
            or invalid (zero / infinite).
        reporting_methods: Reporting-method identifiers
            (e.g. mean, quantile-95).
        output_data_file: CSV file to write the EMCC predictions to.
        uxhw_distance_file: CSV file produced by
            :func:`compute_uxhw_distances`.
        asymptotic_dist_file: CSV file produced by
            :func:`generate_asymptotic_distance_distributions`.
        distance_type: Distance metric used (written to the output
            CSV for traceability).
    """
    load_uxhw_distances(
        benchmarking_variables=benchmarking_variables,
        uxhw_distance_file=uxhw_distance_file,
        asymptotic_dist_file=asymptotic_dist_file,
    )
    for variable in benchmarking_variables:
        new_emcc_data: list[dict[str, object]] = []
        for record in variable.uxhw_distances.records:
            # Use binned when asked and fall back to the plain Wasserstein
            # distance for missing/invalid values.
            if (
                use_binned_uxhw
                and record.uxhw_binned_distance is not None
                and record.uxhw_binned_distance != 0
                and not math.isinf(record.uxhw_binned_distance)
            ):
                distance = record.uxhw_binned_distance
            else:
                distance = record.uxhw_distance

            # Create a new dictionary for each reporting method
            for method in reporting_methods:
                method_dic: dict[str, object] = {
                    BenchmarkingVariables.UXHW_CONF: record.uxhw_conf,
                    BenchmarkingVariables.UXHW_DISTANCE: record.uxhw_distance,
                    BenchmarkingVariables.UXHW_BINNED_DISTANCE: (
                        record.uxhw_binned_distance
                    ),
                    # Annotate the blow-up marker into the in-memory emcc_data
                    # so the report can exclude degraded rows. This is in-memory
                    # only and not written to output_data.csv (see
                    # BenchmarkingVariables.BLOW_UP_REASON). UXHW_CONF is kept
                    # so the row survives the timing join and only its EMCC.
                    # contribution is excluded.
                    BenchmarkingVariables.BLOW_UP_REASON: record.blow_up_reason,
                }
                method_dic[EquivMC.REPORTING_METHOD] = method
                method_dic[EquivMC.EMCC_PREDICTED] = compute_emcc_prediction(
                    variable.asymptotic_distribution.value_for(method),
                    distance,
                )
                new_emcc_data.append(method_dic)

        # Replace the original list with the expanded one
        variable.emcc_results.emcc_data = new_emcc_data

    # Write predictions to a file
    with open(output_data_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                BenchmarkingVariables.VARIABLE_DESCRIPTION,
                BenchmarkingVariables.UXHW_CONF,
                BenchmarkingVariables.VARIABLE_TYPE,
                EquivMC.DISTANCE_TYPE,
                EquivMC.REPORTING_METHOD,
                BenchmarkingVariables.UXHW_DISTANCE,
                BenchmarkingVariables.UXHW_BINNED_DISTANCE,
                EquivMC.EMCC_PREDICTED,
            ]
        )
        for variable in benchmarking_variables:
            for emcc_dic in variable.emcc_results.emcc_data:
                writer.writerow(
                    [
                        variable.description,
                        repr(emcc_dic[BenchmarkingVariables.UXHW_CONF]),
                        variable.type,
                        distance_type,
                        emcc_dic[EquivMC.REPORTING_METHOD],
                        emcc_dic[BenchmarkingVariables.UXHW_DISTANCE],
                        emcc_dic[BenchmarkingVariables.UXHW_BINNED_DISTANCE],
                        emcc_dic[EquivMC.EMCC_PREDICTED],
                    ]
                )


def generate_asymptotic_distance_distributions(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    ground_truth_db_path: str,
    ground_truth_type: str,
    distance_type: str,
    asymptotic_dist_file: str,
    n_processors: int,
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
    demo_cli_args: str,
) -> None:
    """
    Compute the asymptotic distance distribution per variable.

    If a Central Limit Theorem exists for the variable with respect to
    the distance metric, computes the limiting distribution of
    ``sqrt(N) * distance(MC(N), GT)``, where ``GT`` is the ground
    truth and ``MC(N)`` is an MC simulation with ``N`` samples.

    For distributions and the Wasserstein-1 metric, uses the
    Brownian-bridge identity
    ``sqrt(N) * W1(MC(N), GT) ~ ∫ |B(t)| dQ(t)``.

    For scalars, the distance from the ground truth for quantities
    such as the mean and quantile is normally distributed, so
    ``sqrt(N) * | MC(N) - GT | ~ HalfNormal(σ)``.

    Args:
        benchmarking_variables: Variables to process.
        ground_truth_db_path: SQLite database containing ground-truth
            samples.
        ground_truth_type: Representation type string used to select
            the appropriate ground-truth loader.
        distance_type: Distance metric (passed to the Brownian-bridge
            asymptotic computation).
        asymptotic_dist_file: CSV file to write the per-variable
            asymptotic statistics to.
        n_processors: Number of parallel worker processes used by
            :func:`sample_generator.generate_scalar_samples` to draw
            scalar Monte Carlo samples.
        path_to_application: Root path of the application source tree
            (forwarded to ``generate_scalar_samples``).
        native_executable_name: Filename of the compiled native
            binary (forwarded to ``generate_scalar_samples``).
        native_executable_dir: Directory containing the native binary
            (forwarded to ``generate_scalar_samples``).
        demo_cli_args: Per-application command-line argument prefix
            (forwarded to ``generate_scalar_samples``).
    """
    csv_rows = []

    for variable in benchmarking_variables:
        # Load ground truth
        monte_carlo = ground_truth_type == RepresentationTypes.MONTE_CARLO
        ground_truth = _load_ground_truth(
            db_path=ground_truth_db_path,
            table=ground_truth_type,
            target_expression=variable.name,
            expression_type=variable.type,
            monte_carlo=monte_carlo,
        )
        is_normal = False
        std = None

        if variable.type == VariableTypes.DISTRIBUTION:
            # Compute the asymptotic distance using the Brownian bridge
            asymptotic_dist = _compute_asymptotic_distribution_brownian_bridge(
                ground_truth,
                num_points=1000,
                num_samples=1000,
                distance_type=distance_type,
            )
            # Use inverse_cdf so reported quantiles interpolate.
            # DistributionalValue.quantile is a non-interpolating step lookup
            # and would shift the reported quantiles.
            treat_as_samples = (
                getattr(asymptotic_dist, "representation_type", None)
                == RepresentationTypes.SAMPLES
            )
            asymptotic_mean = asymptotic_dist.mean
            if asymptotic_mean is not None:
                mean_quantile = asymptotic_dist.cdf(
                    asymptotic_mean, treat_as_samples=treat_as_samples
                )
            else:
                mean_quantile = None
            quantile_95 = asymptotic_dist.inverse_cdf(
                ReportingNumbers.QUANTILE_95, treat_as_samples=treat_as_samples
            )
            quantile_99 = asymptotic_dist.inverse_cdf(
                ReportingNumbers.QUANTILE_99, treat_as_samples=treat_as_samples
            )

            variable.asymptotic_distribution.samples = asymptotic_dist.positions

            # Save asymptotic distribution data for each distribution
            np.save(
                f"{variable.formatted_description}-asymptotic.npy",
                asymptotic_dist.positions,
            )

        elif variable.type == VariableTypes.SCALAR:
            # Scalars use an empirical approach: draw samples and, if they are
            # normally distributed, estimate the standard deviation.
            test_size = 1000
            samples = generate_scalar_samples(
                variable=variable,
                sizes=[test_size],
                repetitions=10_000,
                n_processors=n_processors,
                path_to_application=path_to_application,
                native_executable_name=native_executable_name,
                native_executable_dir=native_executable_dir,
                demo_cli_args=demo_cli_args,
            )[test_size]
            mean, std, is_normal = _compute_asymptotic_distribution_scalar_empirical(
                samples, test_size, ground_truth.dv.positions[0]
            )

            # Convert distances to units of basis points
            std *= EquivMC.BASIS_POINT_CONVERSION_FACTOR / np.abs(
                ground_truth.dv.positions[0]
            )

            half_norm_dist = halfnorm(scale=std)
            asymptotic_mean = std * np.sqrt(2 / np.pi)
            mean_quantile = half_norm_dist.cdf(asymptotic_mean)
            quantile_95 = half_norm_dist.ppf(ReportingNumbers.QUANTILE_95)
            quantile_99 = half_norm_dist.ppf(ReportingNumbers.QUANTILE_99)

        # Populate the dataclass, casting numpy scalars to plain `float` to
        # match the `float | None` fields. For distribution variables,
        # `is_normal` and `scale` keep their loop-head defaults (`False` /
        # `None`).
        asymptotic = variable.asymptotic_distribution
        asymptotic.mean = None if asymptotic_mean is None else float(asymptotic_mean)
        asymptotic.quantile_95 = None if quantile_95 is None else float(quantile_95)
        asymptotic.quantile_99 = None if quantile_99 is None else float(quantile_99)
        asymptotic.mean_quantile = (
            None if mean_quantile is None else float(mean_quantile)
        )
        asymptotic.is_normal = is_normal
        asymptotic.scale = None if std is None else float(std)

        csv_rows.append(
            [
                variable.description,
                asymptotic_mean,
                quantile_95,
                quantile_99,
                mean_quantile,
                is_normal,
                std,
            ]
        )

    # Write asymptotic distance distribution data to CSV
    with open(asymptotic_dist_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                BenchmarkingVariables.VARIABLE_DESCRIPTION,
                ReportingMethods.MEAN,
                ReportingMethods.QUANTILE_95,
                ReportingMethods.QUANTILE_99,
                EquivMC.MEAN_QUANTILE,
                AsymptoticDistanceDistribution.IS_NORMAL,
                AsymptoticDistanceDistribution.SCALE,
            ]
        )
        writer.writerows(csv_rows)


def compute_equivalent_mc(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    output_data_file: str,
    asymptotic_dist_file: str,
    ground_truth_db_path: str,
    has_analytic_ground_truth: bool,
    ground_truth_type: str,
    adversary_db_path: str,
    tracing_db_path: str,
    representation_types: list[str],
    representation_sizes: list[int],
    correlations: list[str],
    num_parallel_workers: int,
    n_adversaries: int,
    distance_type: str,
    use_binned_uxhw: bool,
    use_clt: bool,
    reporting_methods: list[str],
    plots_dir: str,
    plot_comparison_distributions: bool = False,
    plot_adversary_distances: bool = False,
    plot_distributions: bool = False,
) -> None:
    """
    Compute equivalent Monte Carlo counts across variables.

    Loads the EMCC data and the asymptotic-distance distributions
    written by the earlier pipeline stages and delegates to
    :func:`load_data_and_compute_equivalent_mc`.

    Only the plotting flags are documented below. The remaining arguments are
    forwarded verbatim to :func:``load_data_and_compute_equivalent_mc``.

    Args:
        plot_comparison_distributions: Generate comparison-distribution
            plots (UxHw vs MC vs ground truth).
        plot_adversary_distances: Generate adversary-distance plots.
        plot_distributions: Generate representative-MC distribution
            plots (distribution-typed outputs only). Maps to the
            ``plot_distributions`` gate in ``equivalent_mc_utils``.
    """
    load_emcc_data(
        benchmarking_variables=benchmarking_variables,
        output_data_file=output_data_file,
    )
    load_asymptotic_dist(
        benchmarking_variables=benchmarking_variables,
        asymptotic_dist_file=asymptotic_dist_file,
    )
    load_data_and_compute_equivalent_mc(
        {
            "ground_truth_database_path": ground_truth_db_path,
            "ground_truth_table_name": (
                "WeightedSamples"
                if has_analytic_ground_truth or ground_truth_type == "WeightedSamples"
                else "MonteCarlo"
            ),
            "benchmarking_variables": benchmarking_variables,
            "adversary_database_path": adversary_db_path,
            "uxhw_database_path": tracing_db_path,
            "uxhw_ur_types": representation_types,
            "uxhw_ur_sizes": representation_sizes,
            "correlations": correlations,
            "n_processes": num_parallel_workers,
            "n_adversaries": n_adversaries,
            "plot_comparison_distributions": plot_comparison_distributions,
            "ground_truth_type": ground_truth_type,
            "plot_adversary_distances": plot_adversary_distances,
            "use_adaptive_steps": True,
            "distance_type": distance_type,
            "use_binned_uxhw": use_binned_uxhw,
            "use_clt": use_clt,
            "reporting_methods": reporting_methods,
            "auto_prefix": True,
            "adversary_size_step": n_adversaries,
            "adversary_size_min": 1,
            "adversary_size_max": 50000,
            "adversary_table_name": "MonteCarlo",
            "uxhw_table_names": ["TracingTable"],
            "output_file": output_data_file,
            "plot_distributions": plot_distributions,
            "plots_dir": plots_dir,
        }
    )
