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
from os import path
from typing import TypedDict
from tabulate import tabulate
import signaloid.distributional_information_plotting.plot_wrapper as plot_wrapper
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from signaloid.benchmarking.equivalent_mc.equivalent_monte_carlo import (
    EquivalentMonteCarlo,
)
from signaloid.benchmarking.equivalent_mc.load import (
    _load_uxhw_distributions,
    _load_ground_truth,
)
from signaloid.benchmarking.equivalent_mc.equivalent_mc_utils import (
    _generate_adversary_list,
    _generate_equivalent_mc_plots,
    _compute_shared_xlim,
)
from signaloid.benchmarking.config import (
    RepresentationTypes,
)

from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
from signaloid.benchmarking.distribution_helpers.representation_health import (
    _representation_blow_up_reason,
)

# Printed in the Mean / Variance columns for a blown-up (degraded)
# representation, in place of the overflowing statistics.
BLOW_UP_TABLE_TOKEN = "blow-up / excluded"


class LoadDataComputeEquivalentMCArgs(TypedDict):
    """
    Bundle of inputs for :func:`load_data_and_compute_equivalent_mc`: database
    locations/tables, the UxHw sweep parameters, distance/reporting settings,
    and plotting toggles.
    """

    benchmarking_variables: list[BenchmarkingVariable]
    ground_truth_database_path: str
    ground_truth_table_name: str
    adversary_database_path: str
    adversary_table_name: str
    adversary_size_step: int
    adversary_size_min: int
    adversary_size_max: int | None
    uxhw_database_path: str
    uxhw_table_names: list[str]
    uxhw_ur_types: list[str]
    uxhw_ur_sizes: list[int]
    correlations: list[str]
    distance_type: str
    n_processes: int
    n_adversaries: int
    output_file: str | None
    auto_prefix: bool
    plot_comparison_distributions: bool
    use_clt: bool
    reporting_methods: list[str]
    ground_truth_type: str
    plot_adversary_distances: bool
    use_adaptive_steps: bool
    plot_distributions: bool
    use_binned_uxhw: bool
    plots_dir: str


def _print_uxhw_table(uxhw: list[TaggedDistributionalValue]) -> None:
    """
    Print a table of the loaded UxHw configurations and their mean/variance.

    Blown-up representations are shown as ``"blow-up / excluded"`` instead of
    their (overflowing) statistics.

    Args:
        uxhw: The loaded UxHw configurations to tabulate.
    """
    table_data = [["UR_type", "UR_order", "Correlation Tracking", "Mean", "Variance"]]
    alignments = ["left", "right", "left", "left", "left"]
    # representation_type / correlation_tracking come from the carrier metadata
    # (never off a Distribution). Numeric stats come from the carried dv.
    for uxhw_conf in uxhw:
        # The dv is freshly re-loaded here (it does not carry the Step-6
        # inf-distance flag), so re-detect a blow-up. A blown representation
        # parks mass at ~1e303, where reading .mean / .variance overflows and
        # crashes the sweep. Emit a "blow-up / excluded" row instead.
        if _representation_blow_up_reason(uxhw_conf.dv) is not None:
            mean_cell: str = BLOW_UP_TABLE_TOKEN
            variance_cell: str = BLOW_UP_TABLE_TOKEN
        else:
            mean_cell = str(uxhw_conf.dv.mean)
            variance_cell = str(uxhw_conf.dv.variance)
        table_data.append(
            [
                str(uxhw_conf.representation_type),
                str(uxhw_conf.dv.UR_order),
                str(uxhw_conf.correlation_tracking),
                mean_cell,
                variance_cell,
            ]
        )
    print(
        tabulate(table_data, headers="firstrow", tablefmt="simple", colalign=alignments)
    )
    print()


def _find_common_prefix(args: LoadDataComputeEquivalentMCArgs) -> str:
    """
    Derive a shared filename prefix from the database paths.

    Args:
        args: The equivalent-MC arguments (uses the database paths and the
            ``auto_prefix`` flag).

    Returns:
        The common basename prefix of the databases, or ``""`` when
        ``auto_prefix`` is disabled.
    """
    database_common_prefix = ""
    if args["auto_prefix"]:
        args_list = [args["ground_truth_database_path"], args["uxhw_database_path"]]
        args_list.append(args["adversary_database_path"])
        database_common_prefix = path.basename(path.commonprefix(args_list))
        print(f"Detected common prefix from databases: '{database_common_prefix}'")
    return database_common_prefix


def _resolve_plot_path(filename: str, plots_dir: str) -> str:
    """
    Prepend ``plots_dir`` to a plot filename when set.

    Args:
        filename: The plot filename.
        plots_dir: Directory to place the plot in. Unchanged when empty.

    Returns:
        The joined path, or ``filename`` unchanged when ``plots_dir`` is empty.
    """
    if plots_dir:
        return os.path.join(plots_dir, filename)
    return filename


def _generate_distribution_plots(
    variable: BenchmarkingVariable,
    ground_truth: TaggedDistributionalValue,
    uxhw_from_table: list[TaggedDistributionalValue],
    plots_dir: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Generate the ground-truth plot and a plot per UxHw configuration.

    Args:
        variable: The benchmarking variable being plotted.
        ground_truth: The ground-truth distribution.
        uxhw_from_table: The UxHw configurations loaded from the database.
        plots_dir: Directory to write the plots into. Defaults to the current working directory when empty.
        xlim: Limits applied to the x-axis for every plot. ``None`` falls back to per-plot auto-scaling.
    """
    try:
        print("Generating Ground Truth Plot.")
        gt_plot_path = _resolve_plot_path(
            f"{variable.formatted_description}-ground-truth.png", plots_dir
        )
        plot_wrapper.plot(
            plot_data=PlotData(ground_truth.dv),
            path=gt_plot_path,
            save=True,
            xlim=xlim,
        )
    except Exception as e:
        print(
            f"Failed to generate ground truth plot for "
            f"{variable.description} with error: {e}"
        )

    # The per-config filename embeds the carrier's config string (its repr,
    # e.g. "Athens-16" or "Athens-16-Autocorrelation") so it matches exactly
    # what report_writer reconstructs from the results table.
    for dist in uxhw_from_table:
        assert dist.correlation_tracking is not None
        plot_name = _resolve_plot_path(
            f"{variable.formatted_description}-{dist}.png",
            plots_dir,
        )
        try:
            plot_wrapper.plot(
                plot_data=PlotData(dist.dv),
                path=plot_name,
                save=True,
                xlim=xlim,
            )
        except Exception as e:
            print(f"Failed to generate plot {plot_name} with error: {e}")


def load_data_and_compute_equivalent_mc(args: LoadDataComputeEquivalentMCArgs) -> None:
    """
    Run the full equivalent-MC analysis for each benchmarking variable.

    For every variable: loads the ground truth, adversaries, and UxHw
    distributions from the databases, prints the UxHw table, computes and
    reports the EMCC, and generates the configured plots.

    Args:
        args: The equivalent-MC arguments (database locations, sweep
            parameters, distance/reporting settings, and plotting toggles).
    """
    optional_args = {}
    database_common_prefix = None
    n_processes = args["n_processes"]
    n_adversaries = args["n_adversaries"]
    reporting_methods = args["reporting_methods"]
    database_common_prefix = _find_common_prefix(args=args)
    plots_dir = args["plots_dir"]
    ground_truth_is_mc = (
        True if args["ground_truth_type"] == RepresentationTypes.MONTE_CARLO else False
    )
    distance_type = args["distance_type"]

    # Loop through variables
    for variable in args["benchmarking_variables"]:
        print("=================================================================")
        print("Traced expression:", variable.name)
        print("Expression description:", variable.description)
        print("Expression type:", variable.type)
        print()

        # Load ground truth
        ground_truth = _load_ground_truth(
            db_path=args["ground_truth_database_path"],
            table=args["ground_truth_table_name"],
            target_expression=variable.name,
            expression_type=variable.type,
            monte_carlo=ground_truth_is_mc,
        )

        # Load adversaries
        adversary_list: list[TaggedDistributionalValue] = _generate_adversary_list(
            args=args, target_expr=variable.name, expr_type=variable.type
        )
        optional_args["adversary_mc"] = adversary_list

        # Get all distributions from databases
        uxhw = _load_uxhw_distributions(
            db_path=args["uxhw_database_path"],
            tables=args["uxhw_table_names"],
            target_expr=variable.name,
            ur_types=args["uxhw_ur_types"],
            ur_sizes=args["uxhw_ur_sizes"],
        )

        # A single x-axis domain shared by the ground-truth, UxHw, and
        # representative MC plots. The adversary distributions contribute an
        # outlier-robust range so Monte Carlo tails do not blow up the domain.
        shared_xlim = None
        if args["plot_distributions"]:
            shared_xlim = _compute_shared_xlim(
                ground_truth=ground_truth,
                uxhw=uxhw,
                adversaries=adversary_list,
            )
            _generate_distribution_plots(
                variable, ground_truth, uxhw, plots_dir, xlim=shared_xlim
            )

        _print_uxhw_table(uxhw=uxhw)

        # Compute
        emcc_prefix = database_common_prefix
        if plots_dir and emcc_prefix:
            emcc_prefix = os.path.join(plots_dir, emcc_prefix)
        emcc = EquivalentMonteCarlo(
            ground_truth=ground_truth,
            uxhw_data=uxhw,
            variable=variable,
            distance_type=distance_type,
            use_binned_uxhw=args["use_binned_uxhw"],
            n_processes=n_processes,
            n_adversaries=n_adversaries,
            prefix=emcc_prefix,
            reporting_methods=reporting_methods,
            use_clt=args["use_clt"],
            adversary_size_step=args["adversary_size_step"],
            adversary_size_min=args["adversary_size_min"],
            adversary_size_max=args["adversary_size_max"],
            **optional_args,
        )

        emcc.compute_distance_data(use_adaptive_steps=args["use_adaptive_steps"])
        emcc.compute_and_report_emmc()

        # Generate plots
        _generate_equivalent_mc_plots(
            emcc=emcc,
            args=args,
            target_expr=variable.name,
            expr_description=variable.description,
            distance_type=distance_type,
            xlim=shared_xlim,
        )
