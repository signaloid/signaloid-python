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

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from signaloid.benchmarking.config import (
    RepresentationTypes,
    DistanceMetrics,
    ReportingMethods,
    Correlations,
    DEFAULT_UXHW_SDK_PATH,
    DEFAULT_GROUND_TRUTH_SIZE,
    DEFAULT_ADVERSARY_MC_SIZE,
    DEFAULT_MAX_NUM_WEIGHTED_SAMPLES,
    DEFAULT_NUM_ADVERSARIES,
    DEFAULT_MAX_JUPITER_SIZE,
    DEFAULT_ADVERSARY_MAX_SIZE_SCALAR,
)

# Valid choices for the sweep args, as module constants so
# `create_argument_parser` (CLI `choices=`) and `validate_args` (file-supplied
# values, which argparse does not check) share one source.
REPRESENTATION_TYPE_CHOICES = [
    RepresentationTypes.ATHENS,
    RepresentationTypes.ATLAS,
    RepresentationTypes.JUPITER,
    RepresentationTypes.EUROPA,
]
CORRELATION_CHOICES = [
    Correlations.DISABLED,
    Correlations.AUTOCORRELATION,
]
REPORTING_METHOD_CHOICES = [
    ReportingMethods.MEAN,
    ReportingMethods.QUANTILE_95,
    ReportingMethods.QUANTILE_99,
]


def create_argument_parser() -> ArgumentParser:
    """
    Build the argument parser for the benchmarking CLI.

    Returns:
        The configured ``ArgumentParser``.
    """
    parser = ArgumentParser(
        prog="signaloid-benchmarking",
        description="Benchmark UxHw applications",
    )

    # Optional YAML config file whose values populate argparse defaults (see
    # benchmark_application.load_config), so explicit CLI flags still win.
    # Defined first so --help lists it near the top.
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default=None,
        help=(
            "YAML configuration file. Keys populate arguments "
            "with matching names. Precedence: built-in "
            "defaults < configuration file < explicit CLI flags."
        ),
    )

    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        default=False,
        help=(
            "Resolve parameters and print the effective "
            "configuration and the benchmark matrix that the "
            "benchmark will use, then exit without running."
        ),
    )

    # UxHw configurations.
    #
    # The four sweep args below are not required= (a --config file supplies
    # them via set_defaults, which cannot satisfy argparse's required=).
    # Presence and membership are validated in validate_args() after the merge.
    #
    # They use "store" (not "extend") with nargs="+" so an explicit CLI value
    # fully replaces a file-supplied default.
    parser.add_argument(
        "-u",
        "--representation-types",
        dest="representation_types",
        choices=REPRESENTATION_TYPE_CHOICES,
        type=str,
        nargs="+",
        help="Uncertain representation types to evaluate.",
    )

    parser.add_argument(
        "-s",
        "--representation-sizes",
        dest="representation_sizes",
        type=int,
        nargs="+",
        help="Uncertain representation sizes to evaluate.",
    )

    parser.add_argument(
        "-c",
        "--uncertainty-correlation_types",
        dest="correlations",
        type=str,
        choices=CORRELATION_CHOICES,
        nargs="+",
        help="Uncertain representation correlations to evaluate",
    )

    parser.add_argument(
        "-r",
        "--reporting-methods",
        dest="reporting_methods",
        type=str,
        choices=REPORTING_METHOD_CHOICES,
        nargs="+",
        help="Reporting methods.",
    )

    parser.add_argument(
        "--ground-truth-size",
        dest="ground_truth_size",
        type=int,
        default=DEFAULT_GROUND_TRUTH_SIZE,
        help="Size of ground truth.",
    )

    parser.add_argument(
        "--num-adversaries",
        dest="n_adversaries",
        type=int,
        default=DEFAULT_NUM_ADVERSARIES,
        help="Number of adversaries to evaluate.",
    )

    parser.add_argument(
        "--max-jupiter-size",
        dest="max_jupiter_size",
        type=int,
        default=DEFAULT_MAX_JUPITER_SIZE,
        help="Maximum representation size to benchmark for Jupiter.",
    )

    parser.add_argument(
        "--max-num-weighted-samples",
        dest="max_num_weighted_samples",
        type=int,
        default=DEFAULT_MAX_NUM_WEIGHTED_SAMPLES,
        help="Maximum number of weighted samples to use for ground truth.",
    )

    parser.add_argument(
        "--adversary-mc-size",
        dest="adversary_mc_size",
        type=int,
        default=DEFAULT_ADVERSARY_MC_SIZE,
        help="Size of adversary Monte Carlo array.",
    )

    parser.add_argument(
        "--adversary-max-size-scalar",
        dest="adversary_max_size_scalar",
        type=int,
        default=DEFAULT_ADVERSARY_MAX_SIZE_SCALAR,
        help="Maximum size of adversarial MC for scalar outputs.",
    )

    # Multiprocessing. Default is None (unset) which resolves to the detected
    # core count in Benchmark.get_machine_info. When set, -j sets the max for
    # every worker pool (compile, the three MC sample-generation stages, and the
    # EMCC adversary-distance stage).
    parser.add_argument(
        "-j",
        "--jobs",
        "--num-parallel-workers",
        dest="num_parallel_workers",
        type=int,
        default=None,
        help=(
            "Number of max parallel workers. " "Defaults to the detected core count."
        ),
    )

    # Ground Truth Type
    parser.add_argument(
        "--ground-truth-type",
        dest="ground_truth_type",
        type=str,
        default=RepresentationTypes.MONTE_CARLO,
        choices=[RepresentationTypes.MONTE_CARLO, RepresentationTypes.WEIGHTED_SAMPLES],
        help=f"Type of ground truth ({RepresentationTypes.MONTE_CARLO} or {RepresentationTypes.WEIGHTED_SAMPLES}). Default is {RepresentationTypes.MONTE_CARLO}.",
    )

    parser.add_argument(
        "--distance-type",
        dest="distance_type",
        type=str,
        default=DistanceMetrics.WASSERSTEIN_1,
        choices=[
            DistanceMetrics.WASSERSTEIN_1,
            DistanceMetrics.WASSERSTEIN_2,
        ],
        help="Distance function to use for as the accuracy metric for comparing distributions.",
    )

    parser.add_argument(
        "--has-analytic-ground-truth",
        dest="has_analytic_ground_truth",
        action="store_true",
        default=False,
        help="Application has analytic ground truth.",
    )

    parser.add_argument(
        "--use-binned-uxhw",
        dest="use_binned_uxhw",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Use the binned Wasserstein-1 computation for UxHw "
            "distances. Pass --no-use-binned-uxhw (or set "
            "use_binned_uxhw: false in a config) to disable."
        ),
    )

    parser.add_argument(
        "--use-clt",
        dest="use_clt",
        action="store_true",
        default=False,
        help=(
            "Estimate the equivalent Monte Carlo count from the asymptotic "
            "(CLT / Brownian-bridge) distance distribution instead of "
            "measuring it via explicit adversary MC simulation."
        ),
    )

    # Paths
    parser.add_argument(
        "--path-to-application",
        dest="path_to_application",
        type=str,
        help="Application to benchmark.",
    )

    parser.add_argument(
        "--path-to-uxhw-sdk",
        dest="path_to_uxhw_sdk",
        default=DEFAULT_UXHW_SDK_PATH,
        help="Path to the Signaloid UxHw SDK.",
    )

    parser.add_argument(
        "--path-to-pin",
        dest="path_to_pin",
        type=str,
        default=None,
        help=(
            "Path to the Intel PIN kit (sets PIN_ROOT for the timing "
            "script, which uses it to count dynamic instructions). When "
            "omitted, an inherited PIN_ROOT is used. If neither is set "
            "the timing run errors with 'Intel Pin not found'."
        ),
    )

    parser.add_argument(
        "--path-to-ground-truth-file",
        dest="path_to_ground_truth_file",
        type=str,
        help="Path to ground truth file",
    )

    parser.add_argument(
        "--demo-cli-args",
        dest="demo_cli_args",
        type=str,
        default="",
        help="Extra command-line arguments passed to both native-MC and UxHw executions.",
    )

    # Plotting controls. Each gates one plot call in the equivalent-MC stage
    # (the gates in equivalent_mc_utils). Disabled by default.
    # BooleanOptionalAction adds the --no-... variant, so each is toggleable
    # both ways and settable from a config via set_defaults.
    parser.add_argument(
        "--plot-distance-vs-asymptotic",
        dest="plot_distance_vs_asymptotic",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Plot the empirical equivalent-MC distance distribution "
            "against its asymptotic prediction (Brownian-bridge for "
            "distribution outputs, half-normal for scalars)."
        ),
    )

    parser.add_argument(
        "--plot-adversary-distances",
        dest="plot_adversary_distances",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Generate adversary-distance plots during the "
            "equivalent-MC stage (empty under --use-clt)."
        ),
    )

    parser.add_argument(
        "--plot-representative-mc",
        dest="plot_representative_mc",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Plot a representative equivalent Monte Carlo run (the MC sample set "
            "whose distance to ground truth matches the UxHw result) "
            "per distribution-typed output."
        ),
    )

    parser.add_argument(
        "--google-credentials",
        dest="google_credentials",
        type=str,
        default=None,
        help=(
            "Path to a Google Cloud service-account JSON file for the Sheets "
            "upload. Falls back to the GOOGLE_APPLICATION_CREDENTIALS "
            "environment variable. Required when --write-sheets is set."
        ),
    )

    parser.add_argument(
        "--write-sheets",
        dest="write_sheets",
        action="store_true",
        default=False,
        help=(
            "Optional Google Sheets upload at pipeline step 15 (see README.md). "
            "Requires a credentials file (via --google-credentials or "
            "GOOGLE_APPLICATION_CREDENTIALS) and the `sheets` package "
            "to be installed."
        ),
    )

    return parser


def _validate_sweep_args(args: Namespace) -> None:
    """Validate the four sweep args after the config layer has merged.


    Raises:
        ValueError: If any sweep arg is missing after the merge, is not a
            non-empty list, (for representation_sizes) contains a
            non-integer, or contains a value outside its allowed enum
            membership.
    """
    required_sweep_args = {
        "representation_types": REPRESENTATION_TYPE_CHOICES,
        "representation_sizes": None,
        "correlations": CORRELATION_CHOICES,
        "reporting_methods": REPORTING_METHOD_CHOICES,
    }
    for dest, allowed in required_sweep_args.items():
        values = getattr(args, dest, None)
        if values is None:
            raise ValueError(
                f"Error! '{dest}' must be provided via the CLI or a "
                "--config file (it has no default)."
            )
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Error! '{dest}' must be a non-empty list (a YAML list "
                f"or CLI nargs), got {type(values).__name__}."
            )
        if dest == "representation_sizes" and any(
            not isinstance(value, int) for value in values
        ):
            raise ValueError(
                f"Error! '{dest}' must contain only integers, " f"got {values}."
            )
        if allowed is None:
            continue
        invalid = [value for value in values if value not in allowed]
        if invalid:
            raise ValueError(
                f"Error! '{dest}' contains invalid values {invalid}. "
                f"Allowed values are {allowed}."
            )


def validate_args(args: Namespace) -> None:
    """
    Validate the merged arguments before the pipeline runs.

    Checks the worker count, the sweep args, the analytic-ground-truth /
    weighted-samples pairing, and the binned-distance / metric pairing.

    Args:
        args: The parsed and config-merged arguments.

    Raises:
        ValueError: If any of those constraints is violated.
    """
    # -j is None when unset (get_machine_info resolves it to the core count).
    # When given explicitly it must be positive, else the worker pools get a
    # non-positive max_workers and crash deep in the run.
    if args.num_parallel_workers is not None and args.num_parallel_workers < 1:
        raise ValueError(
            "Error! -j/--jobs/--num-parallel-workers must be >= 1, got "
            f"{args.num_parallel_workers}."
        )

    _validate_sweep_args(args)

    # Analytic ground truth means utilizing weighted samples
    if args.has_analytic_ground_truth:
        if args.ground_truth_type != RepresentationTypes.WEIGHTED_SAMPLES:
            raise ValueError("Error! Analytic ground truth must have weighted samples.")

    if args.use_binned_uxhw and (args.distance_type != DistanceMetrics.WASSERSTEIN_1):
        raise ValueError(
            "Error! Binned distance computations only supported"
            f" for {DistanceMetrics.WASSERSTEIN_1}."
        )
