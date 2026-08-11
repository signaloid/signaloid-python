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

import datetime
import os
import sys
import traceback

import yaml

from argparse import ArgumentParser, Namespace

from signaloid.benchmarking.automation.benchmark import (
    Benchmark,
)
from signaloid.benchmarking.automation.analysis import (
    compute_emcc_predictions,
    compute_equivalent_mc,
    compute_speedups,
    compute_uxhw_distances,
    generate_asymptotic_distance_distributions,
    load_emcc_data,
)
from signaloid.benchmarking.automation.arguments import (
    create_argument_parser,
    validate_args,
)
from signaloid.benchmarking.automation.database_generator import (
    generate_adversary_database,
    generate_ground_truth_database,
    generate_uxhw_tracing_database,
)
from signaloid.benchmarking.automation.measurement_loader import (
    load_measurement_dicts,
    load_timing_data_to_dfs,
)
from signaloid.benchmarking.automation.report_writer import (
    TARGET_FOLDER_ID,
    TEMPLATE_ID,
    resolve_google_credentials_path,
    write_results_to_markdown,
    write_results_to_spreadsheet,
)

TOTAL_STEPS = 15
LOG_FILE_PREFIX = "benchmarking_automation_error"


def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _log_step(step: int, description: str) -> None:
    tag = f"[Step {step:2d}/{TOTAL_STEPS}]"
    if _use_color():
        tag = f"\033[1;32m{tag}\033[0m"
    print(f"{tag} {description}")


def _resolve_sheets_credentials(args: Namespace) -> str | None:
    """
    Resolve Google credentials up-front when ``--write-sheets`` is set.

    Raising here (rather than after 14 pipeline steps) surfaces a
    misconfiguration the user should fix first.

    Args:
        args: The parsed command-line arguments.

    Returns:
        The resolved credentials path when ``--write-sheets`` is set, or
        ``None`` when the user opted out.

    Raises:
        RuntimeError: If ``--write-sheets`` is set but no credentials file
            resolves on disk, or the Drive folder / Sheets template IDs are
            unset.
    """
    if not args.write_sheets:
        return None
    credentials_path = resolve_google_credentials_path(
        google_credentials=args.google_credentials,
    )
    if credentials_path is None:
        raise RuntimeError(
            "--write-sheets was set but no Google credentials file "
            "could be resolved. Provide a path via "
            "--google-credentials or set "
            "GOOGLE_APPLICATION_CREDENTIALS to an existing file."
        )
    if not TARGET_FOLDER_ID or not TEMPLATE_ID:
        raise RuntimeError(
            "--write-sheets requires the target Google Drive folder and "
            "Sheets template IDs. Set UXHW_SHEETS_DRIVE_FOLDER_ID and "
            "UXHW_SHEETS_TEMPLATE_ID."
        )
    return credentials_path


def load_config(parser: ArgumentParser) -> Namespace:
    """Resolve arguments with config-file layering.

    Performs a two-pass parse so the precedence is
    ``defaults < config file < explicit CLI flags``: a first
    ``parse_known_args`` pass reads ``--config``. If present, the YAML
    file's keys populate argparse defaults via ``set_defaults`` (so any
    CLI flag still overrides them). A second ``parse_args`` pass applies
    the CLI on top.

    Args:
        parser: The argument parser from ``create_argument_parser``.

    Returns:
        The fully resolved arguments namespace.

    Raises:
        ValueError: If ``--config`` points to a missing file, the file
            is not a YAML mapping, or it contains keys that do not map to
            a known argument ``dest`` (catches typos like
            ``representaiton_sizes``).
    """
    pre, _ = parser.parse_known_args()
    if pre.config:
        config_path = os.path.expanduser(pre.config)
        try:
            with open(config_path, encoding="utf-8") as config_file:
                file_config = yaml.safe_load(config_file) or {}
        except FileNotFoundError as error:
            raise ValueError(f"Config file not found: {config_path}") from error
        if not isinstance(file_config, dict):
            raise ValueError(
                "Config file must contain a YAML mapping (key/value "
                f"pairs) at the top level, got {type(file_config).__name__}."
            )
        known_dests = {action.dest for action in parser._actions} - {"help"}
        unknown_keys = set(file_config) - known_dests
        if unknown_keys:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown_keys)}. "
                f"Known keys are {sorted(known_dests)}."
            )
        parser.set_defaults(**file_config)
    return parser.parse_args()


def _print_effective_config(args: Namespace) -> None:
    """Print the resolved config and the swept benchmark matrix.

    Used by ``--print-config`` to let the user confirm the merged
    configuration (defaults + config file + CLI) before committing to a
    full pipeline run.

    Args:
        args: The fully resolved arguments namespace.
    """
    print("Effective configuration:")
    for dest in sorted(vars(args)):
        print(f"  {dest}: {getattr(args, dest)}")

    matrix = [
        (representation_type, representation_size, correlation, reporting_method)
        for representation_type in args.representation_types
        for representation_size in args.representation_sizes
        for correlation in args.correlations
        for reporting_method in args.reporting_methods
    ]
    print(
        "\nBenchmark matrix "
        "(representation_type x representation_size x correlation x "
        f"reporting_method): {len(matrix)} combinations:"
    )
    for (
        representation_type,
        representation_size,
        correlation,
        reporting_method,
    ) in matrix:
        print(
            f"  {representation_type} | size={representation_size} | "
            f"{correlation} | {reporting_method}"
        )


def main() -> None:
    """
    CLI entry point: parse arguments and run the benchmarking pipeline.

    Handles ``--print-config`` (print the resolved config and exit) and
    ``--write-sheets`` pre-flight, then runs the pipeline. On failure, writes
    a timestamped traceback to ``logs/`` and exits non-zero.
    """
    parser = create_argument_parser()
    args = load_config(parser)
    validate_args(args)

    if args.print_config:
        _print_effective_config(args)
        sys.exit(0)

    # Fail early on misconfigured --write-sheets: credential resolution depends
    # only on args + env + filesystem, so raise here, not after 14 steps.
    credentials_path = _resolve_sheets_credentials(args)

    try:
        _run_pipeline(args, credentials_path=credentials_path)
    except Exception:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"{LOG_FILE_PREFIX}_{timestamp}.log")
        with open(log_file, "w") as f:
            f.write(f"Benchmarking automation failed at " f"{timestamp}\n")
            f.write(f"Arguments: {vars(args)}\n\n")
            traceback.print_exc(file=f)

        print(f"\nError: pipeline failed. " f"Full traceback written to {log_file}")
        traceback.print_exc()
        sys.exit(1)


def _run_pipeline(args: Namespace, *, credentials_path: str | None) -> None:
    """Run the full benchmarking pipeline.

    Args:
        args: Parsed command-line arguments.
        credentials_path: Resolved Google credentials path, or
            ``None`` when the user did not opt in to ``--write-sheets``.
    """
    benchmark = Benchmark(
        path_to_application=args.path_to_application,
        path_to_uxhw_sdk=args.path_to_uxhw_sdk,
        path_to_pin=args.path_to_pin,
        has_analytic_ground_truth=(args.has_analytic_ground_truth),
        path_to_ground_truth_file=(args.path_to_ground_truth_file),
        ground_truth_type=args.ground_truth_type,
        distance_type=args.distance_type,
        use_binned_uxhw=args.use_binned_uxhw,
        ground_truth_size=args.ground_truth_size,
        adversary_mc_size=args.adversary_mc_size,
        adversary_max_size_scalar=(args.adversary_max_size_scalar),
        use_clt=args.use_clt,
        representation_types=(args.representation_types),
        representation_sizes=(args.representation_sizes),
        reporting_methods=args.reporting_methods,
        correlations=args.correlations,
        max_num_weighted_samples=(args.max_num_weighted_samples),
        num_parallel_workers=(args.num_parallel_workers),
        n_adversaries=args.n_adversaries,
        max_jupiter_size=args.max_jupiter_size,
        demo_cli_args=args.demo_cli_args,
        google_credentials=args.google_credentials,
    )

    # Get the machine information
    _log_step(1, "Collecting machine info...")
    benchmark.get_machine_info()

    # Worker count bounding the three MC sample-generation pools (ground-truth,
    # asymptotic-distance, adversary-DB). get_machine_info has resolved an unset
    # -j to the core count, so this caps the MC stages at -j without exceeding
    # detected cores.
    if benchmark.num_parallel_workers is None:
        raise RuntimeError("num_parallel_workers is unset after get_machine_info().")
    mc_worker_count = min(benchmark.num_parallel_workers, benchmark.n_processors)

    # Get the benchmarking information
    _log_step(2, "Loading application info...")
    benchmark.get_application_info()

    # Bind the common run_timing_script kwargs upfront. Each database
    # generator passes only the mode-specific flag (tracing=True) plus any
    # per-variable index.
    _bound_run_timing_script = benchmark.bind_timing_script()

    # Generate the Ground Truth database
    _log_step(3, "Generating ground truth database...")
    generate_ground_truth_database(
        has_analytic_ground_truth=benchmark.has_analytic_ground_truth,
        has_native_mc=benchmark.has_native_mc,
        benchmarking_variables=benchmark.benchmarking_variables,
        path_to_application=benchmark.path_to_application,
        path_to_ground_truth_file=getattr(benchmark, "path_to_ground_truth_file", ""),
        ground_truth_size=benchmark.ground_truth_size,
        ground_truth_db_path=benchmark.ground_truth_db_path,
        ground_truth_type=benchmark.ground_truth_type,
        max_num_weighted_samples=benchmark.max_num_weighted_samples,
        n_processors=mc_worker_count,
        n_adversaries=benchmark.n_adversaries,
        adversary_max_size_scalar=benchmark.adversary_max_size_scalar,
        use_clt=benchmark.use_clt,
        native_executable_name=benchmark.native_executable_name,
        native_executable_dir=benchmark.native_executable_dir,
        demo_cli_args=benchmark.demo_cli_args,
        run_timing_script=_bound_run_timing_script,
    )

    # Generate the UxHw tracing database
    _log_step(4, "Generating UxHw tracing database...")
    generate_uxhw_tracing_database(
        benchmarking_variables=benchmark.benchmarking_variables,
        tracing_db_path=benchmark.tracing_db_path,
        run_timing_script=_bound_run_timing_script,
    )

    # Compute asymptotic distance distributions
    _log_step(
        5,
        "Computing asymptotic distance distributions...",
    )
    generate_asymptotic_distance_distributions(
        benchmarking_variables=benchmark.benchmarking_variables,
        ground_truth_db_path=benchmark.ground_truth_db_path,
        ground_truth_type=benchmark.ground_truth_type,
        distance_type=benchmark.distance_type,
        asymptotic_dist_file=benchmark.asymptotic_dist_file,
        n_processors=mc_worker_count,
        path_to_application=benchmark.path_to_application,
        native_executable_name=benchmark.native_executable_name,
        native_executable_dir=benchmark.native_executable_dir,
        demo_cli_args=benchmark.demo_cli_args,
    )

    # Compute UxHw distances
    _log_step(6, "Computing UxHw distances...")
    compute_uxhw_distances(
        benchmarking_variables=benchmark.benchmarking_variables,
        distance_type=benchmark.distance_type,
        ground_truth_db_path=benchmark.ground_truth_db_path,
        ground_truth_type=benchmark.ground_truth_type,
        tracing_db_path=benchmark.tracing_db_path,
        representation_types=benchmark.representation_types,
        representation_sizes=benchmark.representation_sizes,
        uxhw_distance_file=benchmark.uxhw_distance_file,
    )

    # Compute EMCC predictions
    _log_step(7, "Computing EMCC predictions...")
    compute_emcc_predictions(
        benchmarking_variables=benchmark.benchmarking_variables,
        use_binned_uxhw=benchmark.use_binned_uxhw,
        reporting_methods=benchmark.reporting_methods,
        output_data_file=benchmark.output_data_file,
        uxhw_distance_file=benchmark.uxhw_distance_file,
        asymptotic_dist_file=benchmark.asymptotic_dist_file,
        distance_type=benchmark.distance_type,
    )

    # Generate the adversary database
    _log_step(8, "Generating adversary database...")
    generate_adversary_database(
        has_native_mc=benchmark.has_native_mc,
        adversary_mc_size=benchmark.adversary_mc_size,
        adversary_db_path=benchmark.adversary_db_path,
        benchmarking_variables=benchmark.benchmarking_variables,
        n_processors=mc_worker_count,
        n_adversaries=benchmark.n_adversaries,
        ground_truth_size=benchmark.ground_truth_size,
        adversary_max_size_scalar=benchmark.adversary_max_size_scalar,
        use_clt=benchmark.use_clt,
        path_to_application=benchmark.path_to_application,
        native_executable_name=benchmark.native_executable_name,
        native_executable_dir=benchmark.native_executable_dir,
        demo_cli_args=benchmark.demo_cli_args,
        run_timing_script=_bound_run_timing_script,
    )

    # Compute Equivalent Monte Carlo
    _log_step(9, "Computing equivalent Monte Carlo...")
    compute_equivalent_mc(
        benchmarking_variables=benchmark.benchmarking_variables,
        output_data_file=benchmark.output_data_file,
        asymptotic_dist_file=benchmark.asymptotic_dist_file,
        ground_truth_db_path=benchmark.ground_truth_db_path,
        has_analytic_ground_truth=benchmark.has_analytic_ground_truth,
        ground_truth_type=benchmark.ground_truth_type,
        adversary_db_path=benchmark.adversary_db_path,
        tracing_db_path=benchmark.tracing_db_path,
        representation_types=benchmark.representation_types,
        representation_sizes=benchmark.representation_sizes,
        correlations=benchmark.correlations,
        num_parallel_workers=benchmark.num_parallel_workers,
        n_adversaries=benchmark.n_adversaries,
        distance_type=benchmark.distance_type,
        use_binned_uxhw=benchmark.use_binned_uxhw,
        use_clt=benchmark.use_clt,
        reporting_methods=benchmark.reporting_methods,
        plots_dir=benchmark.plots_dir,
        # User-facing flag names map to the internal EMCC keys, which keep
        # their original spellings.
        plot_comparison_distributions=args.plot_distance_vs_asymptotic,
        plot_adversary_distances=args.plot_adversary_distances,
        plot_distributions=args.plot_representative_mc,
    )

    # Generate the UxHw timing data
    _log_step(10, "Generating UxHw timing data...")
    benchmark.generate_uxhw_timings()

    # Generate the Monte Carlo timing data
    _log_step(11, "Generating native MC timing data...")
    benchmark.generate_native_mc_timings()

    # Load all measurements
    _log_step(12, "Loading measurement data...")
    load_emcc_data(
        benchmarking_variables=benchmark.benchmarking_variables,
        output_data_file=benchmark.output_data_file,
    )
    benchmark.uxhw_version = load_measurement_dicts(
        benchmarking_variables=benchmark.benchmarking_variables,
        intermediate_path=benchmark.intermediate_timings_path(),
        json_path=benchmark.json_timings_path(),
        logs_dir=benchmark.logs_dir,
        demo_cli_args=benchmark.demo_cli_args,
        representation_sizes=benchmark.representation_sizes,
        representation_types=benchmark.representation_types,
        correlations=benchmark.correlations,
    )

    # Load equivalent MC timing data from native executions. emcc_data is
    # already populated by step 12 above. Note that load_emcc_data is
    # idempotent.
    _log_step(13, "Loading timing data...")
    load_timing_data_to_dfs(
        benchmarking_variables=benchmark.benchmarking_variables,
    )

    # Compute per-record speedups
    _log_step(14, "Computing results...")
    compute_speedups(
        benchmarking_variables=benchmark.benchmarking_variables,
    )

    # Write results to markdown and spreadsheet
    _log_step(15, "Writing output files...")
    write_results_to_markdown(
        benchmarking_variables=benchmark.benchmarking_variables,
        results_dir=benchmark.results_dir,
    )
    if credentials_path is None:
        print("Skipping Google Sheets upload; pass --write-sheets to opt in.")
    else:
        write_results_to_spreadsheet(
            benchmarking_variables=benchmark.benchmarking_variables,
            credentials_path=credentials_path,
            reporting_methods=benchmark.reporting_methods,
            application_name=benchmark.application_name,
            application_version=benchmark.application_version,
            uxhw_version=benchmark.uxhw_version,
            machine_name=benchmark.machine_name,
            git_repo_remote=benchmark.git_repo_remote,
            plots_dir=benchmark.plots_dir,
        )


if __name__ == "__main__":
    main()
