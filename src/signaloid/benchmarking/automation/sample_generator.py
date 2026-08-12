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

import warnings
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np
from signaloid.distributional.distributional import DistributionalValue

from signaloid.benchmarking.automation.benchmarking_utils import (
    _run_mc_simulation,
    _run_scalar_native,
)
from signaloid.benchmarking.types import BenchmarkingVariable
from signaloid.benchmarking.config import (
    EquivMC,
    VariableTypes,
)
from signaloid.benchmarking.distribution_helpers.collapse import (
    _collapse_asymptotically_optimal_w1,
)

# Per-process MC sub-job size used by ``_run_distribution_mc_flat`` to
# split large draws across the worker pool. Module-level so tests can
# patch it.
_MC_CHUNK_SIZE = 1_000_000


def generate_database_native_mc(
    *,
    size: int,
    database_path: str,
    benchmarking_variables: list[BenchmarkingVariable],
    n_processors: int,
    n_adversaries: int,
    ground_truth_size: int,
    adversary_max_size_scalar: int,
    use_clt: bool,
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
    demo_cli_args: str,
    n_steps_scalar: int = 100,
    weighted_samples: bool = False,
    num_weighted_samples: int = 100_000,
    ground_truth: bool = False,
) -> None:
    """
    Generate a Monte Carlo database by executing the native binary.

    Splits ``benchmarking_variables`` into distribution-typed and
    scalar-typed groups, runs each group through its respective
    parallel-MC helper, and writes the collected samples to a SQLite
    database in either weighted-sample or raw-MC format.

    Args:
        size: Number of Monte Carlo samples to draw per distribution
            variable.
        database_path: Filesystem path at which to write the database.
        benchmarking_variables: Variables to populate with samples.
        n_processors: Number of parallel worker processes.
        n_adversaries: Number of independent adversary runs per scalar
            sample size.
        ground_truth_size: Sample size used for the ground-truth scalar
            pass.
        adversary_max_size_scalar: Upper bound used when generating the
            geometric series of scalar sample sizes.
        use_clt: When ``True``, scalar sample sizes are taken
            from the pre-computed EMCC predictions rather than the
            geometric schedule.
        path_to_application: Root path of the application source tree.
        native_executable_name: Filename of the compiled native binary.
        native_executable_dir: Directory containing the native binary.
        demo_cli_args: Per-application command-line argument prefix
            prepended to each variable's own CLA when invoking the
            native executable.
        n_steps_scalar: Number of different scalar sample sizes to
            evaluate.
        weighted_samples: When ``True``, convert raw MC samples to
            weighted samples before persisting.
        num_weighted_samples: Target weighted-sample support size.
        ground_truth: When ``True``, generate the ground-truth
            database. Otherwise the adversarial MC database.
    """
    # Split variables by type, preserving original index for stable
    # ordering.
    dist_vars: list[tuple[int, BenchmarkingVariable]] = []
    scalar_vars: list[tuple[int, BenchmarkingVariable]] = []
    for var_idx, variable in enumerate(benchmarking_variables):
        if variable.type == VariableTypes.DISTRIBUTION:
            dist_vars.append((var_idx, variable))
        elif variable.type == VariableTypes.SCALAR:
            scalar_vars.append((var_idx, variable))
        else:
            print(
                f"Warning! BenchmarkingVariable {variable.name} " f"type not supported!"
            )

    if dist_vars:
        _run_distribution_mc_flat(
            dist_vars=dist_vars,
            size=size,
            weighted_samples=weighted_samples,
            num_weighted_samples=num_weighted_samples,
            benchmarking_variables=benchmarking_variables,
            n_processors=n_processors,
            path_to_application=path_to_application,
            native_executable_name=native_executable_name,
            native_executable_dir=native_executable_dir,
            demo_cli_args=demo_cli_args,
        )

    if scalar_vars:
        _run_scalar_mc_flat(
            scalar_vars=scalar_vars,
            n_steps_scalar=n_steps_scalar,
            ground_truth=ground_truth,
            benchmarking_variables=benchmarking_variables,
            n_processors=n_processors,
            n_adversaries=n_adversaries,
            ground_truth_size=ground_truth_size,
            adversary_max_size_scalar=adversary_max_size_scalar,
            use_clt=use_clt,
            path_to_application=path_to_application,
            native_executable_name=native_executable_name,
            native_executable_dir=native_executable_dir,
            demo_cli_args=demo_cli_args,
        )

    # Write to database. Imported here to avoid a top-level circular
    # import: database_generator.py imports generate_database_native_mc
    # from this module.
    from signaloid.benchmarking.automation.database_generator import (
        generate_database_from_mc_samples,
        generate_database_from_weighted_samples,
    )

    if weighted_samples:
        generate_database_from_weighted_samples(database_path, benchmarking_variables)
    else:
        generate_database_from_mc_samples(database_path, benchmarking_variables)

    # Clean up
    for variable in benchmarking_variables:
        variable.distribution_samples.empty_values()


def _determine_scalar_sample_sizes(
    *,
    variable: BenchmarkingVariable,
    n_steps_scalar: int,
    use_clt: bool,
    adversary_max_size_scalar: int,
    ground_truth_size: int,
) -> list[int]:
    """Determine which sample sizes to use for a scalar variable.

    In the ``use_clt`` branch the sizes come from the (uncapped) EMCC
    predictions, which can run into the billions for a high-fidelity
    representation. Such a count cannot be executed natively (``-M`` is a
    signed 32-bit ``int``, and running more iterations than the ground truth is
    infeasible anyway), so each predicted size is clamped to
    ``ground_truth_size`` for *execution*. The reported ``EMCC_PREDICTED`` value
    is computed elsewhere and left uncapped.

    Args:
        variable: The scalar variable whose sample sizes are resolved.
        n_steps_scalar: Number of points in the geometric schedule used
            when ``use_clt`` is ``False``.
        use_clt: When ``True``, take sizes from the EMCC predictions.
            Otherwise build a geometric schedule.
        adversary_max_size_scalar: Upper bound of the geometric
            schedule.
        ground_truth_size: Feasible upper bound for any executed size.
            Predicted sizes above it are clamped down to it.

    Returns:
        Sorted, de-duplicated list of sample sizes to execute.
    """
    sizes: list[int]
    if use_clt:
        sizes = []
        for dic in variable.emcc_results.emcc_data:
            predicted = dic[EquivMC.EMCC_PREDICTED]
            if predicted > ground_truth_size:
                warnings.warn(
                    f"EMCC-predicted sample size {predicted} for "
                    f"'{variable.description}' exceeds the ground-truth "
                    f"size {ground_truth_size}; clamping the executed "
                    f"adversary size to {ground_truth_size} (the reported "
                    f"EMCC prediction is unchanged).",
                    stacklevel=2,
                )
                predicted = ground_truth_size
            sizes.append(predicted)
    else:
        sizes = np.geomspace(
            1,
            adversary_max_size_scalar,
            num=n_steps_scalar,
            dtype=int,
        ).tolist()

    sizes = sorted(set(sizes))

    return sizes


def _run_distribution_mc_flat(
    *,
    dist_vars: list[tuple[int, BenchmarkingVariable]],
    size: int,
    weighted_samples: bool,
    num_weighted_samples: int,
    benchmarking_variables: list[BenchmarkingVariable],
    n_processors: int,
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
    demo_cli_args: str,
) -> None:
    """
    Run MC sampling for all distribution variables in a shared pool.

    Chunks each variable's ``size`` samples into sub-jobs of up to 1M,
    then submits the union of all sub-jobs across variables to one
    ``ProcessPoolExecutor`` so that ``-j N`` is saturated even when
    each variable alone would fit in a single chunk.
    """
    if size <= 0:
        warnings.warn(
            f"Skipping distribution MC for {len(dist_vars)} variables: "
            f"requested sample size is {size} (must be > 0).",
            stacklevel=2,
        )
        return

    max_size = _MC_CHUNK_SIZE
    quotient, remainder = divmod(size, max_size)
    sub_sizes_per_var = [max_size] * quotient + ([remainder] if remainder else [])

    # Flat list of (var_idx, chunk_idx, sub_size, cla) work items across all
    # variables.
    work_items: list[tuple[int, int, int, str]] = []
    for var_idx, variable in dist_vars:
        cla = f"{demo_cli_args} {variable.cla}".strip()
        for chunk_idx, sub_size in enumerate(sub_sizes_per_var):
            work_items.append((var_idx, chunk_idx, sub_size, cla))

    print(
        f"Running {len(work_items)} MC simulations across "
        f"{len(dist_vars)} distributional variables using {n_processors} "
        f"parallel processes"
    )
    if weighted_samples:
        # Log ``num_weighted_samples`` directly, since that is the collapse
        # size (``n_dirac_deltas``), including when it exceeds ``size`` (an
        # explicit oversampling request the collapse accepts).
        print(
            f"Converting {size} samples into {num_weighted_samples} "
            f"weighted samples for {len(dist_vars)} variables."
        )

    def finalize(var_idx: int, chunk_map: dict[int, list[float]]) -> None:
        variable = benchmarking_variables[var_idx]
        values: list[float] = []
        for chunk_idx in sorted(chunk_map.keys()):
            values.extend(chunk_map.pop(chunk_idx))

        if not values:
            raise RuntimeError(
                f"No MC samples collected for variable "
                f"'{variable.description}' (var_idx={var_idx}); "
                f"all simulation chunks returned empty results."
            )

        if weighted_samples:
            dv = DistributionalValue.from_samples(np.array(values))
            collapsed = _collapse_asymptotically_optimal_w1(
                dv, n_dirac_deltas=num_weighted_samples
            )
            variable.distribution_samples.set_weighted_values(
                collapsed.positions.tolist(), collapsed.masses.tolist()
            )
        else:
            variable.distribution_samples.set_values(values)

    # var_idx -> {chunk_idx -> samples}. Popped once a variable's last chunk
    # arrives, so raw samples are not retained for all variables at once.
    results: dict[int, dict[int, list[float]]] = defaultdict(dict)
    remaining_chunks = {var_idx: len(sub_sizes_per_var) for var_idx, _ in dist_vars}
    completed_vars = 0
    # The weighted-sample collapse is numpy-heavy and independent across
    # variables, so overlap finalize work with the remaining MC chunks.
    finalize_pool = (
        ThreadPoolExecutor(max_workers=n_processors) if weighted_samples else None
    )
    finalize_futures: list = []
    # ``try/finally`` guards ``finalize_pool.shutdown()`` so a re-raised
    # exception from the inner ``with ProcessPoolExecutor`` block does not leak
    # the thread pool, which lives in this outer scope.
    try:
        with ProcessPoolExecutor(max_workers=n_processors) as executor:
            future_to_key = {}
            for var_idx, chunk_idx, sub_size, cla in work_items:
                # Globally-unique index so _run_mc_simulation's temp-dir
                # prefix and error messages disambiguate across variables.
                global_index = f"v{var_idx}_c{chunk_idx}"
                fut = executor.submit(
                    _run_mc_simulation,
                    (global_index, sub_size, cla),
                    path_to_application,
                    native_executable_name,
                    native_executable_dir,
                )
                future_to_key[fut] = (var_idx, chunk_idx)

            for future in as_completed(future_to_key):
                var_idx, chunk_idx = future_to_key[future]
                try:
                    results[var_idx][chunk_idx] = future.result()
                except Exception as exc:
                    # Fail fast: absorbing a partial failure (substituting
                    # ``[]``) would corrupt the database with under-sampled
                    # variables.
                    raise RuntimeError(
                        f"MC simulation failed "
                        f"(var_idx={var_idx}, chunk={chunk_idx}): "
                        f"{exc}"
                    ) from exc
                remaining_chunks[var_idx] -= 1
                if remaining_chunks[var_idx] == 0:
                    completed_vars += 1
                    variable = benchmarking_variables[var_idx]
                    print(
                        f"  [{completed_vars}/{len(dist_vars)}] MC "
                        f"complete: {variable.description}"
                    )
                    chunk_map = results.pop(var_idx)
                    if finalize_pool is not None:
                        finalize_futures.append(
                            finalize_pool.submit(finalize, var_idx, chunk_map)
                        )
                    else:
                        finalize(var_idx, chunk_map)

        if finalize_pool is not None:
            for fut in finalize_futures:
                fut.result()
    finally:
        if finalize_pool is not None:
            finalize_pool.shutdown()


def _run_scalar_mc_flat(
    *,
    scalar_vars: list[tuple[int, BenchmarkingVariable]],
    n_steps_scalar: int,
    ground_truth: bool,
    benchmarking_variables: list[BenchmarkingVariable],
    n_processors: int,
    n_adversaries: int,
    ground_truth_size: int,
    adversary_max_size_scalar: int,
    use_clt: bool,
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
    demo_cli_args: str,
) -> None:
    """
    Run scalar MC sampling for all scalar variables in a shared pool.

    Flattens ``(size, repetition)`` combinations across every scalar
    variable into one work list so the pool stays saturated instead of
    idling between variables.
    """
    # Initialize per-variable output dicts and collect work items
    work_items: list[tuple[int, int, str, int]] = []
    remaining_runs: dict[int, int] = {}
    run_id_counter = 0
    for var_idx, variable in scalar_vars:
        if ground_truth:
            sizes = [ground_truth_size]
            repetitions = 1
        else:
            repetitions = n_adversaries
            sizes = _determine_scalar_sample_sizes(
                variable=variable,
                n_steps_scalar=n_steps_scalar,
                use_clt=use_clt,
                adversary_max_size_scalar=adversary_max_size_scalar,
                ground_truth_size=ground_truth_size,
            )

        variable.distribution_samples.scalar_output_dict = {s: [] for s in sizes}
        cla = f"{demo_cli_args} {variable.cla}".strip()
        var_runs = 0
        for size in sizes * repetitions:
            work_items.append((var_idx, size, cla, run_id_counter))
            run_id_counter += 1
            var_runs += 1
        remaining_runs[var_idx] = var_runs

    if not work_items:
        return

    print(
        f"Running {len(work_items)} scalar simulations across "
        f"{len(scalar_vars)} variables using {n_processors} "
        f"parallel processes"
    )

    completed_vars = 0
    with ProcessPoolExecutor(max_workers=n_processors) as executor:
        future_to_var = {}
        for var_idx, size, cla, run_id in work_items:
            fut = executor.submit(
                _run_scalar_native,
                path_to_application,
                native_executable_name,
                native_executable_dir,
                cla,
                size,
                run_id,
            )
            future_to_var[fut] = var_idx

        for future in as_completed(future_to_var):
            var_idx = future_to_var[future]
            variable = benchmarking_variables[var_idx]
            try:
                sample_size, value = future.result()
                if value is not None:
                    variable.distribution_samples.scalar_output_dict[
                        sample_size
                    ].append(value)
            except Exception as exc:
                # Fail fast: absorbing a partial failure would leave the
                # variable under-sampled relative to
                # ``n_adversaries`` * len(sizes).
                raise RuntimeError(
                    f"Scalar MC run failed for " f"{variable.description}: {exc}"
                ) from exc
            finally:
                remaining_runs[var_idx] -= 1
                if remaining_runs[var_idx] == 0:
                    completed_vars += 1
                    print(
                        f"  [{completed_vars}/{len(scalar_vars)}] "
                        f"scalar MC complete: {variable.description}"
                    )


def generate_scalar_samples(
    *,
    variable: BenchmarkingVariable,
    sizes: list[int],
    repetitions: int,
    n_processors: int,
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
    demo_cli_args: str,
) -> dict[int, list[float]]:
    """
    Generate Monte Carlo samples for a scalar variable using the
    native executable with parallel processing.

    Runs the native executable with ``-M <size>`` for each combination
    of ``sizes`` and ``repetitions``, collecting a single scalar
    output per run. Each size is executed ``repetitions`` times in
    parallel via ``ProcessPoolExecutor``, producing multiple
    independent scalar values per sample size (e.g. one per
    adversary).

    Args:
        variable: The benchmarking variable whose ``cla`` field
            supplies the command-line arguments passed to the native
            executable.
        sizes: List of Monte Carlo sample sizes to evaluate. Each size
            is passed as the ``-M`` argument to the native executable.
        repetitions: Number of independent runs per sample size. For
            ground truth generation this is typically 1. For adversary
            generation it equals ``n_adversaries``.
        n_processors: Number of parallel worker processes.
        path_to_application: Root path of the application source tree.
        native_executable_name: Filename of the compiled native
            binary.
        native_executable_dir: Directory containing the native binary.
        demo_cli_args: Per-application command-line argument prefix
            prepended to ``variable.cla`` when invoking the native
            executable.

    Returns:
        A dict mapping each sample size to a list of scalar output
        values collected across all repetitions for that size.
    """

    scalar_output_dict: dict[int, list[float]] = {size: [] for size in sizes}

    # Use multiprocessing to speedup scalar adversary array creation
    with ProcessPoolExecutor(max_workers=n_processors) as executor:
        futures = []
        for run_id, size in enumerate(sizes * repetitions):
            futures.append(
                executor.submit(
                    _run_scalar_native,
                    path_to_application,
                    native_executable_name,
                    native_executable_dir,
                    f"{demo_cli_args} {variable.cla}".strip(),
                    size,
                    run_id,
                )
            )

        for future in as_completed(futures):
            sample_size, value = future.result()
            if value is not None:
                scalar_output_dict[sample_size].append(value)

    return scalar_output_dict
