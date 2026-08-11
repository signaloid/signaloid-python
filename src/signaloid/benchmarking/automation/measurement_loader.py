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

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from signaloid.benchmarking.automation.benchmarking_utils import (
    parse_timing_intermediate_stream,
)
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    UxhwDistanceRecord,
)
from signaloid.benchmarking.config import (
    AsymptoticDistanceDistribution,
    BenchmarkingVariables,
    EquivMC,
    ReportingMethods,
    TimingFormat,
    VariableTypes,
)


def load_measurement_dicts(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    intermediate_path: str,
    json_path: str,
    logs_dir: str,
    demo_cli_args: str,
    representation_sizes: list[int],
    representation_types: list[str],
    correlations: list[str],
) -> str:
    """
    Load measurement information from the transient intermediate file,
    emit the canonical JSON artifact, and populate BenchmarkingVariable
    objects.

    The canonical artifact is a single JSON document: a top-level object
    containing session-invariant fields (application identity, SDK
    versions, target rep count) alongside a ``runs`` array of per-run
    records. Session-level META keys are validated for consistency
    across runs and lifted out of each run dict into the top level.

    Args:
        benchmarking_variables: Variables to populate with measurement
            data.
        intermediate_path: Absolute path of the transient intermediate
            timings file written by ``get-timings.sh``.
        json_path: Absolute path of the canonical JSON timings artifact
            to write.
        logs_dir: Directory containing timing script logs (used in error
            messages).
        demo_cli_args: Demo-level command-line arguments string.
        representation_sizes: Configured representation sizes.
        representation_types: Configured representation types.
        correlations: Configured correlation modes.

    Returns:
        The ``uxhw_version`` lifted from the canonical document so the
        caller can store it as session metadata.

    Raises:
        RuntimeError: If the timing intermediate file is missing, if
            the parsed run count is smaller than
            ``len(benchmarking_variables) * 2``, or propagated from
            ``load_measurement_data`` (mismatched native-MC count or
            missing EMCC pre-load).
    """
    print(f"Loading measurements from {intermediate_path}")

    if not os.path.isfile(intermediate_path):
        stderr_log = os.path.join(logs_dir, "timing_script_stderr.log")
        raise RuntimeError(
            f"Timing intermediate not found at {intermediate_path}. "
            f"The bash timing script likely failed before writing it. "
            f"See {stderr_log} for details."
        )

    with open(intermediate_path, "r") as file:
        runs = parse_timing_intermediate_stream(file)

    # Keep the last two runs per variable: one UxHw-timing and one
    # native-MC-timing invocation.
    num = len(benchmarking_variables) * 2
    if len(runs) < num:
        raise RuntimeError(
            f"Error! {len(runs)} runs of UxHw timing data found, " f"expected {num}."
        )
    runs = runs[-num:]

    # Promote session-level META keys to the top-level document. Warn (don't
    # fail) on disagreement across runs, and pick the first non-empty value in
    # run order so placeholders never override a real value (deterministically).
    session: dict = {}
    for key in TimingFormat.SESSION_META_KEYS:
        values = [run.get(key, "") for run in runs]
        distinct = set(values)
        if len(distinct) > 1:
            print(f"Warning! {key} differs across runs: {sorted(distinct)}")
        session[key] = next((v for v in values if v != ""), "")
    for run in runs:
        for key in TimingFormat.SESSION_META_KEYS:
            run.pop(key, None)

    uxhw_version: str = session[TimingFormat.META_KEY_UXHW_SDK_VERSION]

    # Write the canonical JSON document before consuming the runs.
    document = {**session, TimingFormat.JSON_KEY_RUNS: runs}
    with open(json_path, "w") as json_file:
        json.dump(document, json_file, indent=2)
        json_file.write("\n")

    # Populate BenchmarkingVariable objects with measurement data.
    load_measurement_data(
        benchmarking_variables=benchmarking_variables,
        runs=runs,
        demo_cli_args=demo_cli_args,
        representation_sizes=representation_sizes,
        representation_types=representation_types,
        correlations=correlations,
    )

    # The JSON is now the canonical artifact, so remove the intermediate. Only
    # reached after a successful parse + write, so a failure keeps it around.
    try:
        os.remove(intermediate_path)
    except OSError:
        pass

    return uxhw_version


def load_measurement_data(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    runs: list[dict],
    demo_cli_args: str,
    representation_sizes: list[int],
    representation_types: list[str],
    correlations: list[str],
) -> None:
    """
    Populate BenchmarkingVariable objects with measurement data from
    parsed runs.

    Precondition: each variable's ``emcc_results.equiv_mc_list`` must
    already be populated (typically via ``analysis.load_emcc_data``)
    so the final native-MC count check has a meaningful expected value.
    The orchestrator (``benchmark_application.py`` or
    ``Benchmark.compute_*`` callers) is responsible for pre-calling
    ``analysis.load_emcc_data``.

    Args:
        benchmarking_variables: Variables to populate.
        runs: list of run dicts produced by
            ``parse_timing_intermediate_stream``.
        demo_cli_args: Demo-level command-line arguments string.
        representation_sizes: Configured representation sizes (used
            only in the final invariant check).
        representation_types: Configured representation types (used
            only in the final invariant check).
        correlations: Configured correlation modes (used only in the
            final invariant check).

    Raises:
        RuntimeError: If the parsed native-MC count disagrees with the
            EMCC-derived expectation, or if any non-zero native-MC
            measurements were parsed while ``equiv_mc_list`` is empty
            for every variable (indicates missing EMCC pre-load).
    """
    native_mc_counter = 0
    uxhw_counter = 0
    for run in runs:
        # A UxHw run's recorded CLA includes a standalone `-T` tracing flag.
        # Drop that token so it matches the variable CLA, which has none. Match
        # on the whole token (not a substring) so application arguments that
        # merely contain "-T" (e.g. `-Threads`, `-T5`, a path with `-T`) are
        # preserved rather than corrupted.
        raw_cla = run.get(TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS, "")
        cla = " ".join(tok for tok in raw_cla.split() if tok != "-T")

        for measurement in run.get(TimingFormat.JSON_KEY_MEASUREMENTS, []):
            config = measurement[TimingFormat.JSON_KEY_MEASUREMENT_CONFIG]
            time = measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME]
            db_time = measurement[TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME]
            e2e_time = measurement[TimingFormat.JSON_KEY_MEASUREMENT_E2E_TIME]
            db_dyn_inst_count = measurement[
                TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT
            ]
            pin_dyn_inst_count = measurement[
                TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT
            ]

            if config.startswith("Native-MC"):
                # Native-MC rows have no database time or database
                # dynamic instruction count. Ignore those fields.
                if None in (time, e2e_time, pin_dyn_inst_count):
                    continue
                for variable in benchmarking_variables:
                    # Collapse internal whitespace, matching the normalization
                    # of `cla` above, so YAML CLAs with accidental double
                    # spaces still match.
                    full_cla = " ".join(f"{demo_cli_args} {variable.cla}".split())
                    if full_cla == cla:
                        variable.timing_measurements.append(
                            config=config,
                            time=time,
                            e2e_time=e2e_time,
                            pin_dyn_inst_count=pin_dyn_inst_count,
                        )
                        native_mc_counter += 1
            else:
                # UxHw path: only ingest rows with all five numeric fields.
                # Reference and Native rows carry `?` in some columns and are
                # intentionally skipped.
                if None in (
                    time,
                    db_time,
                    e2e_time,
                    db_dyn_inst_count,
                    pin_dyn_inst_count,
                ):
                    continue
                for variable in benchmarking_variables:
                    full_cla = " ".join(f"{demo_cli_args} {variable.cla}".split())
                    if full_cla == cla:
                        variable.timing_measurements.append(
                            config=config,
                            time=time,
                            db_time=db_time,
                            e2e_time=e2e_time,
                            db_dyn_inst_count=db_dyn_inst_count,
                            pin_dyn_inst_count=pin_dyn_inst_count,
                        )
                        uxhw_counter += 1
    num_uxhw_configs = (
        len(benchmarking_variables)
        * len(representation_sizes)
        * len(representation_types)
        * len(correlations)
    )
    if uxhw_counter != num_uxhw_configs:
        raise RuntimeError(
            f"Error! Measurement data was obtained for {uxhw_counter} "
            f"UxHw configurations, expected {num_uxhw_configs}"
        )
    num_native_mc_configs: int = int(
        np.sum(
            [
                len(variable.emcc_results.equiv_mc_list)
                for variable in benchmarking_variables
            ]
        )
    )
    if native_mc_counter > 0 and num_native_mc_configs == 0:
        raise RuntimeError(
            f"Parsed {native_mc_counter} native-MC measurement rows but "
            "BenchmarkingVariable.emcc_results.equiv_mc_list is empty for "
            "every variable. Call analysis.load_emcc_data() before "
            "load_measurement_data()."
        )
    if native_mc_counter != num_native_mc_configs:
        raise RuntimeError(
            f"Error! Measurement data was obtained {native_mc_counter} "
            f"native MC configurations, expected {num_native_mc_configs}"
        )


def load_timing_data_to_dfs(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
) -> None:
    """
    Cross-join per-variable timing measurements into ``emcc_data``.

    For each variable, walk its ``timing_measurements.measurement_dict``
    and merge each row into the matching ``emcc_data`` record. Native-MC
    configurations are matched on the trailing integer (the equivalent
    Monte Carlo count) and prefix their column names with ``"Native "``.
    UxHw configurations are matched on the UxHw configuration string.

    Precondition: each variable's ``emcc_results.emcc_data`` must be
    populated (typically via ``analysis.load_emcc_data``). Otherwise
    there is nothing to join into and every row would be silently
    skipped. The orchestrator is responsible for pre-calling
    ``analysis.load_emcc_data``.

    Args:
        benchmarking_variables: Variables whose ``timing_measurements``
            and ``emcc_data`` are joined in place.

    Raises:
        RuntimeError: If every variable has an empty
            ``emcc_results.emcc_data`` (indicates missing EMCC pre-load).
    """
    if benchmarking_variables and not any(
        variable.emcc_results.emcc_data for variable in benchmarking_variables
    ):
        raise RuntimeError(
            "Cannot join timing measurements: BenchmarkingVariable."
            "emcc_results.emcc_data is empty for every variable. Call "
            "analysis.load_emcc_data() before load_timing_data_to_dfs()."
        )

    for variable in benchmarking_variables:
        for dist, values in variable.timing_measurements.measurement_dict.items():
            dist_str = repr(dist).strip("'")
            is_native = dist_str.startswith("Native-MC-")

            match_val: str | int
            # Determine match key and value
            if is_native:
                native_count_str = dist_str.split("-")[-1]
                try:
                    match_val = int(native_count_str)
                except ValueError:
                    print(
                        f"Warning: could not parse an MC count from native-MC "
                        f"config {dist_str!r}; skipping this measurement."
                    )
                    continue
                if (
                    variable.emcc_results.emcc_data
                    and EquivMC.EMCC in variable.emcc_results.emcc_data[0]
                ):
                    match_key = EquivMC.EMCC
                elif (
                    variable.emcc_results.emcc_data
                    and EquivMC.EMCC_PREDICTED in variable.emcc_results.emcc_data[0]
                ):
                    match_key = EquivMC.EMCC_PREDICTED
                else:
                    print(
                        f"Warning: Neither {EquivMC.EMCC} nor {EquivMC.EMCC_PREDICTED} found"
                    )
                    continue
            else:
                match_key = BenchmarkingVariables.UXHW_CONF
                match_val = dist

            # Update matching records. repr() + .strip("'") normalises both
            # plain strings and non-string objects in emcc_data to the same
            # bare configuration text.
            found_match = False
            for record in variable.emcc_results.emcc_data:
                record_val = record.get(match_key)
                if record_val == match_val or repr(record_val).strip("'") == dist_str:
                    found_match = True
                    for col, val in values.items():
                        col_name = f"Native {col}" if is_native else col
                        record[col_name] = val

            if not found_match:
                print(f"Warning: No timing data found for {dist_str}")


def load_asymptotic_dist(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    asymptotic_dist_file: str,
) -> None:
    """
    Load asymptotic distribution data into each variable's
    ``asymptotic_distribution`` dataclass if not already populated.

    For distribution-typed variables, also loads the asymptotic-sample
    NumPy file produced by the asymptotic-distance pipeline.

    A variable counts as already populated when its
    ``asymptotic_distribution.quantile_95`` is not ``None``. The generator
    always sets ``quantile_95`` (from a ``quantile`` / ``ppf`` call that never
    returns ``None``), making it a more reliable "producer has run" sentinel
    than ``mean``, which can legitimately be ``None``.

    Args:
        benchmarking_variables: Variables to populate.
        asymptotic_dist_file: Path to the asymptotic-distance CSV file.

    Raises:
        FileNotFoundError: If ``asymptotic_dist_file`` is missing and
            at least one variable still needs loading. (No exception
            is raised when every variable already has
            ``asymptotic_distribution.quantile_95`` populated — the
            function returns early before any file access.)
    """
    # Skip the whole CSV read if every variable already has data.
    if all(
        variable.asymptotic_distribution.quantile_95 is not None
        for variable in benchmarking_variables
    ):
        return

    print(
        "Warning: Asymptotic distance distribution data not found in Benchmark object."
    )
    print(f"Loading asymptotic distance data from {asymptotic_dist_file}")

    try:
        asymptotic_df = pd.read_csv(asymptotic_dist_file)
    except FileNotFoundError:
        print(f"Error: File not found at {asymptotic_dist_file}")
        print("Cannot continue without asymptotic distance data. Terminating.")
        raise

    records_by_variable: defaultdict = defaultdict(list)
    for record in asymptotic_df.to_dict("records"):
        var_name = record[BenchmarkingVariables.VARIABLE_DESCRIPTION]
        records_by_variable[var_name].append(record)

    for variable in benchmarking_variables:
        if variable.asymptotic_distribution.quantile_95 is not None:
            continue

        if variable.type == VariableTypes.DISTRIBUTION:
            samples = np.load(f"{variable.formatted_description}-asymptotic.npy")
            variable.asymptotic_distribution.samples = samples

        variable_records: list[dict] = records_by_variable[variable.description]
        if not variable_records:
            print(f"Warning: No asymptotic data found for '{variable.description}'")
            continue

        # Take the first matching record (matches legacy behaviour).
        record = variable_records[0]
        asymptotic = variable.asymptotic_distribution
        asymptotic.mean = record.get(ReportingMethods.MEAN)
        asymptotic.quantile_95 = record.get(ReportingMethods.QUANTILE_95)
        asymptotic.quantile_99 = record.get(ReportingMethods.QUANTILE_99)
        asymptotic.mean_quantile = record.get(EquivMC.MEAN_QUANTILE)
        asymptotic.is_normal = record.get(AsymptoticDistanceDistribution.IS_NORMAL)
        asymptotic.scale = record.get(AsymptoticDistanceDistribution.SCALE)


def load_uxhw_distances(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    uxhw_distance_file: str,
    asymptotic_dist_file: str,
) -> None:
    """
    Load UxHw distance distribution data into each variable's
    ``uxhw_distances.records`` if not already populated.

    Also pre-loads asymptotic distribution data (intra-module call to
    :func:`load_asymptotic_dist`), preserving the legacy invocation
    ordering at the orchestrator's call sites.

    Args:
        benchmarking_variables: Variables to populate.
        uxhw_distance_file: Path to the UxHw distances CSV file.
        asymptotic_dist_file: Path to the asymptotic-distance CSV file
            (forwarded to :func:`load_asymptotic_dist`).

    Raises:
        FileNotFoundError: If ``uxhw_distance_file`` is missing and at
            least one variable still needs UxHw-distance hydration, or
            propagated from :func:`load_asymptotic_dist` if
            ``asymptotic_dist_file`` is missing and any variable still
            needs asymptotic hydration. (Returns early without raising
            when every variable's relevant stage is already populated.)
    """
    load_asymptotic_dist(
        benchmarking_variables=benchmarking_variables,
        asymptotic_dist_file=asymptotic_dist_file,
    )

    # Skip CSV read if no variable needs hydration.
    if not any(
        not variable.uxhw_distances.records for variable in benchmarking_variables
    ):
        return

    print("Warning: UxHw distance distribution data not found in Benchmark object.")
    print(f"Loading UxHw distance data from {uxhw_distance_file}")

    try:
        distance_df = pd.read_csv(uxhw_distance_file)
    except FileNotFoundError:
        print(f"Error: File not found at {uxhw_distance_file}")
        print("Cannot continue without UxHw distances data. Terminating.")
        raise

    records_by_variable: defaultdict = defaultdict(list)
    for record in distance_df.to_dict("records"):
        var_name = record[BenchmarkingVariables.VARIABLE_DESCRIPTION]
        records_by_variable[var_name].append(record)

    for variable in benchmarking_variables:
        if variable.uxhw_distances.records:
            continue

        variable_records = records_by_variable[variable.description]
        if not variable_records:
            print(f"Warning: No UxHw distance data found for '{variable.description}'")
            continue

        for record in variable_records:
            binned = record.get(BenchmarkingVariables.UXHW_BINNED_DISTANCE)
            if pd.isna(binned):
                binned = None
            variable.uxhw_distances.records.append(
                UxhwDistanceRecord(
                    uxhw_conf=record.get(BenchmarkingVariables.UXHW_CONF),
                    uxhw_distance=record.get(BenchmarkingVariables.UXHW_DISTANCE),
                    uxhw_binned_distance=binned,
                )
            )
