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

from contextlib import contextmanager
from typing import Any, Iterable, Iterator
import subprocess
import os
import shlex
import pandas as pd
from signaloid.benchmarking.config import (
    BenchmarkingVariables,
    EquivMC,
    TimingFormat,
)
from signaloid.distributional.distributional import DistributionalValue
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
import re
import numpy as np
import tempfile
import csv

_SAMPLES_KEY = "_samples"


def _symlink_application_inputs(inputs_dir: str, dest_dir: str) -> None:
    """
    Symlink an application's ``inputs/`` files into a run directory.

    The native binary writes its samples to a *relative* ``data.out`` in
    its working directory. A stale ``data.out`` left in ``inputs/`` (the
    bash timing layer runs the binary there) must never be symlinked in:
    the binary's ``fopen("data.out", "w")`` would follow the symlink and
    write *through* it to the single shared file, so concurrent workers
    (``-j > 1``) would clobber one another and different variables would
    end up with identical samples.

    Args:
        inputs_dir: The application's ``inputs/`` directory.
        dest_dir: The per-run working directory to populate with symlinks.
    """
    # Resolve to absolute paths so the symlink targets are valid even when
    # a relative application path was provided (os.symlink stores the
    # target verbatim, so a relative target would resolve against dest_dir).
    inputs_dir = os.path.abspath(inputs_dir)
    dest_dir = os.path.abspath(dest_dir)
    if not os.path.isdir(inputs_dir):
        return
    for entry in os.listdir(inputs_dir):
        if entry == EquivMC.MC_OUTPUT_FILENAME:
            # Never expose the binary's output file as an input.
            continue
        os.symlink(
            os.path.join(inputs_dir, entry),
            os.path.join(dest_dir, entry),
        )


def _read_pin_inst_count(path: str) -> float:
    """
    Read an inscount.out file produced by Intel PIN and return the count.

    Args:
        path: Path to the inscount.out file written by PIN's inscount0.so
            tool. Both absolute and relative paths are accepted. Relative
            paths are resolved against the current working directory. The
            file contains a single line of the form ``Count <integer>``.

    Returns:
        The instruction count as a float.

    Raises:
        ValueError: If the file content does not match the expected format.
        OSError: If the file cannot be opened.
    """
    with open(path) as f:
        content = f.read().strip()
    if not content.startswith("Count "):
        raise ValueError(f"Unexpected inscount.out format in {path!r}: {content!r}")
    return float(content.removeprefix("Count "))


def _resolve_pin_inst_from_samples(
    samples: dict[str, list[str]],
) -> float | None:
    """
    Average per-iteration PIN instruction counts from collected sample files.

    Each ``samples`` value is a list of file paths whose counts are summed to
    give that iteration's total. The per-iteration totals are then averaged.

    Args:
        samples: Mapping from iteration index to a list of inscount.out
            paths whose counts are summed for that iteration.

    Returns:
        The averaged instruction count, or ``None`` if ``samples`` is empty.
    """
    if not samples:
        return None
    per_iteration_totals: list[float] = []
    for paths in samples.values():
        iteration_total = sum(_read_pin_inst_count(p) for p in paths)
        per_iteration_totals.append(iteration_total)
    return sum(per_iteration_totals) / len(per_iteration_totals)


def _resolve_time_from_samples(
    samples: dict[str, list[str]],
) -> float | None:
    """
    Average per-iteration float values from collected SAMPLE values.

    Like :func:`_resolve_pin_inst_from_samples` but for value-typed SAMPLE
    entries, where each value is a literal float string rather than a file path.
    Values are summed within an iteration, then averaged across iterations.
    Being field-agnostic, it serves the elapsedTime, databaseTime, and
    databaseDynInstCount fields uniformly.

    Args:
        samples: Mapping from iteration index to a list of literal
            float-string values emitted by the bash side as SAMPLE values for a
            value-typed field.

    Returns:
        The averaged value, or ``None`` if ``samples`` is empty or every value
        is non-numeric.
    """
    if not samples:
        return None
    per_iteration_totals: list[float] = []
    for tokens in samples.values():
        valid: list[float] = []
        for value_str in tokens:
            try:
                valid.append(float(value_str))
            except ValueError:
                continue
        if not valid:
            continue
        per_iteration_totals.append(sum(valid))
    if not per_iteration_totals:
        return None
    return sum(per_iteration_totals) / len(per_iteration_totals)


def parse_timing_intermediate_stream(lines: Iterable[str]) -> list[dict]:
    """
    Parse the tagged line stream written by get-timings.sh into a list
    of per-run dicts.

    A new run begins whenever a `META timestamp ...` line is seen. All
    subsequent META lines attach to that run, SAMPLE lines accumulate
    per-iteration tokens keyed by (config, field), and MEASUREMENT
    lines append to the run's `measurements` list. When a MEASUREMENT
    line arrives, any field whose value is the ``?`` sentinel is
    resolved from the accumulated SAMPLE data for the matching
    (config, field): path-typed samples (e.g. ``pinDynInstCount``) are
    resolved by reading and summing the referenced inscount.out files
    per iteration before averaging. Value-typed samples (e.g.
    ``elapsedTime``, ``databaseTime``, ``databaseDynInstCount``) are
    float-parsed in place and averaged the same way. Session-level
    META keys (see
    TimingFormat.SESSION_META_KEYS) are preserved on every run dict
    at this stage and are promoted to the top level by the caller
    that writes the canonical JSON document.

    Args:
        lines: Iterable of raw lines (with or without trailing
            newline). Accepts a file object, a list of strings, or any
            other iterable so the source can be swapped from a file to
            a live pipe without touching this logic.

    Returns:
        A list of run dicts, each carrying the META keys at the top
        level and a `measurements` list of per-config dicts.
    """
    runs: list[dict] = []
    current: dict | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        tag, _, rest = line.partition(" ")
        if tag == TimingFormat.META_TAG:
            key, _, value = rest.partition(" ")
            if key == TimingFormat.META_KEY_TIMESTAMP:
                if current is not None:
                    _finalise_run(current)
                    runs.append(current)
                current = {
                    TimingFormat.JSON_KEY_MEASUREMENTS: [],
                    _SAMPLES_KEY: {},
                }
            if current is None:
                # META line before any timestamp: be permissive and
                # open a fresh run so keys are not dropped.
                current = {
                    TimingFormat.JSON_KEY_MEASUREMENTS: [],
                    _SAMPLES_KEY: {},
                }
            current[key] = value
        elif tag == TimingFormat.SAMPLE_TAG:
            if current is None:
                continue
            tokens = rest.split(maxsplit=3)
            if len(tokens) != 4:
                continue
            s_config, s_field, s_iteration, s_token = tokens
            sample_key = (s_config, s_field)
            iteration_map: dict[str, list[str]] = current[_SAMPLES_KEY].setdefault(
                sample_key, {}
            )
            iteration_map.setdefault(s_iteration, []).append(s_token)
        elif tag == TimingFormat.MEASUREMENT_TAG:
            if current is None:
                continue
            tokens = rest.split()
            if len(tokens) != 6:
                continue
            config, t_val, db_t, e2e_t, db_i, pin_i = tokens

            def _num(token: str) -> float | None:
                if token == TimingFormat.MISSING_VALUE:
                    return None
                return float(token)

            time_value = _num(t_val)
            if time_value is None:
                sample_key = (config, TimingFormat.SAMPLE_FIELD_TIME)
                iteration_map = current[_SAMPLES_KEY].pop(sample_key, {})
                time_value = _resolve_time_from_samples(iteration_map)

            db_time_value = _num(db_t)
            if db_time_value is None:
                sample_key = (config, TimingFormat.SAMPLE_FIELD_DB_TIME)
                iteration_map = current[_SAMPLES_KEY].pop(sample_key, {})
                db_time_value = _resolve_time_from_samples(iteration_map)

            db_inst_value = _num(db_i)
            if db_inst_value is None:
                sample_key = (
                    config,
                    TimingFormat.SAMPLE_FIELD_DB_DYN_INST_COUNT,
                )
                iteration_map = current[_SAMPLES_KEY].pop(sample_key, {})
                db_inst_value = _resolve_time_from_samples(iteration_map)

            pin_inst_value = _num(pin_i)
            if pin_inst_value is None:
                sample_key = (
                    config,
                    TimingFormat.SAMPLE_FIELD_PIN_INST,
                )
                iteration_map = current[_SAMPLES_KEY].pop(sample_key, {})
                pin_inst_value = _resolve_pin_inst_from_samples(iteration_map)

            current[TimingFormat.JSON_KEY_MEASUREMENTS].append(
                {
                    TimingFormat.JSON_KEY_MEASUREMENT_CONFIG: config,
                    TimingFormat.JSON_KEY_MEASUREMENT_TIME: time_value,
                    TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME: db_time_value,
                    TimingFormat.JSON_KEY_MEASUREMENT_E2E_TIME: _num(e2e_t),
                    TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT: (
                        db_inst_value
                    ),
                    TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT: (
                        pin_inst_value
                    ),
                }
            )
    if current is not None:
        _finalise_run(current)
        runs.append(current)
    return runs


def _finalise_run(run: dict) -> None:
    """Remove internal bookkeeping keys before a run dict is returned.

    Args:
        run: A run dict that may contain the private ``_SAMPLES_KEY``
            entry added during parsing.
    """
    run.pop(_SAMPLES_KEY, None)


def get_git_remote(directory: str) -> str | None:
    """
    Return the normalised ``origin`` remote URL of a Git repository.

    Strips the protocol, any embedded token, and the ``.git`` suffix.

    Args:
        directory: Path to the repository to inspect.

    Returns:
        The normalised remote URL, or ``None`` if ``directory`` is not a Git
        repository or has no ``origin`` remote.
    """

    if not os.path.isdir(os.path.join(directory, ".git")):
        print("Not a Git repository")
        return None

    try:
        # Run the 'git remote get-url origin' command
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=directory,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Handle SSH format: git@github.com:user/repo.git
        if "git@" in remote_url:
            remote_url = remote_url.split("git@")[1]
            remote_url = remote_url.replace(":", "/")
        # Handle HTTPS format: https://github.com/user/repo.git
        elif remote_url.startswith("https://"):
            # Remove the protocol
            remote_url = remote_url.replace("https://", "")
            # Remove token if present (TOKEN@github.com -> github.com)
            if "@" in remote_url:
                remote_url = remote_url.split("@", 1)[1]

        # Remove .git suffix if present
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        print(f"Remote URL: {remote_url}")
        return remote_url

    except subprocess.CalledProcessError:
        print("No remote repository found")
        return None


@contextmanager
def _native_mc_run(
    prefix: str,
    path: str,
    executable: str,
    executable_dir: str,
    cla: str,
    sample_size: int,
) -> Iterator[tuple["subprocess.CompletedProcess[str]", str]]:
    """
    Run the native MC binary in a throwaway working directory.

    Sets up an isolated temp dir, symlinks the application's ``inputs/``. Not
    the binary's own ``data.out`` (see _symlink_application_inputs). Then, runs
    ``<executable> <cla> -M <sample_size>`` there. The temp dir (and the yielded
    file) live only for the ``with`` body.

    Args:
        prefix: Prefix for the temporary directory name.
        path: Application root (its ``inputs/`` is symlinked in).
        executable: Native binary filename.
        executable_dir: Directory containing the native binary.
        cla: Command-line arguments passed to the binary.
        sample_size: Monte Carlo sample count (passed as ``-M``).

    Yields:
        A tuple of (completed process, path to the binary's ``data.out``).
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as work_dir:
        _symlink_application_inputs(os.path.join(path, "inputs"), work_dir)

        argv = [
            os.path.join(executable_dir, executable),
            *shlex.split(cla),
            "-M",
            str(sample_size),
        ]
        result = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
        yield result, os.path.join(work_dir, EquivMC.MC_OUTPUT_FILENAME)


def _run_scalar_native(
    path: str,
    executable: str,
    executable_dir: str,
    cla: str,
    sample_size: int,
    run_id: int,
) -> tuple[int, float | None]:
    """
    Run a native MC execution whose output is a scalar.

    Args:
        path: Application root (its ``inputs/`` is symlinked in).
        executable: Native binary filename.
        executable_dir: Directory containing the native binary.
        cla: Command-line arguments passed to the binary.
        sample_size: Monte Carlo sample count.
        run_id: Identifier for this run, used in warnings and the temp-dir name.

    Returns:
        A tuple of (sample_size, scalar value), where the value is ``None`` if
        the run failed or its output could not be parsed.
    """
    value: float | None = None
    with _native_mc_run(
        f"scalar_run_{run_id}_", path, executable, executable_dir, cla, sample_size
    ) as (result, data_file):
        if result.returncode != 0:
            print(
                f"Warning: Scalar run {run_id} (size={sample_size}) returned non-zero "
                f"exit code: {result.stderr}"
            )

        try:
            with open(data_file, "r") as f:
                for line_no, line in enumerate(f):
                    if line_no == 1:
                        value = float(line.strip())
                        break
        except Exception as e:
            print(
                f"Error reading {EquivMC.MC_OUTPUT_FILENAME} in "
                f"{os.path.dirname(data_file)}: {e}"
            )

    return sample_size, value


def clean_data_for_sheets(rows: list[list[Any]]) -> list[list[Any]]:
    """
    Clean row data for Google Sheets API compatibility.

    Renders distribution objects via ``repr``, blanks out NaN cells, and casts
    numpy scalars to plain floats.

    Args:
        rows: Rows of cell values to clean.

    Returns:
        The cleaned rows.
    """
    cleaned_rows: list[list[Any]] = []
    for row in rows:
        cleaned_row: list[Any] = []
        for cell in row:
            if isinstance(cell, (DistributionalValue, TaggedDistributionalValue)):
                cleaned_row.append(repr(cell))
            elif pd.isna(cell):
                cleaned_row.append("")
            elif isinstance(cell, (np.integer, np.floating)):
                cleaned_row.append(float(cell))
            else:
                cleaned_row.append(cell)
        cleaned_rows.append(cleaned_row)
    return cleaned_rows


def expand_array_expressions(data: list[dict]) -> list[dict]:
    """
    Expand array expressions like 'outputVariables[0:5]' into individual entries.

    Args:
        data: list of dictionaries with 'Expression' key containing array expressions

    Returns:
        Expanded list with individual entries for each array index
    """
    expanded = []

    for item in data:
        expression = item["Expression"]

        # Check if expression contains array slice notation [start:end]
        array_match = re.match(r"(\w+)\[(\d+):(\d+)\]", expression)

        if array_match:
            array_name = array_match.group(1)
            start_idx = int(array_match.group(2))
            end_idx = int(array_match.group(3))

            # Create individual entries for each index
            for i in range(start_idx, end_idx + 1):
                new_item = item.copy()
                new_item["Expression"] = f"{array_name}[{i}]"
                expanded.append(new_item)
        else:
            # Not an array expression, keep as-is
            expanded.append(item)

    return expanded


def _run_mc_simulation(
    work_item: tuple[str, int, str],
    path_to_application: str,
    native_executable_name: str,
    native_executable_dir: str,
) -> list[float]:
    """
    Run a single MC simulation in a separate process.

    Args:
        work_item: A ``(index, sample_size, cla)`` tuple describing the run.
        path_to_application: Application root (its ``inputs/`` is symlinked in).
        native_executable_name: Native binary filename.
        native_executable_dir: Directory containing the native binary.

    Returns:
        The scalar values parsed from the run's ``data.out``.

    Raises:
        RuntimeError: If the simulation exits with a non-zero status.
    """
    index, sub_size, cla = work_item

    values: list[float] = []
    with _native_mc_run(
        f"mc_sim_{index}_",
        path_to_application,
        native_executable_name,
        native_executable_dir,
        cla,
        sub_size,
    ) as (result, data_file):
        if result.returncode != 0:
            raise RuntimeError(
                f"Simulation {index} returned non-zero exit code: {result.stderr}"
            )

        # Read results
        if os.path.exists(data_file):
            with open(data_file, "r") as file:
                for line_no, line in enumerate(file):
                    if line_no > 0:  # Skip header
                        try:
                            values.append(float(line.strip()))
                        except ValueError:
                            continue

    return values


def compute_emcc_prediction(
    asymptotic_dist_quantity: float | None,
    distance: float,
) -> int:
    """
    Predict the equivalent Monte Carlo count from an asymptotic quantity.

    Computes ``(asymptotic_dist_quantity / distance) ** 2``, clamped to a
    minimum of 1. Returns 1 if the inputs are missing or invalid (e.g. a
    ``None`` quantity or a zero distance).

    Args:
        asymptotic_dist_quantity: The asymptotic-distance statistic, or
            ``None`` when unavailable.
        distance: The UxHw-to-ground-truth distance.

    Returns:
        The predicted EMCC (at least 1).
    """
    try:
        emcc_predicted = int((asymptotic_dist_quantity / distance) ** 2)  # type: ignore[operator] # noqa: E501
        # Check results make sense
        if emcc_predicted < 1:
            print("Warning! Predicted EMCC values are zero!")
            emcc_predicted = 1
    except (TypeError, ZeroDivisionError, ValueError) as e:
        print(f"Warning. EMCC prediction failed with: {e}.")
        emcc_predicted = 1

    return emcc_predicted


def write_uxhw_distance_file(
    filename: str, benchmarking_variables: list[BenchmarkingVariable]
) -> None:
    """
    Write UxHw distance data to a CSV file.

    Args:
        filename: Path to the CSV file to write.
        benchmarking_variables: Benchmarking variables to write, each with a
            ``description`` and ``uxhw_distances``.
    """
    header = [
        BenchmarkingVariables.VARIABLE_DESCRIPTION,
        BenchmarkingVariables.UXHW_CONF,
        BenchmarkingVariables.UXHW_DISTANCE,
        BenchmarkingVariables.UXHW_BINNED_DISTANCE,
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        rows = [
            [
                variable.description,
                repr(record.uxhw_conf),
                record.uxhw_distance,
                (
                    ""
                    if record.uxhw_binned_distance is None
                    else record.uxhw_binned_distance
                ),
            ]
            for variable in benchmarking_variables
            for record in variable.uxhw_distances.records
        ]
        writer.writerows(rows)
