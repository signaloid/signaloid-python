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
import shlex
import subprocess
import sys

from signaloid.benchmarking.types import BenchmarkingVariable
from signaloid.benchmarking.config import (
    EquivMC,
    TimingFormat,
    get_repo_root,
    get_resources_dir,
)


def export_timing_env(
    *,
    path_to_uxhw_sdk: str,
    path_to_pin: str | None,
    path_to_application: str,
    application_name: str,
    application_version: str,
    max_jupiter_size: int,
    results_dir: str,
    logs_dir: str,
    tracing_db_path: str,
) -> str:
    """
    Export the environment variables needed by the timing bash scripts.

    Called after ``get_application_info()`` so all paths and config are
    available. ``TRACING_DB_ABS`` is preserved when already set. Otherwise
    ``tracing_db_path`` is used and exported.

    Args:
        path_to_uxhw_sdk: Path to the UxHw SDK.
        path_to_pin: Path to the Intel PIN kit, exported as ``PIN_ROOT``.
            ``None`` leaves any inherited ``PIN_ROOT`` in place. PIN is
            mandatory, so the run errors if neither is set.
        path_to_application: Path to the application source tree.
        application_name: Application name (e.g. ``Finance-...``).
        application_version: Application version string.
        max_jupiter_size: Maximum Jupiter limit.
        results_dir: Directory where results artifacts are written.
        logs_dir: Directory where logs are written.
        tracing_db_path: Default tracing-database path. Overridden when
            ``TRACING_DB_ABS`` is already set.

    Returns:
        The resolved tracing-database path (also written to
        ``TRACING_DB_ABS``). Callers should store it back to
        ``self.tracing_db_path``.
    """
    os.environ["SIGNALOID_PYTHON_DIR"] = os.fspath(get_repo_root())
    os.environ["BENCHMARKING_RESOURCES_DIR"] = str(get_resources_dir())
    os.environ["PATH_TO_UXHW_SDK"] = path_to_uxhw_sdk
    # PIN is mandatory: when no path is given we leave any shell-set
    # PIN_ROOT in place. There is no built-in default. get-timings.sh
    # will error clearly if neither is set.
    if path_to_pin:
        os.environ["PIN_ROOT"] = path_to_pin
    # The timing bash layer shells out to `python3 -m
    # signaloid.benchmarking...`. We pass our own interpreter so those
    # subprocesses use this venv (with the benchmarking dependencies) even when
    # the venv is not on PATH / not activated.
    os.environ["BENCHMARKING_PYTHON"] = sys.executable
    os.environ["APPLICATION_PATH"] = path_to_application
    os.environ["APPLICATION_NAME"] = application_name
    os.environ["APPLICATION_VERSION"] = application_version
    os.environ["PROGRAM"] = "main"
    os.environ["CLA_FOR_MULTIPLE_EXECUTIONS"] = "-M"
    os.environ["MAX_JUPITER_LIMIT"] = str(max_jupiter_size)
    os.environ["APPEND_TO_OUTPUT_FILE"] = "0"
    os.environ["NATIVE_MC_REPETITION"] = "1"
    os.environ["RESULTS_DIR"] = results_dir
    os.environ["LOGS_DIR"] = logs_dir
    resolved_tracing_db_path = os.environ.get("TRACING_DB_ABS", tracing_db_path)
    os.environ["TRACING_DB_ABS"] = resolved_tracing_db_path
    return resolved_tracing_db_path


def run_timing_script(
    *,
    all_outputs_cla: str,
    benchmarking_variables: list[BenchmarkingVariable],
    demo_cli_args: str,
    representation_types: list[str],
    representation_sizes: list[int],
    correlations: list[str],
    logs_dir: str,
    intermediate_timings_path: str,
    timing: bool = False,
    tracing: bool = False,
    native_mc_timing: bool = False,
    variable_index: int = 1,
) -> None:
    """
    Run the timing bash script with the appropriate environment
    variables and mode-specific bash array variables.

    Args:
        all_outputs_cla: The all-outputs invocation CLA string.
        benchmarking_variables: Variables to run timing against.
        demo_cli_args: Demo-level command-line arguments string.
        representation_types: Configured representation types.
        representation_sizes: Configured representation sizes.
        correlations: Configured correlation modes.
        logs_dir: Directory where the bash stderr log is written.
        intermediate_timings_path: Absolute path of the transient
            intermediate timings file the bash script writes.
        timing: When ``True``, run the per-variable UxHw-timing pass.
        tracing: When ``True``, run the per-variable UxHw-tracing
            pass.
        native_mc_timing: When ``True``, run the per-variable native-MC
            timing pass.
        variable_index: Index into ``benchmarking_variables`` for the
            per-variable passes (tracing, timing, native_mc_timing).

    Raises:
        RuntimeError: If the bash timing script exits with a non-zero
            status code. The captured stderr (from
            ``timing_script_stderr.log``) is included in the message.
    """
    # Determine skip flags
    skip_uxhw = 1
    skip_native_mc = 1
    skip_tracing = 1

    command_line_arguments = all_outputs_cla
    variables = benchmarking_variables
    native_mc_sizes: str = "50 100"

    if tracing:
        skip_tracing = 0
        command_line_arguments = benchmarking_variables[variable_index].cla
        variables = [benchmarking_variables[variable_index]]
    if timing:
        skip_uxhw = 0
        command_line_arguments = "-T " + benchmarking_variables[variable_index].cla
        variables = [benchmarking_variables[variable_index]]
    if native_mc_timing:
        skip_native_mc = 0
        command_line_arguments = benchmarking_variables[variable_index].cla
        native_mc_sizes = " ".join(
            map(
                str,
                benchmarking_variables[variable_index].emcc_results.equiv_mc_list,
            )
        )
        variables = [benchmarking_variables[variable_index]]

    # Set mode-specific scalar env vars
    if demo_cli_args:
        command_line_arguments = f"{demo_cli_args} {command_line_arguments}".strip()
    os.environ["CLA"] = command_line_arguments
    os.environ["SKIP_UXHW"] = str(skip_uxhw)
    os.environ["SKIP_UXHW_TRACING"] = str(skip_tracing)
    os.environ["SKIP_NATIVE_MC"] = str(skip_native_mc)

    # Wire the timing-intermediate tokens and file path through to
    # the bash script so its emitted tags match what the Python
    # reader expects (see TimingFormat in config.py).
    os.environ[TimingFormat.META_TAG_ENV_VAR] = TimingFormat.META_TAG
    os.environ[TimingFormat.MEASUREMENT_TAG_ENV_VAR] = TimingFormat.MEASUREMENT_TAG
    os.environ[TimingFormat.SAMPLE_TAG_ENV_VAR] = TimingFormat.SAMPLE_TAG
    os.environ[TimingFormat.INTERMEDIATE_FILE_ENV_VAR] = intermediate_timings_path

    # Build bash arrays (cannot be env vars) and source the script
    representations = " ".join(map(str, representation_types))
    rep_sizes = " ".join(map(str, representation_sizes))
    correlations_str = " ".join(map(str, correlations))

    traces_lines = ""
    for variable in variables:
        native_mc_sizes = " ".join(map(str, variable.emcc_results.equiv_mc_list))
        traces_lines += (
            "    'addDistValueTrace "
            + f'{variable.name}   "{variable.file_name}:{variable.line_number}"'
            + "'\n"
        )
    if len(native_mc_sizes) == 0:
        native_mc_sizes = "50"

    bash_cmd = f"""
TRACES=(
{traces_lines})
REFERENCE_PRECISIONS=({native_mc_sizes})
REPRESENTATION_TYPES=({representations})
REPRESENTATION_SIZES=({rep_sizes})
CORRELATION_TRACKING_TYPES=({correlations_str})
. $SIGNALOID_PYTHON_DIR/src/signaloid/benchmarking/benchmark_timing/get-timings.sh
"""
    stderr_log = os.path.join(logs_dir, "timing_script_stderr.log")
    # Use tee so stderr streams to terminal
    # (preserving interactive prompts) and is
    # also captured to a log file for debugging.
    quoted_stderr_log = shlex.quote(stderr_log)
    wrapped_cmd = f"{{ {bash_cmd.strip()} ; }} " f"2> >(tee {quoted_stderr_log} >&2)"
    result = subprocess.call(["bash", "-c", wrapped_cmd])
    # After the first timing script call, switch to append mode
    os.environ["APPEND_TO_OUTPUT_FILE"] = "1"
    if result != 0:
        stderr_content = ""
        if os.path.isfile(stderr_log):
            with open(stderr_log, "r") as f:
                stderr_content = f.read().strip()
        error_message = "Timing script failed " f"(exit code {result})."
        if stderr_content:
            error_message += "\nstderr:\n" + stderr_content
        error_message += (
            "\nSee also:\n" f"  - {logs_dir}/exec.stderr\n" f"  - {logs_dir}/opt.err"
        )
        raise RuntimeError(error_message)


def compile_native(
    *,
    path_to_application: str,
    num_parallel_workers: int,
) -> tuple[str, str, str, bool]:
    """
    Compile the application for native execution MC.

    Tries the following in order:

    1.  If a Makefile exists at the application root,
        run ``make local-build``.
    2.  Fall back to building a gcc command from
        config.mk.

    Args:
        path_to_application: Path to the application source tree.
        num_parallel_workers: Number of parallel ``make -j`` workers.

    Returns:
        Tuple ``(native_executable_name, native_compilation_command,
        native_executable_dir, has_native_mc)``. The caller should
        store these back to the corresponding ``Benchmark`` attributes.
        When the gcc-fallback path returns early (no ``SOURCES`` in
        ``config.mk``), the tuple is ``("", "", "", False)``.
    """
    makefile_path = os.path.join(path_to_application, "Makefile")

    use_makefile = os.path.isfile(makefile_path)
    if use_makefile:
        print(
            "Makefile detected. Attempting to compile native executable using 'make local-build'."
        )
        native_executable_name = "demo-native-mc"
        native_compilation_command = f"make -j{num_parallel_workers} local-build"
    else:
        print(
            "No Makefile detected. Attempting to compile native executable using gcc and config.mk."
        )
        config_path = f"{path_to_application}" "/src/config.mk"

        # Use make to expand variables (handles both plain lists
        # and Make functions like $(wildcard ...), $(filter-out ...), etc.)
        def _make_expand(var: str) -> str:
            result = subprocess.run(
                [
                    "make",
                    "-f",
                    config_path,
                    "-f",
                    "-",
                    f"print-{var}",
                ],
                capture_output=True,
                text=True,
                cwd=f"{path_to_application}/src",
                input=f"print-{var}:\n\t@echo $({var})\n",
            )
            # Empty stdout means "unset" only from a *successful* make (e.g. no
            # SOURCES, hence no native MC). A non-zero return (e.g. missing
            # config.mk) also yields empty stdout and must not be read as unset.
            if result.returncode != 0:
                raise RuntimeError(f"make failed: {result.stderr}")
            return result.stdout.strip()

        sources_str = _make_expand("SOURCES")
        if not sources_str:
            return ("", "", "", False)
        sources = sources_str.split()
        sources.append("uxhw.c")

        cflags_str = _make_expand("CFLAGS")
        cflags = cflags_str.split() if cflags_str else []

        # config.mk's -I/-L paths are written relative to src/ (like SOURCES),
        # but we compile from the application root, so rebase relative include
        # and library search paths onto src/ to match the src/-prefixed sources
        # below. Absolute paths (e.g. -I/opt/local/include) are left untouched.
        def _rebase_search_path(flag: str) -> str:
            for opt in ("-I", "-L"):
                if flag.startswith(opt):
                    path = flag[len(opt) :]
                    if path and not os.path.isabs(path):
                        return f"{opt}src/{path}"
            return flag

        cflags = [_rebase_search_path(f) for f in cflags]

        native_executable_name = "demo-native-mc"
        sources = [f"src/{s}" for s in sources]
        parts = (
            [
                "gcc",
                "-O3",
                "-o",
                native_executable_name,
                "-Isrc",
                "-I/opt/local/include",
            ]
            + cflags
            + sources
            + [
                "-L/opt/local/lib",
                "-lgsl",
                "-lgslcblas",
                "-lm",
            ]
        )
        native_compilation_command = " ".join(parts)

    # Symlink input files into the application root, skipping any that already
    # exist as non-symlinks (don't overwrite e.g. README.md). Also skip the
    # binary's own data.out: symlinking a stale one would let a run there write
    # back through it to the shared inputs/data.out (see
    # benchmarking_utils._symlink_application_inputs).
    subprocess.call(
        # ``inputs/*`` is relative to cwd (this call's ``cwd=path_to_application``).
        # Prefixing path_to_application would double it.
        f"for f in inputs/*; do "
        f'b="$(basename "$f")"; '
        f'if [ "$b" = "{EquivMC.MC_OUTPUT_FILENAME}" ]; then continue; fi; '
        f'if [ -e "$b" ] && [ ! -L "$b" ]; then '
        f'echo "WARNING: skipping symlink for $b (file already exists in application root)"; '
        f"continue; fi; "
        f'ln -sf "$f" .; done',
        shell=True,
        cwd=path_to_application,
        stdout=None,
        stderr=subprocess.DEVNULL,
    )

    # Always compile from the application root
    native_executable_dir = path_to_application

    # Clear any inherited jobserver flags to avoid
    # "warning: jobserver unavailable: using -j1" in sub-makes.
    clean_env = {**os.environ}
    clean_env.pop("MAKEFLAGS", None)

    print(
        f"Compiling native executable {native_executable_name} at {native_executable_dir}"
    )
    if use_makefile:
        clean_result = subprocess.run(
            f"make -j{num_parallel_workers} clean",
            shell=True,
            cwd=native_executable_dir,
            capture_output=True,
            text=True,
            env=clean_env,
        )
        if clean_result.returncode != 0:
            print(
                f"Warning: 'make clean' failed (exit code {clean_result.returncode}). "
                f"Continuing with build."
            )
    result = subprocess.run(
        native_compilation_command,
        shell=True,
        cwd=native_executable_dir,
        capture_output=True,
        text=True,
        env=clean_env,
    )
    if result.returncode != 0:
        print("Native compilation failed " f"(exit code {result.returncode}).")
        print(result.stdout)
        print(result.stderr)
        has_native_mc = False
    else:
        # Verify the executable was produced and resolve its location
        root_path = os.path.join(path_to_application, native_executable_name)
        src_path = os.path.join(path_to_application, "src", native_executable_name)
        if os.path.isfile(root_path):
            native_executable_dir = path_to_application
        elif os.path.isfile(src_path):
            native_executable_dir = os.path.join(path_to_application, "src")
        else:
            print(
                f"Warning: Compilation succeeded but '{native_executable_name}' "
                f"not found in {path_to_application} or {path_to_application}/src."
            )
            has_native_mc = False
            return (
                native_executable_name,
                native_compilation_command,
                native_executable_dir,
                has_native_mc,
            )
        has_native_mc = True
        print(
            f"Native compilation succeeded. "
            f"Executable: {native_executable_dir}/{native_executable_name}"
        )
    return (
        native_executable_name,
        native_compilation_command,
        native_executable_dir,
        has_native_mc,
    )
