#!/usr/bin/env bash

# * ==========================================================================
# * This script measures the runtime of a given application repository on
# *    different core configurations
# * ==========================================================================

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# ===========================================================================
# Utility functions
# ===========================================================================

die() {
    echo "$*" >&2
    exit 1
}

warn() {
    echo "$*" >&2
}

log_section() {
    echo "==============================================================="
    echo "=== $* ==="
    echo "==============================================================="
}

# Require a variable to be set and non-empty. Exits with an error message if not.
# Usage: require_var VAR_NAME
require_var() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        die "Please set \$$var_name, via env or $SCRIPT_DIR/get_timings_local.sh"
    fi
}

# Warn if a variable is not set, but don't exit.
# Usage: warn_var VAR_NAME
warn_var() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        warn "No \$$var_name set. Set via env or $SCRIPT_DIR/get_timings_local.sh"
    fi
}

# Set an array variable to a default value if it is not already set.
# Usage: default_array ARRAY_NAME default_val1 default_val2 ...
default_array() {
    local var_name="$1"
    shift
    if eval "[ -z \${${var_name}+x} ]"; then
        eval "$var_name=(\"\$@\")"
    fi
    local count
    count=$(eval "echo \${#${var_name}[@]}")
    echo "Testing $count $var_name"
}

# Compute the average of a Python-style list string, e.g. "[1,2,3,]".
# Usage: compute_average "[1.0,2.0,3.0,]"
#
# Surviving callers (B1 / B5 boundary): only the two extrapolation-source
# sites in run_native_mc_benchmarks (last_measured_time and
# last_measured_instructions). Per-iteration values consumed by
# emit_measurement now flow through the SAMPLE tag and are averaged in
# Python (see signaloid.benchmarking.automation.benchmarking_utils
# `parse_timing_intermediate_stream`).
compute_average() {
    "${BENCHMARKING_PYTHON:-python3}" -c "values = $1; print(sum(values) / len(values))"
}

# Scale repetitions so the total measurement time approaches $target_total,
# clamped to [$min_reps, $max_reps]. Falls back to $max_reps if
# $single_time is not a positive number.
# Usage: reps=$(compute_time_scaled_repetitions <single_time> <target_total> <min_reps> <max_reps>)
compute_time_scaled_repetitions() {
    awk -v s="$1" -v t="$2" -v min_r="$3" -v max_r="$4" \
        'BEGIN {
            if (s+0 <= 0) { print max_r; exit }
            r = t / s;
            r = (r == int(r)) ? r : int(r) + 1;
            if (r < min_r) r = min_r;
            if (r > max_r) r = max_r;
            print r;
        }'
}

unset MAKEFLAGS
MAKE_JOBS="-j$(nproc)"

# Run a UxHw make build command, redirecting compiler warnings to UXHW_BUILD_LOG.
# Usage: uxhw_make [make args...]
uxhw_make() {
    make -s $MAKE_JOBS -f "$UXHW_MAKEFILE" "$@" >/dev/null 2>>"$UXHW_BUILD_LOG"
    # Move compiler-generated opt.err to logs directory if present
    if [[ -f opt.err ]]; then
        mv opt.err "$LOGS_DIR/opt.err"
    fi
}

# cd to the application input directory if it exists.
cd_to_input_dir() {
    if [[ -e "$APPLICATION_PATH/$INPUT_DIR" ]]; then
        cd "$APPLICATION_PATH/$INPUT_DIR"
    fi
}

# Check UxHw compilation output for errors.
check_uxhw_compilation() {
    if grep -q 'Could not find' "$LOGS_DIR/opt.err" 2>/dev/null; then
        die "Compilation failed, see $LOGS_DIR/opt.err"
    fi
    echo "ok"
}

# Insert trace lines into an m config file.
# Usage: add_traces_to_config <config_file>
add_traces_to_config() {
    local config_file="$1"
    # Announce the traced expressions once per distinct trace set (compact:
    # "expr @ file:line"), not on every per-config .m rebuild.
    if [[ "${TRACES[*]}" != "${_ANNOUNCED_TRACES:-}" ]]; then
        local summary="" t expr loc
        for t in "${TRACES[@]}"; do
            loc=$(sed -E 's/.*"([^"]*)".*/\1/' <<<"$t")
            expr=$(sed -E 's/^addDistValueTrace[[:space:]]+//; s/[[:space:]]*"[^"]*".*//' <<<"$t")
            summary+="${summary:+, }$expr @ $loc"
        done
        echo "Tracing: $summary"
        _ANNOUNCED_TRACES="${TRACES[*]}"
    fi
    for TRACE in "${TRACES[@]}"; do
        sed -i "/--\s*INSERT TRACES AFTER THIS/a $TRACE" "$config_file"
    done
}

# Write a UxHw emulator m-config file inline (tracing/timing), then append $TRACES.
#
# Replaces the old run-montecarlo.m template + cp/sed chain. The template's
# srecl / loadDwarfBin / loadMapFile directives were UxHw-SDK ("sf")
# RISC-V-image loaders that the LLVM `opt` transformation (which consumes this
# config via --m-config-file) ignores — the tracing/timing passes always
# carried the literal "program-name" placeholders for them and still produced
# correct output — so they are dropped. setMcExecModeIterations stays commented
# out: the tracing/timing passes never enable MC mode.
# Config-file whitespace is not significant, so single-space formatting is used.
#
# Usage: write_emulator_config <out_file> <db_file> <db_table>
write_emulator_config() {
    local out_file="$1" db_file="$2" db_table="$3"
    cat > "$out_file" <<EOF
noDetach 1

--
-- UR Type and Order do not affect this pass; no Monte Carlo seed is set.
--
newNode riscv UncertaintyRepresentationRQHR 64
cacheOff
ff
sizemem 50000000

ChangeGuestWorkingDir "../inputs"
# setMcExecModeIterations
setDbFilename "$db_file" "$db_table"
-- INSERT TRACES AFTER THIS
run
on
dumpInstructionDist
q
EOF
    add_traces_to_config "$out_file"
}

# Check if a Jupiter configuration should be skipped.
# Returns 0 (true) if it should be skipped, 1 (false) otherwise.
# Usage: should_skip_representation REPRESENTATION_TYPE CORRELATION_TRACKING_TYPE REPRESENTATION_SIZE
should_skip_representation() {
    local repr_type="$1"
    local corr_type="$2"
    local repr_size="$3"

    if [[ "$repr_type" == "Jupiter" ]]; then
        if [[ "$corr_type" == "Autocorrelation" ]]; then
            return 0
        fi
        if [[ "$repr_size" -gt "$MAX_JUPITER_LIMIT" ]]; then
            return 0
        fi
    fi
    return 1
}

# Run hyperfine and return the mean time.
# Usage: mean_time=$(run_hyperfine_mean "command to benchmark")
run_hyperfine_mean() {
    local cmd="$1"
    shift
    rm -f hyperfine.json
    hyperfine "$@" --warmup 2 --style none --export-json hyperfine.json "$cmd" >/dev/null 2>>"${LOGS_DIR:-/tmp}/hyperfine.log"
    jq '.results[0].mean' hyperfine.json
}

# Extract the seconds value from a `CPU time used:` line in one or
# more benchmark stdout files, via the parse_cpu_time Python helper.
# Prefixes PYTHONPATH so the module resolves when the script is
# sourced standalone (without the poetry venv). With one positional
# arg the first match in that file is printed; with `--sum` followed
# by paths, every match across every file is summed.
# Usage:
#   t=$(parse_cpu_time "$LOGS_DIR/$EXEC_STDOUT")
#   total=$(parse_cpu_time --sum "${stdout_files[@]}")
parse_cpu_time() {
    PYTHONPATH="$SIGNALOID_PYTHON_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        "${BENCHMARKING_PYTHON:-python3}" -m signaloid.benchmarking.automation.parse_cpu_time "$@"
}

# Read a single-value DB metric via the read_db_metrics Python helper.
# Same PYTHONPATH-prefix shape as parse_cpu_time.
# Usage: t=$(read_db_metric host-wallclock "$UXHW_DB_BASE.db")
read_db_metric() {
    local metric=$1 db=$2
    PYTHONPATH="$SIGNALOID_PYTHON_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        "${BENCHMARKING_PYTHON:-python3}" -m signaloid.benchmarking.automation.read_db_metrics \
        "$db" --metric "$metric"
}

# ===========================================================================
# Argument parsing
# ===========================================================================

usage() {
    cat <<'HELP'
Usage: get-timings.sh [-h] [-a APPLICATION_PATH] [-c APPLICATION_VERSION] [-n APPLICATION_NAME]

Options:
  -h    Show this help message
  -a    Path to the application/demo repository
  -c    Application version string
  -n    Application name
HELP
}

parse_args() {
    while getopts ":ha:c:n:" option; do
        case $option in
        h)
            usage
            exit 0
            ;;
        a) APPLICATION_PATH=$OPTARG ;;
        c) APPLICATION_VERSION=$OPTARG ;;
        n) APPLICATION_NAME=$OPTARG ;;
        \?)
            echo "Error: Invalid option -$OPTARG" >&2
            usage >&2
            exit 1
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument" >&2
            usage >&2
            exit 1
            ;;
        esac
    done
}

# ===========================================================================
# Validation and setup
# ===========================================================================

validate_required_vars() {
    require_var SIGNALOID_PYTHON_DIR
    require_var PATH_TO_UXHW_SDK
    require_var APPLICATION_PATH
    require_var APPLICATION_NAME
    require_var APPLICATION_VERSION
    warn_var CLA
    require_var CLA_FOR_MULTIPLE_EXECUTIONS
    require_var PROGRAM
    require_var REFERENCE_PRECISIONS
    require_var TRACES
}

copy_build_resources() {
    cp "$RESOURCES_DIR/C0/$INIT_S" ./
    cp "$RESOURCES_DIR/common/$STARTUP_CPP" ./
    cp "$RESOURCES_DIR/C0Pro/$UXHW_MAKEFILE" ./
    cp "$RESOURCES_DIR/C0/$MAKEFILE" ./
}

# Fallback tag strings for the case where this script is run directly
# (e.g. via get-timing-template.sh) rather than from Python. The Python
# driver overrides these via environment variables so the tokens match
# the TimingFormat class in src/signaloid/benchmarking/config.py.
: "${TIMING_META_TAG:=META}"
: "${TIMING_MEASUREMENT_TAG:=MEASUREMENT}"
: "${TIMING_SAMPLE_TAG:=SAMPLE}"

# Emit a single `META <key> <value>` line to the intermediate timing file.
# The tag strings and target path are supplied by Python via env vars (see
# the TimingFormat class in src/signaloid/benchmarking/config.py).
# Usage: emit_meta key value...
emit_meta() {
    local key="$1"
    shift
    echo "$TIMING_META_TAG $key $*" >>"$TIMING_INTERMEDIATE_FILE"
}

# Emit a single `MEASUREMENT config time db_time e2e_time db_inst pin_inst`
# line to the intermediate timing file. Missing numeric fields must be
# passed as the literal `?` character (TimingFormat.MISSING_VALUE).
# Usage: emit_measurement config time db_time e2e_time db_inst pin_inst
emit_measurement() {
    echo "$TIMING_MEASUREMENT_TAG $*" >>"$TIMING_INTERMEDIATE_FILE"
}

# Emit a single `SAMPLE config field iteration <value-or-path>` line to
# the intermediate timing file. The 4th token is a literal float value
# (value-typed variant, e.g. elapsedTime) or a file path (path-typed
# variant, e.g. pinDynInstCount). The Python reader collects these and,
# at MEASUREMENT time, resolves any `?` sentinel in the matching field
# by reading/averaging the accumulated samples.
# Usage: emit_sample config field iteration value-or-path
emit_sample() {
    echo "$TIMING_SAMPLE_TAG $*" >>"$TIMING_INTERMEDIATE_FILE"
}

write_timings_header() {
    if [[ $SKIP_UXHW -eq 0 ]] || [[ $SKIP_NATIVE_MC -eq 0 ]] || [[ $SKIP_UXHW_TRACING -eq 0 ]]; then
        emit_meta timestamp "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        emit_meta applicationName "$APPLICATION_NAME"
        emit_meta applicationVersion "$APPLICATION_VERSION"
        emit_meta uxhwSdkVersion "$(tr -d '"' < "$PATH_TO_UXHW_SDK/.sdk_installed_release")"
        emit_meta commandLineArguments "$CLA"
        emit_meta commandLineArgumentsHash "$CLA_HASH"
        # $REPETITION is the UxHw-loop target at session start. The
        # actual rep count used per measurement can be rescaled later
        # (see the per-testcase `uxhw_repetition` warmup scaling in
        # run_uxhw_benchmarks) and native-MC rows use
        # $NATIVE_MC_REPETITION, which is computed dynamically
        # per-precision. This field therefore documents the target,
        # not the per-run ground truth.
        emit_meta uxhwTargetRepetitions "$REPETITION"
    fi
}

# ===========================================================================
# Benchmark: UxHw cores
# ===========================================================================

# Build the canonical UxHw config string "<type>-<size>[-<correlation>]".
# Disabled correlation tracking is the implicit default and is omitted, so it
# yields e.g. "Athens-16"; Autocorrelation yields "Athens-16-Autocorrelation".
# Must stay in lockstep with TaggedDistributionalValue.__repr__ (Python side),
# since this string is the join key between timing and EMCC data.
uxhw_config_string() {
    local repr_type="$1" repr_size="$2" corr_type="$3"
    if [[ "$corr_type" == "Autocorrelation" ]]; then
        printf '%s-%s-%s' "$repr_type" "$repr_size" "$corr_type"
    else
        printf '%s-%s' "$repr_type" "$repr_size"
    fi
}

build_uxhw_testcase_binary() {
    local repr_type="$1"
    local repr_size="$2"
    local corr_type="$3"

    uxhw_make clean

    local args=(
        REPRESENTATION_TYPE="$repr_type"
        REPRESENTATION_SIZE="$repr_size"
        CORRELATION_TRACKING="$corr_type"
        M_CONFIG_FILE="$M_CONFIG_FILE"
        TARGET_ARCH="$TARGET_ARCH"
        ENABLE_UNCERTAIN_TYPE_MODIFIER="$ENABLE_UNCERTAIN_TYPE_MODIFIER"
    )
    uxhw_make "${args[@]}"

    mv "$PROGRAM" "$PROGRAM-$(uxhw_config_string "$repr_type" "$repr_size" "$corr_type")"

    uxhw_make clean
}

run_uxhw_benchmarks() {
    log_section "Get timing from UxHw cores."

    UXHW_TESTCASES=()

    for CORRELATION_TRACKING_TYPE in "${CORRELATION_TRACKING_TYPES[@]}"; do
        for REPRESENTATION_TYPE in "${REPRESENTATION_TYPES[@]}"; do
            for REPRESENTATION_SIZE in "${REPRESENTATION_SIZES[@]}"; do
                if should_skip_representation "$REPRESENTATION_TYPE" "$CORRELATION_TRACKING_TYPE" "$REPRESENTATION_SIZE"; then
                    continue
                fi

                build_uxhw_testcase_binary "$REPRESENTATION_TYPE" "$REPRESENTATION_SIZE" "$CORRELATION_TRACKING_TYPE"

                UXHW_TESTCASES+=("$(uxhw_config_string "$REPRESENTATION_TYPE" "$REPRESENTATION_SIZE" "$CORRELATION_TRACKING_TYPE")")
            done
        done
    done

    for UXHW_TESTCASE in "${UXHW_TESTCASES[@]}"; do
        echo "---> Timing $UXHW_TESTCASE"

        # Run with UxHw cores to generate timing measurements and dynamic
        # instruction count for x86_64
        cd_to_input_dir

        # Warmup run to scale repetitions for this specific UxHw testcase by
        # measured single-run time. The resulting count is testcase-local and
        # may differ across UxHw configurations within the same session.
        rm -f "$UXHW_DB_BASE".db
        "$APPLICATION_PATH/src/$PROGRAM-$UXHW_TESTCASE" $CLA 1>"$LOGS_DIR/$EXEC_STDOUT" 2>"$LOGS_DIR/$EXEC_STDERR"
        local warmup_time
        local uxhw_repetition
        warmup_time=$(grep -a 'CPU time used:' "$LOGS_DIR/$EXEC_STDOUT" \
            | sed -E 's/.*CPU time used: *([0-9]+\.[0-9]+).*/\1/')
        uxhw_repetition=$(compute_time_scaled_repetitions "$warmup_time" "$TIMING_TARGET_TOTAL_TIME" "$TIMING_MIN_REPETITIONS" "$TIMING_MAX_REPETITIONS")
        echo "Scaled repetitions=$uxhw_repetition (warmup ${warmup_time}s, target ${TIMING_TARGET_TOTAL_TIME}s)"

        for ((i = 0; i < uxhw_repetition; i++)); do
            [ -t 1 ] && echo -ne "\rRepetitions ($((i + 1))/$uxhw_repetition)"
            rm -f "$UXHW_DB_BASE".db

            "$APPLICATION_PATH/src/$PROGRAM-$UXHW_TESTCASE" $CLA 1>"$LOGS_DIR/$EXEC_STDOUT" 2>"$LOGS_DIR/$EXEC_STDERR"

            local timing_value
            timing_value=$(parse_cpu_time "$LOGS_DIR/$EXEC_STDOUT")
            emit_sample "$UXHW_TESTCASE" "elapsedTime" "$i" "$timing_value"

            local db_timing_value
            db_timing_value=$(read_db_metric host-wallclock "$UXHW_DB_BASE.db")
            emit_sample "$UXHW_TESTCASE" "databaseTime" "$i" "$db_timing_value"

            $INSTRUCTION_COUNT_COMMAND "$APPLICATION_PATH/src/$PROGRAM-$UXHW_TESTCASE" $CLA &>/dev/null
            if [[ ! -f inscount.out ]]; then
                die "inscount.out not found after PIN execution"
            fi
            local uxhw_sample_path="$LOGS_DIR/inscount-$UXHW_TESTCASE-$i-$$.out"
            mv inscount.out "$uxhw_sample_path"
            emit_sample "$UXHW_TESTCASE" "pinDynInstCount" "$i" "$uxhw_sample_path"
        done
        echo

        cd "$APPLICATION_PATH/src"

        cd_to_input_dir
        PROCESS_E2E_TIME=$(run_hyperfine_mean "$APPLICATION_PATH/src/$PROGRAM-$UXHW_TESTCASE $CLA" --shell=none)

        # The emulated RISC-V dynamic instruction count (dbDynInstCount, the 5th
        # field) was produced by the now-removed UxHw `sf` emulator pass; emit
        # the literal 0 it carried in the SDK-absent case rather than a `?`
        # sentinel that has no samples to resolve.
        emit_measurement "$UXHW_TESTCASE" "?" "?" "$PROCESS_E2E_TIME" 0 "?"

        rm -f "$UXHW_DB_BASE.db"
        rm -f sunflower-*.out
    done

    rm -f ./*.m
    cd "$APPLICATION_PATH/src"
}

# ===========================================================================
# Benchmark: UxHw tracing (accuracy data)
# ===========================================================================

run_uxhw_tracing() {
    cd "$APPLICATION_PATH/src"

    log_section "Get accuracy data from UxHw Cores."

    write_emulator_config "$M_CONFIG_FILE_TRACING" "$TRACING_DB_ABS" "TracingTable"

    local tracing_binaries=()
    local tracing_dbs=()
    local tracing_configs=()

    # -O2 verification: the tracing build uses -O0 (the OPTFLAGS default in
    # Makefile.pro) so the addDistValueTrace file:line directives resolve
    # against unoptimised debug info, but real deployments compile at -O2.
    # Optimisation must not change the values the uncertainty machinery
    # computes, so we build a parallel -O2 binary per config and later check
    # (warn-only) that its Ux strings byte-match the -O0 ones. TRACING_VERIFY_OPTFLAGS
    # overrides the level compared against; -gdwarf-4 is kept so tracing still resolves.
    local verify_optflags="${TRACING_VERIFY_OPTFLAGS:-"-O2 -gdwarf-4"}"
    local verify_binaries=()
    local verify_dbs=()
    local verify_baseline_dbs=()
    local verify_configs=()
    # Configs whose -O2 build or run failed: skipped for comparison and
    # reported (as failing) in the verification summary below.
    local verify_failed_configs=()

    # Phase 1: Serial compilation — build each config and rename the binary
    for CORRELATION_TRACKING_TYPE in "${CORRELATION_TRACKING_TYPES[@]}"; do
        for REPRESENTATION_TYPE in "${REPRESENTATION_TYPES[@]}"; do
            for REPRESENTATION_SIZE in "${REPRESENTATION_SIZES[@]}"; do
                if should_skip_representation "$REPRESENTATION_TYPE" "$CORRELATION_TRACKING_TYPE" "$REPRESENTATION_SIZE"; then
                    continue
                fi

                local suffix
                suffix="$(uxhw_config_string "$REPRESENTATION_TYPE" "$REPRESENTATION_SIZE" "$CORRELATION_TRACKING_TYPE")"
                local binary_name="$PROGRAM-tracing-$suffix"
                local per_config_db="${TRACING_DB_ABS%.db}-$suffix.db"
                local per_config_m="run-tracing-$suffix.m"

                # Create per-config .m file pointing to its own DB
                write_emulator_config "$per_config_m" "$per_config_db" "TracingTable"

                uxhw_make clean

                local args=(
                    REPRESENTATION_TYPE="$REPRESENTATION_TYPE"
                    REPRESENTATION_SIZE="$REPRESENTATION_SIZE"
                    CORRELATION_TRACKING="$CORRELATION_TRACKING_TYPE"
                    M_CONFIG_FILE="$per_config_m"
                    TARGET_ARCH="$TARGET_ARCH"
                    ENABLE_TRACING=ON
                    ENABLE_UNCERTAIN_TYPE_MODIFIER="$ENABLE_UNCERTAIN_TYPE_MODIFIER"
                    STATS_DB_FILENAME="$per_config_db"
                    STATS_DB_TABLENAME="Emulator_Execution_Info"
                )
                echo "Compiling $binary_name"
                uxhw_make "${args[@]}"

                mv "$PROGRAM" "$binary_name"

                uxhw_make clean

                tracing_binaries+=("$binary_name")
                tracing_dbs+=("$per_config_db")
                tracing_configs+=("$suffix")

                # Build the -O2 verification binary for this same config, into
                # its own DB, so its Ux strings can be compared against the
                # -O0 baseline above.
                local verify_binary_name="$binary_name-O2"
                local verify_db="${TRACING_DB_ABS%.db}-$suffix-O2.db"
                local verify_m="run-tracing-$suffix-O2.m"

                write_emulator_config "$verify_m" "$verify_db" "TracingTable"

                local verify_args=(
                    REPRESENTATION_TYPE="$REPRESENTATION_TYPE"
                    REPRESENTATION_SIZE="$REPRESENTATION_SIZE"
                    CORRELATION_TRACKING="$CORRELATION_TRACKING_TYPE"
                    M_CONFIG_FILE="$verify_m"
                    TARGET_ARCH="$TARGET_ARCH"
                    ENABLE_TRACING=ON
                    ENABLE_UNCERTAIN_TYPE_MODIFIER="$ENABLE_UNCERTAIN_TYPE_MODIFIER"
                    STATS_DB_FILENAME="$verify_db"
                    STATS_DB_TABLENAME="Emulator_Execution_Info"
                    OPTFLAGS="$verify_optflags"
                )
                echo "Compiling $verify_binary_name (verification, OPTFLAGS='$verify_optflags')"
                if uxhw_make "${verify_args[@]}" && [[ -f "$PROGRAM" ]]; then
                    mv "$PROGRAM" "$verify_binary_name"
                    verify_binaries+=("$verify_binary_name")
                    verify_dbs+=("$verify_db")
                    verify_baseline_dbs+=("$per_config_db")
                    verify_configs+=("$suffix")
                else
                    warn "WARNING: verification build failed (OPTFLAGS='$verify_optflags') for config '$suffix'; skipping its Ux-string check."
                    verify_failed_configs+=("$suffix (build failed)")
                    rm -f "$verify_m" "$verify_db" "$PROGRAM"
                fi

                uxhw_make clean
            done
        done
    done

    # Phase 2: Parallel execution
    local pids=()
    for i in "${!tracing_binaries[@]}"; do
        local binary="${tracing_binaries[$i]}"
        local suffix="${tracing_configs[$i]}"
        (
            if [[ -e "$APPLICATION_PATH/$INPUT_DIR" ]]; then
                cd "$APPLICATION_PATH/$INPUT_DIR"
            fi
            echo "Tracing $binary execution"
            "$APPLICATION_PATH/src/$binary" $CLA \
                1>"$APPLICATION_PATH/src/$binary.stdout" \
                2>"$APPLICATION_PATH/src/$binary.stderr"
        ) &
        pids+=($!)
    done

    # Verification (-O2) executions run in their own batch so a failure here is
    # a warning, not fatal, and does not trip the die() below.
    local verify_pids=()
    for i in "${!verify_binaries[@]}"; do
        local binary="${verify_binaries[$i]}"
        (
            if [[ -e "$APPLICATION_PATH/$INPUT_DIR" ]]; then
                cd "$APPLICATION_PATH/$INPUT_DIR"
            fi
            echo "Tracing $binary execution (verification)"
            "$APPLICATION_PATH/src/$binary" $CLA \
                1>"$APPLICATION_PATH/src/$binary.stdout" \
                2>"$APPLICATION_PATH/src/$binary.stderr"
        ) &
        verify_pids+=($!)
    done

    # Wait for all and check for failures
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "Error: tracing process $pid failed"
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        die "One or more tracing executions failed"
    fi

    # A failed -O2 execution is warn-only; its config is dropped from the
    # verification set so the comparison below does not read a partial DB.
    for i in "${!verify_pids[@]}"; do
        if ! wait "${verify_pids[$i]}"; then
            warn "WARNING: -O2 verification run failed for config '${verify_configs[$i]}'; skipping its Ux-string check."
            verify_failed_configs+=("${verify_configs[$i]} (run failed)")
            verify_dbs[$i]=""
        fi
    done

    cd "$APPLICATION_PATH/src"

    # Phase 2.5: Verify the -O2 Ux strings byte-match the -O0 baseline. This is
    # warn-only (guarded with `|| true`) and must run before the merge below,
    # which consumes (deletes) the per-config baseline DBs.
    for i in "${!verify_dbs[@]}"; do
        [[ -z "${verify_dbs[$i]}" ]] && continue
        PYTHONPATH="$SIGNALOID_PYTHON_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
            "${BENCHMARKING_PYTHON:-python3}" -m signaloid.benchmarking.automation.compare_tracing_ux_strings \
            "${verify_baseline_dbs[$i]}" "${verify_dbs[$i]}" \
            --baseline-label O0 --candidate-label O2 --config "${verify_configs[$i]}" || true
    done

    # Report configs whose -O2 verification could not run (build or run
    # failure). These are skipped for comparison but surfaced here as failing
    # so an -O2 build/run regression is not silently swallowed.
    if [[ ${#verify_failed_configs[@]} -gt 0 ]]; then
        warn "WARNING: -O2 ux-string verification FAILED (build/run) for ${#verify_failed_configs[@]} config(s): ${verify_failed_configs[*]}"
    fi

    # Phase 3: Merge per-config DBs into final tracing DB
    PYTHONPATH="$SIGNALOID_PYTHON_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        "${BENCHMARKING_PYTHON:-python3}" -m signaloid.benchmarking.automation.merge_tracing_dbs \
        "$TRACING_DB_ABS" "${tracing_dbs[@]}"

    # Cleanup temporary binaries and config files
    for i in "${!tracing_binaries[@]}"; do
        rm -f "${tracing_binaries[$i]}" "${tracing_binaries[$i]}.stdout" "${tracing_binaries[$i]}.stderr"
        rm -f "run-tracing-${tracing_configs[$i]}.m"
    done

    # Cleanup -O2 verification artifacts. Their DBs are not consumed by the
    # merge above (only the -O0 DBs are), so remove them explicitly here.
    for i in "${!verify_binaries[@]}"; do
        rm -f "${verify_binaries[$i]}" "${verify_binaries[$i]}.stdout" "${verify_binaries[$i]}.stderr"
        rm -f "${TRACING_DB_ABS%.db}-${verify_configs[$i]}-O2.db"
        rm -f "run-tracing-${verify_configs[$i]}-O2.m"
    done
}

# ===========================================================================
# Benchmark: Native MC
# ===========================================================================

run_native_mc_benchmarks() {
    cd "$APPLICATION_PATH/src"

    # Collect all C and C++ sources recursively, excluding build artifacts.
    # Compile C and C++ files separately to avoid C99/C++ incompatibilities,
    # then link all object files together.
    local c_sources=()
    mapfile -t c_sources < <(find . -name '*.c' -not -path '*/build/*')
    local cxx_sources=()
    local has_cxx=false
    local cxx_files
    mapfile -t cxx_files < <(find . \( -name '*.cc' -o -name '*.cpp' \) -not -path '*/build/*')
    for f in "${cxx_files[@]}"; do
        if [[ -f "$f" ]]; then
            cxx_sources+=("$f")
            has_cxx=true
        fi
    done

    local common_flags=(
        "${NATIVE_CFLAGS[@]}"
        -Wall -Wextra -Wpedantic
        -g -O3
        -I. -I/opt/local/include
    )
    local link_flags=(
        -L/opt/local/lib
        -lgsl -lgslcblas
        -lm
        -frecord-gcc-switches
    )

    local compiled=false
    if $has_cxx; then
        # Compile C and C++ to object files separately, then link with c++
        local obj_files=()
        local compile_ok=true
        for f in "${c_sources[@]}"; do
            if ! cc "${common_flags[@]}" -c "$f" -o "${f%.c}.o" 2>>"$NATIVE_MC_BUILD_LOG"; then
                compile_ok=false; break
            fi
            obj_files+=("${f%.c}.o")
        done
        if $compile_ok; then
            for f in "${cxx_sources[@]}"; do
                if ! c++ "${common_flags[@]}" -c "$f" -o "${f%.*}.o" 2>>"$NATIVE_MC_BUILD_LOG"; then
                    compile_ok=false; break
                fi
                obj_files+=("${f%.*}.o")
            done
        fi
        if $compile_ok; then
            if c++ "${obj_files[@]}" "${link_flags[@]}" -o "$PROGRAM-native" 2>>"$NATIVE_MC_BUILD_LOG"; then
                compiled=true
            fi
        fi
        # Clean up object files
        rm -f "${obj_files[@]}"
    else
        # Pure C project — compile and link in one step
        if cc "${c_sources[@]}" "${common_flags[@]}" "${link_flags[@]}" -o "$PROGRAM-native" 2>>"$NATIVE_MC_BUILD_LOG"; then
            compiled=true
        fi
    fi

    if $compiled; then
        echo "Compilation succeeded"
    else
        echo "Compilation failed, checking for demo-native-mc executable... (see $NATIVE_MC_BUILD_LOG)"
        # Check both src/ and application root (where make local-build places it)
        if [ -f "./demo-native-mc" ]; then
            echo "Found demo-native-mc in src/, using it as $PROGRAM-native"
            cp ./demo-native-mc "./$PROGRAM-native"
        elif [ -f "$APPLICATION_PATH/demo-native-mc" ]; then
            echo "Found demo-native-mc in application root, using it as $PROGRAM-native"
            cp "$APPLICATION_PATH/demo-native-mc" "./$PROGRAM-native"
        else
            die "demo-native-mc not found either"
        fi
    fi

    cd_to_input_dir

    # Run the native program for Monte Carlo Mode. The extrapolation threshold
    # is the largest reference precision that is directly measured and
    # precisions above it are linearly extrapolated. It defaults to 1000000 but
    # can be overridden via the EXTRAPOLATION_THRESHOLD environment variable to
    # trade accuracy for a shorter run.
    local extrapolation_threshold=${EXTRAPOLATION_THRESHOLD:-1000000}
    if ! [[ "$extrapolation_threshold" =~ ^[0-9]+$ ]]; then
        die "EXTRAPOLATION_THRESHOLD must be a non-negative integer, got '$extrapolation_threshold'"
    fi
    echo "---> Using extrapolation threshold = $extrapolation_threshold"
    local last_measured_precision=""
    local last_measured_time=""
    local last_measured_e2e_time=""
    local last_measured_instructions=""

    for REFERENCE_PRECISION in "${REFERENCE_PRECISIONS[@]}"; do
        # Extrapolate for large precisions using the last measured values
        if [ "$REFERENCE_PRECISION" -gt "$extrapolation_threshold" ] && [ -n "$last_measured_precision" ]; then
            echo "---> Extrapolating for precision = $REFERENCE_PRECISION from $last_measured_precision"

            local scaling_factor
            scaling_factor=$(awk -v ref="$REFERENCE_PRECISION" -v prev="$last_measured_precision" \
                'BEGIN { printf "%.10f", ref / prev }')
            local result
            result=$(awk -v time_val="$last_measured_time" -v scale="$scaling_factor" \
                'BEGIN { printf "%.10f", time_val * scale }')
            local process_e2e_time
            process_e2e_time=$(awk -v e2e_time="$last_measured_e2e_time" -v scale="$scaling_factor" \
                'BEGIN { printf "%.10f", e2e_time * scale }')
            local pin_dynamic_instructions
            pin_dynamic_instructions=$(awk -v instructions="$last_measured_instructions" -v scale="$scaling_factor" \
                'BEGIN { printf "%.10f", instructions * scale }')

            emit_measurement "Native-MC-$REFERENCE_PRECISION" "$result" "?" "$process_e2e_time" "?" "$pin_dynamic_instructions"
            continue
        fi

        echo "---> Timing Native-MC-$REFERENCE_PRECISION"

        local test_time
        test_time=$("$APPLICATION_PATH/src/$PROGRAM-native" $CLA $CLA_FOR_MULTIPLE_EXECUTIONS "$REFERENCE_PRECISION" | grep -oP 'CPU time used: \K[0-9.]+ seconds' | awk '{print $1}')

        # The native binary must print a "CPU time used: <n> seconds" line for
        # timing to work. An empty/zero test_time would divide by zero below.
        # This fails with a clear message (e.g. a non-conforming or failed
        # native build that fell back to a stale prebuilt binary).
        if [[ -z "$test_time" ]] || ! awk -v t="$test_time" 'BEGIN { exit !(t > 0) }'; then
            die "Native-MC timing: '$PROGRAM-native' produced no parseable 'CPU time used: <seconds> seconds' line (got: '${test_time:-<empty>}'). Check $LOGS_DIR/native-mc-build.log and confirm the native binary prints its CPU time."
        fi
        # Scale repetitions based on inverse of single sample run time
        local const_time=10
        NATIVE_MC_REPETITION=$(echo "$const_time $test_time" | awk '{result = $1/$2; print int(result) + (result > int(result))}')
        NATIVE_MC_REPETITION=$((NATIVE_MC_REPETITION < 400 ? NATIVE_MC_REPETITION : 400))

        echo "Running timing measurements"
        local time_array="["
        for ((i = 1; i <= NATIVE_MC_REPETITION; i++)); do
            [ -t 1 ] && echo -ne "\rRepetitions ($i/$NATIVE_MC_REPETITION)"
            local time_val
            time_val=$("$APPLICATION_PATH/src/$PROGRAM-native" $CLA $CLA_FOR_MULTIPLE_EXECUTIONS "$REFERENCE_PRECISION" | grep -oP 'CPU time used: \K[0-9.]+ seconds' | awk '{print $1}')
            emit_sample "Native-MC-$REFERENCE_PRECISION" "elapsedTime" "$i" "$time_val"
            time_array+="$time_val,"
        done
        time_array+="]"
        echo

        rm -f inscount.out

        # Use at most 20 repetitions for dynamic instruction count.
        local native_mc_repetition_pin=$((NATIVE_MC_REPETITION < 20 ? NATIVE_MC_REPETITION : 20))
        local native_mc_config="Native-MC-$REFERENCE_PRECISION"
        # Tracked locally so larger precisions in the same loop can
        # extrapolate from this row (Python is the source of truth for
        # the JSON field via SAMPLE lines; this average is bash-only).
        local pin_dyn_inst_array="["
        echo "Running dynamic instruction measurements"
        for ((i = 1; i <= native_mc_repetition_pin; i++)); do
            [ -t 1 ] && echo -ne "\rRepetitions ($i/$native_mc_repetition_pin)"
            $INSTRUCTION_COUNT_COMMAND "$APPLICATION_PATH/src/$PROGRAM-native" $CLA $CLA_FOR_MULTIPLE_EXECUTIONS "$REFERENCE_PRECISION" &>/dev/null
            if [[ ! -f inscount.out ]]; then
                die "inscount.out not found after PIN execution"
            fi
            pin_dyn_inst_array+=$(sed 's/Count //' inscount.out)
            pin_dyn_inst_array+=","
            local native_mc_sample_path="$LOGS_DIR/inscount-native-mc-$REFERENCE_PRECISION-$i-$$.out"
            mv inscount.out "$native_mc_sample_path"
            emit_sample "$native_mc_config" "pinDynInstCount" "$i" "$native_mc_sample_path"
        done
        echo
        pin_dyn_inst_array+="]"

        local process_e2e_time
        process_e2e_time=$(run_hyperfine_mean "$APPLICATION_PATH/src/$PROGRAM-native $CLA $CLA_FOR_MULTIPLE_EXECUTIONS $REFERENCE_PRECISION")

        # Time and PIN instruction count are both resolved by Python
        # from the SAMPLE lines emitted above. The `?` sentinels trigger
        # resolution via the accumulated elapsedTime and pinDynInstCount
        # samples for this config.
        emit_measurement "$native_mc_config" "?" "?" "$process_e2e_time" "?" "?"

        # Store values for potential extrapolation. The bash-side average
        # is computed here solely to populate last_measured_time, which
        # feeds the awk-based extrapolation arithmetic above. Python
        # independently computes the authoritative average from SAMPLE lines.
        last_measured_precision=$REFERENCE_PRECISION
        # B1 / B5 boundary: this `compute_average` callsite stays in bash
        # because the awk-based extrapolation block above (parallel to B5's
        # awk-stays-in-bash decision) consumes the average inline as control
        # flow input. Do not migrate to SAMPLE: the JSON field is already
        # authoritative via the SAMPLE lines emitted above.
        last_measured_time=$(compute_average "$time_array")
        last_measured_e2e_time=$process_e2e_time
        # B1 / B5 boundary: same reasoning as last_measured_time above.
        last_measured_instructions=$(compute_average "$pin_dyn_inst_array")
    done
}

# ===========================================================================
# Main
# ===========================================================================

main() {
    echo ==============================================================================
    echo Initialising...

    parse_args "$@"

    #   OUTPUT DIRECTORIES
    RESULTS_DIR="${RESULTS_DIR:-$APPLICATION_PATH/src}"
    LOGS_DIR="${LOGS_DIR:-$APPLICATION_PATH/src}"
    mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

    #   FILES
    INIT_S=init.S
    STARTUP_CPP=startup.cpp
    UXHW_MAKEFILE=Makefile.pro
    MAKEFILE=Makefile
    UXHW_DB_BASE=signaloidUxHwExecutionStatistics
    EXEC_STDOUT=exec.stdout
    EXEC_STDERR=exec.stderr
    UXHW_BUILD_LOG="$LOGS_DIR/uxhw-build.log"
    NATIVE_MC_BUILD_LOG="$LOGS_DIR/native-mc-build.log"

    TARGET_ARCH=x86_64-unknown-linux-gnu
    M_CONFIG_FILE=file.m

    REPETITION=20
    NATIVE_MC_REPETITION=20

    TIMING_TARGET_TOTAL_TIME=${TIMING_TARGET_TOTAL_TIME:-30}
    TIMING_MIN_REPETITIONS=${TIMING_MIN_REPETITIONS:-2}
    TIMING_MAX_REPETITIONS=${TIMING_MAX_REPETITIONS:-20}

    # Set defaults if not defined
    REQUIRED_UXHW_SDK_VERSION=${REQUIRED_UXHW_SDK_VERSION:-'"1.1.10-icelake-server"'}

    SKIP_UXHW=${SKIP_UXHW:-0}
    SKIP_NATIVE_MC=${SKIP_NATIVE_MC:-1}
    SKIP_UXHW_TRACING=${SKIP_UXHW_TRACING:-1}
    APPEND_TO_OUTPUT_FILE=${APPEND_TO_OUTPUT_FILE:-0}
    MAX_JUPITER_LIMIT=${MAX_JUPITER_LIMIT:-32}
    ENABLE_UNCERTAIN_TYPE_MODIFIER=${ENABLE_UNCERTAIN_TYPE_MODIFIER:-OFF}

    default_array REPRESENTATION_TYPES Athens
    default_array REPRESENTATION_SIZES 16 32 64 128 256 512
    default_array CORRELATION_TRACKING_TYPES Disabled Autocorrelation

    [[ -f "$SCRIPT_DIR/get_timings_local.sh" ]] && source "$SCRIPT_DIR/get_timings_local.sh"

    validate_required_vars

    ORIG_PWD=$(pwd)
    RESOURCES_DIR="${BENCHMARKING_RESOURCES_DIR:-$SCRIPT_DIR/../../assets/template/coreClass}"

    CLA_HASH=$(echo -n "$CLA" | md5sum | cut -d ' ' -f1)

    # The tracing database name contains the application version hash and a
    # hash generated by the CLA used to generate the database.
    TRACING_DB="$UXHW_DB_BASE-$APPLICATION_VERSION-$CLA_HASH-tracing"
    TRACING_DB_ABS="${TRACING_DB_ABS:-$RESULTS_DIR/$TRACING_DB.db}"
    M_CONFIG_FILE_TRACING=run-tracing.m

    # Python owns the canonical output path and passes the intermediate
    # file path here via $TIMING_INTERMEDIATE_FILE. Fall back to
    # $RESULTS_DIR if the env var is unset so the script remains
    # runnable via get-timing-template.sh.
    if [[ -z "${TIMING_INTERMEDIATE_FILE:-}" ]]; then
        TIMING_INTERMEDIATE_FILE="$RESULTS_DIR/$APPLICATION_NAME-$APPLICATION_VERSION-timings.intermediate"
    fi

    # Get timings for the UxHw (UxHw) cores.
    cd "$APPLICATION_PATH/src"

    # Stage build resources (Makefile, startup, etc.) into the application's
    # src/ before any step that consumes them. On a fresh checkout the
    # application does not ship these files; they live in $RESOURCES_DIR.
    copy_build_resources

    write_emulator_config "$M_CONFIG_FILE" "timing-dummy.db" "TimingTable"

    if [[ $APPEND_TO_OUTPUT_FILE -eq 0 ]]; then
        rm -f "$TIMING_INTERMEDIATE_FILE"
    fi

    write_timings_header

    #   Set PATH_TO_UXHW_SDK in Makefile.uxhw
    PATH_TO_UXHW_SDK_ALT=$(echo "$PATH_TO_UXHW_SDK" | sed 's#/#\\/#g')
    sed -i 's/^PATH_TO_UXHW_SDK\s*.*/PATH_TO_UXHW_SDK='"$PATH_TO_UXHW_SDK_ALT"'/g' "$UXHW_MAKEFILE"

    INPUT_DIR="inputs"
    echo "\$INPUT_DIR is $INPUT_DIR"

    # PIN_ROOT must be provided: the Python tool exports it from
    # --path-to-pin, or set it in the environment for a standalone run.
    # There is no built-in default (the check below fails clearly if unset).
    PIN_ROOT="${PIN_ROOT:-}"
    PIN_TOOL="$PIN_ROOT/source/tools/ManualExamples/obj-intel64/inscount0.so"
    INSTRUCTION_COUNT_COMMAND="$PIN_ROOT/pin -t $PIN_TOOL --"

    # Check Pin tool availability for benchmarks that need it
    if [[ $SKIP_UXHW -eq 0 ]] || [[ $SKIP_NATIVE_MC -eq 0 ]]; then
        if [[ ! -x "$PIN_ROOT/pin" ]]; then
            die "Intel Pin not found at '$PIN_ROOT/pin'. Set PIN_ROOT (or pass --path-to-pin) to a Pin kit directory containing the 'pin' binary."
        fi
        if [[ ! -f "$PIN_TOOL" ]]; then
            die "Pin inscount0 tool not found at $PIN_TOOL. Build it with: make -C $PIN_ROOT/source/tools/ManualExamples obj-intel64/inscount0.so"
        fi
    fi

    # Run benchmark sections
    if [[ $SKIP_UXHW -eq 0 ]]; then
        run_uxhw_benchmarks
    fi

    if [[ $SKIP_UXHW_TRACING -eq 0 ]]; then
        run_uxhw_tracing
    fi

    if [[ $SKIP_NATIVE_MC -eq 0 ]]; then
        run_native_mc_benchmarks
    fi

    cd "$ORIG_PWD"
}

main "$@"
