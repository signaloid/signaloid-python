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
from enum import Enum
from pathlib import Path

# Default location of the UxHw SDK checkout. Both the ``--path-to-uxhw-sdk``
# CLI flag and the ``Benchmark`` constructor fall back to this, so the default
# is defined in exactly one place.
DEFAULT_UXHW_SDK_PATH = "~/project-uxhw-sdk"

# Shared numeric defaults for the benchmarking + ground-truth tools, used by
# both the ``Benchmark`` constructor and the argparse defaults.
DEFAULT_GROUND_TRUTH_SIZE = 1_000_000
DEFAULT_ADVERSARY_MC_SIZE = 1_000_000
DEFAULT_MAX_NUM_WEIGHTED_SAMPLES = 1_000_000
DEFAULT_NUM_ADVERSARIES = 100
DEFAULT_MAX_JUPITER_SIZE = 32
DEFAULT_ADVERSARY_MAX_SIZE_SCALAR = 100_000


class VariableTypes:
    DISTRIBUTION = "Distribution"
    SCALAR = "Scalar"


class CoreLibraryRepresentationTypes(Enum):
    """
    Subset of the core library's ``UxHwCoreRepresentationType`` enum: the
    benchmarked representation types plus ``NoUncertaintyTracking``.

    The integer values mirror the core library and must not be renumbered.
    """

    ATHENS = 4
    JUPITER = 6
    ATLAS = 7
    EUROPA = 8
    NO_UNCERTAINTY_TRACKING = 9


class RepresentationTypes:
    """
    String names for the representation types used across benchmarking: the
    core-library types (Athens, Jupiter, Atlas, Europa) and the Monte-Carlo /
    ground-truth types (WeightedSamples, Samples, MonteCarlo).
    """

    # Representation types supported by the core library and benchmarked here.
    ATHENS = "Athens"
    JUPITER = "Jupiter"
    ATLAS = "Atlas"
    EUROPA = "Europa"

    # The UxHw SDK writes the canonical representation strings
    # ("Athens"/"Atlas"/"Jupiter"/"Europa") directly to the DB UR_Type column,
    # so no name translation is needed and these maps stay empty.
    TO_UXHW_DB: dict[str, str] = {}
    FROM_UXHW_DB = {v: k for k, v in TO_UXHW_DB.items()}

    @classmethod
    def to_uxhw_db(cls, rep_type: str) -> str:
        """Convert a representation type to its UxHw Database name."""
        return cls.TO_UXHW_DB.get(rep_type, rep_type)

    @classmethod
    def from_uxhw_db(cls, uxhw_name: str) -> str:
        """Convert a UxHw Database name back to the standard representation type."""
        return cls.FROM_UXHW_DB.get(uxhw_name, uxhw_name)

    # Ground-truth / Monte-Carlo representation types.
    WEIGHTED_SAMPLES = "WeightedSamples"
    SAMPLES = "Samples"
    MONTE_CARLO = "MonteCarlo"


CORE_TO_STRING_REPRESENTATION = {
    CoreLibraryRepresentationTypes.ATHENS.value: RepresentationTypes.ATHENS,
    CoreLibraryRepresentationTypes.JUPITER.value: RepresentationTypes.JUPITER,
    CoreLibraryRepresentationTypes.ATLAS.value: RepresentationTypes.ATLAS,
}

STRING_TO_CORE_REPRESENTATION = {v: k for k, v in CORE_TO_STRING_REPRESENTATION.items()}


class ReportingMethods:
    MEAN = "Mean"
    QUANTILE_95 = "Quantile-95"
    QUANTILE_99 = "Quantile-99"


class ReportingNumbers:
    QUANTILE_95 = 0.95
    QUANTILE_99 = 0.99


class Correlations:
    DISABLED = "Disabled"
    AUTOCORRELATION = "Autocorrelation"


class DistanceMetrics:
    WASSERSTEIN_1 = "Wasserstein-1"
    WASSERSTEIN_2 = "Wasserstein-2"
    BINNED_WASSERSTEIN_1 = "Binned_Wasserstein-1"


class BenchmarkingVariables:
    VARIABLE = "Variable"
    VARIABLE_TYPE = "Variable Type"
    VARIABLE_DESCRIPTION = "Variable Description"
    UXHW_CONF = "UxHw Conf"
    UXHW_DISTANCE = "UxHw Distance"
    UXHW_BINNED_DISTANCE = "Binned UxHw Distance"
    # Human-readable reason a UxHw configuration's representation blew up.
    # ``None`` / NaN means healthy. A non-empty string marks the row as
    # degraded so it is flagged in the report rather than dropped (dropping it
    # would break the full-matrix join in measurement_loader). In-memory only:
    # not written to the CSVs, so it does not survive a re-load.
    BLOW_UP_REASON = "Blow-Up Reason"


class EquivMC:
    EMCC = "EMCC"
    EMCC_PREDICTED = "EMCC Predicted"
    REPORTING_METHOD = "Reporting Method"
    DISTANCE_TYPE = "Distance Type"
    PERCENTAGE_MC_BEATS_UXHW = "% MC beats UxHw"
    MEAN_QUANTILE = "Mean Quantile"
    BASIS_POINT_CONVERSION_FACTOR = 10_000
    TRACING_TABLE = "TracingTable"
    # Filename the native binary writes its Monte Carlo samples to,
    # relative to its working directory (see the demo's ``common.c``).
    MC_OUTPUT_FILENAME = "data.out"


class SignaloidYaml:
    TRACE_VARIABLES = "TraceVariables"
    BENCHMARKING_VARIABLES = "BenchmarkingVariables"
    FILE = "File"
    EXPRESSION = "Expression"
    LINE_NUMBER = "LineNumber"
    ALL_OUTPUTS = "BenchmarkingAllOutputs"
    VARIABLE_NAME = "VariableName"
    VARIABLE_DESCRIPTION = "VariableDescription"
    OUTPUT_OBJECT = "OutputObject"
    COMMAND_LINE_ARGUMENTS = "CommandLineArguments"


class AsymptoticDistanceDistribution:
    IS_NORMAL = "Is normal"
    SCALE = "Scale"


class Measurements:
    NATIVE_IN_APP_TIME = "Native In Application Time"
    DB_TIME = "Database Time"
    SPEEDUP = "Speedup"
    IN_APP_TIME = "In Application Time"
    E2E_TIME = "End-to-End Time"
    NATIVE_E2E_TIME = "Native End-to-End Time"
    PIN_DYN_COUNT = "PIN Dyn. Inst. Count"
    DB_DYN_COUNT = "Database Dyn. Inst. Count"
    NATIVE_PIN_COUNT = "Native PIN Dyn. Inst. Count"


class ReportSheetTabs:
    """Worksheet tab names in the benchmarking-report Google Sheets template.

    ``report_writer`` looks worksheets up by these names (and the Plot Triptych
    write targets one by name) instead of by positional index, so the tool
    survives the template being reordered or having tabs added / removed.

    The values MUST match the tab names in the Drive template
    (``UXHW_SHEETS_TEMPLATE_ID``) exactly. See the automation README's Google
    Sheets section for the template link and how to read the exact tab names.

    Only ``ASSUMPTIONS_CONFIG``, ``TIMING_PERFORMANCE`` (duplicated per
    reporting method), and ``PLOT_TRIPTYCH`` are written by the tool. The rest
    are listed for completeness and are populated manually / statically.
    """

    ASSUMPTIONS_CONFIG = "Assumptions, Configuration, and Terminology"
    SUMMARY_GITHUB = "Summary for Copying into GitHub"
    TIMING_PERFORMANCE = "Timing Performance"
    PLOT_TRIPTYCH = "Plot Triptych"
    C_CODE_SIZE = "C Code Size Estimate"
    NRE = "Non-Recurring Engineering (NRE) Monetary Cost"


class MetadataRowLabels:
    """Column-A labels on the ``ReportSheetTabs.ASSUMPTIONS_CONFIG`` sheet whose
    column B the uploader fills in.

    Each target row is matched by a case-insensitive substring search for these
    strings. Each constant holds only the stable part of the label, since the
    template adds a list-number prefix and trailing descriptive text.
    """

    GITHUB_REPOSITORY = "GitHub Repository"
    GIT_HASH = "Git Hash"
    UXHW_SDK_VERSION = "UxHw SDK version"
    MACHINE_TYPE = "Machine Type"


class TimingFormat:
    """
    Wire format shared between get-timings.sh (which emits tagged lines into a
    transient intermediate file) and the Python reader (which parses them and
    writes the canonical JSON artifact).

    The canonical output is one JSON document per benchmark session: a top-level
    object with the session-invariant fields (application identity, SDK version,
    target repetition count) plus a ``runs`` array of per-run records, each with
    its own timestamp, command-line arguments, and ``measurements`` array.

    These constants reach bash via environment variables set in
    ``signaloid.benchmarking.automation.build.run_timing_script``, so this class
    is the single source of truth for both sides.
    """

    # Tags prefixed onto each line of the intermediate file.
    META_TAG = "META"
    MEASUREMENT_TAG = "MEASUREMENT"
    # SAMPLE lines carry per-iteration raw values that Python averages
    # during parse. Format:
    #   SAMPLE <config> <field> <iteration> <value-or-path>
    # where the 4th token is a literal float (value-typed variant, used
    # for elapsed time) or a file path (path-typed variant, used for
    # PIN instruction counts).
    SAMPLE_TAG = "SAMPLE"

    # Names of the environment variables used to pass the tag strings
    # and the intermediate file path to the bash script.
    META_TAG_ENV_VAR = "TIMING_META_TAG"
    MEASUREMENT_TAG_ENV_VAR = "TIMING_MEASUREMENT_TAG"
    SAMPLE_TAG_ENV_VAR = "TIMING_SAMPLE_TAG"
    INTERMEDIATE_FILE_ENV_VAR = "TIMING_INTERMEDIATE_FILE"

    # Field names used in SAMPLE lines to identify the measured
    # quantity. These are the values emitted as the second token of a
    # SAMPLE line by the bash script. Value-typed (literal float)
    # fields: elapsedTime, databaseTime, databaseDynInstCount.
    # Path-typed (file path) field: pinDynInstCount (declared below
    # near the JSON keys for grouping reasons).
    SAMPLE_FIELD_TIME = "elapsedTime"
    SAMPLE_FIELD_DB_TIME = "databaseTime"
    SAMPLE_FIELD_DB_DYN_INST_COUNT = "databaseDynInstCount"

    # Filename suffixes appended to "<app>-<version>".
    INTERMEDIATE_SUFFIX = "-timings.intermediate"
    JSON_SUFFIX = "-timings.json"

    # Keys emitted as `META <key> <value>` lines in the intermediate.
    META_KEY_TIMESTAMP = "timestamp"
    META_KEY_APPLICATION_NAME = "applicationName"
    META_KEY_APPLICATION_VERSION = "applicationVersion"
    META_KEY_UXHW_SDK_VERSION = "uxhwSdkVersion"
    META_KEY_COMMAND_LINE_ARGUMENTS = "commandLineArguments"
    META_KEY_COMMAND_LINE_ARGUMENTS_HASH = "commandLineArgumentsHash"
    # Configured UxHw repetition target at session start. The per-measurement
    # count can differ (get-timings.sh rescales ``REPETITION``), so treat this
    # as the session target, not a per-run ground truth.
    META_KEY_UXHW_TARGET_REPETITIONS = "uxhwTargetRepetitions"

    # META keys whose values are invariant across every run in a
    # benchmarking session. The Python reader lifts these out of the
    # per-run dicts and places them at the top level of the JSON
    # document, validating consistency across runs in the process.
    SESSION_META_KEYS = (
        META_KEY_APPLICATION_NAME,
        META_KEY_APPLICATION_VERSION,
        META_KEY_UXHW_SDK_VERSION,
        META_KEY_UXHW_TARGET_REPETITIONS,
    )

    # Field names in the canonical JSON document. Non-session META
    # keys carry over verbatim into each run record.
    JSON_KEY_RUNS = "runs"
    JSON_KEY_MEASUREMENTS = "measurements"
    JSON_KEY_MEASUREMENT_CONFIG = "config"
    JSON_KEY_MEASUREMENT_TIME = "time"
    JSON_KEY_MEASUREMENT_DB_TIME = "dbTime"
    JSON_KEY_MEASUREMENT_E2E_TIME = "e2eTime"
    JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT = "dbDynInstCount"
    JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT = "pinDynInstCount"

    # Field name for PIN dynamic instruction count samples emitted via
    # SAMPLE lines (used as the <field> token in the wire format).
    SAMPLE_FIELD_PIN_INST = "pinDynInstCount"

    # Sentinel used in the intermediate for missing numeric fields
    # (e.g. database time is not measured for native runs).
    MISSING_VALUE = "?"


def get_repo_root() -> Path:
    """
    Find the repository root.

    Checks the SIGNALOID_PYTHON_DIR env var first, then walks upward from
    this file looking for pyproject.toml.

    Returns:
        The repository root directory.

    Raises:
        RuntimeError: If the repository root cannot be located.
    """
    env_override = os.environ.get("SIGNALOID_PYTHON_DIR")
    if env_override:
        p = Path(env_override).resolve()
        if p.is_dir():
            return p

    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise RuntimeError(
        "Could not find repository root. Set SIGNALOID_PYTHON_DIR or run from a repo checkout."
    )


def get_resources_dir() -> Path:
    """
    Path to the bundled coreClass build-template assets.

    ``assets/`` is a verbatim vendored copy of the build-template assets.
    Resolved relative to this package, so it works from both a source checkout
    and an installed wheel.

    Returns:
        The ``assets/template/coreClass`` directory inside this package.
    """
    return Path(__file__).parent / "assets" / "template" / "coreClass"
