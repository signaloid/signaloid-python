PROGRAM="main"
CLA="-T"
CLA_FOR_MULTIPLE_EXECUTIONS=""

# APPLICATION_NAME and APPLICATION_VERSION are required by get-timings.sh.
# When invoked via the Python tool these are exported from
# Benchmark.export_timing_env(). For standalone use, set them here (or
# replace with explicit values, e.g. APPLICATION_NAME="my-demo",
# APPLICATION_VERSION="0.1.0").
APPLICATION_NAME=$(basename "$APPLICATION_PATH" | sed 's/Signaloid-Demo-//')
APPLICATION_VERSION=$(git -C "$APPLICATION_PATH" rev-parse --short=7 HEAD 2>/dev/null \
    || date '+%Y-%m-%d-%H-%M-%S')

# Intel PIN kit location (PIN_ROOT). PIN is optional and off by default.
# The Python tool passes it through only with --measure-dynamic-instructions.
# For standalone use, uncomment the line below and point it at your Pin
# kit directory. Leave it commented out to skip the instruction count.
# export PIN_ROOT="/path/to/pin-external-<version>-gcc-linux"

TRACES=(
    'addDistValueTrace variableName        "main.c:124"'
)

# Reference precisions (Monte Carlo sample counts) to benchmark. Counts up to
# and including EXTRAPOLATION_THRESHOLD (default 1000000) are directly measured.
# Larger counts are linearly extrapolated. Export EXTRAPOLATION_THRESHOLD to
# set:
#     export EXTRAPOLATION_THRESHOLD=200000
REFERENCE_PRECISIONS=(50 500)

SKIP_UXHW=0
SKIP_UXHW_TRACING=1
SKIP_NATIVE_MC=1

APPEND_TO_OUTPUT_FILE=1

REPRESENTATION_TYPES=(Athens Jupiter)
REPRESENTATION_SIZES=(64 128 256 512)
CORRELATION_TRACKING_TYPES=(Disabled Autocorrelation)
MAX_JUPITER_LIMIT=32

# You need to source get-timings.sh for the variables set above to take
# effect. Its location is resolved from the installed `signaloid` package, so
# this works for both a pip install and a source checkout. We try
# BENCHMARKING_PYTHON, then `python3`, then `python`, since PEP 394 is not
# honoured on every platform. A candidate has to both run and import the
# package, so the loop settles on one that can do the real work below.
GET_TIMINGS_SH=""
for BENCHMARKING_PYTHON in "${BENCHMARKING_PYTHON:-}" python3 python; do
    [ -n "$BENCHMARKING_PYTHON" ] || continue
    GET_TIMINGS_SH=$("$BENCHMARKING_PYTHON" -c 'import signaloid.benchmarking.config as config
print(config.get_timing_script())' 2>/dev/null) && [ -n "$GET_TIMINGS_SH" ] && break
    GET_TIMINGS_SH=""
done

if [ -z "$GET_TIMINGS_SH" ]; then
    echo "Could not locate get-timings.sh. Set \$BENCHMARKING_PYTHON to a Python 3 interpreter that can import the signaloid package." >&2
    return 1 2>/dev/null || exit 1
fi

export BENCHMARKING_PYTHON
. "$GET_TIMINGS_SH"
