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

# Intel PIN kit location (PIN_ROOT). When invoked via the Python tool
# this is exported from --path-to-pin. For standalone use, set it here.
# Uncomment and point at your Pin kit directory:
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

# You need to source it to get the env variables to correctly work, also set the correct path to the shell script
. $SIGNALOID_PYTHON_DIR/src/signaloid/benchmarking/benchmark_timing/get-timings.sh
