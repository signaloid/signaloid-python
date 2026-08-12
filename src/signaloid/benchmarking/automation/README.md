# Benchmarking Automation

The signaloid.benchmarking.automation package enables the benchmarking
applications running with Signaloid UxHw technology for computing with
distributions, against Monte Carlo baselines. The tool compiles the subject
application from source code for native Monte Carlo execution, generates the
ground truth result dataset and databases used for the comparison to the Monte
Carlo baselines. It runs the UxHw variant of the application, computes
equivalent Monte Carlo (EMCC) metrics, and collects timing data. The tool writes
all results to CSV data files, Markdown reports, and (optionally) uploads to a
Google Sheet.

## Requirements and dependencies

The tool requires the following system dependencies, Python dependencies, and
third-party dependencies.

### System dependencies
- gcc or g++
- GNU Scientific Library (on Ubuntu: libgsl-dev).
- Python (3.10+; see root-level pyproject.toml)
- GNU Make (make)
- bash
- lscpu
- hyperfine

C and C++ files are compiled separately and linked with `c++`. Applications that
link against GSL need `-lgsl -lgslcblas -lm`.

### Python
This tool is part of the `signaloid.benchmarking` package. Create and activate a
virtual environment, then install the package with pip:
```
python -m venv .venv
source .venv/bin/activate
pip install .
```
For development, you can install in editable mode with `pip install -e .`
instead.

The benchmarking pipeline's application dependencies (`pandas`, `tabulate`,
`tqdm`, `POT`) are non-optional. Optional dependencies is group `sheets`, needed
for the Google Sheets upload (see [Google Sheets](#google-sheets-optional)):
install with `pip install ".[sheets]"`.

Commands in this guide assume you are using the virtual environment, so that the
`signaloid-benchmarking` entry point and `python -m
signaloid.benchmarking.automation` are available.

### Signaloid UxHw SDK

The benchmarking tool needs the Signaloid UxHw SDK to compile applications for
UxHw. Set the path to the UxHw SDK via command-line parameter
`--path-to-uxhw-sdk` (for example, `/opt/Signaloid-UxHw-SDK`).

### Application requirements

A benchmarkable application is a project which can connect to the [Signaloid
Cloud Developer Platform](https://signaloid.io), plus a small amount of
benchmarking metadata. Minimal layout:

```
your-app/
├── src/
│   └── main.c          # UxHw program: writes each output to an array, selects one via `-S <n>`
├── signaloid.yaml      # REQUIRED: declares the trace + benchmarking variables
└── src/config.mk       # OPTIONAL: SOURCES/CFLAGS for the native-MC build
```

The target application must:
1. **Be a Git repository** — the current commit hash versions the output files.
2. **Contain a `signaloid.yaml` at its root** — see [Application
   Configuration](#application-configuration). This is what makes a repo
   benchmarkable. Without it the run aborts at "Loading application info".
3. **Have its source in a `src/` directory.**

The build file is **optional**. Native compilation (for the native-MC baseline)
is resolved in this order: a `Makefile` at the root with a `local-build` target,
else `src/config.mk` with `SOURCES` (and optional `CFLAGS`). The `Makefile` is
**not** required and a `config.mk` alone is enough (as in the [C project
template](https://github.com/signaloid/Signaloid-Demo-General-C)). If neither is
present (or if `config.mk` defines no `SOURCES`), the native-MC baseline is
skipped: the UxHw benchmark and distance analysis still run but it does not
compute the native-vs-UxHw speedup.

See the [Signaloid documentation for more details about using GitHub repositories
with UxHw](https://docs.signaloid.io/docs/api/guides/builds/builds-repository/).


### Intel PIN

Intel PIN is necessary for the dynamic instruction count on any timing run. The
tool uses it via command-line argument `--path-to-pin` or environment variable
`PIN_ROOT`. A timing run fails if neither is set.

The dynamic instruction count (`pinDynInstCount`) is from the `inscount0` tool.

Basic installation instructions:
1. Download a PIN kit for your platform from Intel's
   [Pin binary-instrumentation tool downloads](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html)
   (validated against PIN 4.2).
2. Extract it to a directory `<dir>` of your choice.
3. Set parameter `--path-to-pin <dir>` (or `export PIN_ROOT=<dir>`).
3. Build the `inscount0` counter once (the kit does not ship it prebuilt):


### Google Sheets (Optional)

The benchmarking tool can optionally upload the benchmarking results to a Google
Sheet. It only runs when `--write-sheets` is passed and is is disabled by
default.

To enable it you need:
1. The `sheets` extra installed:
   ```
   pip install ".[sheets]"
   ```
   A plain `pip install .` does **not** pull in the Sheets stack (`gspread`,
   `google-api-python-client`, `oauth2client`).
2. A Google Cloud service-account credentials JSON file, supplied via
   `--google-credentials <path>` or the `GOOGLE_APPLICATION_CREDENTIALS`
   environment variable (whichever resolves to an existing file). There is no
   built-in default — passing `--write-sheets` without a resolvable credentials
   file raises `RuntimeError` at startup (before the pipeline runs).
3. The target Google Drive folder and Sheets template, supplied via the
   `UXHW_SHEETS_DRIVE_FOLDER_ID` and `UXHW_SHEETS_TEMPLATE_ID` environment
   variables. `--write-sheets` without both set also raises `RuntimeError` at
   startup.

The Sheets template is public:
<https://docs.google.com/spreadsheets/d/1MKPkwY_B_m5WH-coqIy19Otpg3IE5yRSKaKGw6wxzL8>.
The tab-name and metadata-row constants in `config.py` (`ReportSheetTabs`,
`MetadataRowLabels`) must match this template exactly. When it changes, read the
full tab names from its **ODS** export, never the XLSX export (XLSX truncates
tab names to 31 characters).


## Source setup

1. Clone this repository and install:
   ```
   git clone <repo-url>
   cd signaloid-python
   git submodule update --init --recursive
   python -m venv .venv
   source .venv/bin/activate
   pip install ".[sheets]"
   ```
   (`pip install .` gives a fully working benchmarking tool. The `[sheets]`
   extra additionally pulls in the optional Sheets stack for the Google Sheets
   upload.)

2. Verify the UxHw SDK is available at its expected path (or note the path for
   use with `--path-to-uxhw-sdk`).

3. Verify that the target application compiles natively:
   ```
   cd /path/to/application
   make local-build   # if using a Makefile
   ```
   The compiled executable (`demo-native-mc`) should be located at the
   application root.

## Usage

Run the tool as a module:
```
python -m signaloid.benchmarking.automation [OPTIONS]
```
or via the installed entry point:
```
signaloid-benchmarking [OPTIONS]
```


### Required Arguments

| Flag | Description |
|---|---|
| `-u`, `--representation-types` | Uncertain representation types. One or more of: `Athens`, `Atlas`, `Jupiter`, `Europa`. |
| `-s`, `--representation-sizes` | Uncertain representation sizes (integers). E.g., `16 32 64 128 256 512`. |
| `-c`, `--uncertainty-correlation_types` | Correlation tracking types. One or more of: `Disabled`, `Autocorrelation`. |
| `-r`, `--reporting-methods` | Reporting methods. One or more of: `Mean`, `Quantile-95`, `Quantile-99`. |
| `--path-to-application` | Path to the application repository to benchmark. |

### Optional Arguments

| Flag | Default | Description |
|---|---|---|
| `--path-to-uxhw-sdk` | `~/project-uxhw-sdk` | Path to the UxHw SDK. |
| `--path-to-pin` | `None` (uses `PIN_ROOT` env) | Path to the Intel PIN kit, exported as `PIN_ROOT` for the timing script's dynamic instruction count. Omit to keep any existing `PIN_ROOT`. A timing run fails clearly if neither `--path-to-pin` nor `PIN_ROOT` is set. |
| `--demo-cli-args` | `""` | Extra command-line arguments passed to both native-MC and UxHw executions. Use this when the demo application requires additional flags (e.g., `--demo-cli-args "--asc-file inputs/blink.asc"`). |
| `--ground-truth-size` | `1` | Number of Monte Carlo samples for ground truth generation. |
| `--ground-truth-type` | `MonteCarlo` | Type of ground truth: `MonteCarlo` or `WeightedSamples`. |
| `--path-to-ground-truth-file` | — | Path to the Python script that generates analytic ground truth. |
| `--distance-type` | `Wasserstein-1` | Distance metric: `Wasserstein-1` or `Wasserstein-2`. |
| `--use-binned-uxhw` / `--no-use-binned-uxhw` | `True` | Use binned Wasserstein-1 for UxHw distances (only valid with `Wasserstein-1`). Pass `--no-use-binned-uxhw` (or `use_binned_uxhw: false` in a config) to disable. |
| `--use-clt` | `False` | Predict the equivalent-MC count from the asymptotic (CLT / Brownian-bridge) distance distribution instead of measuring it via explicit adversary MC. |
| `--adversary-mc-size` | `1` | Number of Monte Carlo samples for the adversary database. |
| `--adversary-max-size-scalar` | `100000` | Maximum adversarial MC size for scalar output variables. |
| `--num-adversaries` | `100` | Number of adversary repetitions. |
| `--max-num-weighted-samples` | `1` | Maximum number of weighted samples for ground truth conversion. |
| `-j`, `--jobs` | `1` | Number of parallel workers for native MC generation. |
| `--config` | `None` | Path to a YAML config file whose keys populate defaults (see [Config Files](#config-files)). |
| `--print-config` | `False` | Resolve defaults + config + CLI, print the effective config and the swept benchmark matrix, then exit without running. |
| `--write-sheets` | `False` | Opt in to the Google Sheets upload step (pipeline step 15). Requires the `sheets` extra, a resolvable credentials file (`--google-credentials` or `GOOGLE_APPLICATION_CREDENTIALS`), and the `UXHW_SHEETS_DRIVE_FOLDER_ID` + `UXHW_SHEETS_TEMPLATE_ID` environment variables. Missing any of these raises `RuntimeError` at startup (before the pipeline runs). |

For an extensive list of command line arguments for his program please run with the `--help` command.

### Example

```
python -m signaloid.benchmarking.automation \
    --path-to-application ~/Signaloid-Demo-Example \
    --path-to-uxhw-sdk ~/project-uxhw-sdk \
    -u Athens Atlas \
    -s 16 32 64 128 256 512 \
    -c Disabled Autocorrelation \
    -r Mean Quantile-95 Quantile-99 \
    --ground-truth-size 1000000 \
    --adversary-mc-size 1000000 \
    --distance-type Wasserstein-1 \
    -j 4
```

## Config Files

Due to the large number of command line arguments this tool contains, one can
create a configuration file with the `--config <file.yaml>` flag that stores the
information for data quality metrics like ground truth size etc. With the
presets in `configs/`, a standard run drops to:

```
signaloid-benchmarking --config configs/full-sweep.yaml \
    --path-to-application ~/Signaloid-Demo-Example
```

**Precedence** is `built-in defaults < config file < explicit CLI flags`. Config
keys are the argparse `dest` names (e.g. `representation_types`, `correlations`,
`use_binned_uxhw`), not the CLI flag spellings. Unknown keys (typos) are
rejected with a clear error rather than silently ignored.

Because the four sweep args (`representation_types`, `representation_sizes`,
`correlations`, `reporting_methods`) can be supplied via the file, they are no
longer `required` on the CLI — but at least one source must provide each, or the
run fails fast with a `ValueError`.

You can find two presets for the configuration YAML in `configs/`:

- `configs/quick.yaml` — a small, fast matrix for smoke-tests.
- `configs/full-sweep.yaml` — the full experiment matrix reused across demos.

Sample schema (every key is optional and omitted keys fall back to defaults):

```yaml
# Benchmark matrix
representation_types: [Athens, Jupiter]
representation_sizes: [64, 128, 256, 512]
correlations: [Disabled, Autocorrelation]
reporting_methods: [Mean, Quantile-95]
distance_type: Wasserstein-1
use_binned_uxhw: true
use_clt: false
n_adversaries: 100  # dest name. The CLI flag is --num-adversaries
# plotting controls (all default off)
plot_distance_vs_asymptotic: false
plot_adversary_distances: false
plot_representative_mc: false
```

Use `--print-config` to resolve defaults + file + CLI, print the effective
configuration and the benchmark matrix that would be swept, then exit without
running:

```
python -m signaloid.benchmarking.automation --config configs/quick.yaml \
    --path-to-application ~/Signaloid-Demo-Example --print-config
```

### Plotting controls

Plotting is off by default, so a plain run no longer pays the plotting cost.
Three independent toggles enable the three plot calls in the equivalent-MC stage
and each is a `BooleanOptionalAction`, so it accepts both the positive and
`--no-...` forms on the CLI and the matching boolean key in a config file:

- `--plot-distance-vs-asymptotic` — empirical equivalent-MC distance
  distribution vs its asymptotic prediction (Brownian-bridge / half-normal).
- `--plot-adversary-distances` — adversary-distance plots (empty under
  `--use-clt`).
- `--plot-representative-mc` — a representative equivalent-MC run matching the
  UxHw–ground-truth distance (distribution outputs only).

## Pipeline Overview

The tool executes the following steps in order (shown as `[Step N/15]` in the
output):

1. **Machine Info** — Detects CPU model and core count via `lscpu`.
2. **Application Info** — Reads `signaloid.yaml`, resolves benchmarking
   variables, compiles the native MC executable, and sets up output paths.
3. **Ground Truth Database** — Generates the reference distribution database
   using (in priority order): analytic formula, then native MC execution. A
   native build (a `Makefile` `local-build` target or `src/config.mk` with
   `SOURCES`) or `--has-analytic-ground-truth` is required.
4. **UxHw Tracing Database** — Runs the application through the UxHw tracing
   pipeline to produce UxHw distributional outputs for each representation
   type/size/correlation combination. If a tracing database already exists, the
   script will prompt before overwriting. The tracing build uses `-O0` (so the
   `addDistValueTrace` `file:line` directives resolve against unoptimised debug
   info); to guard against optimisation changing the traced values, each config
   is also built at `-O2` and its Ux strings are checked (byte-for-byte) against
   the `-O0` ones. Any difference is reported as a warning and does not stop the
   run. If a config's `-O2` build or run fails, that config is skipped and
   reported as failing in an `-O2 ux-string verification FAILED` summary (the
   run still continues). Set `TRACING_VERIFY_OPTFLAGS` to compare against a
   different level.
5. **Asymptotic Distance Distributions** — Computes asymptotic distance
   distributions from the tracing data.
6. **UxHw Distances** — Computes Wasserstein distances between each UxHw
   configuration and the ground truth.
7. **EMCC Predictions** — Predicts equivalent Monte Carlo sample counts from
   asymptotic distances.
8. **Adversary Database** — Generates adversarial Monte Carlo databases for EMCC
   validation.
9. **Equivalent Monte Carlo** — Computes the true EMCC by comparing adversary MC
   distances to UxHw distances.
10. **UxHw Timings** — Measures execution time and dynamic instruction counts
    for each UxHw configuration.
11. **Native MC Timings** — Measures execution time for native MC at each EMCC
    sample size.
12. **Load Measurements** — Loads all measurement data from the timing file.
13. **Load Timing Data** — Loads equivalent MC timing data from native
    executions.
14. **Compute Results** — Merges all timing and EMCC data into final dataframes.
15. **Write Outputs** — Writes results to Markdown unconditionally, and to a
    Google Sheets spreadsheet when `--write-sheets` is passed. The Sheets upload
    is opt-in (default behaviour prints a skip message). When opted in: per
    variable, a "best-of triptych" of three plots — the adversary MC at the
    selected EMCC count, the UxHw distribution for the best UxHw configuration,
    and the ground truth — is uploaded to Google Drive and linked from the
    sheet. The "best" configuration is the one with the maximum speedup against
    the native MC baseline, and the selection is printed to the terminal during
    upload. All other generated plots remain in `results/plots/` and are not
    uploaded. Credentials come from `--google-credentials` or
    `GOOGLE_APPLICATION_CREDENTIALS`, and the target Drive folder + Sheets
    template from `UXHW_SHEETS_DRIVE_FOLDER_ID` + `UXHW_SHEETS_TEMPLATE_ID`.
    Passing `--write-sheets` without all of these resolving raises
    `RuntimeError` at startup, before step 1 runs.

## Application Configuration

The target application must contain a `signaloid.yaml` file at its root. The
file defines which variables to trace and benchmark.

### Required Fields

```yaml
TraceVariables:
  - Expression: "variableName"
    File: "main.c"
    LineNumber: "42"

BenchmarkingAllOutputs:
  - CommandLineArguments: "-S 2"

BenchmarkingVariables:
  - VariableName: "variableName"
    VariableDescription: "Description of the output"
    OutputObject: "Distribution"          # or "Scalar"
    CommandLineArguments: "-S 0"
```

- **`TraceVariables`**: Lists the C expressions to trace, with source file and
  line number.
- **`BenchmarkingAllOutputs`**: The command-line arguments that produce all
  outputs simultaneously (used for tracing runs).
- **`BenchmarkingVariables`**: Maps each traced variable to a human-readable
  description, its output type (`Distribution` or `Scalar`), and the
  command-line arguments to isolate that variable.

Array expressions such as `outputVariables[0:5]` in `TraceVariables` are
automatically expanded into individual entries (`outputVariables[0]`,
`outputVariables[1]`, ..., `outputVariables[5]`).

### Source convention
The tool benchmarks one output at a time, so by convention `main.c`:
- writes each output into an array (e.g. `double outputVariables[N];`), and
- accepts a `-S <index>` command-line argument selecting which output to
  compute/emit.

`signaloid.yaml` ties these together: `TraceVariables` points
`File`/`LineNumber`/`Expression` at the array (`outputVariables[0:N]`), and each
`BenchmarkingVariables` entry pairs a `VariableName` (`outputVariables[i]`) with
the `CommandLineArguments` (`-S i`) that isolate it. The program should also
print a `CPU time used: <seconds> seconds` line to stdout — the timing layer
parses it for the reference timing.

## Bash Timing Scripts

The `src/signaloid/benchmarking/benchmark_timing/` directory contains the
underlying bash scripts used by the pipeline:

- **`get-timings.sh`**: The main timing and compilation driver. It is sourced
  (not executed) by the Python tool with pre-set environment variables. It
  handles UxHw compilation, native MC benchmarking, UxHw tracing, and timing
  collection (the UxHw cores are compiled and timed via UxHw. The dynamic
  instruction count comes from Intel PIN). Compilation warnings are redirected
  to log files (`uxhw-build.log` and `native-mc-build.log` in the `logs/`
  directory). Source files are discovered recursively (excluding `build/`
  directories), and C++ files (`.cc`, `.cpp`) are automatically included when
  present.
- **`get-timing-template.sh`**: A standalone template showing the required
  environment variables and how to source `get-timings.sh` directly from the
  command line.

The UxHw `.m` config files that drive the reference / Monte Carlo / tracing
passes are generated inline by the `write_emulator_config` shell function
(consumed by the UxHw `opt` transform via `--m-config-file`). There is no
separate template file.

## Output Files

Output files are organized into `results/` and `logs/` directories in the
current working directory:

```
results/          # Final outputs
├── timings.json
├── *.db          (ground truth, adversary, tracing)
├── *.csv         (output_data, uxhw_distances, asymptotic_distances)
├── *.md          (markdown reports)
└── plots/
```

| File | Description |
|---|---|
| `output_data.csv` | Full EMCC results with distances, timing data, and speedups for every variable and configuration. |
| `uxhw_distances.csv` | UxHw distance data (Wasserstein distances between each configuration and ground truth). |
| `asymptotic_distances.csv` | Asymptotic distance distribution parameters. |
| `<description>.md` | Per-variable Markdown summary tables. |
| `<app-name>-<hash>-timings.json` | Canonical timing output as a single JSON document (in `results/`). Session-invariant fields (application identity, SDK versions, target UxHw repetition count) sit at the top level. A `runs` array holds per-run records, each with its own `timestamp`, `commandLineArguments`, `commandLineArgumentsHash`, and `measurements` array. The schema is defined by `TimingFormat` in `signaloid/benchmarking/config.py`. |
| `<app-name>-<hash>-timings.intermediate` | Transient flat file the bash timing script writes line-by-line. Python parses it into the JSON above and deletes it on success. It is kept on failure for debugging. |
| `*.db` | SQLite databases for ground truth, adversary, and tracing data (in `<application>/src/`). |

Example `*-timings.json` document:

```json
{
  "applicationName": "call-option",
  "applicationVersion": "a1b2c3d",
  "uxhwSdkVersion": "4.1.2",
  "uxhwTargetRepetitions": "20",
  "runs": [
    {
      "timestamp": "2026-04-15T14:22:10Z",
      "commandLineArguments": "-T 100 --strike 110",
      "commandLineArgumentsHash": "abc123",
      "measurements": [
        {
          "config": "Athens-16-Autocorrelation",
          "time": 1.23,
          "dbTime": 4.56,
          "e2eTime": 7.89,
          "dbDynInstCount": 0.0,
          "pinDynInstCount": 2000.0
        }
      ]
    }
  ]
}
```

Notes on the schema:

- Missing numeric fields (e.g. `dbTime` on native runs) are serialised as
  `null`.
- `dbDynInstCount` for UxHw rows is `0`.
- `uxhwTargetRepetitions` is the UxHw-loop target at session start. The actual
  rep count used per measurement can differ — native-MC rows use
  `NATIVE_MC_REPETITION` (dynamically computed per precision), and the UxHw
  `REPETITION` is rescaled per-testcase from a warmup run inside
  `run_uxhw_benchmarks`. Treat this field as the configured session target
  rather than a per-run ground truth.

### Error Logs

On failure, the pipeline writes a detailed traceback to a timestamped log file:
```
logs/benchmarking_automation_error_<timestamp>.log
```
This file contains the full Python traceback and the command-line arguments
used. Additionally:
- Bash timing script stderr is captured to `logs/timing_script_stderr.log`.
- UxHw compilation warnings are logged to `logs/uxhw-build.log`.
- Native MC compilation warnings are logged to `logs/native-mc-build.log`.
- Per-execution errors can be found in `logs/exec.stderr` and `logs/opt.err`.

## Common Issues

### Native Compilation Fails
- **Missing GSL**: Ensure `libgsl-dev` (or equivalent) is installed and that
  `/opt/local/lib` and `/opt/local/include` are valid paths on your system, or
  that the application's `Makefile` handles library paths.
- **No compilation method found**: The application needs one of: a root-level
  `Makefile` with a `local-build` target, or a `src/config.mk` with `SOURCES`
  defined.
- **Executable not found**: The native executable (`demo-native-mc`) should be
  placed at the application root by `make local-build` or built from
  `config.mk`.
- **Fallback**: If the timing script's ad-hoc compilation fails, it will look
  for a pre-built `demo-native-mc` in both `src/` and the application root
  (where `make local-build` places it).

### Timing Script Fails
- The error message will include captured stderr and point to `exec.stderr` and
  `opt.err` in the `logs/` directory. Check these files for UxHw compilation or
  runtime errors.
- If the tracing database already exists, the script will prompt `Do you want to
  continue execution? (y/n)`. Answer `y` to overwrite or `n` to abort.

### Timing Data Mismatch
- The intermediate timing file is automatically truncated at the start of each
  pipeline run, so stale data from previous runs should not cause issues. If
  problems persist, manually delete the `<app-name>-<hash>-timings.intermediate`
  and `<app-name>-<hash>-timings.json` files in the `results/` directory and
  re-run.

### EMCC Data Not Found
- If the pipeline is interrupted after tracing but before EMCC computation,
  re-running will attempt to load `results/output_data.csv`. If this file does
  not exist or is incomplete, delete it and re-run from the beginning.

### Analytic Ground Truth
- When using `--has-analytic-ground-truth`, you must also pass
  `--ground-truth-type WeightedSamples` and provide
  `--path-to-ground-truth-file`. The ground truth script must accept `<size>
  <output_csv_path>` as positional arguments.

### Parallelism
- The `-j` flag controls parallelism for native MC sample generation only. If
  `-j` exceeds the number of detected CPU cores, a warning is printed. UxHw
  executions are always sequential.

### Binned Distance Computation
- `--use-binned-uxhw` is only compatible with `--distance-type Wasserstein-1`.
  The tool will error if you try to combine it with `Wasserstein-2`.

### Input Files Not Found
- Input files from the application's `inputs/` directory are symlinked into the
  working directory. Files that already exist in the target directory (e.g.,
  `README.md`) are skipped with a warning to avoid overwriting repository files.
  If the demo application expects files at a relative path like
  `inputs/filename`, ensure the demo's default paths do not include the
  `inputs/` prefix, since the files are symlinked flat into the working
  directory.
