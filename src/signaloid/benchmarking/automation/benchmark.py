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

import subprocess
import yaml
import os
import re
import datetime
import functools
import hashlib
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
)
from signaloid.benchmarking.automation.benchmarking_utils import (
    get_git_remote,
    expand_array_expressions,
)
from signaloid.benchmarking.automation.build import (
    compile_native,
    export_timing_env,
    run_timing_script,
)
from signaloid.benchmarking.automation.analysis import (
    load_emcc_data,
)
from signaloid.benchmarking.config import (
    RepresentationTypes,
    Correlations,
    DistanceMetrics,
    ReportingMethods,
    SignaloidYaml,
    VariableTypes,
    TimingFormat,
    DEFAULT_UXHW_SDK_PATH,
    DEFAULT_GROUND_TRUTH_SIZE,
    DEFAULT_ADVERSARY_MC_SIZE,
    DEFAULT_MAX_NUM_WEIGHTED_SAMPLES,
    DEFAULT_NUM_ADVERSARIES,
    DEFAULT_MAX_JUPITER_SIZE,
    DEFAULT_ADVERSARY_MAX_SIZE_SCALAR,
)


class Benchmark:
    """
    Pipeline coordinator for the benchmarking automation tool.

    Holds the resolved configuration for a single application benchmark (paths
    to the application, the UxHw SDK and PIN, the representation types/sizes and
    correlation modes under test, ground-truth and adversary sample sizes, the
    distance metric, and reporting options) and threads it through the
    multi-stage pipeline. The per-stage work (compilation, database generation,
    sample generation, measurement loading, EMCC and UxHw-distance analysis, and
    report writing) lives in the focused ``build``, ``database_generator``,
    ``sample_generator``, ``measurement_loader``, ``analysis`` and
    ``report_writer`` modules. This class owns no per-stage logic itself and
    simply orchestrates those steps.
    """

    def __init__(
        self,
        path_to_application: str,
        path_to_uxhw_sdk: str = DEFAULT_UXHW_SDK_PATH,
        path_to_pin: str | None = None,
        has_analytic_ground_truth: bool = False,
        path_to_ground_truth_file: str = "",
        ground_truth_size: int = DEFAULT_GROUND_TRUTH_SIZE,
        adversary_mc_size: int = DEFAULT_ADVERSARY_MC_SIZE,
        adversary_max_size_scalar: int = DEFAULT_ADVERSARY_MAX_SIZE_SCALAR,
        representation_types: list[str] = [
            RepresentationTypes.ATHENS,
        ],
        representation_sizes: list[int] = [16, 32, 64, 128, 256, 512],
        max_jupiter_size: int = DEFAULT_MAX_JUPITER_SIZE,
        correlations: list[str] = [
            Correlations.DISABLED,
            Correlations.AUTOCORRELATION,
        ],
        ground_truth_type: str = RepresentationTypes.MONTE_CARLO,
        max_num_weighted_samples: int = DEFAULT_MAX_NUM_WEIGHTED_SAMPLES,
        num_parallel_workers: int | None = None,
        distance_type: str = DistanceMetrics.WASSERSTEIN_1,
        use_clt: bool = False,
        reporting_methods: list[str] = [
            ReportingMethods.MEAN,
            ReportingMethods.QUANTILE_95,
            ReportingMethods.QUANTILE_99,
        ],
        n_adversaries: int = DEFAULT_NUM_ADVERSARIES,
        use_binned_uxhw: bool = True,
        demo_cli_args: str = "",
        google_credentials: str | None = None,
    ) -> None:
        """
        Store the resolved benchmark configuration.

        See the class docstring for the configuration groups. Each argument
        sets the correspondingly-named attribute.
        """
        self.path_to_application = os.path.expanduser(path_to_application)
        self.path_to_uxhw_sdk = os.path.expanduser(path_to_uxhw_sdk)
        self.path_to_pin = os.path.expanduser(path_to_pin) if path_to_pin else None
        self.has_analytic_ground_truth = has_analytic_ground_truth
        if self.has_analytic_ground_truth:
            self.path_to_ground_truth_file = os.path.expanduser(
                path_to_ground_truth_file
            )
        self.ground_truth_size = ground_truth_size
        self.adversary_mc_size = adversary_mc_size
        self.adversary_max_size_scalar = adversary_max_size_scalar
        self.all_outputs_cla: str = ""
        self.representation_types = representation_types
        self.representation_sizes = representation_sizes
        self.correlations = correlations
        self.cwd = os.path.abspath(os.getcwd())
        self.results_dir = os.path.join(self.cwd, "results")
        self.logs_dir = os.path.join(self.cwd, "logs")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.ground_truth_type = ground_truth_type
        self.max_num_weighted_samples = max_num_weighted_samples
        # None until get_machine_info resolves it to the detected core count
        # All readers run after that, by which point it is an int.
        self.num_parallel_workers: int | None = num_parallel_workers
        self.output_data_file = os.path.join(self.results_dir, "output_data.csv")
        self.asymptotic_dist_file = os.path.join(
            self.results_dir, "asymptotic_distances.csv"
        )
        self.uxhw_distance_file = os.path.join(self.results_dir, "uxhw_distances.csv")
        self.distance_type = distance_type
        self.use_clt = use_clt
        self.reporting_methods = reporting_methods
        self.variable_types: list[str] = []
        self.n_adversaries = n_adversaries
        self.use_binned_uxhw = use_binned_uxhw
        self.demo_cli_args = demo_cli_args
        self.google_credentials = google_credentials
        self.benchmarking_variables: list[BenchmarkingVariable] = []
        self.max_jupiter_size = max_jupiter_size
        # False until compile_native() sets it True on a successful native-MC
        # build (via Makefile or config.mk).
        self.has_native_mc = False
        # Populated by `load_measurement_dicts` (in `measurement_loader`)
        # via the orchestrator. Consumed by the Google-Sheets writer.
        self.uxhw_version: str = ""

    def get_machine_info(self) -> None:
        """
        Detect the machine model and CPU count, and resolve the parallel-worker
        count.
        """

        result = subprocess.check_output(["lscpu"], text=True)

        # Get the name of the computer model
        match = re.search(r"Model name:\s+(.+)", result)
        if match:
            self.machine_name = match.group(1).strip()
        else:
            self.machine_name = "Unknown"

        # Get the number of CPUs
        match = re.search(r"CPU\(s\)\s*:\s*(.+)", result)
        if match:
            self.n_processors = int(match.group(1).strip())
        else:
            self.n_processors = 1

        # Resolve an unset -j to the detected core count and clamp any request
        # above it.

        num_parallel_workers = (
            self.n_processors
            if self.num_parallel_workers is None
            else self.num_parallel_workers
        )

        if num_parallel_workers > self.n_processors:
            print(
                f"Warning! Requested {num_parallel_workers} parallel workers but only {self.n_processors} detected!"
            )
            num_parallel_workers = self.n_processors

        self.num_parallel_workers = num_parallel_workers

    def load_yaml(self) -> None:
        """
        Parse the application's signaloid.yaml into benchmarking variables.
        """

        self.benchmarking_variables = []
        program = "main"

        signaloid_yaml_path = self.path_to_application + "/signaloid.yaml"
        with open(signaloid_yaml_path, "r") as yaml_file:
            yaml_list = yaml.safe_load(yaml_file)

            # Get trace variable info for file path and line number
            trace_variables = yaml_list[SignaloidYaml.TRACE_VARIABLES]
            trace_variables = expand_array_expressions(trace_variables)

            if SignaloidYaml.BENCHMARKING_VARIABLES in yaml_list:
                if SignaloidYaml.ALL_OUTPUTS not in yaml_list:
                    raise ValueError(
                        f"signaloid.yaml at '{signaloid_yaml_path}' defines "
                        f"'{SignaloidYaml.BENCHMARKING_VARIABLES}' but is missing "
                        f"'{SignaloidYaml.ALL_OUTPUTS}'. An all-outputs invocation "
                        "is required to run combined-output measurements."
                    )
                all_outputs = yaml_list[SignaloidYaml.ALL_OUTPUTS]
                self.all_outputs_cla = all_outputs[0][
                    SignaloidYaml.COMMAND_LINE_ARGUMENTS
                ]
                full_cla = f"{self.demo_cli_args} {self.all_outputs_cla}".strip()
                self.value_id = hashlib.md5(full_cla.encode()).hexdigest()

                # Get the list of BenchmarkingVariable entries
                benchmarking_variables = yaml_list[SignaloidYaml.BENCHMARKING_VARIABLES]

                for var_config in benchmarking_variables:
                    variable_name = var_config[SignaloidYaml.VARIABLE_NAME]
                    trace_variable = next(
                        (
                            elem
                            for elem in trace_variables
                            if elem.get(SignaloidYaml.EXPRESSION) == variable_name
                        ),
                        None,
                    )
                    if trace_variable is None:
                        raise ValueError(
                            f"Benchmarking variable '{variable_name}' has no matching traced variable!"
                        )
                    file_name = trace_variable[SignaloidYaml.FILE]
                    line_number = trace_variable[SignaloidYaml.LINE_NUMBER]
                    path = f"{self.path_to_application}/src/{file_name}"

                    variable_description = var_config[
                        SignaloidYaml.VARIABLE_DESCRIPTION
                    ]
                    variable_type = var_config[SignaloidYaml.OUTPUT_OBJECT]
                    command_line_args = var_config[SignaloidYaml.COMMAND_LINE_ARGUMENTS]

                    # Create BenchmarkingVariable object
                    variable = BenchmarkingVariable(
                        value_id=self.value_id,
                        name=variable_name,
                        description=variable_description,
                        program=program,
                        path=path,
                        line_number=line_number,
                        file_name=file_name,
                        type=variable_type,
                        cla=command_line_args,
                    )
                    self.benchmarking_variables.append(variable)
            else:
                print(
                    "Old Yaml format detected. Falling back to using trace variables as benchmarking variables."
                )
                self.all_outputs_cla = f"-S {len(trace_variables)}"
                full_cla = f"{self.demo_cli_args} {self.all_outputs_cla}".strip()
                self.value_id = hashlib.md5(full_cla.encode()).hexdigest()
                for i, elem in enumerate(trace_variables):
                    file_name = elem[SignaloidYaml.FILE]
                    line_number = elem[SignaloidYaml.LINE_NUMBER]
                    path = f"{self.path_to_application}/src/{file_name}"
                    variable_name = elem[SignaloidYaml.EXPRESSION]
                    # Create BenchmarkingVariable object
                    variable = BenchmarkingVariable(
                        value_id=self.value_id,
                        name=variable_name,
                        description=f"Variable {i}",
                        program=program,
                        path=path,
                        line_number=line_number,
                        file_name=file_name,
                        type=VariableTypes.DISTRIBUTION,
                        cla=f"-S {i}",
                    )
                    self.benchmarking_variables.append(variable)

    @staticmethod
    def _resolve_application_version(path_to_application: str) -> str:
        """
        Resolve a non-empty version string for an application path.

        Uses the git short hash when the path is a git repository, and
        falls back to a date stamp otherwise (or when ``git rev-parse``
        fails). The fallback guarantees a non-empty value, which
        ``get-timings.sh`` requires via ``require_var APPLICATION_VERSION``.

        Args:
            path_to_application: Path to the application repository.

        Returns:
            The git short hash, or a date stamp when unavailable.
        """
        if os.path.isdir(os.path.join(path_to_application, ".git")):
            try:
                return subprocess.check_output(
                    [
                        "git",
                        "-C",
                        path_to_application,
                        "rev-parse",
                        "--short=7",
                        "HEAD",
                    ],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                print(
                    "Error retrieving git hash. "
                    "Falling back to date-based versioning."
                )
        else:
            print(
                "Warning! Application path is not a git repository. "
                "Falling back to date-based versioning."
            )
        return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def get_application_info(self) -> None:
        """
        Resolve application identity and prepare the benchmark's inputs.

        Resolves the application name/version, loads signaloid.yaml, compiles
        the native-MC build, creates the output directories, and derives the
        database paths and shared timing environment.
        """
        self.application_version = self._resolve_application_version(
            self.path_to_application
        )
        application_name = [
            part for part in self.path_to_application.split("/") if part
        ][-1]
        self.application_name = application_name.replace("Signaloid-Demo-", "")

        # Load the signaloid.yaml file
        self.load_yaml()

        # Attempt native compilation
        if self.num_parallel_workers is None:
            raise RuntimeError(
                "num_parallel_workers is unset; call get_machine_info() "
                "before get_application_info()."
            )
        num_parallel_workers = self.num_parallel_workers
        (
            self.native_executable_name,
            self.native_compilation_command,
            self.native_executable_dir,
            self.has_native_mc,
        ) = compile_native(
            path_to_application=self.path_to_application,
            num_parallel_workers=num_parallel_workers,
        )

        # Obtain git remote
        self.git_repo_remote = get_git_remote(self.path_to_application)

        ground_truth_prefix = adversary_prefix = self.application_name

        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Define database paths
        self.tracing_db_path = os.path.join(
            self.results_dir,
            f"uxhwExecutionStatistics-{self.application_version}-{self.value_id}-tracing.db",
        )
        if self.has_native_mc:
            ground_truth_prefix += "-nativeMC"
            adversary_prefix += "-nativeMC"
        else:
            ground_truth_prefix += "-uxhwExecutionStatistics"
            adversary_prefix += "-uxhwExecutionStatistics"

        if self.has_analytic_ground_truth:
            ground_truth_prefix += "-analytic"

        self.ground_truth_db_path = os.path.join(
            self.results_dir,
            f"{ground_truth_prefix}-{self.application_version}-{self.value_id}-ground-{self.ground_truth_size}.db",
        )
        self.adversary_db_path = os.path.join(
            self.results_dir,
            f"{adversary_prefix}-{self.application_version}-{self.value_id}-adv-{self.adversary_mc_size}.db",
        )

        # Export common timing environment variables
        self.tracing_db_path = export_timing_env(
            path_to_uxhw_sdk=self.path_to_uxhw_sdk,
            path_to_pin=self.path_to_pin,
            path_to_application=self.path_to_application,
            application_name=self.application_name,
            application_version=self.application_version,
            max_jupiter_size=self.max_jupiter_size,
            results_dir=self.results_dir,
            logs_dir=self.logs_dir,
            tracing_db_path=self.tracing_db_path,
        )

    def intermediate_timings_path(self) -> str:
        """
        Path of the transient intermediate timings file.

        Returns:
            The absolute path of the intermediate file that bash writes
            line-by-line and Python parses into the canonical JSON artifact.
        """
        filename = (
            f"{self.application_name}-{self.application_version}"
            f"{TimingFormat.INTERMEDIATE_SUFFIX}"
        )
        return os.path.join(self.results_dir, filename)

    def json_timings_path(self) -> str:
        """
        Path of the canonical JSON timings artifact.

        Returns:
            The absolute path of the JSON artifact produced by the Python
            reader.
        """
        filename = (
            f"{self.application_name}-{self.application_version}"
            f"{TimingFormat.JSON_SUFFIX}"
        )
        return os.path.join(self.results_dir, filename)

    def bind_timing_script(self) -> "functools.partial[None]":
        """
        Bind the run_timing_script kwargs shared by every timing pass.

        Every caller (UxHw timing, native-MC timing, and the tracing /
        adversary / ground-truth database generators) passes the same eight
        kwargs sourced from this Benchmark.

        Returns:
            A partial that still needs the per-call arguments: the
            ``variable_index`` and exactly one mode flag (``timing`` /
            ``native_mc_timing`` / ``tracing`` / ``adversary_mc`` /
            ``ground_truth``).
        """
        return functools.partial(
            run_timing_script,
            all_outputs_cla=self.all_outputs_cla,
            benchmarking_variables=self.benchmarking_variables,
            demo_cli_args=self.demo_cli_args,
            representation_types=self.representation_types,
            representation_sizes=self.representation_sizes,
            correlations=self.correlations,
            logs_dir=self.logs_dir,
            intermediate_timings_path=self.intermediate_timings_path(),
        )

    def generate_uxhw_timings(self) -> None:
        """
        Generate the UxHw timing data for every benchmarking variable.
        """
        print("Generating timing database.")
        bound_run_timing_script = self.bind_timing_script()
        for i in range(len(self.benchmarking_variables)):
            bound_run_timing_script(variable_index=i, timing=True)

    def generate_native_mc_timings(self) -> None:
        """
        Generate the native Monte Carlo timing data.

        If EMCC data is not found on the Benchmark object, attempts to load it
        from the equivalent-MC output file and populate the benchmarking
        variables.
        """

        load_emcc_data(
            benchmarking_variables=self.benchmarking_variables,
            output_data_file=self.output_data_file,
        )

        print("Generating native Monte Carlo timing data")
        bound_run_timing_script = self.bind_timing_script()
        for i, var in enumerate(self.benchmarking_variables):
            bound_run_timing_script(variable_index=i, native_mc_timing=True)
