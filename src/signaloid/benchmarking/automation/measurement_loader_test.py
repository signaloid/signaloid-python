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
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signaloid.benchmarking.automation.measurement_loader import (
    load_asymptotic_dist,
    load_measurement_data,
    load_measurement_dicts,
    load_timing_data_to_dfs,
    load_uxhw_distances,
)
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    UxhwDistanceRecord,
)
from signaloid.benchmarking.config import (
    AsymptoticDistanceDistribution,
    BenchmarkingVariables,
    EquivMC,
    RepresentationTypes,
    ReportingMethods,
    TimingFormat,
    VariableTypes,
)


def _make_variable(
    name: str,
    description: str,
    *,
    cla: str = "",
    var_type: str = VariableTypes.DISTRIBUTION,
) -> BenchmarkingVariable:
    """Create a minimal BenchmarkingVariable for tests."""
    return BenchmarkingVariable(
        name=name,
        description=description,
        cla=cla,
        type=var_type,
    )


class _ConfigLikeObject:
    """Non-string config-like object: __repr__ returns the bare config
    text (matching Distribution.__repr__ in the real pipeline). Used to
    exercise the repr()-based normalization path in load_timing_data_to_dfs."""

    def __init__(self, config: str) -> None:
        self._config = config

    def __repr__(self) -> str:
        return self._config


class TestLoadMeasurementDicts(unittest.TestCase):
    """Exercise load_measurement_dicts (intermediate parse + JSON write)."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_load_measurement_dicts_writes_json_and_returns_versions(
        self,
    ) -> None:
        """End-to-end happy path: parse intermediate, write JSON, return
        versions."""
        variable = _make_variable(
            name="x",
            description="x var",
            cla="-S 0",
        )
        intermediate_path = self.tmp_path / "app-v1-timings.intermediate"
        json_path = self.tmp_path / "app-v1-timings.json"
        logs_dir = self.tmp_path / "logs"
        logs_dir.mkdir()

        intermediate_path.write_text(
            "META timestamp 2026-04-15T14:22:10Z\n"
            "META applicationName app\n"
            "META applicationVersion v1\n"
            "META uxhwSdkVersion uxhw-Y\n"
            "META commandLineArguments -T -S 0\n"
            "META commandLineArgumentsHash abc\n"
            "META uxhwTargetRepetitions 1\n"
            "MEASUREMENT Athens-16 1.0 2.0 3.0 100 200\n"
            "META timestamp 2026-04-15T14:25:00Z\n"
            "META applicationName app\n"
            "META applicationVersion v1\n"
            "META uxhwSdkVersion uxhw-Y\n"
            "META commandLineArguments -S 0\n"
            "META commandLineArgumentsHash abc\n"
            "META uxhwTargetRepetitions 1\n"
        )

        uxhw_version = load_measurement_dicts(
            benchmarking_variables=[variable],
            intermediate_path=str(intermediate_path),
            json_path=str(json_path),
            logs_dir=str(logs_dir),
            demo_cli_args="",
            representation_sizes=[16],
            representation_types=[RepresentationTypes.ATHENS],
            correlations=["Disabled"],
        )

        self.assertEqual(uxhw_version, "uxhw-Y")
        self.assertTrue(json_path.exists())
        # Intermediate is removed on successful parse+write.
        self.assertFalse(intermediate_path.exists())

        document = json.loads(json_path.read_text())
        # Session-level META lifted to top level, runs array preserved.
        self.assertEqual(document[TimingFormat.META_KEY_UXHW_SDK_VERSION], "uxhw-Y")
        self.assertIn(TimingFormat.JSON_KEY_RUNS, document)
        self.assertEqual(len(document[TimingFormat.JSON_KEY_RUNS]), 2)

        # Variable was populated via internal call to load_measurement_data.
        measurement_dict = variable.timing_measurements.measurement_dict
        self.assertIn("Athens-16", measurement_dict)

    def test_load_measurement_dicts_missing_intermediate_raises(self) -> None:
        """Missing intermediate file raises RuntimeError pointing at the
        log."""
        variable = _make_variable(name="x", description="x var")
        intermediate_path = self.tmp_path / "missing.intermediate"
        json_path = self.tmp_path / "out.json"
        logs_dir = self.tmp_path / "logs"
        logs_dir.mkdir()

        with self.assertRaises(RuntimeError) as exc_info:
            load_measurement_dicts(
                benchmarking_variables=[variable],
                intermediate_path=str(intermediate_path),
                json_path=str(json_path),
                logs_dir=str(logs_dir),
                demo_cli_args="",
                representation_sizes=[16],
                representation_types=[RepresentationTypes.ATHENS],
                correlations=["Disabled"],
            )

        message = str(exc_info.exception)
        self.assertIn("missing.intermediate", message)
        self.assertIn("timing_script_stderr.log", message)

    def test_load_measurement_dicts_prefers_first_non_empty_meta(self) -> None:
        """When META values disagree across runs, the first non-empty value
        in run order wins (deterministic; not arbitrary set iteration)."""
        variable = _make_variable(name="x", description="x var", cla="")
        intermediate_path = self.tmp_path / "app-v1-timings.intermediate"
        json_path = self.tmp_path / "app-v1-timings.json"
        logs_dir = self.tmp_path / "logs"
        logs_dir.mkdir()

        # Two runs disagree on uxhwSdkVersion: first is "uxhw-A",
        # second is "uxhw-B". The first non-empty value in run order
        # should win.
        intermediate_path.write_text(
            "META timestamp 2026-04-15T14:22:10Z\n"
            "META applicationName app\n"
            "META applicationVersion v1\n"
            "META uxhwSdkVersion uxhw-A\n"
            "META commandLineArguments -T\n"
            "META commandLineArgumentsHash abc\n"
            "META uxhwTargetRepetitions 1\n"
            "MEASUREMENT Athens-16 1.0 2.0 3.0 100 200\n"
            "META timestamp 2026-04-15T14:25:00Z\n"
            "META applicationName app\n"
            "META applicationVersion v1\n"
            "META uxhwSdkVersion uxhw-B\n"
            "META commandLineArguments \n"
            "META commandLineArgumentsHash abc\n"
            "META uxhwTargetRepetitions 1\n"
        )

        uxhw_version = load_measurement_dicts(
            benchmarking_variables=[variable],
            intermediate_path=str(intermediate_path),
            json_path=str(json_path),
            logs_dir=str(logs_dir),
            demo_cli_args="",
            representation_sizes=[16],
            representation_types=[RepresentationTypes.ATHENS],
            correlations=["Disabled"],
        )

        # First non-empty value in run order wins. Not the second run's
        # value, and not any arbitrary set-iteration pick.
        self.assertEqual(uxhw_version, "uxhw-A")


class TestLoadMeasurementData(unittest.TestCase):
    """Exercise load_measurement_data run-to-variable routing."""

    def test_load_measurement_data_populates_timing_measurements(self) -> None:
        """UxHw and Native-MC measurements are routed to the right
        variable."""
        variable = _make_variable(name="x", description="x var", cla="-S 0")
        variable.emcc_results.equiv_mc_list = [50]

        runs = [
            {
                TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS: "-T -S 0",
                TimingFormat.JSON_KEY_MEASUREMENTS: [
                    {
                        TimingFormat.JSON_KEY_MEASUREMENT_CONFIG: ("Athens-16"),
                        TimingFormat.JSON_KEY_MEASUREMENT_TIME: 1.0,
                        TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME: 2.0,
                        TimingFormat.JSON_KEY_MEASUREMENT_E2E_TIME: 3.0,
                        TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT: (100.0),
                        TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT: (200.0),
                    },
                ],
            },
            {
                TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS: "-S 0",
                TimingFormat.JSON_KEY_MEASUREMENTS: [
                    {
                        TimingFormat.JSON_KEY_MEASUREMENT_CONFIG: "Native-MC-50",
                        TimingFormat.JSON_KEY_MEASUREMENT_TIME: 0.5,
                        TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME: None,
                        TimingFormat.JSON_KEY_MEASUREMENT_E2E_TIME: 1.5,
                        TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT: (None),
                        TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT: (300.0),
                    },
                ],
            },
        ]

        load_measurement_data(
            benchmarking_variables=[variable],
            runs=runs,
            demo_cli_args="",
            representation_sizes=[16],
            representation_types=[RepresentationTypes.ATHENS],
            correlations=["Disabled"],
        )

        measurement_dict = variable.timing_measurements.measurement_dict
        uxhw = measurement_dict["Athens-16"]
        self.assertEqual(uxhw["In Application Time"], 1.0)
        self.assertEqual(uxhw["Database Time"], 2.0)
        self.assertEqual(uxhw["End-to-End Time"], 3.0)
        self.assertEqual(uxhw["Database Dyn. Inst. Count"], 100.0)
        self.assertEqual(uxhw["PIN Dyn. Inst. Count"], 200.0)

        native = measurement_dict["Native-MC-50"]
        self.assertEqual(native["In Application Time"], 0.5)
        self.assertEqual(native["End-to-End Time"], 1.5)
        self.assertEqual(native["PIN Dyn. Inst. Count"], 300.0)


class TestLoadTimingDataToDfs(unittest.TestCase):
    """Exercise load_timing_data_to_dfs joining timings into emcc_data."""

    def test_load_timing_data_to_dfs_joins_measurements_into_emcc_data(
        self,
    ) -> None:
        """Native and UxHw timings are merged into emcc_data records."""
        variable = _make_variable(name="x", description="x var")
        variable.emcc_results.emcc_data = [
            {
                BenchmarkingVariables.UXHW_CONF: "Athens-16",
                EquivMC.EMCC: 50,
            },
        ]
        variable.timing_measurements.append(
            config="Athens-16",
            time=1.0,
            e2e_time=3.0,
            pin_dyn_inst_count=200.0,
            db_time=2.0,
            db_dyn_inst_count=100.0,
        )
        variable.timing_measurements.append(
            config="Native-MC-50",
            time=0.5,
            e2e_time=1.5,
            pin_dyn_inst_count=300.0,
        )

        load_timing_data_to_dfs(benchmarking_variables=[variable])

        record = variable.emcc_results.emcc_data[0]
        # UxHw columns merged without prefix.
        self.assertEqual(record["In Application Time"], 1.0)
        self.assertEqual(record["Database Time"], 2.0)
        # Native columns merged with "Native " prefix.
        self.assertEqual(record["Native In Application Time"], 0.5)
        self.assertEqual(record["Native End-to-End Time"], 1.5)

    def test_load_timing_data_to_dfs_raises_when_emcc_data_empty(self) -> None:
        """If every variable has empty emcc_data, the function must raise
        rather than silently producing unjoined records (signals missing
        analysis.load_emcc_data pre-call)."""
        variable = _make_variable(name="x", description="x var")
        # Populate timing_measurements but leave emcc_data empty.
        variable.timing_measurements.append(
            config="Athens-16",
            time=1.0,
            e2e_time=3.0,
            pin_dyn_inst_count=200.0,
            db_time=2.0,
            db_dyn_inst_count=100.0,
        )

        with self.assertRaises(RuntimeError) as exc_info:
            load_timing_data_to_dfs(benchmarking_variables=[variable])

        self.assertIn("load_emcc_data", str(exc_info.exception))

    def test_load_timing_data_to_dfs_matches_distribution_repr(self) -> None:
        """UxHw records whose UXHW_CONF is a non-string object must still
        match via repr()-based normalization."""
        variable = _make_variable(name="y", description="y var")
        variable.emcc_results.emcc_data = [
            {
                BenchmarkingVariables.UXHW_CONF: _ConfigLikeObject("Athens-16"),
                EquivMC.EMCC: 50,
            },
        ]
        variable.timing_measurements.append(
            config="Athens-16",
            time=4.0,
            e2e_time=5.0,
            pin_dyn_inst_count=600.0,
            db_time=2.0,
            db_dyn_inst_count=100.0,
        )

        load_timing_data_to_dfs(benchmarking_variables=[variable])

        record = variable.emcc_results.emcc_data[0]
        self.assertEqual(record["In Application Time"], 4.0)
        self.assertEqual(record["End-to-End Time"], 5.0)
        self.assertEqual(record["PIN Dyn. Inst. Count"], 600.0)


class TestLoadAsymptoticDist(unittest.TestCase):
    """Exercise load_asymptotic_dist CSV/.npy population."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def _chdir_to_tmp(self) -> None:
        """chdir into the test's temp dir, restoring cwd on cleanup."""
        original_cwd = os.getcwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(self.tmp_path)

    def test_load_asymptotic_dist_populates_variable(self) -> None:
        """CSV records are loaded into the asymptotic_distribution
        dataclass per variable."""
        # The function loads ``<formatted_description>-asymptotic.npy`` from
        # the current working directory for DISTRIBUTION variables and isolate
        # to tmp.
        self._chdir_to_tmp()

        variable = _make_variable(
            name="x",
            description="x var",
            var_type=VariableTypes.DISTRIBUTION,
        )
        samples = np.array([0.1, 0.2, 0.3])
        np.save(f"{variable.formatted_description}-asymptotic.npy", samples)

        asymptotic_csv = self.tmp_path / "asymptotic_distances.csv"
        pd.DataFrame(
            [
                {
                    BenchmarkingVariables.VARIABLE_DESCRIPTION: "x var",
                    ReportingMethods.MEAN: 0.1,
                    ReportingMethods.QUANTILE_95: 0.5,
                    ReportingMethods.QUANTILE_99: 0.9,
                    EquivMC.MEAN_QUANTILE: 0.42,
                    AsymptoticDistanceDistribution.IS_NORMAL: True,
                    AsymptoticDistanceDistribution.SCALE: 0.05,
                }
            ]
        ).to_csv(asymptotic_csv, index=False)

        load_asymptotic_dist(
            benchmarking_variables=[variable],
            asymptotic_dist_file=str(asymptotic_csv),
        )

        asymptotic = variable.asymptotic_distribution
        assert asymptotic.samples is not None  # narrow for type checker
        np.testing.assert_array_equal(asymptotic.samples, samples)
        self.assertEqual(asymptotic.mean, 0.1)
        self.assertEqual(asymptotic.quantile_95, 0.5)
        self.assertEqual(asymptotic.quantile_99, 0.9)
        self.assertEqual(asymptotic.mean_quantile, 0.42)
        self.assertTrue(asymptotic.is_normal)
        self.assertEqual(asymptotic.scale, 0.05)

    def test_load_asymptotic_dist_skips_csv_when_all_populated(self) -> None:
        """If every variable already has
        ``asymptotic_distribution.quantile_95`` populated, the function
        must return without reading the CSV — so a non-existent path must
        not raise."""
        variable = _make_variable(name="x", description="x var")
        variable.asymptotic_distribution.quantile_95 = 0.5

        load_asymptotic_dist(
            benchmarking_variables=[variable],
            asymptotic_dist_file=str(self.tmp_path / "does-not-exist.csv"),
        )

        # Pre-existing data untouched. No FileNotFoundError despite the
        # CSV not existing.
        self.assertEqual(variable.asymptotic_distribution.quantile_95, 0.5)


class TestLoadUxhwDistances(unittest.TestCase):
    """Exercise load_uxhw_distances CSV population and intra-module wiring."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def _chdir_to_tmp(self) -> None:
        """chdir into the test's temp dir, restoring cwd on cleanup."""
        original_cwd = os.getcwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(self.tmp_path)

    def test_load_uxhw_distances_populates_variable(self) -> None:
        """CSV records are loaded into uxhw_distances.records per variable.

        Also covers the intra-module call to ``load_asymptotic_dist`` —
        pre-populating ``asymptotic_distribution.quantile_95``
        short-circuits that side branch so we can focus on the UxHw path
        here.
        """
        self._chdir_to_tmp()

        variable = _make_variable(
            name="x",
            description="x var",
            var_type=VariableTypes.DISTRIBUTION,
        )
        # Pre-populate asymptotic so the intra-module call short-circuits.
        variable.asymptotic_distribution.quantile_95 = 0.5

        uxhw_csv = self.tmp_path / "uxhw_distances.csv"
        pd.DataFrame(
            [
                {
                    BenchmarkingVariables.VARIABLE_DESCRIPTION: "x var",
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    BenchmarkingVariables.UXHW_DISTANCE: 0.123,
                    BenchmarkingVariables.UXHW_BINNED_DISTANCE: 0.234,
                }
            ]
        ).to_csv(uxhw_csv, index=False)

        asymptotic_csv = self.tmp_path / "asymptotic_distances.csv"
        # Empty CSV with required column so the asymptotic loader is a no-op.
        pd.DataFrame([{BenchmarkingVariables.VARIABLE_DESCRIPTION: "ignored"}]).to_csv(
            asymptotic_csv, index=False
        )

        load_uxhw_distances(
            benchmarking_variables=[variable],
            uxhw_distance_file=str(uxhw_csv),
            asymptotic_dist_file=str(asymptotic_csv),
        )

        self.assertEqual(len(variable.uxhw_distances.records), 1)
        record = variable.uxhw_distances.records[0]
        self.assertEqual(record.uxhw_conf, "Athens-16")
        self.assertEqual(record.uxhw_distance, 0.123)
        self.assertEqual(record.uxhw_binned_distance, 0.234)

    def test_load_uxhw_distances_normalises_missing_binned_to_none(
        self,
    ) -> None:
        """Empty binned-distance cells round-trip as ``None``, not NaN.

        pandas reads empty CSV cells as NaN. The loader normalises so
        callers can rely on the typed ``float | None`` contract.
        """
        self._chdir_to_tmp()

        variable = _make_variable(
            name="x",
            description="x var",
            var_type=VariableTypes.DISTRIBUTION,
        )
        variable.asymptotic_distribution.quantile_95 = 0.5

        uxhw_csv = self.tmp_path / "uxhw_distances.csv"
        pd.DataFrame(
            [
                {
                    BenchmarkingVariables.VARIABLE_DESCRIPTION: "x var",
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    BenchmarkingVariables.UXHW_DISTANCE: 0.123,
                    BenchmarkingVariables.UXHW_BINNED_DISTANCE: None,
                }
            ]
        ).to_csv(uxhw_csv, index=False)

        asymptotic_csv = self.tmp_path / "asymptotic_distances.csv"
        pd.DataFrame([{BenchmarkingVariables.VARIABLE_DESCRIPTION: "ignored"}]).to_csv(
            asymptotic_csv, index=False
        )

        load_uxhw_distances(
            benchmarking_variables=[variable],
            uxhw_distance_file=str(uxhw_csv),
            asymptotic_dist_file=str(asymptotic_csv),
        )

        record = variable.uxhw_distances.records[0]
        self.assertIsNone(record.uxhw_binned_distance)

    def test_load_uxhw_distances_invokes_load_asymptotic_dist(self) -> None:
        """Intra-module call wires asymptotic data into the same
        variables."""
        self._chdir_to_tmp()

        variable = _make_variable(
            name="x",
            description="x var",
            var_type=VariableTypes.SCALAR,
        )
        # SCALAR variable: load_asymptotic_dist skips the .npy load and just
        # reads the CSV.

        uxhw_csv = self.tmp_path / "uxhw_distances.csv"
        pd.DataFrame(
            [
                {
                    BenchmarkingVariables.VARIABLE_DESCRIPTION: "x var",
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    BenchmarkingVariables.UXHW_DISTANCE: 0.1,
                    BenchmarkingVariables.UXHW_BINNED_DISTANCE: 0.2,
                }
            ]
        ).to_csv(uxhw_csv, index=False)

        asymptotic_csv = self.tmp_path / "asymptotic_distances.csv"
        pd.DataFrame(
            [
                {
                    BenchmarkingVariables.VARIABLE_DESCRIPTION: "x var",
                    ReportingMethods.MEAN: 0.9,
                    ReportingMethods.QUANTILE_95: 0.95,
                    ReportingMethods.QUANTILE_99: 0.99,
                    EquivMC.MEAN_QUANTILE: 0.5,
                    AsymptoticDistanceDistribution.IS_NORMAL: False,
                    AsymptoticDistanceDistribution.SCALE: 0.7,
                }
            ]
        ).to_csv(asymptotic_csv, index=False)

        load_uxhw_distances(
            benchmarking_variables=[variable],
            uxhw_distance_file=str(uxhw_csv),
            asymptotic_dist_file=str(asymptotic_csv),
        )

        # Both data structures were populated.
        self.assertTrue(variable.uxhw_distances.records)
        self.assertEqual(variable.asymptotic_distribution.mean, 0.9)

    def test_load_uxhw_distances_skips_csv_when_all_populated(self) -> None:
        """If every variable already has uxhw_distances.records populated
        (and ``asymptotic_distribution.quantile_95``, since
        load_uxhw_distances calls load_asymptotic_dist intra-module
        first), neither CSV is read."""
        variable = _make_variable(name="x", description="x var")
        variable.asymptotic_distribution.quantile_95 = 0.5
        placeholder = UxhwDistanceRecord(
            uxhw_conf="placeholder",
            uxhw_distance=0.0,
        )
        variable.uxhw_distances.records = [placeholder]

        load_uxhw_distances(
            benchmarking_variables=[variable],
            uxhw_distance_file=str(self.tmp_path / "missing-uxhw.csv"),
            asymptotic_dist_file=str(self.tmp_path / "missing-asymptotic.csv"),
        )

        # Both stages left untouched. No FileNotFoundError despite both
        # CSV paths being bogus.
        self.assertEqual(variable.uxhw_distances.records, [placeholder])
        self.assertEqual(variable.asymptotic_distribution.quantile_95, 0.5)


if __name__ == "__main__":
    unittest.main()
