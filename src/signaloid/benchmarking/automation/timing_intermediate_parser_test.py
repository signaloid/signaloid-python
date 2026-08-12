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

import io
import json
import tempfile
import unittest
from pathlib import Path

from signaloid.benchmarking.automation.benchmarking_utils import (
    parse_timing_intermediate_stream,
)
from signaloid.benchmarking.config import TimingFormat

UXHW_RUN = """\
META timestamp 2026-04-15T14:22:10Z
META applicationName call-option
META applicationVersion a1b2c3d
META uxhwSdkVersion v4.5.6
META commandLineArguments -T 100 --strike 110
META commandLineArgumentsHash abc123
META uxhwTargetRepetitions 20
MEASUREMENT Athens-16-Autocorrelation 1.23 4.56 7.89 1000 2000
MEASUREMENT Athens-16 1.01 4.02 7.03 950 ?
MEASUREMENT Reference-50-1 0.5 1.0 1.5 100 ?
MEASUREMENT Native-50-1 0.6 ? 1.7 ? ?
"""

NATIVE_MC_RUN = """\
META timestamp 2026-04-15T14:24:55Z
META applicationName call-option
META applicationVersion a1b2c3d
META uxhwSdkVersion v4.5.6
META commandLineArguments 100 --strike 110
META commandLineArgumentsHash abc123
META uxhwTargetRepetitions 20
MEASUREMENT Native-MC-50 0.9 ? 2.3 ? ?
MEASUREMENT Native-MC-100 1.1 ? 2.7 ? ?
"""


class TestParseTimingIntermediateStream(unittest.TestCase):
    """Tests for parse_timing_intermediate_stream: run splitting and meta/measurement parsing."""

    def test_splits_runs_on_timestamp_meta(self) -> None:
        runs = parse_timing_intermediate_stream((UXHW_RUN + NATIVE_MC_RUN).splitlines())
        self.assertEqual(len(runs), 2)
        self.assertEqual(
            runs[0][TimingFormat.META_KEY_TIMESTAMP], "2026-04-15T14:22:10Z"
        )
        self.assertEqual(
            runs[1][TimingFormat.META_KEY_TIMESTAMP], "2026-04-15T14:24:55Z"
        )

    def test_meta_keys_attached_to_current_run(self) -> None:
        runs = parse_timing_intermediate_stream(UXHW_RUN.splitlines())
        run = runs[0]
        self.assertEqual(run[TimingFormat.META_KEY_APPLICATION_NAME], "call-option")
        self.assertEqual(run[TimingFormat.META_KEY_APPLICATION_VERSION], "a1b2c3d")
        self.assertEqual(run[TimingFormat.META_KEY_UXHW_SDK_VERSION], "v4.5.6")
        self.assertEqual(
            run[TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS], "-T 100 --strike 110"
        )
        self.assertEqual(
            run[TimingFormat.META_KEY_COMMAND_LINE_ARGUMENTS_HASH], "abc123"
        )
        self.assertEqual(run[TimingFormat.META_KEY_UXHW_TARGET_REPETITIONS], "20")

    def test_measurement_numerics_parsed_as_float(self) -> None:
        runs = parse_timing_intermediate_stream(UXHW_RUN.splitlines())
        first = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertEqual(
            first[TimingFormat.JSON_KEY_MEASUREMENT_CONFIG], "Athens-16-Autocorrelation"
        )
        self.assertEqual(first[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 1.23)
        self.assertEqual(first[TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME], 4.56)
        self.assertEqual(first[TimingFormat.JSON_KEY_MEASUREMENT_E2E_TIME], 7.89)
        self.assertEqual(
            first[TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT], 1000.0
        )
        self.assertEqual(
            first[TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT], 2000.0
        )

    def test_missing_value_becomes_none(self) -> None:
        runs = parse_timing_intermediate_stream(UXHW_RUN.splitlines())
        measurements = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS]

        reference = next(
            m
            for m in measurements
            if m[TimingFormat.JSON_KEY_MEASUREMENT_CONFIG] == "Reference-50-1"
        )
        self.assertIsNone(
            reference[TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT]
        )

        native = next(
            m
            for m in measurements
            if m[TimingFormat.JSON_KEY_MEASUREMENT_CONFIG] == "Native-50-1"
        )
        self.assertIsNone(native[TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME])
        self.assertIsNone(native[TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT])

    def test_malformed_measurement_lines_are_skipped(self) -> None:
        fixture = """\
META timestamp 2026-04-15T14:22:10Z
MEASUREMENT too few tokens here
MEASUREMENT config 1 2 3 4 5
MEASUREMENT way too many tokens 1 2 3 4 5 6 7
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurements = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS]
        self.assertEqual(len(measurements), 1)
        self.assertEqual(
            measurements[0][TimingFormat.JSON_KEY_MEASUREMENT_CONFIG], "config"
        )

    def test_blank_lines_are_ignored(self) -> None:
        fixture = UXHW_RUN.replace(
            "META applicationVersion a1b2c3d",
            "\nMETA applicationVersion a1b2c3d\n",
        )
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][TimingFormat.META_KEY_APPLICATION_VERSION], "a1b2c3d")

    def test_accepts_file_like_iterable(self) -> None:
        stream = io.StringIO(UXHW_RUN)
        runs = parse_timing_intermediate_stream(stream)
        self.assertEqual(len(runs), 1)

    def test_parsed_run_is_json_serialisable(self) -> None:
        runs = parse_timing_intermediate_stream((UXHW_RUN + NATIVE_MC_RUN).splitlines())
        for run in runs:
            line = json.dumps(run)
            self.assertEqual(json.loads(line), run)


# ---------------------------------------------------------------------------
# SAMPLE tag tests — value-typed elapsedTime variant
# ---------------------------------------------------------------------------

SAMPLE_TIME_RUN = """\
META timestamp 2026-04-15T14:30:00Z
META applicationName call-option
META applicationVersion a1b2c3d
META uxhwSdkVersion v4.5.6
META commandLineArguments 50 --strike 110
META commandLineArgumentsHash abc123
META uxhwTargetRepetitions 3
SAMPLE Native-MC-50 elapsedTime 1 1.0
SAMPLE Native-MC-50 elapsedTime 2 2.0
SAMPLE Native-MC-50 elapsedTime 3 3.0
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
"""


class TestSampleElapsedTime(unittest.TestCase):
    """Tests for value-typed elapsedTime SAMPLE resolution."""

    def test_sample_time_resolves_average_from_question_mark(self) -> None:
        """
        A MEASUREMENT with `?` in the time slot is resolved to the mean of
        the preceding elapsedTime SAMPLE values for the same config.
        """
        runs = parse_timing_intermediate_stream(SAMPLE_TIME_RUN.splitlines())
        self.assertEqual(len(runs), 1)
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_CONFIG], "Native-MC-50"
        )
        # mean of 1.0, 2.0, 3.0 is 2.0
        self.assertAlmostEqual(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 2.0)

    def test_sample_time_non_missing_time_slot_is_not_overridden(self) -> None:
        """
        When the time slot in a MEASUREMENT is an explicit float (not `?`),
        SAMPLE lines for the same config must not override it.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Native-MC-50 elapsedTime 1 1.0
SAMPLE Native-MC-50 elapsedTime 2 9.0
MEASUREMENT Native-MC-50 4.0 ? 5.0 ? 100
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertAlmostEqual(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 4.0)

    def test_sample_time_missing_samples_yields_none(self) -> None:
        """
        A MEASUREMENT with `?` in the time slot and no matching SAMPLE lines
        must leave the time field as None.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertIsNone(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME])

    def test_sample_time_malformed_value_tokens_are_skipped(self) -> None:
        """
        Non-numeric value tokens in elapsedTime SAMPLE lines are silently
        discarded. Only valid float tokens contribute to the average.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Native-MC-50 elapsedTime 1 not_a_number
SAMPLE Native-MC-50 elapsedTime 2 6.0
SAMPLE Native-MC-50 elapsedTime 3 bad
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        # Only the valid token (6.0) contributes — mean is 6.0.
        self.assertAlmostEqual(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 6.0)

    def test_sample_time_all_malformed_values_yield_none(self) -> None:
        """
        When every elapsedTime SAMPLE token is non-numeric the resolved
        time must be None rather than raising an exception.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Native-MC-50 elapsedTime 1 bad
SAMPLE Native-MC-50 elapsedTime 2 also_bad
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertIsNone(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME])

    def test_sample_time_malformed_sample_lines_too_few_tokens_skipped(self) -> None:
        """
        SAMPLE lines with fewer than four tokens are silently skipped and
        do not affect the resolved average.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Native-MC-50 elapsedTime 4.0
SAMPLE Native-MC-50 elapsedTime 1 8.0
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        # Only the well-formed line (8.0) contributes.
        self.assertAlmostEqual(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 8.0)

    def test_sample_time_samples_scoped_to_run(self) -> None:
        """
        SAMPLE lines from a previous run must not bleed into the next run's
        MEASUREMENT resolution.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Native-MC-50 elapsedTime 1 99.0
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
META timestamp 2026-04-15T14:31:00Z
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        self.assertEqual(len(runs), 2)
        first = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        second = runs[1][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertAlmostEqual(first[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 99.0)
        self.assertIsNone(second[TimingFormat.JSON_KEY_MEASUREMENT_TIME])

    def test_sample_time_config_isolation(self) -> None:
        """
        SAMPLE lines for one config must not affect the resolved time of a
        different config in the same run.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Native-MC-50 elapsedTime 1 4.0
MEASUREMENT Native-MC-50 ? ? 5.0 ? 100
MEASUREMENT Native-MC-100 ? ? 6.0 ? 200
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurements = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS]
        mc_50 = next(
            m
            for m in measurements
            if m[TimingFormat.JSON_KEY_MEASUREMENT_CONFIG] == "Native-MC-50"
        )
        mc_100 = next(
            m
            for m in measurements
            if m[TimingFormat.JSON_KEY_MEASUREMENT_CONFIG] == "Native-MC-100"
        )
        self.assertAlmostEqual(mc_50[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 4.0)
        self.assertIsNone(mc_100[TimingFormat.JSON_KEY_MEASUREMENT_TIME])

    def test_sample_value_typed_summed_within_iteration(self) -> None:
        """
        Multiple value-typed SAMPLE lines for the same iteration index are
        summed before averaging across iterations.

        `run_native_benchmarks` relies on this: it emits two ``elapsedTime``
        SAMPLEs per iteration (program runtime + framework overhead) and
        expects the parser to sum them within the iteration before
        averaging across iterations — mirroring the pre-refactor
        `prog + overhead` semantics that fed `compute_average`.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Native-50-1 elapsedTime 0 1.5
SAMPLE Native-50-1 elapsedTime 0 0.3
SAMPLE Native-50-1 elapsedTime 1 2.0
SAMPLE Native-50-1 elapsedTime 1 0.4
MEASUREMENT Native-50-1 ? ? 5.0 ? 100
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        # iter 0: 1.5 + 0.3 = 1.8; iter 1: 2.0 + 0.4 = 2.4; mean = 2.1
        self.assertAlmostEqual(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 2.1)


# ---------------------------------------------------------------------------
# SAMPLE tag tests — path-typed pinDynInstCount variant
# ---------------------------------------------------------------------------


class TestSamplePinDynInstCount(unittest.TestCase):
    """Tests for path-typed pinDynInstCount SAMPLE resolution."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_sample_lines_resolved_to_averaged_pin_inst(self) -> None:
        """SAMPLE lines are read and averaged into pinDynInstCount on MEASUREMENT."""
        sample_0 = self.tmp_path / "inscount-cfg-0.out"
        sample_1 = self.tmp_path / "inscount-cfg-1.out"
        sample_0.write_text("Count 1000\n")
        sample_1.write_text("Count 3000\n")

        fixture = f"""\
META timestamp 2026-04-15T14:22:10Z
SAMPLE cfg pinDynInstCount 0 {sample_0}
SAMPLE cfg pinDynInstCount 1 {sample_1}
MEASUREMENT cfg 1.0 ? 2.0 ? ?
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT], 2000.0
        )

    def test_sample_lines_summed_within_iteration(self) -> None:
        """Multiple SAMPLE lines for the same iteration index are summed before averaging."""
        prog_0 = self.tmp_path / "inscount-prog-0.out"
        overhead_0 = self.tmp_path / "inscount-overhead-0.out"
        prog_1 = self.tmp_path / "inscount-prog-1.out"
        overhead_1 = self.tmp_path / "inscount-overhead-1.out"
        prog_0.write_text("Count 1000\n")
        overhead_0.write_text("Count 500\n")
        prog_1.write_text("Count 2000\n")
        overhead_1.write_text("Count 1000\n")

        fixture = f"""\
META timestamp 2026-04-15T14:22:10Z
SAMPLE Native-50-1 pinDynInstCount 0 {prog_0}
SAMPLE Native-50-1 pinDynInstCount 0 {overhead_0}
SAMPLE Native-50-1 pinDynInstCount 1 {prog_1}
SAMPLE Native-50-1 pinDynInstCount 1 {overhead_1}
MEASUREMENT Native-50-1 0.6 ? 1.7 ? ?
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        # iteration 0: 1000+500=1500, iteration 1: 2000+1000=3000 → avg=2250
        self.assertEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT], 2250.0
        )

    def test_sample_lines_scoped_to_run(self) -> None:
        """SAMPLE lines from one run do not bleed into the next run."""
        sample = self.tmp_path / "inscount.out"
        sample.write_text("Count 5000\n")

        fixture = f"""\
META timestamp 2026-04-15T14:22:10Z
SAMPLE cfg pinDynInstCount 0 {sample}
MEASUREMENT cfg 1.0 ? 2.0 ? ?
META timestamp 2026-04-15T14:25:00Z
MEASUREMENT cfg 1.0 ? 2.0 ? ?
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        first = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        second = runs[1][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertEqual(
            first[TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT], 5000.0
        )
        self.assertIsNone(second[TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT])

    def test_malformed_sample_lines_are_skipped(self) -> None:
        """SAMPLE lines with wrong token count are silently ignored."""
        sample = self.tmp_path / "inscount.out"
        sample.write_text("Count 1000\n")

        fixture = f"""\
META timestamp 2026-04-15T14:22:10Z
SAMPLE too few tokens
SAMPLE cfg pinDynInstCount 0 {sample}
MEASUREMENT cfg 1.0 ? 2.0 ? ?
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_PIN_DYN_INST_COUNT], 1000.0
        )


# ---------------------------------------------------------------------------
# SAMPLE tag tests — value-typed databaseTime variant (B1)
# ---------------------------------------------------------------------------


class TestSampleDatabaseTime(unittest.TestCase):
    """Tests for value-typed databaseTime SAMPLE resolution."""

    def test_sample_db_time_resolves_average_from_question_mark(self) -> None:
        """
        A MEASUREMENT with `?` in the db_t slot is resolved to the mean of
        the preceding databaseTime SAMPLE values for the same config.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Athens-16-Autocorrelation databaseTime 0 1.0
SAMPLE Athens-16-Autocorrelation databaseTime 1 2.0
SAMPLE Athens-16-Autocorrelation databaseTime 2 3.0
MEASUREMENT Athens-16-Autocorrelation 5.0 ? 7.0 100 200
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        # mean of 1.0, 2.0, 3.0 is 2.0
        self.assertAlmostEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME], 2.0
        )

    def test_sample_db_time_explicit_value_not_overridden(self) -> None:
        """
        When the db_t slot in a MEASUREMENT is an explicit float (not `?`),
        SAMPLE lines for the same config must not override it.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE cfg databaseTime 0 1.0
SAMPLE cfg databaseTime 1 9.0
MEASUREMENT cfg 5.0 4.0 7.0 100 200
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertAlmostEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME], 4.0
        )


# ---------------------------------------------------------------------------
# SAMPLE tag tests — value-typed databaseDynInstCount variant (B1)
# ---------------------------------------------------------------------------


class TestSampleDatabaseDynInstCount(unittest.TestCase):
    """Tests for value-typed databaseDynInstCount SAMPLE resolution."""

    def test_sample_db_dyn_inst_count_resolves_average_from_question_mark(
        self,
    ) -> None:
        """
        A MEASUREMENT with `?` in the db_i slot is resolved to the mean of
        the preceding databaseDynInstCount SAMPLE values for the same config.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE Athens-16-Autocorrelation databaseDynInstCount 0 100
SAMPLE Athens-16-Autocorrelation databaseDynInstCount 1 200
SAMPLE Athens-16-Autocorrelation databaseDynInstCount 2 300
MEASUREMENT Athens-16-Autocorrelation 5.0 4.0 7.0 ? 500
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        # mean of 100, 200, 300 is 200
        self.assertAlmostEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT], 200.0
        )

    def test_sample_db_dyn_inst_count_explicit_value_not_overridden(self) -> None:
        """
        When the db_i slot in a MEASUREMENT is an explicit float, SAMPLE
        lines for the same config must not override it.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE cfg databaseDynInstCount 0 100
SAMPLE cfg databaseDynInstCount 1 900
MEASUREMENT cfg 5.0 4.0 7.0 400 500
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertAlmostEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT], 400.0
        )

    def test_sample_mixed_three_slots_resolved_independently(self) -> None:
        """
        A MEASUREMENT with `?` in the time, db_time, and db_dyn_inst_count
        slots must resolve all three from their respective SAMPLE streams
        without cross-contamination.
        """
        fixture = """\
META timestamp 2026-04-15T14:30:00Z
SAMPLE cfg elapsedTime 0 1.0
SAMPLE cfg elapsedTime 1 3.0
SAMPLE cfg databaseTime 0 10.0
SAMPLE cfg databaseTime 1 20.0
SAMPLE cfg databaseDynInstCount 0 100
SAMPLE cfg databaseDynInstCount 1 300
MEASUREMENT cfg ? ? 7.0 ? 500
"""
        runs = parse_timing_intermediate_stream(fixture.splitlines())
        measurement = runs[0][TimingFormat.JSON_KEY_MEASUREMENTS][0]
        self.assertAlmostEqual(measurement[TimingFormat.JSON_KEY_MEASUREMENT_TIME], 2.0)
        self.assertAlmostEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_DB_TIME], 15.0
        )
        self.assertAlmostEqual(
            measurement[TimingFormat.JSON_KEY_MEASUREMENT_DB_DYN_INST_COUNT], 200.0
        )


if __name__ == "__main__":
    unittest.main()
