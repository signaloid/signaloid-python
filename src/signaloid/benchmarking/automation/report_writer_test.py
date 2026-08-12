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
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from signaloid.benchmarking.automation.report_writer import (
    BLOW_UP_DISPLAY_TOKEN,
    _escape_drive_query_value,
    _get_best_speedup_info,
    _get_triptych_image_names,
    _is_blow_up,
    _plot_per_config_token,
    _update_metadata_sheet,
    resolve_google_credentials_path,
    write_results_to_markdown,
)
from signaloid.benchmarking.types import BenchmarkingVariable
from signaloid.benchmarking.config import (
    BenchmarkingVariables,
    EquivMC,
    Measurements,
    VariableTypes,
)


def _make_variable(
    description: str,
    emcc_data: list,
) -> BenchmarkingVariable:
    """
    Create a BenchmarkingVariable fixture with given emcc_data.

    Args:
        description: Human-readable description of the variable.
        emcc_data: List of dicts to populate emcc_data.

    Returns:
        A BenchmarkingVariable instance ready for use in tests.
    """
    variable = BenchmarkingVariable(
        name=description,
        description=description,
    )
    variable.emcc_results.emcc_data = emcc_data
    return variable


def _make_speedup_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the columns _get_best_speedup_info reads."""
    return pd.DataFrame(rows)


# Column A of the assumptions / configuration sheet in the live template
# (https://docs.google.com/spreadsheets/d/1MKPkwY_B_m5WH-coqIy19Otpg3IE5yRSKaKGw6wxzL8):
# the labels carry a list number (whose sequence skips) and trailing text, so
# the metadata write must locate rows by substring rather than fixed position.
_LIVE_METADATA_COLUMN_A = [
    "1. GitHub Repository for Application:",
    "2. Git Hash for Code:",
    "3. Number of distributional inputs:",
    "4. Number of distributional outputs:",
    "5. Example SCCE TaskID for SCCE runs:",
    "7. UxHw SDK version",
    "8. Machine Type",
]


def _metadata_cell(sheet_mock: Mock, cell_range: str) -> Any:
    """Extract a cell value (or None) from a mocked ``sheet.batch_update``."""
    updates = sheet_mock.batch_update.call_args.args[0]
    match = next((u for u in updates if u["range"] == cell_range), None)
    return None if match is None else match["values"][0][0]


def _run_metadata(sheet_mock: Mock, column_a: list[str] | None = None) -> None:
    sheet_mock.col_values.return_value = (
        _LIVE_METADATA_COLUMN_A if column_a is None else column_a
    )
    _update_metadata_sheet(
        sheet=sheet_mock,
        application_version="v1",
        uxhw_version="4.2.1",
        machine_name="test-machine",
        git_repo_remote=None,
    )


class TestWriteResultsToMarkdown(unittest.TestCase):
    """write_results_to_markdown emits one .md file per variable."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_file_created_per_variable(self) -> None:
        """
        Each variable produces one .md file named after its formatted_description.
        """
        var_a = _make_variable("alpha result", [{"col": 1}])
        var_b = _make_variable("beta result", [{"col": 2}])
        write_results_to_markdown(
            benchmarking_variables=[var_a, var_b],
            results_dir=str(self.tmp_path),
        )
        self.assertTrue((self.tmp_path / "alpha-result.md").exists())
        self.assertTrue((self.tmp_path / "beta-result.md").exists())

    def test_markdown_content_contains_headers_and_values(self) -> None:
        """
        The markdown file contains the column header and row values.
        """
        emcc_data = [
            {"Representation": "Athens-16", "EquivMC": 128},
            {"Representation": "Athens-32", "EquivMC": 256},
        ]
        var = _make_variable("my variable", emcc_data)
        write_results_to_markdown(
            benchmarking_variables=[var],
            results_dir=str(self.tmp_path),
        )
        content = (self.tmp_path / "my-variable.md").read_text()
        self.assertIn("Representation", content)
        self.assertIn("EquivMC", content)
        self.assertIn("Athens-16", content)
        self.assertIn("128", content)
        self.assertIn("Athens-32", content)
        self.assertIn("256", content)

    def test_uxhw_conf_column_is_repr_formatted(self) -> None:
        """
        Values in the UxHw Conf column are formatted with repr().
        """
        uxhw_col = BenchmarkingVariables.UXHW_CONF
        emcc_data = [{uxhw_col: 0.95, "EquivMC": 64}]
        var = _make_variable("conf variable", emcc_data)
        write_results_to_markdown(
            benchmarking_variables=[var],
            results_dir=str(self.tmp_path),
        )
        content = (self.tmp_path / "conf-variable.md").read_text()
        self.assertIn(repr(0.95), content)

    def test_zero_variables_writes_no_files(self) -> None:
        """
        Passing an empty list writes no files and raises no exception.
        """
        write_results_to_markdown(
            benchmarking_variables=[],
            results_dir=str(self.tmp_path),
        )
        self.assertEqual(list(self.tmp_path.iterdir()), [])


class TestGetBestSpeedupInfo(unittest.TestCase):
    """_get_best_speedup_info selects the best config / EMCC from a table."""

    def test_get_best_speedup_info_picks_row_with_max_speedup(self) -> None:
        """
        _get_best_speedup_info returns the row whose Speedup is largest.
        """
        df = _make_speedup_df(
            [
                {
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    Measurements.SPEEDUP: 1.5,
                    EquivMC.EMCC: 100,
                },
                {
                    BenchmarkingVariables.UXHW_CONF: "Athens-32",
                    Measurements.SPEEDUP: 4.2,
                    EquivMC.EMCC: 256,
                },
                {
                    BenchmarkingVariables.UXHW_CONF: "Atlas-64",
                    Measurements.SPEEDUP: 2.1,
                    EquivMC.EMCC: 128,
                },
            ]
        )
        best_config, best_emcc = _get_best_speedup_info(df=df)
        self.assertEqual(best_config, "Athens-32")
        self.assertEqual(best_emcc, 256)

    def test_get_best_speedup_info_preserves_correlation_prefix(self) -> None:
        """
        ``best_config`` is returned verbatim from the results table, full
        ``CORRELATION_`` token included. The per-config plot on disk uses the
        short token (``...-Athens-16-OFF.png``). Mapping the results token to
        that short form is the job of ``_plot_per_config_token`` /
        ``_get_triptych_image_names``, not of ``_get_best_speedup_info``,
        which must report the configuration exactly as the results show it.
        """
        df = _make_speedup_df(
            [
                {
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    Measurements.SPEEDUP: 9.0,
                    EquivMC.EMCC: 42,
                },
            ]
        )
        best_config, best_emcc = _get_best_speedup_info(df=df)
        self.assertEqual(best_config, "Athens-16")
        self.assertEqual(best_emcc, 42)

    def test_get_best_speedup_info_falls_back_to_emcc_predicted(self) -> None:
        """
        When EMCC is missing, _get_best_speedup_info reads EMCC_PREDICTED.
        """
        df = _make_speedup_df(
            [
                {
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    Measurements.SPEEDUP: 3.0,
                    EquivMC.EMCC_PREDICTED: 99,
                },
            ]
        )
        best_config, best_emcc = _get_best_speedup_info(df=df)
        self.assertEqual(best_config, "Athens-16")
        self.assertEqual(best_emcc, 99)

    def test_get_best_speedup_info_rejects_non_numeric_emcc(self) -> None:
        """
        A non-numeric EMCC value raises TypeError.
        """
        df = _make_speedup_df(
            [
                {
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    Measurements.SPEEDUP: 1.0,
                    EquivMC.EMCC: "not-a-number",
                },
            ]
        )
        with self.assertRaises(TypeError):
            _get_best_speedup_info(df=df)


class TestBlowUpReporting(unittest.TestCase):
    """Blown-up rows are rendered as "blow-up / excluded" and never win
    "best speedup"."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_is_blow_up_recognises_markers(self) -> None:
        """A reason string is a blow-up. None / NaN / empty are not."""
        self.assertTrue(_is_blow_up("0.5 of mass sits at ..."))
        self.assertFalse(_is_blow_up(None))
        self.assertFalse(_is_blow_up(np.nan))
        self.assertFalse(_is_blow_up(""))

    def test_markdown_renders_blow_up_token(self) -> None:
        """A blow-up-marked row shows the excluded token in the EMCC /
        distance columns rather than a meaningless small distance."""
        emcc_data = [
            {
                BenchmarkingVariables.UXHW_CONF: "Athens-16",
                BenchmarkingVariables.UXHW_DISTANCE: 0.01,
                EquivMC.EMCC_PREDICTED: 128,
                BenchmarkingVariables.BLOW_UP_REASON: None,
            },
            {
                BenchmarkingVariables.UXHW_CONF: "Jupiter-32",
                BenchmarkingVariables.UXHW_DISTANCE: float("inf"),
                EquivMC.EMCC_PREDICTED: 1,
                BenchmarkingVariables.BLOW_UP_REASON: "0.5 of mass sits at ...",
            },
        ]
        var = _make_variable("blow up variable", emcc_data)
        write_results_to_markdown(
            benchmarking_variables=[var],
            results_dir=str(self.tmp_path),
        )
        content = (self.tmp_path / "blow-up-variable.md").read_text()
        self.assertIn(BLOW_UP_DISPLAY_TOKEN, content)
        # The healthy row's real EMCC is preserved.
        self.assertIn("128", content)

    def test_best_speedup_excludes_blown_row(self) -> None:
        """A blown row collapses to EMCC=1 and might post a huge speedup.
        ``_get_best_speedup_info`` must exclude it and pick a healthy row."""
        df = _make_speedup_df(
            [
                {
                    BenchmarkingVariables.UXHW_CONF: "Athens-16",
                    Measurements.SPEEDUP: 4.2,
                    EquivMC.EMCC: 256,
                    BenchmarkingVariables.BLOW_UP_REASON: None,
                },
                {
                    BenchmarkingVariables.UXHW_CONF: "Jupiter-32",
                    Measurements.SPEEDUP: 999.0,
                    EquivMC.EMCC: 1,
                    BenchmarkingVariables.BLOW_UP_REASON: "blown",
                },
            ]
        )
        best_config, best_emcc = _get_best_speedup_info(df=df)
        # The blown 999x row is excluded. The healthy Athens-16 row wins.
        self.assertEqual(best_config, "Athens-16")
        self.assertEqual(best_emcc, 256)


class TestTriptychImageNames(unittest.TestCase):
    """_get_triptych_image_names / _plot_per_config_token build plot paths."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_triptych_image_names_compose_three_paths(self) -> None:
        """
        _get_triptych_image_names returns adversary, config, ground-truth
        PNG paths under the provided plots_dir.
        """
        variable = BenchmarkingVariable(
            name="my variable",
            description="my variable",
        )
        plots_dir = str(self.tmp_path)
        names = _get_triptych_image_names(
            variable=variable,
            best_config="Athens-32",
            best_emcc=256,
            plots_dir=plots_dir,
        )
        self.assertEqual(
            names,
            [
                os.path.join(plots_dir, "my-variable-adversary-256.png"),
                os.path.join(plots_dir, "my-variable-Athens-32.png"),
                os.path.join(plots_dir, "my-variable-ground-truth.png"),
            ],
        )

    def test_plot_per_config_token_passes_correlation_through(self) -> None:
        """The results and generator share one canonical correlation token, so
        the per-config token passes through unchanged."""
        cases = [
            # Disabled correlation is omitted from the config string, and
            # Autocorrelation is written verbatim by both the results table and
            # the generator: both pass through unchanged.
            ("Jupiter-16", "Jupiter-16"),
            ("Jupiter-16-Autocorrelation", "Jupiter-16-Autocorrelation"),
            # A config whose trailing segment is not a correlation token is
            # left untouched (representation type/size are never rewritten).
            ("Athens-32", "Athens-32"),
            # A string with no dash passes through unchanged.
            ("Athens", "Athens"),
        ]
        for best_config, expected in cases:
            with self.subTest(best_config=best_config, expected=expected):
                self.assertEqual(_plot_per_config_token(best_config), expected)

    def test_triptych_per_config_uses_canonical_correlation_token(self) -> None:
        """A Disabled best_config (suffix omitted) yields the on-disk
        ...-Jupiter-16.png name."""
        variable = BenchmarkingVariable(
            name="gg -> gg cross-section (pb)",
            description="gg -> gg cross-section (pb)",
            type=VariableTypes.DISTRIBUTION,
        )
        plots_dir = str(self.tmp_path)
        names = _get_triptych_image_names(
            variable=variable,
            best_config="Jupiter-16",
            best_emcc=10,
            plots_dir=plots_dir,
        )
        formatted = variable.formatted_description
        self.assertEqual(
            names,
            [
                os.path.join(plots_dir, f"{formatted}-adversary-10.png"),
                os.path.join(plots_dir, f"{formatted}-Jupiter-16.png"),
                os.path.join(plots_dir, f"{formatted}-ground-truth.png"),
            ],
        )

    def test_triptych_scalar_adversary_globs_distances_scatter(self) -> None:
        """A scalar's adversary slot resolves to the distances scatter on disk.

        The generator writes the scatter with the raw variable name (which
        contains glob metacharacters) and a suffix encoding ground-truth and
        adversary detail the uploader cannot rebuild, so the slot is located
        by glob rather than reconstructed.
        """
        variable = BenchmarkingVariable(
            name="outputVariables[0]",
            description="gg -> gg cross-section (pb)",
            type=VariableTypes.SCALAR,
        )
        scatter_name = (
            "outputVariables[0]-adversary_distances-"
            "100_weighted_ground_truth_samples-100_adversaries.png"
        )
        # Write the scatter literally (pathlib does not treat [] as a glob).
        (self.tmp_path / scatter_name).write_text("")
        plots_dir = str(self.tmp_path)

        names = _get_triptych_image_names(
            variable=variable,
            best_config="Jupiter-16",
            best_emcc=12345,
            plots_dir=plots_dir,
        )
        formatted = variable.formatted_description
        # Adversary slot is the scatter (NOT a reconstructed -adversary-N.png),
        # found despite the [] metacharacters in the variable name.
        self.assertEqual(names[0], os.path.join(plots_dir, scatter_name))
        # Per-config uses formatted_description + short token. Scatter used
        # the raw name -- the two prefixes legitimately differ.
        self.assertEqual(
            names[1],
            os.path.join(plots_dir, f"{formatted}-Jupiter-16.png"),
        )
        self.assertEqual(
            names[2], os.path.join(plots_dir, f"{formatted}-ground-truth.png")
        )

    def test_triptych_scalar_adversary_picks_most_recent_match(self) -> None:
        """With several scatter matches, the most recently written one wins.

        The lexicographically-last file is deliberately made the OLDER one,
        so this fails if selection falls back to sorting by name.
        """
        variable = BenchmarkingVariable(
            name="outputVariables[0]",
            description="scalar out",
            type=VariableTypes.SCALAR,
        )
        # "2_adversaries" sorts lexicographically AFTER "100_adversaries"
        # ('2' > '1'), so a name-sort would wrongly pick the 2-adversary file.
        newer = self.tmp_path / (
            "outputVariables[0]-adversary_distances-"
            "50_weighted_ground_truth_samples-100_adversaries.png"
        )
        older = self.tmp_path / (
            "outputVariables[0]-adversary_distances-"
            "50_weighted_ground_truth_samples-2_adversaries.png"
        )
        newer.write_text("")
        older.write_text("")
        os.utime(newer, (2000, 2000))
        os.utime(older, (1000, 1000))

        names = _get_triptych_image_names(
            variable=variable,
            best_config="Athens-16",
            best_emcc=7,
            plots_dir=str(self.tmp_path),
        )
        self.assertEqual(names[0], str(newer))

    def test_triptych_scalar_adversary_fallback_when_missing(self) -> None:
        """With no scatter on disk the scalar adversary slot is a clean path.

        It must not raise and must not leak a glob wildcard, so the uploader
        reports it missing (the prior skip-with-warning behavior).
        """
        variable = BenchmarkingVariable(
            name="outputVariables[0]",
            description="scalar out",
            type=VariableTypes.SCALAR,
        )
        names = _get_triptych_image_names(
            variable=variable,
            best_config="Athens-16",
            best_emcc=7,
            plots_dir=str(self.tmp_path),
        )
        self.assertEqual(
            names[0],
            os.path.join(
                str(self.tmp_path), "outputVariables[0]-adversary_distances.png"
            ),
        )
        self.assertNotIn("*", names[0])
        self.assertFalse(os.path.exists(names[0]))


class TestResolveGoogleCredentials(unittest.TestCase):
    """resolve_google_credentials_path resolves explicit / env / default paths."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_resolve_google_credentials_returns_explicit_path(self) -> None:
        """
        A credentials file passed via google_credentials is returned.
        """
        creds_file = self.tmp_path / "creds.json"
        creds_file.write_text("{}")
        resolved = resolve_google_credentials_path(
            google_credentials=str(creds_file),
        )
        self.assertEqual(resolved, str(creds_file))

    def test_resolve_google_credentials_uses_env_var(self) -> None:
        """
        Falls back to GOOGLE_APPLICATION_CREDENTIALS if the explicit arg
        is None.
        """
        creds_file = self.tmp_path / "env-creds.json"
        creds_file.write_text("{}")
        with patch.dict(
            os.environ,
            {"GOOGLE_APPLICATION_CREDENTIALS": str(creds_file)},
        ):
            resolved = resolve_google_credentials_path(google_credentials=None)
        self.assertEqual(resolved, str(creds_file))

    def test_resolve_google_credentials_returns_none_when_missing(self) -> None:
        """
        Returns None (no raise) when no candidate file exists on disk.
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            resolved = resolve_google_credentials_path(
                google_credentials="/nonexistent/path/to/creds.json",
            )
        self.assertIsNone(resolved)

    def test_resolve_google_credentials_no_inputs_returns_none(self) -> None:
        """
        With no explicit arg and no env var, returns None (no org fallback).
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            resolved = resolve_google_credentials_path(google_credentials=None)
        self.assertIsNone(resolved)


class TestEscapeDriveQueryValue(unittest.TestCase):
    """_escape_drive_query_value escapes Drive files().list query values."""

    def test_escape_drive_query_value_passes_safe_strings_through(self) -> None:
        """Strings without special characters are returned unchanged."""
        self.assertEqual(_escape_drive_query_value("uxhw-1.2.3"), "uxhw-1.2.3")
        self.assertEqual(_escape_drive_query_value("MyApp"), "MyApp")

    def test_escape_drive_query_value_escapes_single_quote(self) -> None:
        """Single quotes inside the value are backslash-escaped so the
        Drive ``files().list`` query string is not terminated early."""
        self.assertEqual(_escape_drive_query_value("foo's bar"), "foo\\'s bar")

    def test_escape_drive_query_value_escapes_backslash_before_quote(self) -> None:
        """Backslashes are escaped before single quotes to avoid the
        quote-escape's backslash being doubled into a literal backslash
        plus an unescaped quote."""
        self.assertEqual(_escape_drive_query_value("a\\b"), "a\\\\b")
        self.assertEqual(_escape_drive_query_value("a\\'b"), "a\\\\\\'b")


class TestUpdateMetadataSheet(unittest.TestCase):
    """_update_metadata_sheet writes the session metadata cells by label."""

    def test_metadata_sheet_cell_mapping(self) -> None:
        # Each value lands in column B of the row whose column-A label matches,
        # resolved against the LIVE template's column A. With that layout the
        # rows resolve to B1 repo, B2 git hash, B6 "UxHw SDK version",
        # B7 "Machine Type". Nothing is written at B8 and no shading is applied.
        sheet = Mock()
        _run_metadata(sheet)
        self.assertEqual(_metadata_cell(sheet, "B1"), "")
        self.assertEqual(_metadata_cell(sheet, "B2"), "v1")
        self.assertEqual(_metadata_cell(sheet, "B6"), "4.2.1")
        self.assertEqual(_metadata_cell(sheet, "B7"), "test-machine")
        self.assertIsNone(_metadata_cell(sheet, "B8"))
        sheet.format.assert_not_called()

    def test_metadata_follows_reordered_rows(self) -> None:
        """Reordering the template rows moves the writes with them: the values
        track column-A labels, not fixed cell positions."""
        reordered = [
            "8. Machine Type",  # row 1
            "2. Git Hash for Code:",  # row 2
            "1. GitHub Repository for Application:",  # row 3
            "7. UxHw SDK version",  # row 4
        ]
        sheet = Mock()
        _run_metadata(sheet, column_a=reordered)
        self.assertEqual(_metadata_cell(sheet, "B1"), "test-machine")
        self.assertEqual(_metadata_cell(sheet, "B2"), "v1")
        self.assertEqual(_metadata_cell(sheet, "B3"), "")
        self.assertEqual(_metadata_cell(sheet, "B4"), "4.2.1")

    def test_metadata_missing_label_raises(self) -> None:
        """A template that no longer carries an expected label fails loudly
        (with the missing label named) rather than silently dropping it."""
        without_machine = [
            "1. GitHub Repository for Application:",
            "2. Git Hash for Code:",
            "7. UxHw SDK version",
        ]
        sheet = Mock()
        with self.assertRaises(ValueError) as ctx:
            _run_metadata(sheet, column_a=without_machine)
        self.assertIn("Machine Type", str(ctx.exception))
        sheet.batch_update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
