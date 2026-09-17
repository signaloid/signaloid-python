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

import importlib.machinery
import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from signaloid.benchmarking.automation import benchmark_application
from signaloid.benchmarking.automation.benchmark_application import (
    SHEETS_MODULE_NAMES,
    _resolve_sheets_credentials,
)


def _make_args(*, write_sheets: bool, google_credentials: str | None) -> Namespace:
    """
    Build the argument subset that _resolve_sheets_credentials reads.

    Args:
        write_sheets: Whether the user opted in to the Sheets upload.
        google_credentials: Explicit credentials path, or None.

    Returns:
        A Namespace carrying just those two attributes.
    """
    return Namespace(
        write_sheets=write_sheets,
        google_credentials=google_credentials,
    )


def _find_spec_missing(module_names: set[str]) -> Any:
    """
    Build a find_spec replacement with a fixed view of the Sheets stack.

    Every name in ``SHEETS_MODULE_NAMES`` is answered from
    ``module_names`` alone, so these tests behave identically whether or
    not the `sheets` extra is installed in the running environment. Any
    other name is delegated to the real ``importlib.util.find_spec``.

    Args:
        module_names: Sheets modules to report as not installed.

    Returns:
        A callable suitable for patching ``importlib.util.find_spec``.
    """
    real_find_spec = importlib.util.find_spec
    present_spec = importlib.machinery.ModuleSpec("present", loader=None)

    def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in SHEETS_MODULE_NAMES:
            return None if name in module_names else present_spec
        return real_find_spec(name, *args, **kwargs)

    return fake_find_spec


class TestSheetsPreflight(unittest.TestCase):
    """_resolve_sheets_credentials validates the `sheets` extra up front."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.credentials_file = Path(tmp_dir.name) / "creds.json"
        self.credentials_file.write_text("{}")

    def test_opted_out_skips_the_module_check(self) -> None:
        """
        Without --write-sheets the extra is irrelevant, so a missing
        module must not raise: the run does not touch the upload.
        """
        args = _make_args(write_sheets=False, google_credentials=None)
        with patch(
            "importlib.util.find_spec",
            _find_spec_missing(set(SHEETS_MODULE_NAMES)),
        ):
            self.assertIsNone(_resolve_sheets_credentials(args))

    def test_missing_module_raises_naming_the_extra(self) -> None:
        """
        A missing Sheets module raises at startup, and the message names
        both the module and the command that installs it.
        """
        args = _make_args(
            write_sheets=True,
            google_credentials=str(self.credentials_file),
        )
        with patch(
            "importlib.util.find_spec",
            _find_spec_missing({"gspread"}),
        ):
            with self.assertRaises(RuntimeError) as raised:
                _resolve_sheets_credentials(args)
        message = str(raised.exception)
        self.assertIn("gspread", message)
        self.assertIn('pip install ".[sheets]"', message)

    def test_module_check_precedes_the_credentials_check(self) -> None:
        """
        With no credentials *and* no extra, the missing extra is reported.
        Fixing credentials first would only surface the import error at
        step 15, which is the failure mode this check exists to prevent.
        """
        args = _make_args(write_sheets=True, google_credentials=None)
        with patch(
            "importlib.util.find_spec",
            _find_spec_missing(set(SHEETS_MODULE_NAMES)),
        ):
            with self.assertRaises(RuntimeError) as raised:
                _resolve_sheets_credentials(args)
        self.assertIn("`sheets` extra", str(raised.exception))

    def test_installed_extra_passes_through_to_credentials(self) -> None:
        """
        With every module present the check is transparent: resolution
        continues and returns the credentials path.
        """
        args = _make_args(
            write_sheets=True,
            google_credentials=str(self.credentials_file),
        )
        with patch("importlib.util.find_spec", _find_spec_missing(set())):
            with patch.object(
                benchmark_application, "TARGET_FOLDER_ID", "folder-id"
            ), patch.object(benchmark_application, "TEMPLATE_ID", "template-id"):
                resolved = _resolve_sheets_credentials(args)
        self.assertEqual(resolved, str(self.credentials_file))


if __name__ == "__main__":
    unittest.main()
