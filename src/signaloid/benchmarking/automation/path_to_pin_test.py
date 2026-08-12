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
from typing import Optional

from signaloid.benchmarking.automation.arguments import (
    create_argument_parser,
)
from signaloid.benchmarking.automation.build import export_timing_env
from signaloid.benchmarking.config import (
    Correlations,
    RepresentationTypes,
    ReportingMethods,
)


def _base_argv() -> list[str]:
    """The four required sweep args, with no PIN path."""
    return [
        "-u",
        RepresentationTypes.ATHENS,
        "-s",
        "16",
        "-c",
        Correlations.DISABLED,
        "-r",
        ReportingMethods.MEAN,
    ]


def _export_kwargs(tmp_path: Path, path_to_pin: Optional[str]) -> dict:
    """Minimal valid kwargs for ``export_timing_env``."""
    return dict(
        path_to_uxhw_sdk="~/project-uxhw-sdk",
        path_to_pin=path_to_pin,
        path_to_application=str(tmp_path),
        application_name="demo",
        application_version="abc1234",
        max_jupiter_size=32,
        results_dir=str(tmp_path / "results"),
        logs_dir=str(tmp_path / "logs"),
        tracing_db_path=str(tmp_path / "tracing.db"),
    )


class TestPathToPin(unittest.TestCase):
    """``--path-to-pin`` parsing and its ``PIN_ROOT`` export side effect."""

    def setUp(self) -> None:
        # Snapshot and restore ``os.environ`` so ``export_timing_env`` side
        # effects do not leak into other tests.
        snapshot = dict(os.environ)

        def _restore_environ() -> None:
            os.environ.clear()
            os.environ.update(snapshot)

        self.addCleanup(_restore_environ)

        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_parser_accepts_path_to_pin(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(_base_argv() + ["--path-to-pin", "/opt/pin-test"])
        self.assertEqual(args.path_to_pin, "/opt/pin-test")

    def test_parser_path_to_pin_defaults_to_none(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(_base_argv())
        self.assertIsNone(args.path_to_pin)

    def test_export_timing_env_exports_pin_root(self) -> None:
        export_timing_env(**_export_kwargs(self.tmp_path, "/opt/pin-test"))
        self.assertEqual(os.environ["PIN_ROOT"], "/opt/pin-test")

    def test_export_timing_env_omits_pin_root_when_unset(self) -> None:
        # PIN is mandatory and overridable: with no --path-to-pin we must not
        # touch PIN_ROOT, so get-timings.sh applies its built-in fallback.
        os.environ.pop("PIN_ROOT", None)
        export_timing_env(**_export_kwargs(self.tmp_path, None))
        self.assertNotIn("PIN_ROOT", os.environ)


if __name__ == "__main__":
    unittest.main()
