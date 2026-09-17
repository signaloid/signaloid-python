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


def _export_kwargs(tmp_path: Path, measure_dynamic_instructions: bool) -> dict:
    """Minimal valid kwargs for ``export_timing_env``."""
    return dict(
        path_to_uxhw_sdk="~/project-uxhw-sdk",
        measure_dynamic_instructions=measure_dynamic_instructions,
        path_to_application=str(tmp_path),
        application_name="demo",
        application_version="abc1234",
        max_jupiter_size=32,
        results_dir=str(tmp_path / "results"),
        logs_dir=str(tmp_path / "logs"),
        tracing_db_path=str(tmp_path / "tracing.db"),
    )


class TestMeasureDynamicInstructions(unittest.TestCase):
    """``--measure-dynamic-instructions`` parsing and its ``PIN_ROOT`` effect.

    The flag is the single switch for the dynamic instruction count. These
    tests pin the property the design rests on, which is that a run without
    the flag never measures, whatever ``PIN_ROOT`` holds.
    """

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

    def test_parser_defaults_to_off(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(_base_argv())
        self.assertFalse(args.measure_dynamic_instructions)

    def test_parser_accepts_flag(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(_base_argv() + ["--measure-dynamic-instructions"])
        self.assertTrue(args.measure_dynamic_instructions)

    def test_parser_rejects_removed_path_to_pin(self) -> None:
        # --path-to-pin is gone. Argparse must reject it rather than
        # silently ignoring a stale invocation.
        parser = create_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(_base_argv() + ["--path-to-pin", "/opt/pin"])

    def test_flag_keeps_pin_root(self) -> None:
        os.environ["PIN_ROOT"] = "/opt/pin-test"
        export_timing_env(**_export_kwargs(self.tmp_path, True))
        self.assertEqual(os.environ["PIN_ROOT"], "/opt/pin-test")

    def test_flag_expands_user_in_pin_root(self) -> None:
        os.environ["PIN_ROOT"] = "~/pin-kit"
        export_timing_env(**_export_kwargs(self.tmp_path, True))
        self.assertEqual(os.environ["PIN_ROOT"], os.path.expanduser("~/pin-kit"))

    def test_flag_without_pin_root_raises(self) -> None:
        os.environ.pop("PIN_ROOT", None)
        with self.assertRaises(ValueError) as caught:
            export_timing_env(**_export_kwargs(self.tmp_path, True))
        self.assertIn("PIN_ROOT", str(caught.exception))

    def test_flag_with_empty_pin_root_raises(self) -> None:
        # Set but empty is not a usable kit. Treat it as unset rather than
        # letting the shell look for a kit at "/".
        os.environ["PIN_ROOT"] = ""
        with self.assertRaises(ValueError):
            export_timing_env(**_export_kwargs(self.tmp_path, True))

    def test_no_flag_drops_inherited_pin_root(self) -> None:
        # The whole point of the redesign. A kit exported in the caller's
        # shell profile must not switch the measurement on.
        os.environ["PIN_ROOT"] = "/opt/pin-test"
        export_timing_env(**_export_kwargs(self.tmp_path, False))
        self.assertNotIn("PIN_ROOT", os.environ)

    def test_no_flag_with_no_pin_root_is_fine(self) -> None:
        os.environ.pop("PIN_ROOT", None)
        export_timing_env(**_export_kwargs(self.tmp_path, False))
        self.assertNotIn("PIN_ROOT", os.environ)


if __name__ == "__main__":
    unittest.main()
