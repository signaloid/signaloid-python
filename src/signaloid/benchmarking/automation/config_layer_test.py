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
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import yaml

from signaloid.benchmarking.automation.arguments import (
    create_argument_parser,
    validate_args,
)
from signaloid.benchmarking.automation.benchmark_application import (
    load_config,
)


def _write_config(tmp_path: Path, config: Dict[str, Any]) -> str:
    """Write ``config`` to a YAML file under ``tmp_path`` and return the
    path as a string."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


def _base_config() -> Dict[str, Any]:
    """A minimal valid sweep config (all four required sweep args)."""
    return {
        "representation_types": ["Athens"],
        "representation_sizes": [16, 64],
        "correlations": ["Disabled"],
        "reporting_methods": ["Mean"],
    }


class TestConfigLayer(unittest.TestCase):
    """Config-file merge, validation, and plot/flag defaults for the CLI."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def _set_argv(self, argv: list[str]) -> None:
        """Patch ``sys.argv`` for the duration of the test."""
        p = patch.object(sys, "argv", argv)
        p.start()
        self.addCleanup(p.stop)

    def test_config_file_populates_defaults(self) -> None:
        config_path = _write_config(self.tmp_path, _base_config())
        self._set_argv(["prog", "--config", config_path])

        args = load_config(create_argument_parser())

        self.assertEqual(args.representation_types, ["Athens"])
        self.assertEqual(args.representation_sizes, [16, 64])
        self.assertEqual(args.correlations, ["Disabled"])
        self.assertEqual(args.reporting_methods, ["Mean"])
        # validate_args must accept a file-populated sweep.
        validate_args(args)

    def test_cli_overrides_config_value(self) -> None:
        config_path = _write_config(self.tmp_path, _base_config())
        self._set_argv(
            ["prog", "--config", config_path, "-r", "Quantile-95"],
        )

        args = load_config(create_argument_parser())

        # CLI flag wins over the file value for reporting_methods, while the
        # other file-supplied sweep args are retained.
        self.assertEqual(args.reporting_methods, ["Quantile-95"])
        self.assertEqual(args.representation_types, ["Athens"])

    def test_config_sets_non_sweep_scalar(self) -> None:
        # Config keys are dest names: --num-adversaries has dest
        # n_adversaries, so the config key is n_adversaries (not the flag
        # spelling).
        config = _base_config()
        config["n_adversaries"] = 7
        config_path = _write_config(self.tmp_path, config)
        self._set_argv(["prog", "--config", config_path])

        args = load_config(create_argument_parser())

        self.assertEqual(args.n_adversaries, 7)

    def test_missing_required_after_merge_raises(self) -> None:
        # No config file and no sweep args on the CLI: validate_args must
        # raise a clear ValueError naming the missing dest.
        self._set_argv(["prog"])
        args = load_config(create_argument_parser())
        with self.assertRaisesRegex(ValueError, "representation_types"):
            validate_args(args)

    def test_partial_config_missing_one_sweep_arg_raises(self) -> None:
        config = _base_config()
        del config["correlations"]
        config_path = _write_config(self.tmp_path, config)
        self._set_argv(["prog", "--config", config_path])

        args = load_config(create_argument_parser())
        with self.assertRaisesRegex(ValueError, "correlations"):
            validate_args(args)

    def test_unknown_config_key_rejected(self) -> None:
        config = _base_config()
        # Typo of representation_sizes.
        config["representaiton_sizes"] = [32]
        config_path = _write_config(self.tmp_path, config)
        self._set_argv(["prog", "--config", config_path])

        with self.assertRaisesRegex(ValueError, "representaiton_sizes"):
            load_config(create_argument_parser())

    def test_non_dict_config_rejected(self) -> None:
        # A YAML file that is valid but not a top-level mapping (e.g. a
        # list) must raise a clear ValueError, not a confusing TypeError.
        config_path = self.tmp_path / "config.yaml"
        config_path.write_text("- not\n- a\n- mapping\n")
        self._set_argv(["prog", "--config", str(config_path)])

        with self.assertRaisesRegex(ValueError, "mapping"):
            load_config(create_argument_parser())

    def test_missing_config_file_raises(self) -> None:
        # A missing --config file must raise a clear ValueError, not a raw
        # FileNotFoundError.
        self._set_argv(["prog", "--config", "/nonexistent/config.yaml"])
        with self.assertRaisesRegex(ValueError, "not found"):
            load_config(create_argument_parser())

    def test_config_path_expands_user(self) -> None:
        # --config works with ~ expansion.
        config_path = self.tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(_base_config()))
        with patch.dict(os.environ, {"HOME": str(self.tmp_path)}):
            self._set_argv(["prog", "--config", "~/cfg.yaml"])

            args = load_config(create_argument_parser())
        self.assertEqual(
            args.representation_types,
            _base_config()["representation_types"],
        )

    def test_scalar_sweep_arg_rejected(self) -> None:
        # A config that supplies a scalar where a list is expected (e.g.
        # representation_types: Athens) must raise, not iterate char-by-char.
        config = _base_config()
        config["representation_types"] = "Athens"
        config_path = _write_config(self.tmp_path, config)
        self._set_argv(["prog", "--config", config_path])

        args = load_config(create_argument_parser())
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            validate_args(args)

    def test_invalid_sweep_membership_rejected(self) -> None:
        # File-supplied values bypass argparse's choices= check, so
        # validate_args must catch an out-of-enum representation type.
        config = _base_config()
        config["representation_types"] = ["NOT_A_REAL_TYPE"]
        config_path = _write_config(self.tmp_path, config)
        self._set_argv(["prog", "--config", config_path])

        args = load_config(create_argument_parser())
        with self.assertRaisesRegex(ValueError, "NOT_A_REAL_TYPE"):
            validate_args(args)

    def test_plot_flag_defaults_off(self) -> None:
        for dest in [
            "plot_distance_vs_asymptotic",
            "plot_adversary_distances",
            "plot_representative_mc",
        ]:
            with self.subTest(dest=dest):
                parser = create_argument_parser()
                args = parser.parse_args([])
                self.assertFalse(getattr(args, dest))

    def test_plot_flag_positive_form(self) -> None:
        for dest, flag in [
            ("plot_distance_vs_asymptotic", "--plot-distance-vs-asymptotic"),
            ("plot_adversary_distances", "--plot-adversary-distances"),
            ("plot_representative_mc", "--plot-representative-mc"),
        ]:
            with self.subTest(dest=dest, flag=flag):
                parser = create_argument_parser()
                args = parser.parse_args([flag])
                self.assertTrue(getattr(args, dest))

    def test_plot_flag_negated_form(self) -> None:
        for dest, flag in [
            (
                "plot_distance_vs_asymptotic",
                "--no-plot-distance-vs-asymptotic",
            ),
            ("plot_adversary_distances", "--no-plot-adversary-distances"),
            ("plot_representative_mc", "--no-plot-representative-mc"),
        ]:
            with self.subTest(dest=dest, flag=flag):
                parser = create_argument_parser()
                args = parser.parse_args([flag])
                self.assertFalse(getattr(args, dest))

    def test_plot_flag_settable_from_config(self) -> None:
        config = _base_config()
        config["plot_representative_mc"] = True
        config_path = _write_config(self.tmp_path, config)
        self._set_argv(["prog", "--config", config_path])

        args = load_config(create_argument_parser())
        self.assertTrue(args.plot_representative_mc)

    def test_use_binned_uxhw_defaults_true(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args([])
        self.assertTrue(args.use_binned_uxhw)

    def test_use_binned_uxhw_disableable_from_cli(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--no-use-binned-uxhw"])
        self.assertFalse(args.use_binned_uxhw)

    def test_use_binned_uxhw_disableable_from_config(self) -> None:
        config = _base_config()
        config["use_binned_uxhw"] = False
        config_path = _write_config(self.tmp_path, config)
        self._set_argv(["prog", "--config", config_path])

        args = load_config(create_argument_parser())
        self.assertFalse(args.use_binned_uxhw)


if __name__ == "__main__":
    unittest.main()
