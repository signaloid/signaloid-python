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

from signaloid.benchmarking.automation.benchmarking_utils import (
    _symlink_application_inputs,
)
from signaloid.benchmarking.config import EquivMC

MC_OUTPUT_FILENAME = EquivMC.MC_OUTPUT_FILENAME


class TestSymlinkApplicationInputs(unittest.TestCase):
    """``_symlink_application_inputs`` exposes inputs without leaking
    the binary's ``data.out`` into the shared inputs directory."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_real_inputs_are_symlinked(self) -> None:
        """Genuine input files are symlinked into the run directory."""
        inputs_dir = self.tmp_path / "inputs"
        inputs_dir.mkdir()
        (inputs_dir / "config.json").write_text("{}")
        (inputs_dir / "prices.csv").write_text("1,2,3")
        dest = self.tmp_path / "run"
        dest.mkdir()

        _symlink_application_inputs(str(inputs_dir), str(dest))

        for name in ("config.json", "prices.csv"):
            link = dest / name
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), str(inputs_dir / name))

    def test_data_out_is_not_symlinked(self) -> None:
        """The binary's output file is never exposed as an input."""
        inputs_dir = self.tmp_path / "inputs"
        inputs_dir.mkdir()
        (inputs_dir / MC_OUTPUT_FILENAME).write_text("stale output\n")
        (inputs_dir / "config.json").write_text("{}")
        dest = self.tmp_path / "run"
        dest.mkdir()

        _symlink_application_inputs(str(inputs_dir), str(dest))

        self.assertFalse((dest / MC_OUTPUT_FILENAME).exists())
        self.assertTrue((dest / "config.json").is_symlink())

    def test_worker_write_does_not_leak_into_shared_inputs(self) -> None:
        """Regression: a stale inputs/data.out must not be written *through*.

        Reproduces the parallel-MC corruption precondition. Before the fix the
        helper symlinked inputs/data.out into the run dir, so a worker writing a
        relative ``data.out`` (as the native binary does) wrote through the
        symlink to the single shared file -> concurrent ``-j > 1`` workers
        clobbered one another and variables ended up with identical samples.
        """
        inputs_dir = self.tmp_path / "inputs"
        inputs_dir.mkdir()
        sentinel = "SENTINEL-MUST-NOT-BE-OVERWRITTEN\n"
        shared = inputs_dir / MC_OUTPUT_FILENAME
        shared.write_text(sentinel)
        dest = self.tmp_path / "run"
        dest.mkdir()

        _symlink_application_inputs(str(inputs_dir), str(dest))

        # Emulate the native binary writing its samples to a relative data.out.
        with open(dest / MC_OUTPUT_FILENAME, "w") as f:
            f.write("42.0\n")

        # The shared inputs file must be untouched and isolated by inode.
        self.assertEqual(shared.read_text(), sentinel)
        self.assertFalse((dest / MC_OUTPUT_FILENAME).is_symlink())
        self.assertNotEqual(
            (dest / MC_OUTPUT_FILENAME).stat().st_ino, shared.stat().st_ino
        )

    def test_missing_inputs_dir_is_noop(self) -> None:
        """A missing inputs/ directory is tolerated without error."""
        dest = self.tmp_path / "run"
        dest.mkdir()

        _symlink_application_inputs(str(self.tmp_path / "does-not-exist"), str(dest))

        self.assertEqual(list(dest.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
