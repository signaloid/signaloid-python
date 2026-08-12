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

"""
Regression coverage for the standalone-bash usage path.

bash.s ``validate_required_vars`` requires non-empty
APPLICATION_NAME and APPLICATION_VERSION. The template must
therefore populate both before sourcing ``get-timings.sh``, even
when APPLICATION_PATH is not a git repository.
"""

import os
import subprocess
import tempfile
import unittest

from pathlib import Path

# The template now lives adjacent to this test in the relocated package.
_TEMPLATE = Path(__file__).resolve().parent / "get-timing-template.sh"


def _source_template(application_path: Path, signaloid_python_dir: Path) -> str:
    """Source the template and echo APPLICATION_NAME / APPLICATION_VERSION.

    Returns combined stdout. Tests parse it with simple substring checks.
    """
    cmd = (
        f"source {_TEMPLATE} && "
        'echo "NAME=$APPLICATION_NAME" && '
        'echo "VERSION=$APPLICATION_VERSION"'
    )
    result = subprocess.run(
        ["bash", "-c", cmd],
        env={
            "SIGNALOID_PYTHON_DIR": str(signaloid_python_dir),
            "APPLICATION_PATH": str(application_path),
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", ""),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"template sourcing failed: stderr={result.stderr!r}"
    return result.stdout


class TestGetTimingTemplate(unittest.TestCase):
    """The template populates APPLICATION_NAME / APPLICATION_VERSION even
    when APPLICATION_PATH is not a git repository."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)
        self.stub_signaloid_python_dir = self._make_stub_signaloid_python_dir()

    def _make_stub_signaloid_python_dir(self) -> Path:
        """Provide a stub SIGNALOID_PYTHON_DIR with a no-op get-timings.sh.

        The template ends by sourcing
        ``$SIGNALOID_PYTHON_DIR/src/signaloid/benchmarking/benchmark_timing/get-timings.sh``.
        For these tests we only care about the variable assignments at
        the top of the template, so we point SIGNALOID_PYTHON_DIR at a
        fake tree whose get-timings.sh is empty.
        """
        fake_get_timings = (
            self.tmp_path
            / "src"
            / "signaloid"
            / "benchmarking"
            / "benchmark_timing"
            / "get-timings.sh"
        )
        fake_get_timings.parent.mkdir(parents=True)
        fake_get_timings.write_text("# no-op stub for tests\n")
        return self.tmp_path

    def test_template_application_name_is_derived_from_path(self) -> None:
        app_path = self.tmp_path / "Signaloid-Demo-MyDemo"
        app_path.mkdir()
        out = _source_template(app_path, self.stub_signaloid_python_dir)
        self.assertIn("NAME=MyDemo", out)

    def test_template_application_version_is_non_empty_for_non_git_path(
        self,
    ) -> None:
        app_path = self.tmp_path / "demo"
        app_path.mkdir()
        out = _source_template(app_path, self.stub_signaloid_python_dir)
        version_line = next(
            line for line in out.splitlines() if line.startswith("VERSION=")
        )
        version = version_line.removeprefix("VERSION=")
        self.assertTrue(version, "expected non-empty APPLICATION_VERSION")
        self.assertRegex(
            version,
            r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$",
        )


if __name__ == "__main__":
    unittest.main()
