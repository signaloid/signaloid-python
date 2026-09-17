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
import shutil
import subprocess
import tempfile
import unittest

from pathlib import Path

# The template now lives adjacent to this test in the relocated package.
_TEMPLATE = Path(__file__).resolve().parent / "get-timing-template.sh"


def _source_template(application_path: Path, benchmarking_python: Path) -> str:
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
            "BENCHMARKING_PYTHON": str(benchmarking_python),
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
        self.stub_benchmarking_python = self._make_stub_benchmarking_python()

    def _make_stub_benchmarking_python(self) -> Path:
        """Provide a stub interpreter that resolves a no-op get-timings.sh.

        The template ends by asking ``$BENCHMARKING_PYTHON`` for the path of
        the bundled ``get-timings.sh`` and sourcing the result. For these tests
        we only care about the variable assignments at the top of the template,
        so the stub interpreter ignores its arguments and prints the path of an
        empty get-timings.sh.
        """
        fake_get_timings = self.tmp_path / "get-timings.sh"
        fake_get_timings.write_text("# no-op stub for tests\n")

        stub_python = self.tmp_path / "stub-python"
        stub_python.write_text(f'#!/usr/bin/env bash\necho "{fake_get_timings}"\n')
        stub_python.chmod(0o755)
        return stub_python

    def test_template_application_name_is_derived_from_path(self) -> None:
        app_path = self.tmp_path / "Signaloid-Demo-MyDemo"
        app_path.mkdir()
        out = _source_template(app_path, self.stub_benchmarking_python)
        self.assertIn("NAME=MyDemo", out)

    def test_template_application_version_is_non_empty_for_non_git_path(
        self,
    ) -> None:
        app_path = self.tmp_path / "demo"
        app_path.mkdir()
        out = _source_template(app_path, self.stub_benchmarking_python)
        version_line = next(
            line for line in out.splitlines() if line.startswith("VERSION=")
        )
        version = version_line.removeprefix("VERSION=")
        self.assertTrue(version, "expected non-empty APPLICATION_VERSION")
        self.assertRegex(
            version,
            r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$",
        )


class TestGetTimingTemplateInterpreterDiscovery(unittest.TestCase):
    """The template must find an interpreter without BENCHMARKING_PYTHON set.

    PEP 394 asks for a ``python3`` on the PATH, but it is not honoured
    everywhere, so the template also accepts a ``python`` that is Python 3.
    """

    # The template shells out to these while populating APPLICATION_NAME and
    # APPLICATION_VERSION, so a restricted PATH still has to provide them.
    _REQUIRED_TOOLS = ("basename", "sed", "date")

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

        self.stub_get_timings = self.tmp_path / "get-timings.sh"
        self.stub_get_timings.write_text("# no-op stub for tests\n")

    def _make_bin_dir(self, *interpreter_names: str) -> Path:
        """Build a PATH directory holding only the named interpreters.

        Each named interpreter is a stub that ignores its arguments and prints
        the path of an empty get-timings.sh, standing in for
        ``print(config.get_timing_script())``. The tools the template itself
        shells out to are symlinked in, so nothing else leaks in from the
        real PATH.

        Args:
            interpreter_names: Names to create, e.g. ``"python"``. Passing
                none yields a directory with no interpreter at all.

        Returns:
            The directory to use as the entire PATH.
        """
        bin_dir = self.tmp_path / ("bin-" + "-".join(interpreter_names or ("none",)))
        bin_dir.mkdir()

        for tool in self._REQUIRED_TOOLS:
            resolved = shutil.which(tool)
            assert resolved is not None, f"{tool} is needed to run this test"
            (bin_dir / tool).symlink_to(resolved)

        for name in interpreter_names:
            stub = bin_dir / name
            stub.write_text(f'#!/bin/sh\necho "{self.stub_get_timings}"\n')
            stub.chmod(0o755)

        return bin_dir

    def _source_with_path(self, bin_dir: Path) -> subprocess.CompletedProcess[str]:
        """Source the template with `bin_dir` as the whole PATH.

        BENCHMARKING_PYTHON is deliberately absent so the template has to
        discover an interpreter by name.
        """
        app_path = self.tmp_path / "Signaloid-Demo-MyDemo"
        app_path.mkdir(exist_ok=True)
        bash = shutil.which("bash")
        assert bash is not None, "bash is needed to run this test"
        return subprocess.run(
            [bash, "-c", f'source {_TEMPLATE} && echo "NAME=$APPLICATION_NAME"'],
            env={"PATH": str(bin_dir), "APPLICATION_PATH": str(app_path)},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_python3_on_path_is_used(self) -> None:
        result = self._source_with_path(self._make_bin_dir("python3"))
        self.assertEqual(result.returncode, 0, f"stderr={result.stderr!r}")
        self.assertIn("NAME=MyDemo", result.stdout)

    def test_bare_python_is_used_when_python3_is_absent(self) -> None:
        """The case this class exists for: only `python`, and it is Python 3."""
        result = self._source_with_path(self._make_bin_dir("python"))
        self.assertEqual(result.returncode, 0, f"stderr={result.stderr!r}")
        self.assertIn("NAME=MyDemo", result.stdout)

    def test_no_interpreter_fails_loudly(self) -> None:
        """With no interpreter the template must error, not silently no-op."""
        result = self._source_with_path(self._make_bin_dir())
        self.assertNotEqual(result.returncode, 0, "expected a non-zero exit")
        self.assertIn("Could not locate get-timings.sh", result.stderr)
        self.assertIn("BENCHMARKING_PYTHON", result.stderr)


if __name__ == "__main__":
    unittest.main()
