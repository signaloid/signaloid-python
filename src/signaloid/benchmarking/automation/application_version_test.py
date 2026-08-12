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


import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from signaloid.benchmarking.automation.benchmark import Benchmark

_DATE_VERSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


class TestResolveApplicationVersion(unittest.TestCase):
    """Tests for Benchmark._resolve_application_version fallback and git-hash paths."""

    def setUp(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.tmp_path = Path(tmp_dir.name)

    def test_resolve_application_version_falls_back_to_date_when_not_git_repo(
        self,
    ) -> None:
        version = Benchmark._resolve_application_version(str(self.tmp_path))
        self.assertTrue(version)
        self.assertRegex(
            version,
            _DATE_VERSION_RE,
            msg=f"Expected YYYY-MM-DD-HH-MM-SS date stamp, got {version!r}",
        )

    def test_resolve_application_version_falls_back_to_date_on_git_failure(
        self,
    ) -> None:
        (self.tmp_path / ".git").mkdir()
        with patch.object(
            subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            version = Benchmark._resolve_application_version(str(self.tmp_path))
        self.assertRegex(
            version,
            _DATE_VERSION_RE,
            msg=f"Expected date fallback after git failure, got {version!r}",
        )

    def test_resolve_application_version_returns_short_git_hash_for_repo(
        self,
    ) -> None:
        (self.tmp_path / ".git").mkdir()
        with patch.object(
            subprocess, "check_output", return_value="abcdef0\n"
        ) as mock_check:
            version = Benchmark._resolve_application_version(str(self.tmp_path))
        self.assertEqual(version, "abcdef0")
        args, _kwargs = mock_check.call_args
        cmd = args[0]
        self.assertEqual(cmd[:2], ["git", "-C"])
        self.assertEqual(cmd[2], str(self.tmp_path))
        self.assertEqual(cmd[3:], ["rev-parse", "--short=7", "HEAD"])


if __name__ == "__main__":
    unittest.main()
