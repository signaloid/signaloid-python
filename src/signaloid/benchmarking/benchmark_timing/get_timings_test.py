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
Regression coverage for the build-resources fallback in get-timings.sh.

``get-timings.sh`` resolves the bundled coreClass build templates from
``BENCHMARKING_RESOURCES_DIR`` when the Python tool exports it, and otherwise
falls back to a path relative to its own location. The Python tool always
exports it, so the fallback only runs for standalone bash use and went
unnoticed while it pointed one directory level too high. These tests pin the
fallback to the same directory ``get_resources_dir()`` returns.
"""

import re
import unittest

from pathlib import Path

from signaloid.benchmarking.config import get_resources_dir

_SCRIPT = Path(__file__).resolve().parent / "get-timings.sh"

# Matches the default in `RESOURCES_DIR="${BENCHMARKING_RESOURCES_DIR:-...}"`.
_FALLBACK_PATTERN = re.compile(
    r'RESOURCES_DIR="\$\{BENCHMARKING_RESOURCES_DIR:-(?P<default>[^}]+)\}"'
)


def _fallback_resources_dir() -> Path:
    """Resolve the script's own default for the build-resources directory.

    Returns:
        The directory the fallback expands to, with ``$SCRIPT_DIR``
        substituted for the script's location and ``..`` segments collapsed.
    """
    match = _FALLBACK_PATTERN.search(_SCRIPT.read_text())
    assert match is not None, "could not find the RESOURCES_DIR fallback"
    default = match.group("default")
    assert "$SCRIPT_DIR" in default, f"fallback is not script-relative: {default}"
    return Path(default.replace("$SCRIPT_DIR", str(_SCRIPT.parent))).resolve()


class TestGetTimingsResourcesFallback(unittest.TestCase):
    """The standalone fallback must find the bundled build templates."""

    def test_fallback_directory_exists(self) -> None:
        fallback = _fallback_resources_dir()
        self.assertTrue(
            fallback.is_dir(),
            f"fallback build-resources directory does not exist: {fallback}",
        )

    def test_fallback_matches_python_resolver(self) -> None:
        """The bash fallback and get_resources_dir() must not diverge."""
        self.assertEqual(_fallback_resources_dir(), get_resources_dir().resolve())

    def test_fallback_contains_expected_templates(self) -> None:
        """Guard the files copy_build_resources() reads from that directory."""
        fallback = _fallback_resources_dir()
        for relative_path in (
            "C0/init.S",
            "common/startup.cpp",
            "C0Pro/Makefile.pro",
            "C0/Makefile",
        ):
            with self.subTest(relative_path=relative_path):
                self.assertTrue(
                    (fallback / relative_path).is_file(),
                    f"missing build template: {relative_path}",
                )


if __name__ == "__main__":
    unittest.main()
