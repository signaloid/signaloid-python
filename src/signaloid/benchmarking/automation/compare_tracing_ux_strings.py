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
Compare the Ux strings two builds of the same application print to stdout.

Optimisation must not change the values the
uncertainty machinery computes, so the two builds' Ux strings are expected to
be byte-for-byte identical.

The values are read from each run's **stdout**, not from a tracing database.
A tracing DB only exists when the ``opt`` pass is given ``--enable-tracing``,
and the verification build is deliberately compiled without it. The SDK forces
``OPTFLAGS`` back to ``-O0`` whenever tracing is on, which would defeat the
point of building at ``-O2``.

Ux strings are compared in the order printed. Everything else on stdout is
ignored, which matters because ordinary output lines carry per-build paths
(the stats DB filename, for one) that legitimately differ between the two runs.

Invoked from the bash tracing layer (``get-timings.sh``) as::

    python3 -m signaloid.benchmarking.automation.compare_tracing_ux_strings \
        <baseline_stdout> <candidate_stdout> [--config SUFFIX] \
        [--baseline-label O0] [--candidate-label O2]

Exit status is ``0`` when the two builds produced identical Ux strings and
non-zero when they differ, when neither printed any, or when the output could
not be read. The caller treats a non-zero status as a warning and continues.
"""

import argparse
import re
import sys

# A printed uncertain value renders as `Ux` followed by the hex encoding of its
# representation. The leading type byte varies by representation (Athens, Mercury,
# ...), so the pattern must not pin it to a particular value.
_UX_STRING_PATTERN = re.compile(r"Ux[0-9A-Fa-f]+")

# Cap on how much of a differing Ux string to print. A single Athens-16 value
# is several hundred characters, and a systematic difference would otherwise
# flood the build log.
_REPORTED_MISMATCHES = 5
_REPORTED_PREFIX_LENGTH = 80


def load_ux_strings(path: str) -> list[str]:
    """
    Read every Ux string printed by one run, in the order it was printed.

    Args:
        path: Path to the captured stdout of a single application run.

    Returns:
        The Ux strings found, in print order. Non-Ux output is ignored.
    """
    with open(path, encoding="utf-8", errors="replace") as stream:
        return _UX_STRING_PATTERN.findall(stream.read())


class ComparisonResult:
    """Outcome of comparing two runs' printed Ux strings."""

    def __init__(
        self,
        *,
        matched: int,
        mismatches: list[tuple[int, str, str]],
        baseline_count: int,
        candidate_count: int,
    ) -> None:
        self.matched = matched
        self.mismatches = mismatches
        self.baseline_count = baseline_count
        self.candidate_count = candidate_count

    @property
    def is_identical(self) -> bool:
        """True when both runs printed the same Ux strings in the same order."""
        return not self.mismatches and self.baseline_count == self.candidate_count

    @property
    def is_empty(self) -> bool:
        """
        True when neither run printed a single Ux string.

        :attr:`is_identical` is vacuously true in that case, because there is
        nothing to mismatch and the two counts are equally zero. Callers must
        check this first, or a run that printed nothing reads as a pass.
        """
        return self.is_identical and self.matched == 0


def compare_ux_strings(baseline_stdout: str, candidate_stdout: str) -> ComparisonResult:
    """
    Compare the Ux strings printed by two builds of the same application.

    Args:
        baseline_stdout: Captured stdout of the baseline (``-O0``) run.
        candidate_stdout: Captured stdout of the candidate (``-O2``) run.

    Returns:
        A :class:`ComparisonResult` describing how many values matched and
        which positions differed.
    """
    baseline = load_ux_strings(baseline_stdout)
    candidate = load_ux_strings(candidate_stdout)

    matched = 0
    mismatches: list[tuple[int, str, str]] = []
    for index, (baseline_value, candidate_value) in enumerate(zip(baseline, candidate)):
        if baseline_value == candidate_value:
            matched += 1
        else:
            mismatches.append((index, baseline_value, candidate_value))

    return ComparisonResult(
        matched=matched,
        mismatches=mismatches,
        baseline_count=len(baseline),
        candidate_count=len(candidate),
    )


def _abbreviate(value: str) -> str:
    """Shorten a Ux string for reporting, marking it when truncated."""
    if len(value) <= _REPORTED_PREFIX_LENGTH:
        return value
    return f"{value[:_REPORTED_PREFIX_LENGTH]}... ({len(value)} chars)"


def _report(
    result: ComparisonResult,
    *,
    baseline_label: str,
    candidate_label: str,
    config: str | None,
) -> None:
    """Print a human-readable summary of *result* to stdout."""
    scope = f" for config '{config}'" if config else ""

    if result.is_empty:
        print(
            f"WARNING: ux-string check{scope}: neither the {baseline_label} "
            f"nor the {candidate_label} build printed any Ux string, so "
            f"nothing was verified. Check that the application prints its "
            f"uncertain values."
        )
        return

    if result.is_identical:
        print(
            f"ux-string check{scope}: OK — {result.matched} printed Ux "
            f"string(s) identical between {baseline_label} and "
            f"{candidate_label} builds."
        )
        return

    print(
        f"WARNING: ux-string check{scope}: {baseline_label} and "
        f"{candidate_label} builds printed different Ux strings "
        f"({result.matched} identical, {len(result.mismatches)} differing, "
        f"{result.baseline_count} printed by {baseline_label}, "
        f"{result.candidate_count} by {candidate_label})."
    )
    for index, baseline_value, candidate_value in result.mismatches[
        :_REPORTED_MISMATCHES
    ]:
        print(f"  DIFF at printed value #{index}")
        print(f"    {baseline_label}: {_abbreviate(baseline_value)}")
        print(f"    {candidate_label}: {_abbreviate(candidate_value)}")
    remaining = len(result.mismatches) - _REPORTED_MISMATCHES
    if remaining > 0:
        print(f"  ... and {remaining} further differing value(s)")


def main() -> int:
    """
    Parse arguments, compare the two runs' stdout, and report the result.

    Returns:
        ``0`` when the two builds printed identical Ux strings, ``1`` when
        they differed or when neither printed any, and ``2`` when the
        comparison could not be performed (e.g. a stdout file was missing).
        The bash caller treats any non-zero status as a warning and continues.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Ux strings printed by two builds of the same "
            "application at different optimisation levels, warning on any "
            "byte-level difference."
        ),
    )
    parser.add_argument("baseline_stdout", help="Captured stdout of the -O0 run.")
    parser.add_argument("candidate_stdout", help="Captured stdout of the -O2 run.")
    parser.add_argument(
        "--baseline-label",
        default="O0",
        help="Label for the baseline build in messages (default: O0).",
    )
    parser.add_argument(
        "--candidate-label",
        default="O2",
        help="Label for the candidate build in messages (default: O2).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config suffix, included in messages for context.",
    )
    args = parser.parse_args()

    try:
        result = compare_ux_strings(args.baseline_stdout, args.candidate_stdout)
    except OSError as exc:
        print(
            f"WARNING: ux-string check could not compare "
            f"{args.baseline_stdout} and {args.candidate_stdout}: {exc}",
            file=sys.stderr,
        )
        return 2

    _report(
        result,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        config=args.config,
    )
    return 0 if result.is_identical and not result.is_empty else 1


if __name__ == "__main__":
    sys.exit(main())
