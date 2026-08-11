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
Compare the Ux strings produced by two tracing runs of the same application.

The benchmarking tracing pass compiles the application at ``-O0`` (the
``OPTFLAGS`` default in ``Makefile.pro``) so the ``addDistValueTrace``
``file:line`` directives resolve against unoptimised debug info. Real
deployments compile at ``-O2``. Optimisation must not change the values the
uncertainty machinery computes, so the Ux strings from the ``-O0`` and ``-O2``
builds are expected to be byte-for-byte identical. This module loads the final
(last-written) Ux string for each traced expression from two tracing DBs and
reports any expression whose Ux string differs or is present in only one build.

Invoked from the bash tracing layer (``get-timings.sh``) as::

    python3 -m signaloid.benchmarking.automation.compare_tracing_ux_strings \
        <baseline_db> <candidate_db> [--table TABLE] [--config SUFFIX] \
        [--baseline-label O0] [--candidate-label O2]

Exit status is ``0`` when the two builds produce identical Ux strings and
non-zero when they differ or cannot be compared. The caller treats a non-zero
status as a warning and continues.
"""

import argparse
import sqlite3
import sys
from contextlib import closing

from signaloid.benchmarking.config import EquivMC

# Table holding the per-execution emulator metadata, joined to the tracing
# table on Execution_ID = Execution_Info_Table_ID (see merge_tracing_dbs).
_EXECUTION_INFO_TABLE = "Emulator_Execution_Info"

# Columns that together identify a single traced expression under one emulator
# configuration. Two rows sharing these values are writes to the same logical
# distributional value. The last such write is the final Ux string. This
# mirrors the grouping keys _load_uxhw uses when picking the last write.
_IDENTITY_COLUMNS = (
    "Expression_DeclarationFileName",
    "Expression_Subprogram",
    "Expression_Name",
    "Expression_DeclarationLineNumber",
    "UR_Type",
    "UR_Order",
    "UR_Order_CoreLibrary",
    "CorrelationTracking_Status",
)


def _load_final_ux_strings(db_path: str, table: str) -> dict[tuple[object, ...], str]:
    """
    Load the final Ux string for each traced expression from a tracing DB.

    Joins *table* to ``Emulator_Execution_Info`` on the execution foreign key,
    orders by ``rowid`` (insertion order), and keeps the last write per
    identity group. A program can write a value, jump back, and overwrite it,
    so the last write (not the highest PC / assignment index) is the final
    value — matching ``_load_uxhw``'s ``.last()`` semantics.

    Args:
        db_path: Path to the tracing SQLite database.
        table: Name of the traced-values table (e.g. ``TracingTable``).

    Returns:
        Mapping from the identity tuple (``_IDENTITY_COLUMNS`` order) to the
        last-written ``Dist_Value`` for that expression.
    """
    identity_select = ", ".join(f't."{c}"' for c in _IDENTITY_COLUMNS[:4])
    exec_select = ", ".join(f'e."{c}"' for c in _IDENTITY_COLUMNS[4:])
    query = (
        f"SELECT {identity_select}, {exec_select}, t.Dist_Value "
        f'FROM "{table}" t '
        f'JOIN "{_EXECUTION_INFO_TABLE}" e '
        f"ON e.Execution_ID = t.Execution_Info_Table_ID "
        f"ORDER BY t.rowid"
    )
    final: dict[tuple[object, ...], str] = {}
    with closing(sqlite3.connect(db_path)) as connection:
        for row in connection.execute(query):
            key = tuple(row[: len(_IDENTITY_COLUMNS)])
            # Insertion order is preserved by ORDER BY rowid, so overwriting
            # here leaves the last write as the final value.
            final[key] = row[len(_IDENTITY_COLUMNS)]
    return final


class ComparisonResult:
    """Outcome of comparing two tracing DBs' Ux strings."""

    def __init__(
        self,
        *,
        matched: int,
        mismatches: list[tuple[tuple[object, ...], str, str]],
        only_in_baseline: list[tuple[object, ...]],
        only_in_candidate: list[tuple[object, ...]],
    ) -> None:
        self.matched = matched
        self.mismatches = mismatches
        self.only_in_baseline = only_in_baseline
        self.only_in_candidate = only_in_candidate

    @property
    def is_identical(self) -> bool:
        """True when every expression matched and none was missing on a side."""
        return (
            not self.mismatches
            and not self.only_in_baseline
            and not self.only_in_candidate
        )


def compare_ux_strings(
    baseline_db: str, candidate_db: str, table: str
) -> ComparisonResult:
    """
    Compare the final Ux strings of two tracing DBs for the same application.

    Args:
        baseline_db: Path to the baseline (``-O0``) tracing DB.
        candidate_db: Path to the candidate (``-O2``) tracing DB.
        table: Name of the traced-values table in both DBs.

    Returns:
        A :class:`ComparisonResult` describing matches, byte-level mismatches,
        and expressions present in only one of the two builds.
    """
    baseline = _load_final_ux_strings(baseline_db, table)
    candidate = _load_final_ux_strings(candidate_db, table)

    matched = 0
    mismatches: list[tuple[tuple[object, ...], str, str]] = []
    for key, baseline_value in baseline.items():
        if key not in candidate:
            continue
        if candidate[key] == baseline_value:
            matched += 1
        else:
            mismatches.append((key, baseline_value, candidate[key]))

    only_in_baseline = [key for key in baseline if key not in candidate]
    only_in_candidate = [key for key in candidate if key not in baseline]

    return ComparisonResult(
        matched=matched,
        mismatches=mismatches,
        only_in_baseline=only_in_baseline,
        only_in_candidate=only_in_candidate,
    )


def _format_key(key: tuple[object, ...]) -> str:
    """Render an identity tuple as ``expr @ file:line [UR_Type/UR_Order/...]``."""
    fields = dict(zip(_IDENTITY_COLUMNS, key))
    return (
        f'{fields["Expression_Name"]} @ '
        f'{fields["Expression_DeclarationFileName"]}:'
        f'{fields["Expression_DeclarationLineNumber"]} '
        f'[{fields["UR_Type"]}/{fields["UR_Order"]}/'
        f'{fields["CorrelationTracking_Status"]}]'
    )


def _report(
    result: ComparisonResult,
    *,
    baseline_label: str,
    candidate_label: str,
    config: str | None,
) -> None:
    """Print a human-readable summary of *result* to stdout."""
    scope = f" for config '{config}'" if config else ""
    if result.is_identical:
        print(
            f"ux-string check{scope}: OK — {result.matched} traced "
            f"expression(s) identical between {baseline_label} and "
            f"{candidate_label} builds."
        )
        return

    print(
        f"WARNING: ux-string check{scope}: {baseline_label} and "
        f"{candidate_label} builds produced different Ux strings "
        f"({result.matched} identical, {len(result.mismatches)} differing, "
        f"{len(result.only_in_baseline)} only in {baseline_label}, "
        f"{len(result.only_in_candidate)} only in {candidate_label})."
    )
    for key, baseline_value, candidate_value in result.mismatches:
        print(f"  DIFF {_format_key(key)}")
        print(f"    {baseline_label}: {baseline_value}")
        print(f"    {candidate_label}: {candidate_value}")
    for key in result.only_in_baseline:
        print(f"  ONLY IN {baseline_label}: {_format_key(key)}")
    for key in result.only_in_candidate:
        print(f"  ONLY IN {candidate_label}: {_format_key(key)}")


def main() -> int:
    """
    Parse arguments, compare the two tracing DBs, and report the result.

    Returns:
        ``0`` when the two builds produced identical Ux strings, ``1`` when
        they differed, and ``2`` when the comparison could not be performed
        (e.g. a DB was missing or had an incompatible schema). The bash caller
        treats any non-zero status as a warning and continues.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare the final Ux strings of two tracing DBs built at "
            "different optimisation levels, warning on any byte-level difference."
        ),
    )
    parser.add_argument("baseline_db", help="Baseline (e.g. -O0) tracing DB.")
    parser.add_argument("candidate_db", help="Candidate (e.g. -O2) tracing DB.")
    parser.add_argument(
        "--table",
        default=EquivMC.TRACING_TABLE,
        help=f"Traced-values table name (default: {EquivMC.TRACING_TABLE}).",
    )
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
        result = compare_ux_strings(args.baseline_db, args.candidate_db, args.table)
    except (OSError, sqlite3.Error) as exc:
        print(
            f"WARNING: ux-string check could not compare "
            f"{args.baseline_db} and {args.candidate_db}: {exc}",
            file=sys.stderr,
        )
        return 2

    _report(
        result,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        config=args.config,
    )
    return 0 if result.is_identical else 1


if __name__ == "__main__":
    sys.exit(main())
