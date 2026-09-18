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
Check that every Ux string a tracing run recorded was also printed by it.

The ``-O2`` verification compares the two builds' **stdout**, because the
verification build is compiled without tracing and so has no tracing database
(see ``compare_tracing_ux_strings``). That substitution is only sound while the
SDK prints the same bytes it stores. This module checks exactly that, against
the one run that produces both: the ``-O0`` tracing build writes its traced
values to a database *and* prints them.

A mismatch here means the printed representation has drifted from the stored
one, which would quietly weaken the ``-O2`` check rather than break it. It is
therefore reported as a warning, and the caller continues.

Invoked from the bash tracing layer (``get-timings.sh``) as::

    python3 -m signaloid.benchmarking.automation.check_traced_values_printed \
        <tracing_db> <stdout> [--table TABLE] [--config SUFFIX]

Exit status is ``0`` when every recorded value was printed, ``1`` when any was
not or when the database recorded nothing, and ``2`` when the check could not
be performed.
"""

import argparse
import sqlite3
import sys
from contextlib import closing

from signaloid.benchmarking.automation.compare_tracing_ux_strings import (
    load_ux_strings,
)

# Enough to name which traced expression is missing from stdout.
_IDENTITY_COLUMNS = (
    "Expression_DeclarationFileName",
    "Expression_Name",
    "Expression_DeclarationLineNumber",
)

# A single Athens-16 value runs to several hundred characters, so reports
# truncate. See compare_tracing_ux_strings for the same reasoning.
_REPORTED_PREFIX_LENGTH = 80


def load_recorded_values(db_path: str, table: str) -> dict[str, tuple[object, ...]]:
    """
    Load the distinct Ux strings a tracing run wrote, with one identity each.

    Args:
        db_path: Path to the tracing SQLite database.
        table: Name of the traced-values table (e.g. ``TracingTable``).

    Returns:
        Mapping from Ux string to the identity tuple of an expression that
        produced it. One identity is kept per distinct value, which is all the
        report needs to point at the offending expression.
    """
    columns = ", ".join(f'"{column}"' for column in _IDENTITY_COLUMNS)
    query = f'SELECT {columns}, Dist_Value FROM "{table}"'
    recorded: dict[str, tuple[object, ...]] = {}
    with closing(sqlite3.connect(db_path)) as connection:
        for row in connection.execute(query):
            recorded.setdefault(row[-1], tuple(row[:-1]))
    return recorded


def find_unprinted_values(
    db_path: str, stdout_path: str, table: str
) -> tuple[list[tuple[str, tuple[object, ...]]], int]:
    """
    Find recorded Ux strings that the same run did not print.

    Args:
        db_path: Path to the tracing SQLite database.
        stdout_path: Captured stdout of the same run.
        table: Name of the traced-values table.

    Returns:
        A tuple of (unprinted values with their identities, number of distinct
        values recorded).
    """
    recorded = load_recorded_values(db_path, table)
    printed = set(load_ux_strings(stdout_path))
    unprinted = [
        (value, identity)
        for value, identity in recorded.items()
        if value not in printed
    ]
    return unprinted, len(recorded)


def _format_identity(identity: tuple[object, ...]) -> str:
    """Render an identity tuple as ``expr @ file:line``."""
    file_name, name, line = identity
    return f"{name} @ {file_name}:{line}"


def main() -> int:
    """
    Parse arguments, run the cross-check, and report the result.

    Returns:
        ``0`` when every recorded value was printed, ``1`` when any was not or
        when nothing was recorded, and ``2`` when the check could not run.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Check that every Ux string a tracing run recorded in its "
            "database was also printed to stdout by that same run."
        ),
    )
    parser.add_argument("tracing_db", help="Tracing DB written by the -O0 run.")
    parser.add_argument("stdout_path", help="Captured stdout of the same run.")
    parser.add_argument(
        "--table",
        default="TracingTable",
        help="Name of the traced-values table (default: TracingTable).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config suffix, included in messages for context.",
    )
    args = parser.parse_args()

    scope = f" for config '{args.config}'" if args.config else ""

    try:
        unprinted, recorded_count = find_unprinted_values(
            args.tracing_db, args.stdout_path, args.table
        )
    except (OSError, sqlite3.Error) as exc:
        print(
            f"WARNING: db-vs-stdout check{scope} could not run on "
            f"{args.tracing_db} and {args.stdout_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    if recorded_count == 0:
        print(
            f"WARNING: db-vs-stdout check{scope}: the tracing run recorded no "
            f"Ux strings, so nothing was cross-checked."
        )
        return 1

    if not unprinted:
        print(
            f"db-vs-stdout check{scope}: OK — all {recorded_count} recorded "
            f"Ux string(s) also appear in the run's stdout."
        )
        return 0

    print(
        f"WARNING: db-vs-stdout check{scope}: {len(unprinted)} of "
        f"{recorded_count} recorded Ux string(s) were not printed by the same "
        f"run. The -O2 check compares stdout, so it no longer covers these."
    )
    for value, identity in unprinted:
        print(f"  NOT PRINTED {_format_identity(identity)}")
        print(f"    recorded: {value[:_REPORTED_PREFIX_LENGTH]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
