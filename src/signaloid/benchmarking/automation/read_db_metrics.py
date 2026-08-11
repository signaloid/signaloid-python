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

import argparse
import sqlite3
import sys
import urllib.parse

# Metric name to SQL query. Column and table names must match the schema
# exactly, preserving case, since SQLite is case-sensitive for identifiers in
# some configurations.
_METRIC_QUERIES: dict[str, str] = {
    "host-wallclock": ("SELECT Host_UserTimeElapsedWallClock FROM runtimeStats"),
    "host-wallclock-total": (
        "SELECT TOTAL(Host_UserTimeElapsedWallClock) FROM runtimeStats"
    ),
    "emulated-dyn-inst": ("SELECT EmulatedCPU_DynCnt FROM Emulator_Execution_Info"),
    "emulated-dyn-inst-total": ("SELECT TOTAL(EmulatedCPU_DynCnt) FROM runtimeStats"),
}


def read_db_metric(db_path: str, metric: str) -> float:
    """
    Query a single scalar metric from a UxHw/emulator SQLite database.

    Args:
        db_path: Path to the SQLite database file.
        metric: Metric name (one of the keys in :data:`_METRIC_QUERIES`).

    Returns:
        The queried value as a float.

    Raises:
        KeyError: If *metric* is not a recognised metric name.
        OSError: If the database file cannot be opened.
        sqlite3.OperationalError: If the query fails (e.g. missing table
            or column).
        ValueError: If the query returns no rows or a NULL value.
    """
    if metric not in _METRIC_QUERIES:
        valid = ", ".join(sorted(_METRIC_QUERIES))
        raise KeyError(f"unknown metric {metric!r}; valid choices: {valid}")
    sql = _METRIC_QUERIES[metric]
    # `uri=True` + `mode=ro` opens read-only and raises OperationalError if the
    # file is missing (instead of silently creating an empty DB). The path is
    # percent-encoded so paths containing `?` or `#` are not split into URI
    # query / fragment components.
    quoted_path = urllib.parse.quote(db_path)
    with sqlite3.connect(f"file:{quoted_path}?mode=ro", uri=True) as conn:
        cursor = conn.execute(sql)
        row = cursor.fetchone()
    if row is None or row[0] is None:
        raise ValueError(
            f"query returned no value for metric {metric!r} " f"from {db_path!r}"
        )
    return float(row[0])


def main() -> None:
    """
    Parse command-line arguments and print the queried metric value.

    Intended for the bash layer via ``python3 -m
    signaloid.benchmarking.automation.read_db_metrics``. Prints a single float
    to stdout.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read a single scalar metric from a UxHw/emulator " "SQLite database."
        ),
    )
    parser.add_argument(
        "db_path",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        choices=list(_METRIC_QUERIES),
        help="Metric to read from the database.",
    )
    args = parser.parse_args()
    print(read_db_metric(args.db_path, args.metric))


if __name__ == "__main__":
    try:
        main()
    except (OSError, sqlite3.OperationalError, ValueError, KeyError) as exc:
        print(f"read_db_metrics: {exc}", file=sys.stderr)
        sys.exit(1)
