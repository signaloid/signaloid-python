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
import os
import shutil
import sqlite3
import sys
from contextlib import closing

# Name of the table whose primary key is auto-incremented per source DB (and
# which the foreign keys in the other tracing tables refer to). Pulled out of
# the inline SQL so a schema rename only touches these constants.
_EXECUTION_INFO_TABLE = "Emulator_Execution_Info"
_EXECUTION_ID_COLUMN = "Execution_ID"
_FOREIGN_KEY_COLUMN = "Execution_Info_Table_ID"


def _get_table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    """
    Return the column names of *table* in the order ``PRAGMA`` lists.

    ``PRAGMA table_info`` yields rows of the form
    ``(cid, name, type, notnull, dflt_value, pk)``. The second column is the
    name.

    Args:
        connection: Open connection to the database holding *table*.
        table: Table whose column names to return.

    Returns:
        The column names, in ``PRAGMA table_info`` order.
    """
    cursor = connection.execute(f'PRAGMA table_info("{table}")')
    return [row[1] for row in cursor.fetchall()]


def _list_tables(connection: sqlite3.Connection) -> list[str]:
    """
    Return all user-table names in *connection* (excluding indices).

    Lists ordinary tables only (skipping views and internal tables)
    by filtering ``sqlite_master`` on ``type='table'`` and excluding the
    ``sqlite_*`` prefix.

    Args:
        connection: Open connection to inspect.

    Returns:
        The user-table names, sorted.
    """
    cursor = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def _insert_execution_info(connection: sqlite3.Connection, src_db: str) -> None:
    """
    Insert ``Emulator_Execution_Info`` rows from *src_db* into the target.

    The ``Execution_ID`` auto-increment column is excluded from both the column
    list and the SELECT so the target's ``INTEGER PRIMARY KEY`` assigns a fresh
    value to each inserted row.

    Args:
        connection: Open connection to the target database.
        src_db: Path to the source database to copy rows from.
    """
    columns = _get_table_columns(connection, _EXECUTION_INFO_TABLE)
    insert_columns = [c for c in columns if c != _EXECUTION_ID_COLUMN]
    column_list = ", ".join(f'"{c}"' for c in insert_columns)

    # ATTACH/DETACH around the INSERT scopes the cross-DB access to this
    # statement group and avoids leaving an attached connection between
    # source iterations.
    connection.execute("ATTACH DATABASE ? AS src", (src_db,))
    try:
        connection.execute(
            f'INSERT INTO "{_EXECUTION_INFO_TABLE}" ({column_list}) '
            f"SELECT {column_list} "
            f'FROM src."{_EXECUTION_INFO_TABLE}"'
        )
        # Python's sqlite3 opens an implicit transaction on DML, and DETACH
        # fails while one is open on the attached source, so commit first.
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE src")


def _insert_remapped_table(
    connection: sqlite3.Connection,
    src_db: str,
    table: str,
    id_offset: int,
) -> None:
    """
    Insert *src_db*'s rows for *table*, offsetting the FK column.

    The column list comes from the source's ``PRAGMA table_info``, and the
    SELECT replaces ``Execution_Info_Table_ID`` with
    ``Execution_Info_Table_ID + id_offset``. The offset is the pre-INSERT
    ``MAX(Execution_ID)`` of the target. Since each source starts at
    ``Execution_ID = 1``, adding it places the row at ``old_max + 1``, the new
    ``Execution_ID`` in the target.

    Args:
        connection: Open connection to the target database.
        src_db: Path to the source database to copy rows from.
        table: Table to copy (must contain the FK column).
        id_offset: Value added to each row's ``Execution_Info_Table_ID``.
    """
    # Read the schema from the source DB via a separate connection (rather than
    # the attached alias) so introspection is independent of ATTACH state.
    with closing(sqlite3.connect(src_db)) as src_connection:
        columns = _get_table_columns(src_connection, table)
    column_list = ", ".join(f'"{c}"' for c in columns)
    connection.execute("ATTACH DATABASE ? AS src", (src_db,))
    try:
        # id_offset comes from a SELECT result, not user input, so it is always
        # an integer. It is embedded inline rather than parameter-bound because
        # it appears inside a SELECT projection. Cast to int for safety.
        offset = int(id_offset)
        select_pieces = []
        for column in columns:
            if column == _FOREIGN_KEY_COLUMN:
                select_pieces.append(f'"{_FOREIGN_KEY_COLUMN}" + {offset}')
            else:
                select_pieces.append(f'"{column}"')
        select_list = ", ".join(select_pieces)
        connection.execute(
            f'INSERT INTO "{table}" ({column_list}) '
            f"SELECT {select_list} "
            f'FROM src."{table}"'
        )
        # sqlite3 opens an implicit transaction on DML, and DETACH
        # fails while one is open on the attached source, so commit first.
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE src")


def _insert_or_ignore_table(
    connection: sqlite3.Connection, src_db: str, table: str
) -> None:
    """
    Idempotent merge for tables without the FK column.

    Uses ``INSERT OR IGNORE`` so re-running the merge against an already-merged
    target is a no-op for these value-only lookup-style tables.

    Args:
        connection: Open connection to the target database.
        src_db: Path to the source database to copy rows from.
        table: Table to copy (has no FK column).
    """
    connection.execute("ATTACH DATABASE ? AS src", (src_db,))
    try:
        connection.execute(
            f'INSERT OR IGNORE INTO "{table}" ' f'SELECT * FROM src."{table}"'
        )
        # sqlite3 opens an implicit transaction on DML, and DETACH
        # fails while one is open on the attached source, so commit first.
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE src")


def _merge_one_source(target_db: str, src_db: str) -> None:
    """
    Merge a single source DB into an existing *target_db*.

    Assumes target_db already exists (this is not the first source). Computes
    the FK offset, copies the execution-info row, then walks the remaining
    tables, picking the FK-remapped or ``INSERT OR IGNORE`` branch by
    inspecting each table's columns.

    Args:
        target_db: Path to the existing target database.
        src_db: Path to the source database to merge in.
    """
    with sqlite3.connect(target_db) as connection:
        cursor = connection.execute(
            f'SELECT MAX("{_EXECUTION_ID_COLUMN}") ' f'FROM "{_EXECUTION_INFO_TABLE}"'
        )
        row = cursor.fetchone()
        # MAX() over an empty table is NULL. Treat as 0 so the first offset
        # places the source row at Execution_ID = 1.
        old_max = row[0] if row is not None and row[0] is not None else 0

        _insert_execution_info(connection, src_db)

        # The tables to remap come from the source DB so tables present in the
        # source but absent in the target are not silently dropped. ``closing``
        # is required so the schema-reading connection releases its lock on
        # src_db before the per-table merges ATTACH it on the target connection.
        with closing(sqlite3.connect(src_db)) as src_connection:
            tables = _list_tables(src_connection)
            fk_flags = {
                table: _FOREIGN_KEY_COLUMN in _get_table_columns(src_connection, table)
                for table in tables
                if table != _EXECUTION_INFO_TABLE
            }

        for table, has_fk in fk_flags.items():
            if has_fk:
                _insert_remapped_table(connection, src_db, table, old_max)
            else:
                _insert_or_ignore_table(connection, src_db, table)

        connection.commit()


def merge_tracing_dbs(*, target_db: str, source_dbs: list[str]) -> None:
    """Merge per-config tracing SQLite DBs into *target_db*.

    For the first source DB that exists, *target_db* is created via
    :func:`shutil.copy`. Subsequent source DBs are merged with
    ``Execution_Info_Table_ID`` remapped so that foreign keys point at
    the new ``Execution_ID`` in the target. Missing source DBs print
    a warning to stderr and are skipped. Each source DB is removed
    from disk after it is processed.

    Args:
        target_db: Destination SQLite DB path. May not exist. The
            first present source DB is copied to this path.
        source_dbs: Ordered list of per-config tracing DB paths.
            Each is consumed (deleted) after merging or after being
            reported as missing.

    Raises:
        sqlite3.OperationalError: If a source DB has an incompatible
            schema (e.g. the ``Emulator_Execution_Info`` table is
            absent).
        OSError: If file operations on *target_db* or any source
            fail.
    """
    for src_db in source_dbs:
        if not os.path.isfile(src_db):
            print(
                f"Warning: expected tracing DB {src_db} not found, " f"skipping",
                file=sys.stderr,
            )
            continue
        if not os.path.isfile(target_db):
            shutil.copy(src_db, target_db)
        else:
            _merge_one_source(target_db, src_db)
        # Remove every processed source (including the one used to seed the
        # target). Missing sources are handled by the `continue` above, so
        # os.remove here always has a file to remove.
        os.remove(src_db)


def main() -> None:
    """
    Parse command-line arguments and merge tracing DBs.

    Intended for the bash layer via ``python3 -m
    signaloid.benchmarking.automation.merge_tracing_dbs``. The first positional
    argument is the destination DB. The rest are the per-config source DBs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-config tracing SQLite DBs into a single "
            "target DB, remapping Execution_Info_Table_ID foreign "
            "keys to point at the new Execution_ID in the target."
        ),
    )
    parser.add_argument(
        "target_db",
        help="Destination SQLite DB path (created if absent).",
    )
    parser.add_argument(
        "source_dbs",
        nargs="+",
        help=(
            "One or more per-config source DB paths. Each is removed "
            "from disk after being merged."
        ),
    )
    args = parser.parse_args()
    merge_tracing_dbs(target_db=args.target_db, source_dbs=args.source_dbs)


if __name__ == "__main__":
    try:
        main()
    except (OSError, sqlite3.OperationalError, ValueError) as exc:
        print(f"merge_tracing_dbs: {exc}", file=sys.stderr)
        sys.exit(1)
