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

from contextlib import closing
from sqlite3 import Connection
import sqlite3
from typing import Callable, TypeVar, cast
import pandas as pd
from signaloid.distributional.distributional import DistributionalValue
from signaloid.benchmarking.types import TaggedDistributionalValue
from signaloid.benchmarking.config import (
    VariableTypes,
    RepresentationTypes,
    STRING_TO_CORE_REPRESENTATION,
)
import os
import struct


def _connect_to_database(db_path: str) -> sqlite3.Connection:
    """
    Connect to an existing SQLite database.

    Args:
        db_path: Path to the database file.

    Returns:
        An open connection to the database.

    Raises:
        FileNotFoundError: If the file is missing or not a valid SQLite database.
    """
    try:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"{db_path} does not exist.")

        con = sqlite3.connect(db_path)
        try:
            con.execute("SELECT 1").fetchone()  # verify it's a valid SQLite database
        except Exception:
            con.close()
            raise

        return con

    except FileNotFoundError:
        raise
    except sqlite3.DatabaseError as e:
        raise FileNotFoundError(f"{db_path} is not a valid SQLite database.") from e
    except Exception as e:
        raise FileNotFoundError(f"Could not connect to {db_path}: {e}") from e


_LoadResult = TypeVar("_LoadResult")


def _load_with_connection(
    db_path: str,
    table: str,
    target_expression: str,
    loader: Callable[[sqlite3.Connection, str, str], _LoadResult],
) -> _LoadResult:
    """
    Open ``db_path``, run ``loader`` against the connection, then close it.
    """
    with closing(_connect_to_database(db_path)) as con:
        return loader(con, table, target_expression)


def _load_mc(
    db_path: str, table: str, target_expression: str
) -> TaggedDistributionalValue:
    """
    Load Monte Carlo samples data from SQLite database into a
    TaggedDistributionalValue.
    """
    return _load_with_connection(db_path, table, target_expression, _load_mc_with_con)


def _load_mc_scalar(
    db_path: str, table: str, target_expression: str
) -> list[TaggedDistributionalValue]:
    """
    Load Monte Carlo samples data from SQLite database into a list of
    TaggedDistributionalValue.
    """
    return _load_with_connection(
        db_path, table, target_expression, _load_scalar_mc_with_con
    )


def _load_weighted_samples(
    db_path: str, table: str, target_expression: str
) -> TaggedDistributionalValue:
    """
    Load weighted samples data from SQLite database into a
    TaggedDistributionalValue.
    """
    return _load_with_connection(
        db_path, table, target_expression, _load_weighted_samples_with_con
    )


def _load_weighted_samples_scalar(
    db_path: str, table: str, target_expression: str
) -> list[TaggedDistributionalValue]:
    """
    Load weighted samples data from SQLite database into a list of
    TaggedDistributionalValue.
    """
    return _load_with_connection(
        db_path, table, target_expression, _load_scalar_weighted_samples_with_con
    )


def _load_ground_truth(
    db_path: str,
    table: str,
    target_expression: str,
    monte_carlo: bool,
    expression_type: str,
) -> TaggedDistributionalValue:
    """
    Load ground truth samples, via either mc or weighted samples.
    """
    if monte_carlo:
        if expression_type == VariableTypes.DISTRIBUTION:
            tagged = _load_mc(db_path, table, target_expression)
        else:
            # Ground truth is just a scalar
            tagged_list = _load_mc_scalar(db_path, table, target_expression)
            tagged = _max_mc_count(tagged_list)
    else:
        if expression_type == VariableTypes.DISTRIBUTION:
            tagged = _load_weighted_samples(db_path, table, target_expression)
        elif expression_type == VariableTypes.SCALAR:
            tagged_list = _load_weighted_samples_scalar(
                db_path, table, target_expression
            )
            tagged = _max_mc_count(tagged_list)
        else:
            raise ValueError(f"Error! Expression type {expression_type} not supported!")

    return tagged


def _load_uxhw_distributions(
    db_path: str,
    tables: list[str],
    target_expr: str,
    ur_types: list[str],
    ur_sizes: list[int],
) -> list[TaggedDistributionalValue]:
    """
    Load and concatenate UxHw distributions across DB tables.
    """
    result: list[TaggedDistributionalValue] = []
    for table in tables:
        result.extend(
            _load_uxhw(
                db_path=db_path,
                table=table,
                target_expression=target_expr,
                ur_types=ur_types,
                ur_sizes=ur_sizes,
                value_id=None,
            )
        )
    return result


def _max_mc_count(
    tagged_list: list[TaggedDistributionalValue],
) -> TaggedDistributionalValue:
    """
    Pick the largest-``mc_count`` scalar carrier as ground truth.

    The scalar loaders set ``mc_count`` from the DB row. We read it back off
    the carrier (never off a ``Distribution``) to select the highest-MC scalar.
    """
    valid_objects = [obj for obj in tagged_list if obj.mc_count is not None]
    if not valid_objects:
        raise ValueError("No valid objects with non-None mc_count found")

    def get_mc_count(obj: TaggedDistributionalValue) -> int:
        assert obj.mc_count is not None  # We filtered these out above
        return obj.mc_count

    return max(valid_objects, key=get_mc_count)


def _load_mc_with_con(
    con: Connection, table: str, target_expression: str
) -> TaggedDistributionalValue:
    """
    Load Monte Carlo samples data from SQLite database connection into a
    TaggedDistributionalValue.
    """

    mc_data_select = [
        "Expression_Name",
        "Expression_Subprogram",
        "Expression_DeclarationFileName",
        "Expression_DeclarationLineNumber",
        "MC_Id",
        "Particle_Value",
        "Assignment_Index",
        "ValueId",
    ]

    mc_data_df = pd.read_sql_query(
        f'SELECT {",".join(mc_data_select)} FROM "{table}" WHERE Expression_Name = ?',
        con,
        params=(target_expression,),
    )

    # Note: Monte Carlo data does not need the Emulator_Execution_Info table.

    mc_data_info_df = mc_data_df.groupby(
        [
            "Expression_DeclarationFileName",
            "Expression_Subprogram",
            "Expression_Name",
            "Expression_DeclarationLineNumber",
            "Assignment_Index",
            "ValueId",
        ],
        as_index=False,
    ).aggregate(func={"Particle_Value": list, "MC_Id": len})

    # Multiple subprograms tracing the same expression would be ambiguous.
    uniq_subprogram = mc_data_info_df["Expression_Subprogram"].unique()
    if len(uniq_subprogram) != 1:
        raise ValueError(
            "Data ambiguity error: Expression_Subprogram unique values count after "
            + f"filtering is not 1: len(uniq_subprogram)=={len(uniq_subprogram)}"
        )
    assert len(uniq_subprogram) == 1

    # Multiple declaration line numbers tracing the same expression would be ambiguous.
    uniq_declaration_line_number = mc_data_info_df[
        "Expression_DeclarationLineNumber"
    ].unique()
    if len(uniq_declaration_line_number) != 1:
        raise ValueError(
            "Data ambiguity error: Expression_DeclarationLineNumber unique values count"
            + " after filtering is not 1: "
            + f"len(uniq_declaration_line_number)=={len(uniq_declaration_line_number)}"
        )
    assert len(uniq_declaration_line_number) == 1

    # When an expression has multiple assignments, keep only the last one.
    max_assignment_index = mc_data_info_df["Assignment_Index"].max()
    mc_data_info_df = mc_data_info_df[
        mc_data_info_df["Assignment_Index"] == max_assignment_index
    ]

    mc_data_info_sr = mc_data_info_df.squeeze()

    mc_samples = mc_data_info_sr["Particle_Value"]
    # Create return value
    dv = DistributionalValue.from_samples(mc_samples)

    return TaggedDistributionalValue(
        dv=dv,
        representation_type=RepresentationTypes.MONTE_CARLO,
        representation_size=dv.UR_order,
    )


def _load_scalar_mc_with_con(
    con: Connection, table: str, target_expression: str
) -> list[TaggedDistributionalValue]:
    """
    Load Monte Carlo samples data from SQLite database connection into a list
    of TaggedDistributionalValue (one per unique grouping of metadata).
    """

    group_cols = [
        "Expression_DeclarationFileName",
        "Expression_Subprogram",
        "Expression_Name",
        "Expression_DeclarationLineNumber",
        "Assignment_Index",
        "ValueId",
        "MonteCarlo_Count",
    ]
    mc_data_select = group_cols + ["Particle_Value"]

    # Query the database
    query = (
        f'SELECT {",".join(mc_data_select)} FROM "{table}" WHERE Expression_Name = ?'
    )
    mc_data_df = pd.read_sql_query(query, con, params=(target_expression,))
    if mc_data_df.empty:
        raise ValueError(f"No results found for query: {query}")

    # Group by all these columns, aggregate Particle_Value into list
    grouped_df = mc_data_df.groupby(group_cols, as_index=False).agg(
        {"Particle_Value": list}
    )

    distributions = []
    for _, row in grouped_df.iterrows():
        mc_samples = row["Particle_Value"]
        try:
            dv = DistributionalValue.from_samples(mc_samples)
        except ValueError as e:
            print(
                f"Failed creating DistributionalValue.from_samples: {e}. "
                f"Skipping sample"
            )
            continue
        distributions.append(
            TaggedDistributionalValue(
                dv=dv,
                representation_type=RepresentationTypes.MONTE_CARLO,
                representation_size=dv.UR_order,
                mc_count=row["MonteCarlo_Count"],
            )
        )

    return distributions


def _load_scalar_weighted_samples_with_con(
    con: Connection, table: str, target_expression: str
) -> list[TaggedDistributionalValue]:
    """
    Load weighted samples data from SQLite database connection into a list of
    TaggedDistributionalValue objects, grouped by MonteCarlo_Count.
    """

    weighted_samples_data_select = [
        "Expression_Name",
        "Expression_Subprogram",
        "Expression_DeclarationFileName",
        "Expression_DeclarationLineNumber",
        "ValueId",
        "Position",
        "Weight",
        "MonteCarlo_Count",
    ]

    weighted_samples_data_df = pd.read_sql_query(
        f'SELECT {",".join(weighted_samples_data_select)} FROM "{table}" '
        "WHERE Expression_Name = ?",
        con,
        params=(target_expression,),
    )

    # Group by all key fields including MonteCarlo_Count
    grouped = weighted_samples_data_df.groupby(
        [
            "Expression_DeclarationFileName",
            "Expression_Subprogram",
            "Expression_Name",
            "Expression_DeclarationLineNumber",
            "ValueId",
            "MonteCarlo_Count",
        ],
        as_index=False,
    ).aggregate(func={"Position": list, "Weight": list})

    distributions = []

    for _, row in grouped.iterrows():
        positions = row["Position"]
        masses = row["Weight"]
        value_id = row["ValueId"]
        mc_count = row["MonteCarlo_Count"]
        try:
            dv = DistributionalValue.from_weighted_samples(positions, masses)
            distributions.append(
                TaggedDistributionalValue(
                    dv=dv,
                    representation_type=RepresentationTypes.WEIGHTED_SAMPLES,
                    representation_size=dv.UR_order,
                    mc_count=mc_count,
                )
            )
        except ValueError as e:
            raise ValueError(
                f"{e}: DistributionalValue.from_weighted_samples failed:\n"
                f"{row['Expression_DeclarationFileName']}:\n"
                f"{row['Expression_DeclarationLineNumber']}:\n"
                f"({row['Expression_Name']}), ValueID:{value_id}\n"
            ) from e

    return distributions


def _load_weighted_samples_with_con(
    con: Connection, table: str, target_expression: str
) -> TaggedDistributionalValue:
    """
    Load weighted samples data from SQLite database connection into a
    TaggedDistributionalValue.
    """

    # Select only the needed columns and filter at the DB level.
    query = f"""
    SELECT
        Expression_Subprogram,
        Expression_DeclarationFileName,
        Expression_DeclarationLineNumber,
        Expression_Name,
        ValueId,
        Position,
        Weight,
        COUNT(*) as Id
    FROM "{table}"
    WHERE Expression_Name = ?
    GROUP BY Expression_Subprogram, Expression_DeclarationFileName,
             Expression_DeclarationLineNumber, Expression_Name, ValueId
    """

    # Use parameterized query for safety and potentially better caching
    weighted_samples_data_info_df = pd.read_sql_query(
        query, con, params=(target_expression,)
    )

    # Early exit if no data
    if weighted_samples_data_info_df.empty:
        raise ValueError(f"No data found for expression: {target_expression}")

    # Validate uniqueness (faster on aggregated data)
    uniq_subprogram = weighted_samples_data_info_df["Expression_Subprogram"].unique()
    if len(uniq_subprogram) != 1:
        raise RuntimeError(
            f"Data ambiguity error: Expression_Subprogram unique values count after "
            f"filtering is not 1: len(uniq_subprogram)=={len(uniq_subprogram)}"
        )

    uniq_declaration_line_number = weighted_samples_data_info_df[
        "Expression_DeclarationLineNumber"
    ].unique()
    if len(uniq_declaration_line_number) != 1:
        raise ValueError(
            f"Data ambiguity error: Expression_DeclarationLineNumber unique values count "
            f"after filtering is not 1: len(uniq_declaration_line_number)=={len(uniq_declaration_line_number)}"
        )

    # Get the single row
    row = weighted_samples_data_info_df.iloc[0]

    # Need to fetch position and weight lists separately since GROUP BY doesn't
    # support array aggregation in SQLite
    positions_weights_query = f"""
    SELECT Position, Weight
    FROM "{table}"
    WHERE Expression_Name = ?
    ORDER BY Id
    """
    positions_weights_df = pd.read_sql_query(
        positions_weights_query, con, params=(target_expression,)
    )

    positions = positions_weights_df["Position"].tolist()
    masses = positions_weights_df["Weight"].tolist()

    # Create return value
    try:
        dv = DistributionalValue.from_weighted_samples(positions, masses)
    except ValueError as e:
        raise ValueError(
            f"ValueError {e}: DistributionalValue.from_weighted_samples: "
            f"did not parse UxHw Dist_Value \n"
            f"{row['Expression_DeclarationFileName']}:\n"
            f"{row['Expression_DeclarationLineNumber']}:\n"
            f"({row['Expression_Name']}):\n"
            f" ValueID:{row['ValueId']}\n"
        ) from e

    return TaggedDistributionalValue(
        dv=dv,
        representation_type=RepresentationTypes.WEIGHTED_SAMPLES,
        representation_size=dv.UR_order,
    )


def _filter_duplicate_ux_writes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out duplicate Evaluation rows from a database traced multiple times.

    A repeated trace can leave several rows under one Execution_ID for the same
    expression, sometimes with a different representation type (e.g. Jupiter
    data under an Athens Execution_ID). For such groups we keep the row whose Ux
    UR_type matches the expected UR_Type from Emulator_Execution_Info, falling
    back to the first row when none match (e.g. Atlas stores as Ux06 internally).

    Args:
        df: Merged UxHw-data / execution-info rows, keyed by Execution_ID,
            Expression_Name, ValueId, and Assignment_Index.

    Returns:
        The same rows with each duplicate group reduced to a single row.
    """
    group_cols = [
        "Execution_ID",
        "Expression_Name",
        "ValueId",
        "Assignment_Index",
    ]

    # Check if there are any duplicates at all (fast path)
    if not df.duplicated(subset=group_cols, keep=False).any():
        return df

    result_rows = []

    for _, group in df.groupby(group_cols, sort=False):
        if len(group) == 1:
            result_rows.append(group)
            continue

        # Multiple rows: prefer the one whose Ux UR_type matches UR_Type
        matching = []
        for idx, row in group.iterrows():
            ux_ur_type = _extract_ur_type_from_ux_string(row["Dist_Value"])
            expected_str = RepresentationTypes.from_uxhw_db(row["UR_Type"])
            if expected_str in STRING_TO_CORE_REPRESENTATION:
                expected_int = STRING_TO_CORE_REPRESENTATION[expected_str]
                if ux_ur_type == expected_int:
                    matching.append(idx)

        if matching:
            result_rows.append(group.loc[matching[:1]])
        else:
            # No Ux type matches (e.g., Atlas). Keep the first row
            result_rows.append(group.iloc[:1])

    result = pd.concat(result_rows, ignore_index=True)

    return result


def _load_uxhw(
    db_path: str,
    table: str,
    target_expression: str,
    ur_types: list[str],
    ur_sizes: list[int],
    value_id: str | None = None,
) -> list[TaggedDistributionalValue]:
    """
    Load UxHw traced values from a SQLite database.

    Filters by ``target_expression``, ``ur_types``, and ``ur_sizes``. When
    ``value_id`` is given, picks the row with that ValueId. Otherwise picks the
    last database write for each group.

    Args:
        db_path: Path to the SQLite database file.
        table: Name of the table holding the traced values.
        target_expression: Expression name to filter rows by.
        ur_types: Representation types to include (e.g. ``["Athens"]``).
        ur_sizes: Representation sizes to include.
        value_id: Specific ValueId to select. If ``None``, the last write per
            group is used.

    Returns:
        The matching values as a list of ``TaggedDistributionalValue``.
    """
    # Scope the connection with `closing()` so it is released on every exit
    # path, including the error `raise`s below (the sibling loaders get this
    # from `_load_with_connection`. This one has extra params so it opens its
    # own connection).
    with closing(_connect_to_database(db_path)) as con:
        # Map each requested type to its DB name (currently an identity mapping;
        # see RepresentationTypes.TO_UXHW_DB).
        uxhw_db_ur_types = []
        for ur_type in ur_types:
            uxhw_db_ur_types.append(RepresentationTypes.to_uxhw_db(ur_type))

        uxhw_data_select = [
            "ValueId",
            "Expression_Name",
            "Expression_Subprogram",
            "Expression_DeclarationFileName",
            "Expression_DeclarationLineNumber",
            "Execution_Info_Table_ID",
            "Particle_Value",
            "Dist_Value",
            "Assignment_Index",
        ]

        uxhw_data_query = f'SELECT {",".join(uxhw_data_select)} FROM "{table}" WHERE Expression_Name = ?'
        params: list[str] = [target_expression]
        if value_id is not None:
            uxhw_data_query += " AND ValueId = ?"
            params.append(value_id)

        uxhw_data_df = pd.read_sql_query(uxhw_data_query, con, params=tuple(params))
        if uxhw_data_df.empty:
            raise ValueError("ValueError: " + uxhw_data_query + ": no results ")

        ex_info_select = [
            "Execution_ID",
            "UR_Type",
            "UR_Order",
            "UR_Order_CoreLibrary",
            "CorrelationTracking_Status",
        ]

        if not uxhw_db_ur_types or not ur_sizes:
            raise ValueError("ur_types and ur_sizes must be non-empty")

        ur_type_placeholders = ",".join(["?"] * len(uxhw_db_ur_types))
        size_placeholders = ",".join(["?"] * len(ur_sizes))
        ex_info_query = (
            f'SELECT {",".join(ex_info_select)} FROM "Emulator_Execution_Info" '
            f"WHERE (UR_Type IN ({ur_type_placeholders})) "
            f"AND (UR_Order IN ({size_placeholders}) "
            f"OR UR_Order_CoreLibrary IN ({size_placeholders}))"
        )
        ex_info_params: list[str | int] = [*uxhw_db_ur_types, *ur_sizes, *ur_sizes]

        ex_info_df = pd.read_sql_query(ex_info_query, con, params=tuple(ex_info_params))
        if ex_info_df.empty:
            print(
                f"⚠️  WARNING: Execution info query returned no results:\n{ex_info_query}"
            )
            raise ValueError("No results from Execution info query.")

        uxhw_data_info_df = ex_info_df.merge(
            uxhw_data_df,
            how="inner",
            left_on=["Execution_ID"],
            right_on=["Execution_Info_Table_ID"],
        )

        if uxhw_data_info_df.empty:
            print(
                "⚠️ WARNING: Merge of UxHw and Execution info dataframes returned no results."
            )
            raise ValueError("No results after merging UxHw data and execution info.")

    uxhw_data_info_df = _filter_duplicate_ux_writes(uxhw_data_info_df)

    # With no ValueId anchor, several values may satisfy the filters. Pick the
    # most recently written one per group.
    if value_id is None:
        # `.last()` takes the most recently written row, using row (write) order
        # rather than the largest PC / Assignment_Index: a program can write
        # from a high PC then jump back and overwrite, so the highest PC is not
        # necessarily the last write.
        uxhw_data_info_df = uxhw_data_info_df.groupby(
            [
                "Expression_DeclarationFileName",
                "Expression_Subprogram",
                "Expression_Name",
                "Expression_DeclarationLineNumber",
                "UR_Type",
                "UR_Order",
                "UR_Order_CoreLibrary",
                "CorrelationTracking_Status",
            ],
            as_index=False,
        ).last()

    dist_values_sr = uxhw_data_info_df.apply(  # type: ignore[call-overload]
        _dist_value_from_row, axis="columns", result_type="reduce"
    )

    return cast(list[TaggedDistributionalValue], dist_values_sr.tolist())


def _extract_ur_type_from_ux_string(dist_value: str) -> int | None:
    """
    Extract the UR_type byte from a Ux hex string without full parsing.

    The hex buffer starts at the first character after "Ux", so the UR_type is
    the first byte (hex chars 0-1).

    Args:
        dist_value: The raw ``Dist_Value`` string from the database.

    Returns:
        The UR_type byte as an int, or ``None`` if ``dist_value`` is not a
        parseable Ux string.
    """
    if not isinstance(dist_value, str) or "Ux" not in dist_value:
        return None
    try:
        hex_str = dist_value.split("Ux", 1)[1]
        # Byte 0: UR_type (2 hex chars)
        ur_type_byte = bytes.fromhex(hex_str[0:2])
        return int(struct.unpack(">B", ur_type_byte)[0])
    except (ValueError, struct.error, IndexError):
        return None


def _dist_value_from_row(uxhw_data_info_sr: pd.Series) -> TaggedDistributionalValue:
    """
    Convert a database row of UxHw traced value data into a
    TaggedDistributionalValue.

    Args:
        uxhw_data_info_sr: One merged UxHw-data / execution-info row.

    Returns:
        The row's distribution and metadata as a ``TaggedDistributionalValue``.
    """

    dist_value_raw = uxhw_data_info_sr["Dist_Value"]
    dv = DistributionalValue.parse(dist_value_raw)
    if dv is None:
        raise ValueError(f"Parsing failed for {dist_value_raw!r}.")

    # Read metadata from the DB columns, never off the parsed Distribution.
    representation_type = RepresentationTypes.from_uxhw_db(uxhw_data_info_sr["UR_Type"])
    # UR_Order is the atom count N. For Athens, UR_Order_CoreLibrary
    # is N+3 (it also counts 3 internal sentinel Diracs). Use UR_Order so every
    # output is labelled by build size N, including scalars that collapse to a
    # single atom. External CSVs set both columns to N.
    if representation_type == RepresentationTypes.ATHENS:
        representation_size = uxhw_data_info_sr["UR_Order"]
    else:
        representation_size = uxhw_data_info_sr["UR_Order_CoreLibrary"]

    return TaggedDistributionalValue(
        dv=dv,
        representation_type=representation_type,
        representation_size=representation_size,
        correlation_tracking=uxhw_data_info_sr["CorrelationTracking_Status"],
    )
