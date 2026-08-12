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

import glob
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from signaloid.benchmarking.automation.benchmarking_utils import (
    clean_data_for_sheets,
)
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
from signaloid.benchmarking.config import (
    BenchmarkingVariables,
    EquivMC,
    Measurements,
    MetadataRowLabels,
    ReportSheetTabs,
    VariableTypes,
)
from signaloid.distributional.distributional import DistributionalValue

# The Google Drive folder and Sheets template the upload writes to.
# Deployment-specific, so supply them via environment variables (the upload is
# opt-in via --write-sheets and is validated up front in benchmark_application).
TARGET_FOLDER_ID = os.environ.get("UXHW_SHEETS_DRIVE_FOLDER_ID", "")
TEMPLATE_ID = os.environ.get("UXHW_SHEETS_TEMPLATE_ID", "")

# Token rendered in place of EMCC / distance values for a blown-up (degraded)
# representation row, in both the markdown report and the Sheets export.
BLOW_UP_DISPLAY_TOKEN = "blow-up / excluded"


def _is_blow_up(value: Any) -> bool:
    """
    Whether an emcc_data BLOW_UP_REASON cell marks a degraded row.

    The marker is a non-empty reason string for a blown row. It is ``None``
    (in-memory) or NaN / empty (round-tripped through pandas / CSV) for a
    healthy row, all of which read as "not a blow-up".

    Args:
        value: A ``BLOW_UP_REASON`` cell value.

    Returns:
        ``True`` if the cell marks a blown-up row, else ``False``.
    """
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return str(value).strip() != ""


def write_results_to_markdown(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    results_dir: str,
) -> None:
    """
    Write the results to markdown format.

    Args:
        benchmarking_variables: List of benchmarking variables to write.
        results_dir: Directory in which to write the markdown files.
    """
    print("Writing results to Markdown files...")
    for variable in benchmarking_variables:
        df = pd.DataFrame(variable.emcc_results.emcc_data)
        if BenchmarkingVariables.UXHW_CONF in df.columns:
            df[BenchmarkingVariables.UXHW_CONF] = df[
                BenchmarkingVariables.UXHW_CONF
            ].apply(repr)
        # Show "blow-up / excluded" in the EMCC / distance columns of blown-up
        # rows, so a degraded representation reads as excluded rather than as a
        # meaningless small distance.
        if BenchmarkingVariables.BLOW_UP_REASON in df.columns:
            blown_mask = df[BenchmarkingVariables.BLOW_UP_REASON].apply(_is_blow_up)
            blow_up_columns = [
                col
                for col in (
                    BenchmarkingVariables.UXHW_DISTANCE,
                    BenchmarkingVariables.UXHW_BINNED_DISTANCE,
                    EquivMC.EMCC,
                    EquivMC.EMCC_PREDICTED,
                )
                if col in df.columns
            ]
            if blown_mask.any():
                for col in blow_up_columns:
                    # Cast to object first: assigning a string token into a
                    # numeric column otherwise raises a pandas
                    # incompatible-dtype FutureWarning.
                    df[col] = df[col].astype(object)
                    df.loc[blown_mask, col] = BLOW_UP_DISPLAY_TOKEN
        md_path = os.path.join(results_dir, f"{variable.formatted_description}.md")
        with open(md_path, "w") as f:
            f.write(
                tabulate(
                    df,  # type: ignore[arg-type]
                    headers="keys",
                    tablefmt="github",
                )
            )


def resolve_google_credentials_path(
    *,
    google_credentials: str | None,
) -> str | None:
    """
    Resolve the Google service-account credentials JSON path.

    The resolution order is:

    1. The ``google_credentials`` argument (typically from
       ``--google-credentials``).
    2. The ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.

    The resolved path is returned only when a file actually exists at
    that location. If no candidate path resolves to an existing file,
    a warning is printed and ``None`` is returned (this function never
    raises).

    Args:
        google_credentials: Explicit credentials path, or ``None`` to
            fall back to the environment variable.

    Returns:
        The resolved absolute path to an existing credentials file, or
        ``None`` if no candidate resolves to an existing file.
    """
    if google_credentials:
        json_path = os.path.abspath(os.path.expanduser(google_credentials))
    elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        json_path = os.path.abspath(
            os.path.expanduser(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        )
    else:
        print(
            "Warning: no --google-credentials and no "
            "GOOGLE_APPLICATION_CREDENTIALS set; cannot upload to Google Sheets."
        )
        return None

    if not os.path.isfile(json_path):
        print(f"Warning: Google credentials file not found at " f"'{json_path}'.")
        return None

    return json_path


def _get_credentials(*, credentials_path: str) -> Any:
    """
    Build a Google API service-account client for the given path.

    Args:
        credentials_path: Path to the service-account JSON key file.

    Returns:
        An authorised service-account credentials object.
    """
    from oauth2client.service_account import (  # type: ignore
        ServiceAccountCredentials,
    )

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    return ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)


def _escape_drive_query_value(value: str) -> str:
    """
    Escape a value for safe interpolation in a Drive query string.

    Per the Google Drive API, backslashes and single quotes inside
    string literals must be backslash-escaped. Escape backslashes
    first so the subsequent quote escape does not double-escape them.

    Args:
        value: The raw value to embed in a Drive query.

    Returns:
        The value with backslashes and single quotes escaped.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _get_or_create_folder(
    *,
    drive_service: Any,
    folder_name: str,
    parent_id: str,
) -> str:
    """
    Get the existing Drive folder by name under ``parent_id``, or create it.

    Args:
        drive_service: Authorised Drive API service.
        folder_name: Name of the folder to find or create.
        parent_id: ID of the parent folder to search within / create under.

    Returns:
        The folder's Drive ID.
    """
    safe_name = _escape_drive_query_value(folder_name)
    safe_parent = _escape_drive_query_value(parent_id)
    query = (
        f"name = '{safe_name}' "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and '{safe_parent}' in parents "
        f"and trashed = false"
    )
    # supportsAllDrives + includeItemsFromAllDrives let the lookup see folders
    # on a shared drive (the create/copy calls already opt in). Without them,
    # repeated uploads would mint a new folder every run instead of reusing it.
    response = (
        drive_service.files()
        .list(
            q=query,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    folders = response.get("files", [])

    if folders:
        return str(folders[0]["id"])

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = (
        drive_service.files()
        .create(body=metadata, fields="id", supportsAllDrives=True)
        .execute()
    )
    return str(folder["id"])


def _get_or_create_folder_structure(
    *,
    drive_service: Any,
    uxhw_version: str,
    application_name: str,
) -> str:
    """
    Get or create the ``uxhw-<version>/<application>`` results folder.

    Args:
        drive_service: Authorised Drive API service.
        uxhw_version: UxHw version, naming the top-level folder.
        application_name: Application name, naming the nested folder.

    Returns:
        The Drive ID of the application folder.
    """
    parent_directory = f"uxhw-{uxhw_version}"
    parent_folder_id = _get_or_create_folder(
        drive_service=drive_service,
        folder_name=parent_directory,
        parent_id=TARGET_FOLDER_ID,
    )
    return _get_or_create_folder(
        drive_service=drive_service,
        folder_name=application_name,
        parent_id=parent_folder_id,
    )


def _find_label_row(*, column_a: list[str], label: str) -> int:
    """
    Return the 1-based row of the first column-A cell containing ``label``.

    Matching is case-insensitive and by substring, because the template
    prefixes the labels with a (non-stable) list number and may carry trailing
    descriptive text (see :class:`MetadataRowLabels`).

    Args:
        column_a: The sheet's column-A values, top to bottom.
        label: The distinguishing label substring to find.

    Returns:
        The 1-based row index of the first matching cell.

    Raises:
        ValueError: If no column-A cell contains ``label`` (the template no
            longer has the expected row), so a structural mismatch fails loudly
            at the metadata step rather than silently dropping the value.
    """
    needle = label.casefold()
    for index, cell in enumerate(column_a):
        if needle in (cell or "").casefold():
            return index + 1
    raise ValueError(
        f"Metadata sheet '{ReportSheetTabs.ASSUMPTIONS_CONFIG}' has no column-A "
        f"row containing {label!r}; the report template layout changed. "
        f"Update MetadataRowLabels to match it."
    )


def _update_metadata_sheet(
    *,
    sheet: Any,
    application_version: str,
    uxhw_version: str,
    machine_name: str,
    git_repo_remote: str | None,
) -> None:
    """
    Write session metadata into the assumptions / configuration sheet.

    Each value is written to column B of the row whose column-A label matches
    (by-name via :func:`_find_label_row`), so the write follows the template's
    labels if its rows are reordered instead of being pinned to fixed cells.

    Args:
        sheet: The assumptions / configuration worksheet.
        application_version: Git hash written to the Git Hash row.
        uxhw_version: UxHw SDK version written to the SDK-version row.
        machine_name: Machine type written to the Machine Type row.
        git_repo_remote: Repository URL written to the GitHub Repository row
            (empty string when unknown).
    """
    column_a = sheet.col_values(1)
    label_values = [
        (MetadataRowLabels.GITHUB_REPOSITORY, git_repo_remote or ""),
        (MetadataRowLabels.GIT_HASH, application_version),
        (MetadataRowLabels.UXHW_SDK_VERSION, uxhw_version),
        (MetadataRowLabels.MACHINE_TYPE, machine_name),
    ]
    updates = [
        {
            "range": f"B{_find_label_row(column_a=column_a, label=label)}",
            "values": [[value]],
        }
        for label, value in label_values
    ]
    sheet.batch_update(updates)


def _get_best_speedup_info(*, df: pd.DataFrame) -> tuple[str, int]:
    """
    Get the configuration and EMCC for the best speedup.

    Blown-up rows are excluded first: a degraded representation collapses to
    ``EMCC=1`` and would otherwise win "best speedup" spuriously.

    Args:
        df: Per-configuration results for one variable.

    Returns:
        A tuple of (best config string, its EMCC count).
    """
    if BenchmarkingVariables.BLOW_UP_REASON in df.columns:
        df = df[~df[BenchmarkingVariables.BLOW_UP_REASON].apply(_is_blow_up)]
    idx = df[Measurements.SPEEDUP].idxmax()

    best_config = df[BenchmarkingVariables.UXHW_CONF][idx]
    best_config = (
        repr(best_config)
        if isinstance(best_config, (DistributionalValue, TaggedDistributionalValue))
        else str(best_config)
    )

    try:
        raw_value = df.at[idx, EquivMC.EMCC]
    except KeyError:
        raw_value = df.at[idx, EquivMC.EMCC_PREDICTED]

    if not isinstance(raw_value, (float, int, np.integer, np.floating)):
        error_value = (
            raw_value.decode() if isinstance(raw_value, bytes) else str(raw_value)
        )
        raise TypeError(
            f"Expected float or int. {error_value} is of type " f"{type(raw_value)}"
        )

    return best_config, int(raw_value)


def _write_measurement_data(
    *,
    sheet: Any,
    df: pd.DataFrame,
) -> None:
    """
    Write measurement data to the sheet.

    Blown-up rows have their EMCC / distance cells replaced with the
    "blow-up / excluded" token so a degraded representation does not read as a
    tiny distance or ``EMCC=1``.

    Args:
        sheet: The timing-performance worksheet.
        df: Per-configuration results for one variable.
    """
    sheet_columns = [
        BenchmarkingVariables.UXHW_CONF,
        Measurements.IN_APP_TIME,
        Measurements.DB_TIME,
        Measurements.E2E_TIME,
        BenchmarkingVariables.UXHW_DISTANCE,
        BenchmarkingVariables.UXHW_BINNED_DISTANCE,
        EquivMC.EMCC,
        EquivMC.EMCC_PREDICTED,
        EquivMC.PERCENTAGE_MC_BEATS_UXHW,
        Measurements.NATIVE_IN_APP_TIME,
        Measurements.NATIVE_E2E_TIME,
        Measurements.SPEEDUP,
        Measurements.PIN_DYN_COUNT,
        Measurements.DB_DYN_COUNT,
        Measurements.NATIVE_PIN_COUNT,
    ]

    # Capture which rows are blown-up before the reindex below drops the
    # BLOW_UP_REASON column.
    if BenchmarkingVariables.BLOW_UP_REASON in df.columns:
        blown_flags = (
            df[BenchmarkingVariables.BLOW_UP_REASON].apply(_is_blow_up).tolist()
        )
    else:
        blown_flags = [False] * len(df)

    rows = df.reindex(columns=sheet_columns).values.tolist()
    rows = clean_data_for_sheets(rows)

    # Columns in sheet_columns whose value is meaningless for a blown row:
    # UxHw Distance, Binned UxHw Distance, EMCC, EMCC Predicted.
    blow_up_indices = [
        sheet_columns.index(col)
        for col in (
            BenchmarkingVariables.UXHW_DISTANCE,
            BenchmarkingVariables.UXHW_BINNED_DISTANCE,
            EquivMC.EMCC,
            EquivMC.EMCC_PREDICTED,
        )
    ]
    for row, is_blown in zip(rows, blown_flags):
        if is_blown:
            for col_index in blow_up_indices:
                row[col_index] = BLOW_UP_DISPLAY_TOKEN

    start_row = 4
    end_row = start_row + len(rows) - 1

    sheet.batch_update(
        [
            {
                "range": f"A{start_row}:C{end_row}",
                "values": [row[:3] for row in rows],
            },
            {
                "range": f"E{start_row}:E{end_row}",
                "values": [row[3:4] for row in rows],
            },
            {
                "range": f"G{start_row}:Q{end_row}",
                "values": [row[4:] for row in rows],
            },
        ]
    )
    time.sleep(1)


# The results table and the plot generator now use the same canonical
# correlation tokens (``Disabled`` / ``Autocorrelation``), written verbatim by
# both, so no remapping is needed and this map stays empty. (It previously
# bridged the old verbose/terse spellings, e.g. "CORRELATION_OFF" -> "OFF".)
_RESULTS_TO_PLOT_CORRELATION_TOKEN: dict[str, str] = {}


def _plot_per_config_token(best_config: str) -> str:
    """
    Rewrite a results "UxHw Conf" string to the generator's form.

    Only the trailing correlation token is remapped. The type and size are
    left untouched. Results and generator now share one canonical correlation
    token (``Disabled`` / ``Autocorrelation``), so the map is empty and this is
    effectively a pass-through (kept as the single seam should the two forms
    ever diverge again).

    Args:
        best_config: The results-table "UxHw Conf" string.

    Returns:
        The same string with its correlation token mapped to the generator's
        form.
    """
    head, separator, correlation = best_config.rpartition("-")
    if not separator:
        return best_config
    mapped = _RESULTS_TO_PLOT_CORRELATION_TOKEN.get(correlation, correlation)
    return f"{head}-{mapped}"


def _scalar_adversary_plot(
    *,
    variable: BenchmarkingVariable,
    plots_dir: str,
) -> str:
    """
    Locate a scalar variable's adversary-distances scatter plot.

    Scalars have no per-EMCC adversary distribution plot, only the
    distances-vs-size scatter written by ``_plot_adversary_distances``. Its
    suffix (ground-truth UR order and adversary count) is not reconstructable
    here, so glob for it. ``variable.name`` may contain glob metacharacters
    (e.g. ``outputVariables[0]``) and is escaped before matching.

    Args:
        variable: The scalar benchmarking variable.
        plots_dir: Directory the plot generator wrote into.

    Returns:
        The path of the most recently modified matching scatter plot, or a
        representative non-glob path when none exists (so the caller reports it
        missing rather than raising).
    """
    pattern = os.path.join(
        plots_dir,
        f"{glob.escape(variable.name)}-adversary_distances-*.png",
    )
    matches = glob.glob(pattern)
    if matches:
        # Matches accumulate across reruns. Pick the most recently written to
        # align with the latest generator output (lexicographic order tracks
        # neither recency nor the numeric suffixes in the name).
        return max(matches, key=os.path.getmtime)
    return os.path.join(
        plots_dir,
        f"{variable.name}-adversary_distances.png",
    )


def _get_triptych_image_names(
    *,
    variable: BenchmarkingVariable,
    best_config: str,
    best_emcc: int,
    plots_dir: str,
) -> list[str]:
    """
    Reconstruct the filenames the plot generators wrote.

    The adversary slot differs by variable type: distribution variables get a
    per-EMCC adversary distribution plot (``...-adversary-<emcc>.png``). Scalar
    variables only get the adversary-distances scatter (located by glob).

    Args:
        variable: The benchmarking variable.
        best_config: The best-speedup config string (names the UxHw plot).
        best_emcc: The EMCC count of the best config (names the adversary plot).
        plots_dir: Directory the plot generators wrote into.

    Returns:
        The reconstructed triptych image paths.
    """
    if variable.type == VariableTypes.SCALAR:
        adversary_path = _scalar_adversary_plot(variable=variable, plots_dir=plots_dir)
    else:
        adversary_path = os.path.join(
            plots_dir,
            f"{variable.formatted_description}-adversary-{best_emcc}.png",
        )

    return [
        adversary_path,
        os.path.join(
            plots_dir,
            f"{variable.formatted_description}-"
            f"{_plot_per_config_token(best_config)}.png",
        ),
        os.path.join(
            plots_dir,
            f"{variable.formatted_description}-ground-truth.png",
        ),
    ]


def upload_and_link_images_to_sheet(
    *,
    image_files: list[str],
    sheets_service: Any,
    drive_service: Any,
    folder_id: str,
    sheet_id: str,
) -> None:
    """
    Upload images to a shared Google Drive folder and link them.

    Each image listed in ``image_files`` is uploaded to the folder
    identified by ``folder_id`` and a HYPERLINK formula referencing the
    uploaded file is written into the ``Plot Triptych`` sheet of the
    spreadsheet identified by ``sheet_id``. Files that do not exist on
    disk are skipped with a warning.

    Args:
        image_files: Local paths of PNG files to upload.
        sheets_service: A Google Sheets v4 service client.
        drive_service: A Google Drive v3 service client.
        folder_id: Drive folder ID to upload images into.
        sheet_id: Spreadsheet ID to link the uploaded images from.
    """
    from googleapiclient.http import MediaFileUpload  # type: ignore

    link_rows = []
    for file_name in image_files:
        try:
            if not os.path.exists(file_name):
                print(f"Warning: Image file {file_name} not found, " f"skipping")
                continue

            base_name = os.path.basename(file_name)
            file_metadata = {
                "name": base_name,
                "parents": [folder_id],
            }
            media = MediaFileUpload(file_name, mimetype="image/png")
            uploaded_file = (
                drive_service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )

            file_id = uploaded_file["id"]
            image_url = f"https://drive.google.com/file/d/{file_id}/view"
            # Sheets escapes " inside a string literal by doubling it.
            escaped_url = image_url.replace('"', '""')
            escaped_name = base_name.replace('"', '""')
            link_formula = f'=HYPERLINK("{escaped_url}", "{escaped_name}")'
            link_rows.append([link_formula])

        except Exception as e:
            print(f"Error processing image {file_name}: {e}")

    if link_rows:
        sheets_service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"{ReportSheetTabs.PLOT_TRIPTYCH}!A1",
            valueInputOption="USER_ENTERED",
            body={"values": link_rows},
        ).execute()


def _create_measurement_sheet(
    *,
    sh: Any,
    template_sheet: Any,
    results_df: pd.DataFrame,
    reporting_method: str,
    variable: BenchmarkingVariable,
    index: int,
    sheets_service: Any,
    drive_service: Any,
    folder_id: str,
    sheet_id: str,
    plots_dir: str,
) -> None:
    """
    Create and populate one measurement-data sheet for a reporting method.

    Duplicates the template tab, writes the measurement data, and uploads the
    best-config plot triptych, linking it from the sheet.
    """
    # Insert each duplicate just after the template sheet (relative to its live
    # index) rather than at a hard-coded position, so the layout does not assume
    # the template tab sits at a fixed index.
    emcc_sheet = template_sheet.duplicate(
        insert_sheet_index=template_sheet.index + 1 + index
    )
    time.sleep(1)
    emcc_sheet.update_title(f"{EquivMC.EMCC} {reporting_method}")
    time.sleep(1)

    df = results_df[results_df[EquivMC.REPORTING_METHOD] == reporting_method]

    best_config, best_emcc = _get_best_speedup_info(df=df)
    print(
        f"Best UxHw configuration for {variable.description} "
        f"({reporting_method}): {best_config} (EMCC={best_emcc}). "
        f"Uploading triptych for this configuration."
    )

    _write_measurement_data(sheet=emcc_sheet, df=df)

    image_files = _get_triptych_image_names(
        variable=variable,
        best_config=best_config,
        best_emcc=best_emcc,
        plots_dir=plots_dir,
    )
    upload_and_link_images_to_sheet(
        image_files=image_files,
        sheets_service=sheets_service,
        drive_service=drive_service,
        folder_id=folder_id,
        sheet_id=sheet_id,
    )


def _create_spreadsheet_for_variable(
    *,
    variable: BenchmarkingVariable,
    client: Any,
    drive_service: Any,
    sheets_service: Any,
    folder_id: str,
    reporting_methods: list[str],
    application_version: str,
    uxhw_version: str,
    machine_name: str,
    git_repo_remote: str | None,
    plots_dir: str,
) -> None:
    """
    Create and populate the Google Sheets report for a single variable.

    Copies the template, writes the metadata sheet, and adds one
    measurement-data sheet per reporting method.
    """
    spreadsheet_title = f"{variable.description}"
    copied_file = {
        "name": spreadsheet_title,
        "parents": [folder_id],
    }
    new_file = (
        drive_service.files()
        .copy(
            fileId=TEMPLATE_ID,
            body=copied_file,
            supportsAllDrives=True,
        )
        .execute()
    )

    new_sheet_id = new_file["id"]
    sh = client.open_by_key(new_sheet_id)

    _update_metadata_sheet(
        sheet=sh.worksheet(ReportSheetTabs.ASSUMPTIONS_CONFIG),
        application_version=application_version,
        uxhw_version=uxhw_version,
        machine_name=machine_name,
        git_repo_remote=git_repo_remote,
    )

    results_df = pd.DataFrame(variable.emcc_results.emcc_data)
    template_sheet = sh.worksheet(ReportSheetTabs.TIMING_PERFORMANCE)

    for i, reporting_method in enumerate(reporting_methods):
        _create_measurement_sheet(
            sh=sh,
            template_sheet=template_sheet,
            results_df=results_df,
            reporting_method=reporting_method,
            variable=variable,
            index=i,
            sheets_service=sheets_service,
            drive_service=drive_service,
            folder_id=folder_id,
            sheet_id=new_sheet_id,
            plots_dir=plots_dir,
        )

    sh.del_worksheet(template_sheet)


def write_results_to_spreadsheet(
    *,
    benchmarking_variables: list[BenchmarkingVariable],
    credentials_path: str,
    reporting_methods: list[str],
    application_name: str,
    application_version: str,
    uxhw_version: str,
    machine_name: str,
    git_repo_remote: str | None,
    plots_dir: str,
) -> None:
    """
    Write the results to Google Sheets and upload them.

    Authenticates against the Google APIs using the supplied
    credentials path, creates (or reuses) a Drive folder hierarchy
    keyed on ``uxhw_version`` and ``application_name``, and writes
    one spreadsheet per benchmarking variable into that folder.

    Args:
        benchmarking_variables: Variables whose results to upload.
        credentials_path: Path to a Google service-account JSON file.
            Must exist on disk (see
            :func:`resolve_google_credentials_path`).
        reporting_methods: Reporting methods to emit a sheet for.
        application_name: Application identifier (used as a Drive
            folder name).
        application_version: Application version string (written to
            the metadata sheet).
        uxhw_version: UxHw version string (used as a Drive folder
            name and written to the metadata sheet).
        machine_name: Machine identifier (written to the metadata
            sheet).
        git_repo_remote: Git remote URL of the application repo, or
            ``None`` if unavailable.
        plots_dir: Directory containing the triptych plot PNGs.
    """
    import gspread
    from googleapiclient.discovery import build  # type: ignore

    print("Uploading results to Google Sheets...")
    creds = _get_credentials(credentials_path=credentials_path)
    client = gspread.authorize(creds)
    drive_service = build("drive", "v3", credentials=creds)
    sheets_service = build("sheets", "v4", credentials=creds)

    folder_id = _get_or_create_folder_structure(
        drive_service=drive_service,
        uxhw_version=uxhw_version,
        application_name=application_name,
    )

    for variable in benchmarking_variables:
        _create_spreadsheet_for_variable(
            variable=variable,
            client=client,
            drive_service=drive_service,
            sheets_service=sheets_service,
            folder_id=folder_id,
            reporting_methods=reporting_methods,
            application_version=application_version,
            uxhw_version=uxhw_version,
            machine_name=machine_name,
            git_repo_remote=git_repo_remote,
            plots_dir=plots_dir,
        )
