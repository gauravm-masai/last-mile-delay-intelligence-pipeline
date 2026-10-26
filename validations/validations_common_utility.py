from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_COLUMNS = [
    "source_file",
    "metric_name",
    "metric_value",
    "tolerance",
]

REPORT_COLUMNS = [
    "source_file",
    "metric_name",
    "reference_value",
    "code_value",
    "tolerance",
    "status",
    "comment",
]

def make_report_row(
    source_file,
    metric_name,
    reference_value="",
    code_value="",
    tolerance="",
    status="ERROR",
    comment="",
):
    """Create a report row using the standard report schema."""
    return {
        "source_file": source_file,
        "metric_name": metric_name,
        "reference_value": reference_value,
        "code_value": code_value,
        "tolerance": tolerance,
        "status": status,
        "comment": comment,
    }


def error_row(source_file, metric_name, comment):
    """Create a standardized ERROR report row."""
    return make_report_row(
        source_file=source_file,
        metric_name=metric_name,
        status="ERROR",
        comment=comment,
    )

def load_reference(reference_path):
    """
    Load and validate a reference CSV.

    Returns
    -------
    reference : pd.DataFrame
        Valid reference rows with numeric metric_value and tolerance.

    errors : list[dict]
        Standardized report rows for reference-loading errors.
    """
    reference_path = Path(reference_path)
    errors = []

    if not reference_path.is_file():
        errors.append(
            error_row(
                str(reference_path),
                "__reference_file__",
                "Reference CSV does not exist.",
            )
        )
        return pd.DataFrame(columns=REFERENCE_COLUMNS), errors

    try:
        reference = pd.read_csv(
            reference_path,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        errors.append(
            error_row(
                str(reference_path),
                "__reference_file__",
                f"Could not read reference CSV: {exc}",
            )
        )
        return pd.DataFrame(columns=REFERENCE_COLUMNS), errors

    if reference.columns.tolist() != REFERENCE_COLUMNS:
        errors.append(
            error_row(
                str(reference_path),
                "__reference_schema__",
                "Reference CSV must contain exactly these columns, "
                f"in order: {REFERENCE_COLUMNS}. "
                f"Found: {reference.columns.tolist()}",
            )
        )
        return pd.DataFrame(columns=REFERENCE_COLUMNS), errors

    valid_rows = []

    for index, row in reference.iterrows():
        source_file = row["source_file"].strip()
        metric_name = row["metric_name"].strip()
        row_number = index + 2

        if not source_file or not metric_name:
            errors.append(
                error_row(
                    source_file or str(reference_path),
                    metric_name or f"__reference_row_{row_number}__",
                    f"Reference row {row_number} has a blank "
                    "source_file or metric_name.",
                )
            )
            continue

        try:
            metric_value = float(row["metric_value"])
            tolerance = float(row["tolerance"])
        except (TypeError, ValueError):
            errors.append(
                error_row(
                    source_file,
                    metric_name,
                    f"Reference row {row_number} has a nonnumeric "
                    "metric_value or tolerance.",
                )
            )
            continue

        if not np.isfinite(metric_value):
            errors.append(
                error_row(
                    source_file,
                    metric_name,
                    "Reference metric_value must be finite.",
                )
            )
            continue

        if not np.isfinite(tolerance) or tolerance < 0:
            errors.append(
                error_row(
                    source_file,
                    metric_name,
                    "Reference tolerance must be finite and nonnegative.",
                )
            )
            continue

        valid_rows.append(
            {
                "source_file": source_file,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "tolerance": tolerance,
            }
        )

    validated = pd.DataFrame(
        valid_rows,
        columns=REFERENCE_COLUMNS,
    )

    if not validated.empty:
        duplicate_mask = validated.duplicated(
            ["source_file", "metric_name"],
            keep=False,
        )

        duplicates = validated.loc[duplicate_mask]

        for _, row in duplicates.iterrows():
            errors.append(
                error_row(
                    row["source_file"],
                    row["metric_name"],
                    "Reference contains duplicate "
                    "(source_file, metric_name) entries.",
                )
            )

        validated = validated.loc[~duplicate_mask].copy()

    return validated, errors

def compare_numeric(
    source_file,
    metric_name,
    reference_value,
    code_value,
    tolerance,
):
    """
    Compare two numeric values and return a standardized report row.
    """
    try:
        expected = float(reference_value)
        actual = float(code_value)
        allowed_difference = float(tolerance)
    except (TypeError, ValueError):
        return error_row(
            source_file,
            metric_name,
            "Reference value, code value, or tolerance is nonnumeric.",
        )

    if (
        not np.isfinite(expected)
        or not np.isfinite(actual)
        or not np.isfinite(allowed_difference)
        or allowed_difference < 0
    ):
        return error_row(
            source_file,
            metric_name,
            "Cannot compare non-finite values or a negative tolerance.",
        )

    difference = abs(actual - expected)

    if difference <= allowed_difference:
        status = "PASS"
        comment = (
            f"Values match within tolerance. "
            f"Absolute difference: {difference:.8g}; "
            f"allowed tolerance: {allowed_difference:.8g}."
        )
    else:
        status = "MISMATCH"
        comment = (
            f"Values differ beyond tolerance. "
            f"Absolute difference: {difference:.8g}; "
            f"allowed tolerance: {allowed_difference:.8g}."
        )

    return make_report_row(
        source_file=source_file,
        metric_name=metric_name,
        reference_value=expected,
        code_value=actual,
        tolerance=allowed_difference,
        status=status,
        comment=comment,
    )

def write_report(report_rows, report_path):
    """
    Write the standard validation report, print a summary,
    and return 0 for success or 1 for any non-PASS result.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(
        report_rows,
        columns=REPORT_COLUMNS,
    )

    if report.empty:
        report = pd.DataFrame(
            [
                error_row(
                    str(report_path),
                    "__validation__",
                    "No validation metrics were evaluated.",
                )
            ],
            columns=REPORT_COLUMNS,
        )

    report.to_csv(
        report_path,
        index=False,
    )

    counts = report["status"].value_counts()

    passed = int(counts.get("PASS", 0))
    mismatches = int(counts.get("MISMATCH", 0))
    errors = int(counts.get("ERROR", 0))

    print(
        f"Validation summary: {passed} PASS, "
        f"{mismatches} MISMATCH, {errors} ERROR"
    )
    print(f"Report saved to: {report_path}")

    if mismatches or errors:
        print("Validation failed.")
        return 1

    print("All validations passed.")
    return 0