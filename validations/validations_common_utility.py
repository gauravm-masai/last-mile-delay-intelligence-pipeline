from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SHARED SCHEMAS
# ============================================================

REFERENCE_COLUMNS = [
    "source_file",
    "metric_name",
    "value_type",
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

VALID_VALUE_TYPES = {"REFERENCE", "ACCEPTANCE"}

VALID_STATUSES = {"PASS", "MISMATCH", "ERROR"}

VALID_COMPARISON_OPERATORS = {
    "numeric",
    "exact",
    "categorical",
    "boolean",
    "structural",
    "minimum",
    "maximum",
    "range",
}


# ============================================================
# STANDARDIZED REPORT ROWS
# ============================================================

def make_report_row(
    source_file,
    metric_name,
    reference_value="",
    code_value="",
    tolerance="",
    status="ERROR",
    comment="",
):
    """Create a report row with the standard seven-column schema."""
    if status not in VALID_STATUSES:
        status = "ERROR"
        comment = (
            f"Invalid report status supplied. "
            f"Allowed statuses: {sorted(VALID_STATUSES)}."
        )

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


# ============================================================
# REFERENCE CSV LOADING AND VALIDATION
# ============================================================

def load_reference(reference_path):
    """
    Load and validate the shared reference CSV.

    Required schema:
        source_file, metric_name, value_type, metric_value, tolerance

    value_type:
        REFERENCE  - verified benchmark
        ACCEPTANCE - mandatory acceptance criterion

    A metric may have one REFERENCE row and one ACCEPTANCE row.
    Duplicate rows with the same source_file, metric_name, and
    value_type are invalid.

    Returns
    -------
    reference : pandas.DataFrame
        Valid reference entries with numeric metric_value and tolerance.

    errors : list[dict]
        Standardized ERROR report rows describing reference problems.
    """
    reference_path = Path(reference_path)
    errors = []

    empty_reference = pd.DataFrame(columns=REFERENCE_COLUMNS)

    if not reference_path.is_file():
        errors.append(
            error_row(
                str(reference_path),
                "__reference_file__",
                "Reference CSV does not exist.",
            )
        )
        return empty_reference, errors

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
        return empty_reference, errors

    if reference.columns.tolist() != REFERENCE_COLUMNS:
        errors.append(
            error_row(
                str(reference_path),
                "__reference_schema__",
                "Reference CSV must contain exactly these columns "
                f"in this order: {REFERENCE_COLUMNS}. "
                f"Found: {reference.columns.tolist()}",
            )
        )
        return empty_reference, errors

    valid_rows = []

    for index, row in reference.iterrows():
        row_number = index + 2

        source_file = row["source_file"].strip()
        metric_name = row["metric_name"].strip()
        value_type = row["value_type"].strip().upper()

        row_identifier = (
            metric_name
            or f"__reference_row_{row_number}__"
        )

        if not source_file or not metric_name:
            errors.append(
                error_row(
                    source_file or str(reference_path),
                    row_identifier,
                    f"Reference CSV row {row_number} has a blank "
                    "source_file or metric_name.",
                )
            )
            continue

        if value_type not in VALID_VALUE_TYPES:
            errors.append(
                error_row(
                    source_file,
                    metric_name,
                    f"Reference CSV row {row_number} has invalid "
                    f"value_type {value_type!r}. Expected "
                    f"REFERENCE or ACCEPTANCE.",
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
                    f"Reference CSV row {row_number} has a nonnumeric "
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
                "value_type": value_type,
                "metric_value": metric_value,
                "tolerance": tolerance,
            }
        )

    validated = pd.DataFrame(
        valid_rows,
        columns=REFERENCE_COLUMNS,
    )

    if validated.empty:
        return validated, errors

    # REFERENCE and ACCEPTANCE entries for the same logical metric
    # are permitted. Duplicate entries of the same type are not.
    duplicate_columns = [
        "source_file",
        "metric_name",
        "value_type",
    ]

    duplicate_mask = validated.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    duplicate_rows = validated.loc[duplicate_mask]

    for _, row in duplicate_rows.iterrows():
        errors.append(
            error_row(
                row["source_file"],
                row["metric_name"],
                "Reference CSV contains duplicate entries for "
                f"(source_file, metric_name, value_type) = "
                f"({row['source_file']!r}, "
                f"{row['metric_name']!r}, "
                f"{row['value_type']!r}).",
            )
        )

    validated = validated.loc[~duplicate_mask].copy()

    return validated, errors


# ============================================================
# NUMERIC REFERENCE COMPARISON
# ============================================================

def compare_numeric(
    source_file,
    metric_name,
    reference_value,
    code_value,
    tolerance=0.0001,
):
    """
    Compare numeric values against a reference benchmark.

    A difference beyond tolerance produces MISMATCH, not ERROR.
    Missing, nonnumeric, non-finite, or invalid comparison inputs
    produce ERROR.
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
            "Reference comparison passed: absolute difference "
            f"{difference:.8g} is within tolerance "
            f"{allowed_difference:.8g}."
        )
    else:
        status = "MISMATCH"
        comment = (
            "Reference comparison mismatch: absolute difference "
            f"{difference:.8g} exceeds tolerance "
            f"{allowed_difference:.8g}. This is informational "
            "and does not fail validation by itself."
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


# ============================================================
# ACCEPTANCE COMPARISONS
# ============================================================

def compare_acceptance(
    source_file,
    metric_name,
    code_value,
    operator,
    threshold=None,
    lower_bound=None,
    upper_bound=None,
    tolerance=0.0,
):
    """
    Evaluate an explicitly configured mandatory acceptance criterion.

    Supported operators
    -------------------
    numeric:
        Absolute difference from threshold <= tolerance.

    exact:
        Exact equality with threshold.

    categorical:
        Exact equality with threshold after string conversion.

    boolean:
        Exact equality with the expected Boolean value.

    structural:
        Exact equality with threshold. Use for counts, dimensions,
        column order, feature lists, or other structural values.

    minimum:
        code_value >= threshold.

    maximum:
        code_value <= threshold.

    range:
        lower_bound <= code_value <= upper_bound.

    Acceptance failures are ERROR, not MISMATCH.

    The operator must be supplied by the validator configuration.
    It is never inferred from the metric name.
    """
    if operator not in VALID_COMPARISON_OPERATORS:
        return error_row(
            source_file,
            metric_name,
            f"Invalid acceptance operator {operator!r}. "
            f"Allowed operators: "
            f"{sorted(VALID_COMPARISON_OPERATORS)}.",
        )

    if code_value is None:
        return error_row(
            source_file,
            metric_name,
            "Acceptance check cannot be evaluated because "
            "the code-generated value is missing.",
        )

    if isinstance(code_value, (float, np.floating)):
        if not np.isfinite(code_value):
            return error_row(
                source_file,
                metric_name,
                "Acceptance check cannot be evaluated because "
                "the code-generated value is non-finite.",
            )

    if operator in {"numeric", "minimum", "maximum", "range"}:
        try:
            actual = float(code_value)
        except (TypeError, ValueError):
            return error_row(
                source_file,
                metric_name,
                "Acceptance check requires a numeric code value.",
            )

        if not np.isfinite(actual):
            return error_row(
                source_file,
                metric_name,
                "Acceptance check requires a finite code value.",
            )

        if operator == "numeric":
            if threshold is None:
                return error_row(
                    source_file,
                    metric_name,
                    "Numeric acceptance requires a threshold.",
                )

            try:
                expected = float(threshold)
                allowed_difference = float(tolerance)
            except (TypeError, ValueError):
                return error_row(
                    source_file,
                    metric_name,
                    "Numeric acceptance threshold or tolerance "
                    "is invalid.",
                )

            if (
                not np.isfinite(expected)
                or not np.isfinite(allowed_difference)
                or allowed_difference < 0
            ):
                return error_row(
                    source_file,
                    metric_name,
                    "Numeric acceptance requires a finite threshold "
                    "and nonnegative finite tolerance.",
                )

            passed = abs(actual - expected) <= allowed_difference
            rule = (
                f"absolute difference from {expected:g} "
                f"<= {allowed_difference:g}"
            )
            displayed_threshold = expected
            displayed_tolerance = allowed_difference

        elif operator == "minimum":
            if threshold is None:
                return error_row(
                    source_file,
                    metric_name,
                    "Minimum acceptance requires a threshold.",
                )

            try:
                minimum = float(threshold)
            except (TypeError, ValueError):
                return error_row(
                    source_file,
                    metric_name,
                    "Minimum acceptance threshold is invalid.",
                )

            if not np.isfinite(minimum):
                return error_row(
                    source_file,
                    metric_name,
                    "Minimum acceptance threshold must be finite.",
                )

            passed = actual >= minimum
            rule = f"value >= {minimum:g}"
            displayed_threshold = minimum
            displayed_tolerance = ""

        elif operator == "maximum":
            if threshold is None:
                return error_row(
                    source_file,
                    metric_name,
                    "Maximum acceptance requires a threshold.",
                )

            try:
                maximum = float(threshold)
            except (TypeError, ValueError):
                return error_row(
                    source_file,
                    metric_name,
                    "Maximum acceptance threshold is invalid.",
                )

            if not np.isfinite(maximum):
                return error_row(
                    source_file,
                    metric_name,
                    "Maximum acceptance threshold must be finite.",
                )

            passed = actual <= maximum
            rule = f"value <= {maximum:g}"
            displayed_threshold = maximum
            displayed_tolerance = ""

        else:  # range
            if lower_bound is None or upper_bound is None:
                return error_row(
                    source_file,
                    metric_name,
                    "Range acceptance requires lower_bound and "
                    "upper_bound.",
                )

            try:
                lower = float(lower_bound)
                upper = float(upper_bound)
            except (TypeError, ValueError):
                return error_row(
                    source_file,
                    metric_name,
                    "Range acceptance bounds must be numeric.",
                )

            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower > upper
            ):
                return error_row(
                    source_file,
                    metric_name,
                    "Range acceptance bounds must be finite and "
                    "lower_bound must not exceed upper_bound.",
                )

            passed = lower <= actual <= upper
            rule = f"{lower:g} <= value <= {upper:g}"
            displayed_threshold = f"[{lower:g}, {upper:g}]"
            displayed_tolerance = ""

        status = "PASS" if passed else "ERROR"

        comment = (
            f"Mandatory acceptance {'passed' if passed else 'failed'}: "
            f"{actual:g}; required rule: {rule}."
        )

        return make_report_row(
            source_file=source_file,
            metric_name=metric_name,
            reference_value=displayed_threshold,
            code_value=actual,
            tolerance=displayed_tolerance,
            status=status,
            comment=comment,
        )

    # Exact, categorical, Boolean, and structural comparisons.
    if operator == "boolean":
        if isinstance(code_value, (bool, np.bool_)):
            actual_bool = bool(code_value)
        elif isinstance(code_value, (int, np.integer)) and code_value in (0, 1):
            actual_bool = bool(code_value)
        elif isinstance(code_value, str):
            normalized = code_value.strip().lower()

            if normalized in {"true", "1"}:
                actual_bool = True
            elif normalized in {"false", "0"}:
                actual_bool = False
            else:
                return error_row(
                    source_file,
                    metric_name,
                    "Boolean acceptance received an invalid code value.",
                )
        else:
            return error_row(
                source_file,
                metric_name,
                "Boolean acceptance requires a Boolean or 0/1 value.",
            )

        if isinstance(threshold, (bool, np.bool_)):
            expected_bool = bool(threshold)
        elif threshold in (0, 1, "0", "1"):
            expected_bool = str(threshold).strip() == "1"
        elif isinstance(threshold, str):
            normalized = threshold.strip().lower()

            if normalized == "true":
                expected_bool = True
            elif normalized == "false":
                expected_bool = False
            else:
                return error_row(
                    source_file,
                    metric_name,
                    "Boolean acceptance expected value is invalid.",
                )
        else:
            return error_row(
                source_file,
                metric_name,
                "Boolean acceptance requires an explicit expected "
                "Boolean value.",
            )

        passed = actual_bool == expected_bool
        displayed_actual = actual_bool
        displayed_expected = expected_bool

    elif operator == "categorical":
        if threshold is None:
            return error_row(
                source_file,
                metric_name,
                "Categorical acceptance requires an expected value.",
            )

        displayed_actual = str(code_value)
        displayed_expected = str(threshold)
        passed = displayed_actual == displayed_expected

    else:  # exact or structural
        if threshold is None:
            return error_row(
                source_file,
                metric_name,
                f"{operator.capitalize()} acceptance requires "
                "an expected value.",
            )

        if operator == "structural":
            # Structural expectations may be lists, tuples, mappings,
            # or scalars. Python equality preserves type-specific
            # structural checks when the expected value is structured.
            passed = code_value == threshold
        else:
            passed = type(code_value) is type(threshold) and (
                code_value == threshold
            )

        displayed_actual = repr(code_value)
        displayed_expected = repr(threshold)

    status = "PASS" if passed else "ERROR"

    comment = (
        f"Mandatory {operator} acceptance "
        f"{'passed' if passed else 'failed'}: "
        f"actual={displayed_actual!r}; "
        f"expected={displayed_expected!r}."
    )

    return make_report_row(
        source_file=source_file,
        metric_name=metric_name,
        reference_value=displayed_expected,
        code_value=displayed_actual,
        tolerance="Exact",
        status=status,
        comment=comment,
    )


# ============================================================
# COMPARISON DISPATCHER
# ============================================================

def compare_metric(
    source_file,
    metric_name,
    code_value,
    value_type,
    operator,
    reference_value=None,
    tolerance=0.0001,
    threshold=None,
    lower_bound=None,
    upper_bound=None,
):
    """
    Dispatch a check to reference comparison or acceptance comparison.

    REFERENCE:
        Numeric benchmark comparison. A difference beyond tolerance
        returns MISMATCH, which is informational.

    ACCEPTANCE:
        Mandatory acceptance comparison. A failed rule returns ERROR.

    The operator and all acceptance bounds must be provided explicitly.
    """
    normalized_type = str(value_type).strip().upper()

    if normalized_type == "REFERENCE":
        if reference_value is None:
            return error_row(
                source_file,
                metric_name,
                "Reference comparison cannot be evaluated because "
                "the reference value is missing.",
            )

        if operator not in {"numeric", "exact"}:
            return error_row(
                source_file,
                metric_name,
                "Reference comparisons currently support numeric "
                "benchmarks. Use compare_acceptance for categorical, "
                "Boolean, structural, minimum, maximum, or range checks.",
            )

        if operator == "numeric":
            return compare_numeric(
                source_file=source_file,
                metric_name=metric_name,
                reference_value=reference_value,
                code_value=code_value,
                tolerance=tolerance,
            )

        # Exact reference comparison uses zero tolerance and numeric
        # equality. It is intended for numeric structural quantities
        # such as counts when represented numerically.
        return compare_numeric(
            source_file=source_file,
            metric_name=metric_name,
            reference_value=reference_value,
            code_value=code_value,
            tolerance=0.0,
        )

    if normalized_type == "ACCEPTANCE":
        return compare_acceptance(
            source_file=source_file,
            metric_name=metric_name,
            code_value=code_value,
            operator=operator,
            threshold=threshold,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tolerance=tolerance,
        )

    return error_row(
        source_file,
        metric_name,
        f"Invalid value_type {value_type!r}. Expected REFERENCE "
        "or ACCEPTANCE.",
    )


# ============================================================
# REPORT WRITING AND PROCESS EXIT CODE
# ============================================================

def write_report(report_rows, report_path):
    """
    Write the standard seven-column validation report.

    Exit behavior
    -------------
    0: The report contains PASS and/or MISMATCH, but no ERROR.
    1: The report contains one or more ERROR rows, or no checks ran.

    MISMATCH is informational and does not fail the process.
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

    # Prevent invalid status values from being silently treated as success.
    invalid_status_mask = ~report["status"].isin(VALID_STATUSES)

    for index in report.index[invalid_status_mask]:
        report.at[index, "status"] = "ERROR"
        report.at[index, "comment"] = (
            "Invalid or missing status was found in a report row."
        )

    # Enforce the report schema and order.
    report = report[REPORT_COLUMNS]

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

    if errors:
        print(
            "Validation failed: at least one mandatory acceptance "
            "check or validation integrity check produced ERROR."
        )
        return 1

    if mismatches:
        print(
            "Validation completed with informational reference "
            "MISMATCH results. No ERROR results were found."
        )
        return 0

    print("All validation checks passed.")
    return 0

def run_dual_validation(
    validation_runner,
    reference_path,
    internal_reference_path,
    report_path,
    internal_report_path,
):
    """
    Run official PDF-defined validation and internal validation separately.

    The validation_runner must accept keyword arguments:
        reference_path
        report_path

    Returns:
        A dictionary containing the independent exit codes.

    Exit code 0 means the corresponding report contains no ERROR.
    MISMATCH remains informational.
    """
    print("\n" + "=" * 65)
    print("OFFICIAL PDF-DEFINED VALIDATION")
    print("=" * 65)

    reference_exit_code = validation_runner(
        reference_path=reference_path,
        report_path=report_path,
    )

    print("\n" + "=" * 65)
    print("INTERNAL VALIDATION")
    print("=" * 65)

    internal_exit_code = validation_runner(
        reference_path=internal_reference_path,
        report_path=internal_report_path,
    )

    results = {
        "reference_exit_code": int(reference_exit_code),
        "internal_exit_code": int(internal_exit_code),
    }

    print("\n" + "=" * 65)
    print("DUAL VALIDATION SUMMARY")
    print("=" * 65)
    print(
        "Official report: "
        f"{'PASS (no ERROR)' if results['reference_exit_code'] == 0 else 'ERROR(S) FOUND'}"
    )
    print(
        "Internal report: "
        f"{'PASS (no ERROR)' if results['internal_exit_code'] == 0 else 'ERROR(S) FOUND'}"
    )

    return results