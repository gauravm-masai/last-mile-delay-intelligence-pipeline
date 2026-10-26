from pathlib import Path
import sys

import pandas as pd

from validations.validations_common_utility import (
    compare_numeric,
    error_row,
    load_reference,
    make_report_row,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = ROOT / "references" / "part_a_reference_metrics.csv"
REPORT_PATH = (
    ROOT
    / "validations"
    / "reports"
    / "part_a_validation_report.csv"
)

def load_output(source_file, cache):
    """
    Load an existing output CSV once and cache the result.

    This function only reads generated outputs. It does not regenerate them.
    """
    if source_file in cache:
        return cache[source_file]

    path = ROOT / source_file

    if not path.is_file():
        cache[source_file] = (
            None,
            f"Expected output file does not exist: {source_file}",
        )
        return cache[source_file]

    try:
        if source_file == "outputs/part_a_correlations.csv":
            data = pd.read_csv(path, index_col=0)
        else:
            data = pd.read_csv(path)

        cache[source_file] = (data, None)

    except Exception as exc:
        cache[source_file] = (
            None,
            f"Could not read output file {source_file}: {exc}",
        )

    return cache[source_file]

def extract_check_metric(metric_name, data):
    """Extract a metric from outputs/part_a_checks.csv."""
    required_columns = {"metric", "value"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Checks output must contain columns 'metric' and 'value'."
        )

    if data["metric"].duplicated().any():
        raise ValueError(
            "Checks output contains duplicate metric names."
        )

    matches = data.loc[data["metric"] == metric_name, "value"]

    if matches.empty:
        raise KeyError(
            f"Metric '{metric_name}' is missing from "
            "outputs/part_a_checks.csv."
        )

    return matches.iloc[0]

def extract_group_metric(metric_name, data):
    """
    Extract a shipment count, delayed count, or delay rate
    from outputs/part_a_delay_by_group.csv.

    Expected metric naming:
        shipments__dimension__group
        delayed__dimension__group
        delay_rate__dimension__group
    """
    required_columns = {
        "dimension",
        "group",
        "shipments",
        "delayed",
        "delay_rate",
    }

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Delay evidence output is missing required columns: "
            f"{sorted(required_columns - set(data.columns))}"
        )

    parts = metric_name.split("__", maxsplit=2)

    if len(parts) != 3:
        raise ValueError(
            f"Invalid group metric name: '{metric_name}'."
        )

    metric, dimension, group = parts

    if metric not in {"shipments", "delayed", "delay_rate"}:
        raise ValueError(
            f"Unsupported group metric '{metric}' in '{metric_name}'."
        )

    # Convert keys to strings so numeric hub tiers match reference labels.
    dimensions = data["dimension"].astype(str).str.strip()
    groups = data["group"].astype(str).str.strip()

    if data.assign(
        _dimension=dimensions,
        _group=groups,
    ).duplicated(["_dimension", "_group"]).any():
        raise ValueError(
            "Delay evidence output contains duplicate dimension/group keys."
        )

    matches = data.loc[
        (dimensions == dimension)
        & (groups == group),
        metric,
    ]

    if matches.empty:
        raise KeyError(
            f"Group '{group}' in dimension '{dimension}' is missing "
            "from outputs/part_a_delay_by_group.csv."
        )

    return matches.iloc[0]

def extract_correlation_metric(metric_name, data):
    """
    Extract one correlation from outputs/part_a_correlations.csv.

    Expected metric naming:
        corr__column1__column2
    """
    parts = metric_name.split("__", maxsplit=2)

    if len(parts) != 3 or parts[0] != "corr":
        raise ValueError(
            f"Invalid correlation metric name: '{metric_name}'."
        )

    _, column_1, column_2 = parts

    if column_1 not in data.index:
        raise KeyError(
            f"Correlation row '{column_1}' is missing."
        )

    if column_2 not in data.columns:
        raise KeyError(
            f"Correlation column '{column_2}' is missing."
        )

    return data.loc[column_1, column_2]

def extract_metric(reference_row, output_cache):
    """Dispatch metric extraction based on its source_file."""
    source_file = reference_row["source_file"]
    metric_name = reference_row["metric_name"]

    data, load_error = load_output(source_file, output_cache)

    if load_error:
        raise FileNotFoundError(load_error)

    if source_file == "outputs/part_a_checks.csv":
        return extract_check_metric(metric_name, data)

    if source_file == "outputs/part_a_delay_by_group.csv":
        return extract_group_metric(metric_name, data)

    if source_file == "outputs/part_a_correlations.csv":
        return extract_correlation_metric(metric_name, data)

    raise ValueError(
        f"Unsupported source_file in reference: '{source_file}'."
    )

def report_unreferenced_check_metrics(reference, output_cache):
    """
    Preserve the existing check that reports unexpected metrics
    in outputs/part_a_checks.csv.
    """
    source_file = "outputs/part_a_checks.csv"
    data, load_error = load_output(source_file, output_cache)

    if load_error or data is None:
        # Missing/unreadable files are already reported for their
        # reference metrics; do not create a duplicate error here.
        return []

    if not {"metric", "value"}.issubset(data.columns):
        return []

    referenced_metrics = set(
        reference.loc[
            reference["source_file"] == source_file,
            "metric_name",
        ]
    )

    rows = []

    for _, actual_row in data.iterrows():
        metric_name = str(actual_row["metric"])

        if metric_name not in referenced_metrics:
            rows.append(
                make_report_row(
                    source_file=source_file,
                    metric_name=metric_name,
                    code_value=actual_row["value"],
                    status="MISMATCH",
                    comment=(
                        "Output contains a metric that is not defined "
                        "in the reference CSV."
                    ),
                )
            )

    return rows

def validate_part_a():
    """Validate Part A outputs against the approved reference CSV."""
    reference, reference_errors = load_reference(REFERENCE_PATH)

    report_rows = list(reference_errors)

    output_cache = {}

    for _, reference_row in reference.iterrows():
        source_file = reference_row["source_file"]
        metric_name = reference_row["metric_name"]
        expected = reference_row["metric_value"]
        tolerance = reference_row["tolerance"]

        try:
            actual = extract_metric(reference_row, output_cache)

            report_row = compare_numeric(
                source_file=source_file,
                metric_name=metric_name,
                reference_value=expected,
                code_value=actual,
                tolerance=tolerance,
            )

        except Exception as exc:
            report_row = error_row(
                source_file,
                metric_name,
                str(exc),
            )

            # Include the expected reference value even when extraction
            # fails, so the report remains useful for debugging.
            report_row["reference_value"] = expected
            report_row["tolerance"] = tolerance

        report_rows.append(report_row)

    report_rows.extend(
        report_unreferenced_check_metrics(reference, output_cache)
    )

    return write_report(report_rows, REPORT_PATH)

if __name__ == "__main__":
    sys.exit(validate_part_a())