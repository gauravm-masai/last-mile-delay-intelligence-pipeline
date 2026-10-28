from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
REFERENCES = ROOT / "references"

REFERENCE_PATH = REFERENCES / "part_d_reference_metrics.csv"
REPORT_PATH = (
    ROOT / "validations" / "reports" / "part_d_validation_report.csv"
)

sys.path.insert(0, str(ROOT))

from validations.validations_common_utility import (
    load_reference,
    compare_numeric,
    make_report_row,
    error_row,
    write_report,
)


SEGMENT_CODES = {
    "Standard": 1,
    "Watch": 2,
    "Escalate": 3,
}

CITY_CODES = {
    "Kolkata": 1,
    "Nagpur": 2,
    "Kochi": 3,
    "Dehradun": 4,
}

EXPECTED_MISMATCHES = {
    "HUB07": {
        "city": "Kolkata",
        "hub_tier": 1,
        "segment": "Watch",
        "expected_segment": "Standard",
    },
    "HUB13": {
        "city": "Nagpur",
        "hub_tier": 2,
        "segment": "Escalate",
        "expected_segment": "Watch",
    },
    "HUB15": {
        "city": "Kochi",
        "hub_tier": 2,
        "segment": "Escalate",
        "expected_segment": "Watch",
    },
    "HUB22": {
        "city": "Dehradun",
        "hub_tier": 3,
        "segment": "Watch",
        "expected_segment": "Escalate",
    },
}


def add_error(report_rows, source_file, metric_name, comment):
    """Add a standardized ERROR row."""
    report_rows.append(
        error_row(source_file, metric_name, comment)
    )


def add_mismatch(
    report_rows,
    source_file,
    metric_name,
    reference_value,
    code_value,
    comment,
):
    """Add a standardized MISMATCH row."""
    report_rows.append(
        make_report_row(
            source_file=source_file,
            metric_name=metric_name,
            reference_value=reference_value,
            code_value=code_value,
            tolerance=0,
            status="MISMATCH",
            comment=comment,
        )
    )


def read_existing_csv(path, source_file, report_rows):
    """
    Read an existing CSV.

    This function never executes the implementation or regenerates
    the output being validated.
    """
    if not path.is_file():
        add_error(
            report_rows,
            source_file,
            "__file__",
            f"Required existing file is missing: {path}",
        )
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        add_error(
            report_rows,
            source_file,
            "__file__",
            f"Could not read CSV: {exc}",
        )
        return None


def finite_number(value):
    """Return a finite float, or None when conversion is impossible."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if np.isfinite(result) else None


def record_metric(metrics, source_file, metric_name, value):
    """Store a metric using the shared reference key."""
    metrics[(source_file, metric_name)] = value


def extract_metrics(report_rows):
    """
    Extract actual metrics from existing Part D files.

    Returns a dictionary keyed by (source_file, metric_name).
    It does not rerun any Part D implementation.
    """
    metrics = {}

    # ---------------------------------------------------------
    # 1. Hub profile
    # ---------------------------------------------------------
    source = "hub_profile.csv"
    profile = read_existing_csv(
        OUTPUTS / source, source, report_rows
    )

    if profile is not None:
        required = {
            "hub_id",
            "shipments",
            "mean_delivery_hours",
            "delay_rate",
            "mean_distance_km",
            "mean_n_stops",
        }

        if not required.issubset(profile.columns):
            add_error(
                report_rows,
                source,
                "__schema__",
                f"Missing columns: {sorted(required - set(profile.columns))}",
            )
        else:
            record_metric(
                metrics, source, "row_count", len(profile)
            )

            shipments = pd.to_numeric(
                profile["shipments"], errors="coerce"
            )

            record_metric(
                metrics,
                source,
                "shipments_per_hub",
                shipments.iloc[0] if len(shipments) else None,
            )

            # A numeric benchmark alone cannot establish that every
            # hub has the expected shipment count. Check every row.
            if (
                shipments.isna().any()
                or not shipments.eq(50).all()
            ):
                add_mismatch(
                    report_rows,
                    source,
                    "all_hubs_have_50_shipments",
                    50,
                    shipments.tolist(),
                    "Every hub must have exactly 50 shipments.",
                )

            for hub_id in ("HUB01", "HUB13", "HUB17"):
                hub_rows = profile.loc[
                    profile["hub_id"].astype(str) == hub_id
                ]

                if len(hub_rows) != 1:
                    add_error(
                        report_rows,
                        source,
                        f"{hub_id}.__row__",
                        f"Expected exactly one row for {hub_id}; "
                        f"found {len(hub_rows)}.",
                    )
                    continue

                row = hub_rows.iloc[0]

                for column in (
                    "mean_delivery_hours",
                    "delay_rate",
                    "mean_distance_km",
                    "mean_n_stops",
                ):
                    record_metric(
                        metrics,
                        source,
                        f"{hub_id}.{column}",
                        finite_number(row[column]),
                    )

    # ---------------------------------------------------------
    # 2. KMeans separation evidence
    # ---------------------------------------------------------
    source = "part_d_separation.csv"
    separation = read_existing_csv(
        OUTPUTS / source, source, report_rows
    )

    if separation is not None:
        required = {
            "K",
            "across_over_within",
            "cluster_sizes",
        }

        if not required.issubset(separation.columns):
            add_error(
                report_rows,
                source,
                "__schema__",
                f"Missing columns: {sorted(required - set(separation.columns))}",
            )
        else:
            record_metric(
                metrics, source, "row_count", len(separation)
            )

            for k in (2, 3, 4, 5):
                rows = separation.loc[separation["K"] == k]

                if len(rows) != 1:
                    add_error(
                        report_rows,
                        source,
                        f"K{k}.__row__",
                        f"Expected one row for K={k}; found {len(rows)}.",
                    )
                    continue

                row = rows.iloc[0]

                record_metric(
                    metrics,
                    source,
                    f"K{k}.across_over_within",
                    finite_number(row["across_over_within"]),
                )

                raw_sizes = str(row["cluster_sizes"]).strip()

                try:
                    sizes = [
                        int(value)
                        for value in raw_sizes.split("-")
                    ]

                    if any(size <= 0 for size in sizes):
                        raise ValueError(
                            "Cluster sizes must be positive."
                        )

                except ValueError:
                    add_error(
                        report_rows,
                        source,
                        f"K{k}.cluster_sizes",
                        "Expected hyphen-separated positive integers.",
                    )
                    continue

                for rank, size in enumerate(sizes, start=1):
                    record_metric(
                        metrics,
                        source,
                        f"K{k}.cluster_size_rank_{rank}",
                        size,
                    )

                if len(sizes) != k:
                    add_error(
                        report_rows,
                        source,
                        f"K{k}.cluster_sizes",
                        f"K={k} must have {k} cluster sizes; "
                        f"found {len(sizes)}.",
                    )

                if sum(sizes) != 24:
                    add_error(
                        report_rows,
                        source,
                        f"K{k}.cluster_sizes_total",
                        f"Cluster sizes must sum to 24; found {sum(sizes)}.",
                    )

    # ---------------------------------------------------------
    # 3. Segment profiles
    # ---------------------------------------------------------
    source = "segment_profile.csv"
    segment_profile = read_existing_csv(
        OUTPUTS / source, source, report_rows
    )

    if segment_profile is not None:
        required = {
            "segment",
            "hubs",
            "mean_delivery_hours",
            "delay_rate",
            "mean_distance_km",
            "mean_n_stops",
        }

        if not required.issubset(segment_profile.columns):
            add_error(
                report_rows,
                source,
                "__schema__",
                f"Missing columns: "
                f"{sorted(required - set(segment_profile.columns))}",
            )
        else:
            record_metric(
                metrics, source, "row_count", len(segment_profile)
            )

            for segment in ("Standard", "Watch", "Escalate"):
                rows = segment_profile.loc[
                    segment_profile["segment"].astype(str) == segment
                ]

                if len(rows) != 1:
                    add_error(
                        report_rows,
                        source,
                        f"{segment}.__row__",
                        f"Expected one row for {segment}; found {len(rows)}.",
                    )
                    continue

                row = rows.iloc[0]

                for column in (
                    "hubs",
                    "mean_delivery_hours",
                    "delay_rate",
                    "mean_distance_km",
                    "mean_n_stops",
                ):
                    record_metric(
                        metrics,
                        source,
                        f"{segment}.{column}",
                        finite_number(row[column]),
                    )

            expected_segments = {"Standard", "Watch", "Escalate"}
            actual_segments = set(
                segment_profile["segment"].dropna().astype(str)
            )

            if actual_segments != expected_segments:
                add_mismatch(
                    report_rows,
                    source,
                    "segment_names",
                    sorted(expected_segments),
                    sorted(actual_segments),
                    "Segment names must be Standard, Watch and Escalate.",
                )

    # ---------------------------------------------------------
    # 4. Tier-versus-segment mismatch details
    # ---------------------------------------------------------
    source = "tier_vs_segment_mismatch.csv"
    mismatches = read_existing_csv(
        OUTPUTS / source, source, report_rows
    )

    if mismatches is not None:
        required = {
            "hub_id",
            "segment",
            "hub_city",
            "hub_tier",
            "expected_segment",
        }

        if not required.issubset(mismatches.columns):
            add_error(
                report_rows,
                source,
                "__schema__",
                f"Missing columns: {sorted(required - set(mismatches.columns))}",
            )
        else:
            record_metric(
                metrics, source, "row_count", len(mismatches)
            )

            if mismatches["hub_id"].duplicated().any():
                add_error(
                    report_rows,
                    source,
                    "unique_hub_ids",
                    "Mismatch output contains duplicate hub IDs.",
                )

            actual_hubs = set(mismatches["hub_id"].astype(str))
            expected_hubs = set(EXPECTED_MISMATCHES)

            if actual_hubs != expected_hubs:
                add_mismatch(
                    report_rows,
                    source,
                    "mismatch_hub_ids",
                    sorted(expected_hubs),
                    sorted(actual_hubs),
                    "Mismatch hub IDs must match the PDF benchmark.",
                )

            for hub_id, expected in EXPECTED_MISMATCHES.items():
                rows = mismatches.loc[
                    mismatches["hub_id"].astype(str) == hub_id
                ]

                if len(rows) != 1:
                    add_error(
                        report_rows,
                        source,
                        f"{hub_id}.__row__",
                        f"Expected one row for {hub_id}; found {len(rows)}.",
                    )
                    continue

                row = rows.iloc[0]

                actual_values = {
                    "city": str(row["hub_city"]),
                    "hub_tier": finite_number(row["hub_tier"]),
                    "segment": str(row["segment"]),
                    "expected_segment": str(row["expected_segment"]),
                }

                for field, expected_value in expected.items():
                    if actual_values[field] != expected_value:
                        add_mismatch(
                            report_rows,
                            source,
                            f"{hub_id}.{field}",
                            expected_value,
                            actual_values[field],
                            f"{hub_id} {field} must match the PDF.",
                        )

                record_metric(
                    metrics,
                    source,
                    f"{hub_id}.city_code",
                    CITY_CODES.get(actual_values["city"]),
                )
                record_metric(
                    metrics,
                    source,
                    f"{hub_id}.hub_tier",
                    actual_values["hub_tier"],
                )
                record_metric(
                    metrics,
                    source,
                    f"{hub_id}.segment_code",
                    SEGMENT_CODES.get(actual_values["segment"]),
                )
                record_metric(
                    metrics,
                    source,
                    f"{hub_id}.expected_segment_code",
                    SEGMENT_CODES.get(
                        actual_values["expected_segment"]
                    ),
                )

    # ---------------------------------------------------------
    # 5. Raw new-shipment input profile
    # ---------------------------------------------------------
    source = "data/new_shipments.csv"
    new_shipments = read_existing_csv(
        DATA / "new_shipments.csv", source, report_rows
    )

    if new_shipments is not None:
        required = {
            "shipment_id",
            "service_type",
            "payment_mode",
            "parcel_weight_kg",
            "distance_km",
            "declared_value",
        }

        if not required.issubset(new_shipments.columns):
            add_error(
                report_rows,
                source,
                "__schema__",
                f"Missing columns: {sorted(required - set(new_shipments.columns))}",
            )
        else:
            record_metric(
                metrics, source, "row_count", len(new_shipments)
            )
            record_metric(
                metrics,
                source,
                "missing_distance_km",
                int(new_shipments["distance_km"].isna().sum()),
            )
            record_metric(
                metrics,
                source,
                "missing_parcel_weight_kg",
                int(new_shipments["parcel_weight_kg"].isna().sum()),
            )
            record_metric(
                metrics,
                source,
                "missing_declared_value",
                int(new_shipments["declared_value"].isna().sum()),
            )
            record_metric(
                metrics,
                source,
                "missing_payment_mode",
                int(new_shipments["payment_mode"].isna().sum()),
            )
            record_metric(
                metrics,
                source,
                "service_type_unique_spellings",
                int(new_shipments["service_type"].nunique(dropna=True)),
            )

            if new_shipments["shipment_id"].duplicated().any():
                add_error(
                    report_rows,
                    source,
                    "unique_shipment_ids",
                    "Input contains duplicate shipment IDs.",
                )

    # ---------------------------------------------------------
    # 6. Model reload check
    # ---------------------------------------------------------
    source = "part_d_reload_check.csv"
    reload_check = read_existing_csv(
        OUTPUTS / source, source, report_rows
    )

    if reload_check is not None:
        if not {"metric", "value"}.issubset(reload_check.columns):
            add_error(
                report_rows,
                source,
                "__schema__",
                "Expected columns: metric,value.",
            )
        else:
            for metric_name in (
                "test_rows_scored",
                "reload_max_abs_probability_diff",
                "threshold",
            ):
                rows = reload_check.loc[
                    reload_check["metric"].astype(str) == metric_name
                ]

                if len(rows) != 1:
                    add_error(
                        report_rows,
                        source,
                        metric_name,
                        f"Expected exactly one row for {metric_name}; "
                        f"found {len(rows)}.",
                    )
                    continue

                record_metric(
                    metrics,
                    source,
                    metric_name,
                    finite_number(rows.iloc[0]["value"]),
                )

    # ---------------------------------------------------------
    # 7. Scoring output
    # ---------------------------------------------------------
    source = "new_shipment_actions.csv"
    actions = read_existing_csv(
        OUTPUTS / source, source, report_rows
    )

    if actions is not None:
        required_columns = [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action",
        ]

        if list(actions.columns) != required_columns:
            add_error(
                report_rows,
                source,
                "__schema__",
                f"Expected columns in order: {required_columns}; "
                f"found: {list(actions.columns)}.",
            )

        if not set(required_columns).issubset(actions.columns):
            add_error(
                report_rows,
                source,
                "__required_columns__",
                "Cannot validate scoring output because required columns "
                "are missing.",
            )
        else:
            record_metric(
                metrics, source, "row_count", len(actions)
            )
            record_metric(
                metrics,
                source,
                "unique_shipment_ids",
                int(actions["shipment_id"].nunique()),
            )

            if actions["shipment_id"].duplicated().any():
                add_error(
                    report_rows,
                    source,
                    "unique_shipment_ids",
                    "Scoring output contains duplicate shipment IDs.",
                )

            probabilities = pd.to_numeric(
                actions["delay_probability"], errors="coerce"
            )

            if probabilities.isna().any():
                add_error(
                    report_rows,
                    source,
                    "delay_probability",
                    "Probability contains missing or nonnumeric values.",
                )
            elif not probabilities.between(0, 1).all():
                add_error(
                    report_rows,
                    source,
                    "delay_probability",
                    "Every probability must be between 0 and 1 inclusive.",
                )

            if actions[required_columns].isna().any().any():
                add_error(
                    report_rows,
                    source,
                    "required_values_present",
                    "Required scoring output columns contain missing values.",
                )

            # Check actions using the stored flags and segment labels.
            # Do not recompute flags from rounded probabilities.
            valid_flags = pd.to_numeric(
                actions["flagged"], errors="coerce"
            )

            if (
                valid_flags.isna().any()
                or not valid_flags.isin([0, 1]).all()
            ):
                add_error(
                    report_rows,
                    source,
                    "flagged",
                    "Flag values must be numeric 0 or 1.",
                )
            else:
                expected_actions = []

                for flag, segment in zip(
                    valid_flags, actions["segment"].astype(str)
                ):
                    if flag == 0:
                        expected_actions.append("No action")
                    elif segment == "Escalate":
                        expected_actions.append(
                            "Escalate to hub manager"
                        )
                    else:
                        expected_actions.append(
                            "Proactive customer call"
                        )

                if actions["action"].astype(str).tolist() != expected_actions:
                    add_mismatch(
                        report_rows,
                        source,
                        "action_rules",
                        "Actions consistent with flags and segments",
                        "One or more actions are inconsistent",
                        "Flagged Escalate shipments must be escalated; "
                        "other flagged shipments receive a proactive call; "
                        "unflagged shipments require no action.",
                    )

            # Check that scoring preserves input shipment IDs and order.
            if new_shipments is not None:
                if (
                    actions["shipment_id"].astype(str).tolist()
                    != new_shipments["shipment_id"].astype(str).tolist()
                ):
                    add_mismatch(
                        report_rows,
                        source,
                        "shipment_ids_and_order",
                        "Input shipment IDs in input order",
                        "IDs or order differ",
                        "Scoring output must preserve input shipment IDs "
                        "and row order.",
                    )

            # Validate segment assignment against hub_segments.csv.
            hub_segments = read_existing_csv(
                OUTPUTS / "hub_segments.csv",
                "hub_segments.csv",
                report_rows,
            )

            if hub_segments is not None:
                required_mapping = {"hub_id", "segment"}

                if not required_mapping.issubset(hub_segments.columns):
                    add_error(
                        report_rows,
                        "hub_segments.csv",
                        "__schema__",
                        "Expected hub_id and segment columns.",
                    )
                elif hub_segments["hub_id"].duplicated().any():
                    add_error(
                        report_rows,
                        "hub_segments.csv",
                        "unique_hub_ids",
                        "Hub segment mapping contains duplicate hub IDs.",
                    )
                else:
                    mapping = hub_segments.set_index("hub_id")["segment"]
                    expected_segments = actions["hub_id"].map(mapping)

                    if expected_segments.isna().any():
                        add_error(
                            report_rows,
                            source,
                            "hub_segment_mapping",
                            "At least one scored hub has no segment mapping.",
                        )
                    elif not np.array_equal(
                        actions["segment"].astype(str).to_numpy(),
                        expected_segments.astype(str).to_numpy(),
                    ):
                        add_mismatch(
                            report_rows,
                            source,
                            "hub_segment_mapping",
                            "Segment matches hub_segments.csv",
                            "One or more segment assignments differ",
                            "Every shipment's segment must match its hub.",
                        )

    return metrics


def main():
    report_rows = []

    # Load and validate the reference file.
    reference, reference_errors = load_reference(REFERENCE_PATH)
    report_rows.extend(reference_errors)

    if reference.empty and not reference_errors:
        add_error(
            report_rows,
            str(REFERENCE_PATH),
            "__reference__",
            "Reference CSV contains no usable metrics.",
        )

    metrics = extract_metrics(report_rows)

    # Compare every reference metric individually.
    for _, reference_row in reference.iterrows():
        source_file = reference_row["source_file"]
        metric_name = reference_row["metric_name"]
        key = (source_file, metric_name)

        if key not in metrics:
            add_error(
                report_rows,
                source_file,
                metric_name,
                "Reference metric could not be extracted from the "
                "existing output. Check the file, metric name and schema.",
            )
            continue

        actual_value = metrics[key]

        if actual_value is None:
            add_error(
                report_rows,
                source_file,
                metric_name,
                "Actual metric is missing, nonnumeric or non-finite.",
            )
            continue

        report_rows.append(
            compare_numeric(
                source_file=source_file,
                metric_name=metric_name,
                reference_value=reference_row["metric_value"],
                code_value=actual_value,
                tolerance=reference_row["tolerance"],
            )
        )

    # Ensure the report is never silently successful when nothing was checked.
    if not reference_errors and reference.empty:
        add_error(
            report_rows,
            str(REFERENCE_PATH),
            "__validation__",
            "No reference metrics were evaluated.",
        )

    exit_code = write_report(report_rows, REPORT_PATH)

    if exit_code == 0:
        print("PASS: Part D reference validation.")
    else:
        print("FAIL: Part D reference validation.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())