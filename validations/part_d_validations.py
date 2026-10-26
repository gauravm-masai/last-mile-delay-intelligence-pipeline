from pathlib import Path
import os
import ast
import sys
import subprocess
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
REFERENCES = ROOT / "references"
REPORTS = ROOT / "validations/reports"

REFERENCE_PATH = REFERENCES / "part_d_reference_metrics.csv"

TOLERANCE = 0.01

def validate_hub_profile():
    """
    Validate hub_profile.csv:
    1. Exactly 24 rows.
    2. Every hub has 50 shipments.
    3. HUB01, HUB13 and HUB17 match reference metrics
       within the reference tolerance.
    """

    profile = pd.read_csv(OUTPUTS / "hub_profile.csv")
    reference = pd.read_csv(REFERENCE_PATH)

    errors = []

    # --------------------------------------------------
    # Check 1: Row count
    # --------------------------------------------------

    if len(profile) != 24:
        errors.append(
            f"Expected 24 hubs, found {len(profile)}."
        )

    # --------------------------------------------------
    # Check 2: Shipments per hub
    # --------------------------------------------------

    if "shipments" not in profile.columns:
        errors.append("Missing column: shipments.")
    else:
        incorrect_shipments = profile[
            profile["shipments"] != 50
        ]

        if not incorrect_shipments.empty:
            errors.append(
                "Every hub must have 50 shipments. "
                f"Incorrect hubs: "
                f"{incorrect_shipments['hub_id'].tolist()}"
            )

    # --------------------------------------------------
    # Check 3: Reference spot checks
    # --------------------------------------------------

    # Metric labels in the reference table mapped to
    # the corresponding hub_profile.csv columns.
    metric_columns = {
        "hours": "mean_delivery_hours",
        "delay rate": "delay_rate",
        "distance": "mean_distance_km",
        "stops": "mean_n_stops",
    }

    reference_hubs = ["HUB01", "HUB13", "HUB17"]

    required_reference_columns = {
        "Item", "Reference", "Tolerance"
    }

    if not required_reference_columns.issubset(
        reference.columns
    ):
        errors.append(
            "Reference CSV must contain columns: "
            "Item, Reference, Tolerance."
        )
    else:
        for hub_id in reference_hubs:

            # Locate the reference row for this hub.
            matching_rows = reference[
                reference["Item"].astype(str).str.contains(
                    hub_id, case=False, regex=False
                )
                & reference["Item"].astype(str).str.contains(
                    "hub_profile.csv",
                    case=False,
                    regex=False
                )
            ]

            if matching_rows.empty:
                errors.append(
                    f"No hub_profile reference found for {hub_id}."
                )
                continue

            # The PDF reference table stores all four
            # metric values in one Reference cell.
            reference_text = str(
                matching_rows.iloc[0]["Reference"]
            )

            tolerance_text = str(
                matching_rows.iloc[0]["Tolerance"]
            )

            # Extract the numeric tolerance, e.g. ±0.01.
            tolerance_match = re.search(
                r"[\u00b1+\-]?\s*(\d+(?:\.\d+)?)",
                tolerance_text
            )

            tolerance = (
                float(tolerance_match.group(1))
                if tolerance_match
                else TOLERANCE
            )

            # Find the actual row for this hub.
            actual_rows = profile[
                profile["hub_id"] == hub_id
            ]

            if actual_rows.empty:
                errors.append(
                    f"{hub_id} is missing from hub_profile.csv."
                )
                continue

            actual_row = actual_rows.iloc[0]

            for label, column in metric_columns.items():

                # Extract the reference value following
                # its label, e.g. "hours 45.685".
                pattern = (
                    re.escape(label)
                    + r"\s*[:=]?\s*"
                    + r"(-?\d+(?:\.\d+)?)"
                )

                value_match = re.search(
                    pattern,
                    reference_text,
                    flags=re.IGNORECASE
                )

                if not value_match:
                    errors.append(
                        f"Could not read reference value for "
                        f"{hub_id}, metric '{label}'."
                    )
                    continue

                expected = float(value_match.group(1))
                actual = float(actual_row[column])

                if abs(actual - expected) > tolerance:
                    errors.append(
                        f"{hub_id} {column}: expected "
                        f"{expected}, found {actual}; "
                        f"tolerance ±{tolerance}."
                    )

    # --------------------------------------------------
    # Report result
    # --------------------------------------------------

    passed = len(errors) == 0

    if passed:
        print("PASS: hub_profile validation.")
    else:
        print("FAIL: hub_profile validation.")
        for error in errors:
            print(f" - {error}")

    return {
        "validation": "hub_profile",
        "passed": passed,
        "errors": errors,
    }


def validate_part_d_separation():
    """Validate separation results against the reference CSV."""
    output_path = OUTPUTS / "part_d_separation.csv"

    results = pd.read_csv(output_path)
    reference = pd.read_csv(REFERENCE_PATH)

    errors = []

    # Check required output columns.
    required_columns = {
        "K", "across_over_within", "cluster_sizes"
    }

    if not required_columns.issubset(results.columns):
        errors.append(
            "part_d_separation.csv is missing required columns."
        )
        return {
            "validation": "part_d_separation",
            "passed": False,
            "errors": errors,
        }

    # The reference CSV has columns Item, Reference, Tolerance.
    if not {"Item", "Reference", "Tolerance"}.issubset(
        reference.columns
    ):
        errors.append(
            "Reference CSV must contain Item, Reference and "
            "Tolerance columns."
        )
        return {
            "validation": "part_d_separation",
            "passed": False,
            "errors": errors,
        }

    # Exactly four output rows are required.
    if len(results) != 4:
        errors.append(
            f"Expected 4 rows, found {len(results)}."
        )

    # Find the reference row for each metric group.
    items = reference["Item"].astype(str)

    def find_reference_row(metric_name, k_group):
        mask = (
            items.str.contains(
                "part_d_separation.csv",
                case=False,
                regex=False
            )
            & items.str.contains(
                metric_name,
                case=False,
                regex=False
            )
            & items.str.contains(
                k_group,
                case=False,
                regex=False
            )
        )

        rows = reference.loc[mask]

        if len(rows) != 1:
            errors.append(
                f"Expected one reference row for "
                f"{metric_name}, K={k_group}; found {len(rows)}."
            )
            return None

        return rows.iloc[0]

    # Parse the paired values directly from each reference row.
    def parse_pair(reference_value, pattern):
        return re.findall(pattern, str(reference_value))

    reference_specs = [
        ("cluster_sizes", "2 / 3", [2, 3]),
        ("across_over_within", "2 / 3", [2, 3]),
        ("cluster_sizes", "4 / 5", [4, 5]),
        ("across_over_within", "4 / 5", [4, 5]),
    ]

    expected_sizes = {}
    expected_ratios = {}
    ratio_tolerances = {}

    for metric_name, k_group, ks in reference_specs:
        row = find_reference_row(metric_name, k_group)

        if row is None:
            continue

        reference_value = row["Reference"]

        if metric_name == "cluster_sizes":
            values = parse_pair(
                reference_value,
                r"\d+(?:-\d+)+"
            )

            if len(values) != 2:
                errors.append(
                    f"Could not parse cluster sizes for K={k_group}."
                )
                continue

            for k, value in zip(ks, values):
                expected_sizes[k] = value

        else:
            values = parse_pair(
                reference_value,
                r"\d+(?:\.\d+)?"
            )

            if len(values) != 2:
                errors.append(
                    f"Could not parse ratios for K={k_group}."
                )
                continue

            tolerance_text = str(row["Tolerance"])
            tolerance_match = re.search(
                r"[±+-]?\s*(\d+(?:\.\d+)?)",
                tolerance_text
            )

            if not tolerance_match:
                errors.append(
                    f"Could not parse ratio tolerance for K={k_group}."
                )
                continue

            tolerance = float(tolerance_match.group(1))

            for k, value in zip(ks, values):
                expected_ratios[k] = float(value)
                ratio_tolerances[k] = tolerance

    # Compare actual results with values loaded from the reference.
    for k in [2, 3, 4, 5]:
        rows = results[results["K"] == k]

        if len(rows) != 1:
            errors.append(
                f"K={k}: expected one output row, found {len(rows)}."
            )
            continue

        actual = rows.iloc[0]

        if k in expected_sizes:
            actual_sizes = str(actual["cluster_sizes"]).strip()

            if actual_sizes != expected_sizes[k]:
                errors.append(
                    f"K={k}: expected cluster sizes "
                    f"{expected_sizes[k]}, found {actual_sizes}."
                )

        if k in expected_ratios:
            actual_ratio = float(actual["across_over_within"])
            expected_ratio = expected_ratios[k]
            tolerance = ratio_tolerances[k]

            if abs(actual_ratio - expected_ratio) > tolerance:
                errors.append(
                    f"K={k}: expected ratio {expected_ratio}, "
                    f"found {actual_ratio}; tolerance ±{tolerance}."
                )

    passed = len(errors) == 0

    if passed:
        print("PASS: part_d_separation validation.")
    else:
        print("FAIL: part_d_separation validation.")
        for error in errors:
            print(" -", error)

    return {
        "validation": "part_d_separation",
        "passed": passed,
        "errors": errors,
    }


def validate_segment_profile():
    """Compare segment_profile.csv with the reference metrics."""
    profile = pd.read_csv(OUTPUTS / "segment_profile.csv")
    reference = pd.read_csv(REFERENCE_PATH)

    errors = []

    required_columns = {
        "segment", "hubs", "mean_delivery_hours",
        "delay_rate", "mean_distance_km", "mean_n_stops"
    }

    if not required_columns.issubset(profile.columns):
        errors.append("segment_profile.csv is missing required columns.")
        return {
            "validation": "segment_profile",
            "passed": False,
            "errors": errors,
        }

    if not {"Item", "Reference", "Tolerance"}.issubset(reference.columns):
        errors.append("Reference CSV must contain Item, Reference and Tolerance.")
        return {
            "validation": "segment_profile",
            "passed": False,
            "errors": errors,
        }

    # Find the reference row for each segment.
    reference_items = reference["Item"].astype(str)

    segment_specs = {
        "Standard": {
            "hubs": "hubs",
            "mean_delivery_hours": "hours",
            "delay_rate": "delay rate",
            "mean_distance_km": "distance",
            "mean_n_stops": "stops",
        },
        "Watch": {
            "hubs": "hubs",
            "mean_delivery_hours": "hours",
            "delay_rate": "delay rate",
            "mean_distance_km": "distance",
            "mean_n_stops": "stops",
        },
        "Escalate": {
            "hubs": "hubs",
            "mean_delivery_hours": "hours",
            "delay_rate": "delay rate",
            "mean_distance_km": "distance",
            "mean_n_stops": "stops",
        },
    }

    for segment, metric_labels in segment_specs.items():

        mask = (
            reference_items.str.contains(
                f"Segment {segment}",
                case=False,
                regex=False
            )
        )

        rows = reference.loc[mask]

        if len(rows) != 1:
            errors.append(
                f"Expected one reference row for segment {segment}; "
                f"found {len(rows)}."
            )
            continue

        reference_row = rows.iloc[0]
        reference_text = str(reference_row["Reference"])

        tolerance_match = re.search(
            r"±\s*(\d+(?:\.\d+)?)",
            str(reference_row["Tolerance"])
        )
        tolerance = (
            float(tolerance_match.group(1))
            if tolerance_match
            else 0.01
        )

        actual_rows = profile[
            profile["segment"].astype(str) == segment
        ]

        if len(actual_rows) != 1:
            errors.append(
                f"Expected one output row for segment {segment}; "
                f"found {len(actual_rows)}."
            )
            continue

        actual_row = actual_rows.iloc[0]

        # The reference text contains values in this order:
        # hubs, hours, delay rate, distance, stops.
        values = re.findall(
            r"-?\d+(?:\.\d+)?",
            reference_text
        )

        if len(values) < 5:
            errors.append(
                f"Could not parse five reference values for {segment}."
            )
            continue

        expected_values = {
            "hubs": float(values[0]),
            "mean_delivery_hours": float(values[1]),
            "delay_rate": float(values[2]),
            "mean_distance_km": float(values[3]),
            "mean_n_stops": float(values[4]),
        }

        for column, expected in expected_values.items():
            actual = float(actual_row[column])

            if abs(actual - expected) > tolerance:
                errors.append(
                    f"{segment} {column}: expected {expected}, "
                    f"found {actual}; tolerance ±{tolerance}."
                )

    passed = len(errors) == 0

    print(
        "PASS: segment profile validation."
        if passed else "FAIL: segment profile validation."
    )
    for error in errors:
        print(" -", error)

    return {
        "validation": "segment_profile",
        "passed": passed,
        "errors": errors,
    }


def validate_tier_segment_mismatches():
    """Check mismatch hubs against the reference CSV."""
    mismatch_path = OUTPUTS / "tier_vs_segment_mismatch.csv"

    mismatches = pd.read_csv(mismatch_path)
    reference = pd.read_csv(REFERENCE_PATH)

    errors = []

    required_columns = {
        "hub_id", "segment", "hub_city", "hub_tier"
    }

    if not required_columns.issubset(mismatches.columns):
        errors.append(
            "tier_vs_segment_mismatch.csv is missing required columns."
        )
        return {
            "validation": "tier_segment_mismatches",
            "passed": False,
            "errors": errors,
        }

    if not {"Item", "Reference", "Tolerance"}.issubset(reference.columns):
        errors.append("Reference CSV must contain Item, Reference and Tolerance.")
        return {
            "validation": "tier_segment_mismatches",
            "passed": False,
            "errors": errors,
        }

    item_text = reference["Item"].astype(str)

    rows = reference[
        item_text.str.contains(
            "Tier-versus-segment mismatches",
            case=False,
            regex=False
        )
    ]

    if len(rows) != 1:
        errors.append(
            "Could not find a unique tier-versus-segment reference row."
        )
    else:
        expected_text = str(rows.iloc[0]["Reference"])
        expected_hubs = set(
            re.findall(r"\bHUB\d+\b", expected_text)
        )

        actual_hubs = set(mismatches["hub_id"].astype(str))

        if len(mismatches) != 4:
            errors.append(
                f"Expected 4 mismatch rows, found {len(mismatches)}."
            )

        if actual_hubs != expected_hubs:
            errors.append(
                f"Expected hubs {sorted(expected_hubs)}, "
                f"found {sorted(actual_hubs)}."
            )

    passed = len(errors) == 0

    print(
        "PASS: tier-versus-segment mismatch validation."
        if passed else "FAIL: tier-versus-segment mismatch validation."
    )
    for error in errors:
        print(" -", error)

    return {
        "validation": "tier_segment_mismatches",
        "passed": passed,
        "errors": errors,
    }


def validate_segment_naming():
    """Verify segment names follow ascending mean delay rate."""

    profile = pd.read_csv(OUTPUTS / "hub_profile.csv")
    segments = pd.read_csv(OUTPUTS / "hub_segments.csv")

    errors = []

    required_profile = {"hub_id", "delay_rate"}
    required_segments = {"hub_id", "segment"}

    if not required_profile.issubset(profile.columns):
        errors.append("hub_profile.csv is missing hub_id or delay_rate.")
    if not required_segments.issubset(segments.columns):
        errors.append("hub_segments.csv is missing hub_id or segment.")

    if errors:
        return {
            "validation": "segment_naming",
            "passed": False,
            "errors": errors,
        }

    combined = profile[["hub_id", "delay_rate"]].merge(
        segments[["hub_id", "segment"]],
        on="hub_id",
        how="inner",
        validate="one_to_one"
    )

    if len(combined) != len(profile):
        errors.append(
            "Not every hub in hub_profile.csv has exactly one segment."
        )

    segment_means = (
        combined.groupby("segment")["delay_rate"]
        .mean()
        .sort_values()
    )

    expected_order = ["Standard", "Watch", "Escalate"]

    if list(segment_means.index) != expected_order:
        errors.append(
            "Segments must be ordered by ascending mean delay rate "
            "as Standard, Watch, Escalate. Actual order: "
            f"{list(segment_means.index)}."
        )

    if set(segments["segment"].dropna().unique()) != set(expected_order):
        errors.append(
            "The segment names must be exactly Standard, Watch and Escalate."
        )

    passed = len(errors) == 0

    print(
        "PASS: segment naming validation."
        if passed else "FAIL: segment naming validation."
    )
    for error in errors:
        print(" -", error)

    return {
        "validation": "segment_naming",
        "passed": passed,
        "errors": errors,
    }

def validate_model_packaging():
    reload_path = OUTPUTS / "part_d_reload_check.csv"
    feature_path = MODELS / "feature_order.csv"

    model_files = [
        MODELS / "delay_model.pkl",
        MODELS / "delay_threshold.csv",
        MODELS / "feature_order.csv",
        MODELS / "impute_values.csv",
    ]

    errors = []

    # --------------------------------------------------
    # Check 1: All four model files exist
    # --------------------------------------------------

    for path in model_files:
        if not os.path.isfile(path):
            errors.append(f"Required model file is missing: {path}")

    # --------------------------------------------------
    # Check 2: Reload-check CSV
    # --------------------------------------------------

    if not os.path.isfile(reload_path):
        errors.append(f"Missing file: {reload_path}")
    else:
        reload_df = pd.read_csv(reload_path)

        if not {"metric", "value"}.issubset(reload_df.columns):
            errors.append(
                "part_d_reload_check.csv must have metric,value columns."
            )
        else:
            metrics = reload_df.set_index("metric")["value"]

            required_metrics = [
                "test_rows_scored",
                "reload_max_abs_probability_diff",
                "threshold",
            ]

            for metric in required_metrics:
                if metric not in metrics.index:
                    errors.append(
                        f"Missing reload-check metric: {metric}"
                    )

            if "test_rows_scored" in metrics.index:
                actual_rows = float(metrics["test_rows_scored"])

                if actual_rows != 180:
                    errors.append(
                        f"Expected 180 test rows, found {actual_rows}."
                    )

            if "reload_max_abs_probability_diff" in metrics.index:
                difference = float(
                    metrics["reload_max_abs_probability_diff"]
                )

                if difference > 1e-9:
                    errors.append(
                        "Reload probability difference exceeds 1e-9: "
                        f"{difference}."
                    )

    # --------------------------------------------------
    # Check 3: Read pinned FEATURES from Part C source
    # --------------------------------------------------

    # Update this path if FEATURES is defined in a different file.
    source_path = ROOT / "src/shipment_prep.py"

    if not os.path.isfile(feature_path):
        errors.append(f"Missing file: {feature_path}")

    if not os.path.isfile(source_path):
        errors.append(
            f"Could not find {source_path} to verify the pinned FEATURES list. "
            "Update source_path to the file where FEATURES is defined."
        )
    elif os.path.isfile(feature_path):
        with open(source_path, "r", encoding="utf-8") as file:
            source_code = file.read()

        syntax_tree = ast.parse(source_code)
        pinned_features = None

        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "FEATURES":
                        try:
                            pinned_features = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass

            elif isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "FEATURES"
                ):
                    try:
                        pinned_features = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass

        if pinned_features is None:
            errors.append(
                f"Could not extract a literal FEATURES list from {source_path}."
            )
        else:
            saved_df = pd.read_csv(feature_path)

            if "feature" not in saved_df.columns:
                errors.append(
                    "feature_order.csv must contain a 'feature' column."
                )
            else:
                saved_features = saved_df["feature"].tolist()

                if len(pinned_features) != 17:
                    errors.append(
                        f"Expected 17 pinned features, found "
                        f"{len(pinned_features)}."
                    )

                if saved_features != pinned_features:
                    errors.append(
                        "feature_order.csv does not match the pinned "
                        "FEATURES list in the exact order."
                    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    passed = len(errors) == 0

    if passed:
        print("PASS: model packaging validation.")
    else:
        print("FAIL: model packaging validation.")
        for error in errors:
            print(" -", error)

    return {
        "validation": "model_packaging",
        "passed": passed,
        "errors": errors,
    }


def validate_new_shipment_actions():
    """
    Validate the existing new_shipment_actions.csv output.

    This function only reads existing files. It does not run prediction,
    retrain a model, or regenerate any output.
    """
    errors = []

    input_path = DATA / "new_shipments.csv"
    output_path = OUTPUTS / "new_shipment_actions.csv"
    threshold_path = MODELS / "delay_threshold.csv"
    hub_segments_path = OUTPUTS / "hub_segments.csv"

    # Required output columns, in the required order
    expected_columns = [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]

    # Check that required files exist
    required_files = {
        "input shipments": input_path,
        "prediction output": output_path,
        "saved threshold": threshold_path,
        "hub segments": hub_segments_path,
    }

    for name, path in required_files.items():
        if not path.exists():
            errors.append(f"Missing {name} file: {path}")

    if errors:
        return {
            "validation": "new_shipment_actions",
            "passed": False,
            "errors": errors,
        }

    # Read existing files; do not run predict_delay.py
    try:
        input_df = pd.read_csv(input_path)
        output_df = pd.read_csv(output_path)
        threshold_df = pd.read_csv(threshold_path)
        hub_segments_df = pd.read_csv(hub_segments_path)
    except Exception as exc:
        return {
            "validation": "new_shipment_actions",
            "passed": False,
            "errors": [f"Could not read validation files: {exc}"],
        }

    # 1. Validate output columns
    if list(output_df.columns) != expected_columns:
        errors.append(
            "Output columns do not match the required columns and order. "
            f"Expected {expected_columns}; "
            f"found {list(output_df.columns)}."
        )

    # 2. Validate row count
    if len(input_df) != 48:
        errors.append(
            f"Expected 48 input shipments; found {len(input_df)}."
        )

    if len(output_df) != 48:
        errors.append(
            f"Expected 48 output rows; found {len(output_df)}."
        )

    if len(input_df) != len(output_df):
        errors.append(
            "Input and output row counts do not match."
        )

    # Further checks require the expected columns
    if not all(col in output_df.columns for col in expected_columns):
        return {
            "validation": "new_shipment_actions",
            "passed": False,
            "errors": errors,
        }

    # 3. Check missing values
    if output_df[expected_columns].isnull().any().any():
        missing_columns = (
            output_df[expected_columns]
            .columns[
                output_df[expected_columns].isnull().any()
            ]
            .tolist()
        )
        errors.append(
            f"Output contains missing values in: {missing_columns}."
        )

    # 4. Check shipment IDs are unique
    if output_df["shipment_id"].duplicated().any():
        errors.append("Output contains duplicate shipment IDs.")

    # 5. Check input and output shipment IDs and preserve input order
    if "shipment_id" not in input_df.columns:
        errors.append("Input file does not contain shipment_id.")
    else:
        input_ids = input_df["shipment_id"].tolist()
        output_ids = output_df["shipment_id"].tolist()

        if input_ids != output_ids:
            errors.append(
                "Output shipment IDs do not match the input shipment IDs "
                "in the same order."
            )

    # 6. Validate saved threshold
    if "threshold" not in threshold_df.columns:
        errors.append(
            "delay_threshold.csv does not contain a 'threshold' column."
        )
    elif len(threshold_df) != 1:
        errors.append(
            "delay_threshold.csv must contain exactly one threshold value."
        )
    else:
        threshold = pd.to_numeric(
            threshold_df.loc[0, "threshold"],
            errors="coerce",
        )

        if pd.isna(threshold) or not 0 <= threshold <= 1:
            errors.append(
                "The saved threshold must be a numeric value between 0 and 1."
            )
        else:
            # 7. Validate probabilities and flags
            probabilities = pd.to_numeric(
                output_df["delay_probability"],
                errors="coerce",
            )

            if probabilities.isna().any():
                errors.append(
                    "delay_probability contains missing or non-numeric values."
                )
            elif not probabilities.between(0, 1).all():
                errors.append(
                    "delay_probability values must be between 0 and 1."
                )
            else:
                expected_flags = (
                    probabilities >= threshold
                ).astype(int)

                actual_flags = pd.to_numeric(
                    output_df["flagged"],
                    errors="coerce",
                )

                if (
                    actual_flags.isna().any()
                    or not actual_flags.isin([0, 1]).all()
                    or not actual_flags.astype(int).equals(
                        expected_flags.rename(actual_flags.name)
                    )
                ):
                    errors.append(
                        "flagged values do not match the saved threshold."
                    )

    # 8. Validate segment values against the hub-segment mapping
    required_mapping_columns = {"hub_id", "segment"}

    if not required_mapping_columns.issubset(hub_segments_df.columns):
        errors.append(
            "hub_segments.csv must contain hub_id and segment columns."
        )
    elif hub_segments_df["hub_id"].duplicated().any():
        errors.append(
            "hub_segments.csv contains duplicate hub IDs."
        )
    else:
        expected_segments = output_df["hub_id"].map(
            hub_segments_df.set_index("hub_id")["segment"]
        )

        if expected_segments.isna().any():
            errors.append(
                "Some output hub IDs are missing from hub_segments.csv."
            )
        else:
            mismatch = (
                output_df["segment"].astype(str).to_numpy()
                != expected_segments.astype(str).to_numpy()
            )

            if mismatch.any():
                errors.append(
                    "Some output segments do not match their hub's segment."
                )

    # 9. Validate action rules
    if "flagged" in output_df.columns:
        flags = pd.to_numeric(output_df["flagged"], errors="coerce")

        if flags.notna().all():
            expected_actions = []

            for flag, segment in zip(flags, output_df["segment"]):
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

            if output_df["action"].astype(str).tolist() != expected_actions:
                errors.append(
                    "Action values do not match the required flag/segment rules."
                )

    if errors:
        print("FAIL: new shipment actions validation.")
        for error in errors:
            print(f"- {error}")
    else:
        print("PASS: new shipment actions validation.")

    return {
        "validation": "new_shipment_actions",
        "passed": len(errors) == 0,
        "errors": errors,
    }

if __name__ == "__main__":
    validate_hub_profile()
    validate_part_d_separation()
    validate_segment_profile()
    validate_tier_segment_mismatches()
    validate_segment_naming()
    validate_model_packaging()
    validate_new_shipment_actions()