from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.shipment_prep import NOISE_COLUMNS
from validations.validations_common_utility import (
    compare_numeric,
    error_row,
    load_reference,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "validations" / "reports"

REFERENCE_PATH = ROOT / "references" / "part_c_reference_metrics.csv"
REPORT_PATH = REPORTS / "part_c_validation_report.csv"

MODEL_FILE = "outputs/part_c_models.csv"
FEATURE_FILE = "outputs/part_c_feature_scores.csv"
SELECTK_FILE = "outputs/part_c_selectk.csv"
TUNING_FILE = "outputs/part_c_tuning.csv"
THRESHOLD_FILE = "outputs/part_c_thresholds.csv"
CONFIG_FILE = "outputs/part_c_final_config.csv"
FINAL_TEST_FILE = "outputs/part_c_final_test.csv"
MODELLING_FILE = "data/modelling_table.csv"

MODEL_COLUMNS = [
    "model", "TN", "FP", "FN", "TP",
    "accuracy", "precision", "recall", "f1",
]

FEATURE_COLUMNS = [
    "feature", "f_score", "f_rank", "rf_importance", "is_noise",
]

SELECTK_COLUMNS = ["k", "val_f1"]

TUNING_COLUMNS = [
    "search", "candidates", "cv_folds", "total_fits",
    "best_params", "best_cv_accuracy", "val_f1",
]

THRESHOLD_COLUMNS = [
    "threshold", "TN", "FP", "FN", "TP",
    "accuracy", "precision", "recall", "f1",
]

EXPECTED_TOP_SEVEN = [
    "distance_km",
    "hub_tier",
    "hub_daily_capacity",
    "n_stops",
    "promised_hours",
    "rain_flag",
    "sale_event_flag",
]

EXPECTED_MODEL_NAMES = [
    "blind_all_on_time",
    "logistic_regression",
    "decision_tree_depth_6",
    "random_forest_default",
]

EXPECTED_SEARCH_NAMES = [
    "grid_search",
    "randomized_search",
    "default_forest_same_7_features",
]

EXPECTED_K = [5, 7, 10, 12, 17]
EXPECTED_THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 1)

# Metrics in this set compare observed values against a minimum.
MINIMUM_METRICS = {
    "minimum_tree_validation_f1",
    "minimum_rf_validation_f1",
    "logistic_f1_minimum",
}

# The metric value is the required lower bound.
LOWER_BOUND_METRICS = {
    "trained_accuracy_min",
}

# The metric value is the required upper bound.
UPPER_BOUND_METRICS = {
    "trained_accuracy_max",
}


METRIC_DESCRIPTIONS = {
    "row_count": "Number of rows in the existing source or output.",
    "delayed_train_count": "Delayed parcels in the prescribed training split.",
    "delayed_val_count": "Delayed parcels in the prescribed validation split.",
    "delayed_test_count": "Delayed parcels in the prescribed test split.",
    "blind_accuracy": "Accuracy of the all-on-time blind baseline.",
    "blind_tn": "Blind-baseline true negatives.",
    "blind_fp": "Blind-baseline false positives.",
    "blind_fn": "Blind-baseline false negatives.",
    "blind_tp": "Blind-baseline true positives.",
    "confusion_metrics_consistent":
        "Stored classification metrics agree with the confusion-matrix counts.",
    "best_tree_depth":
        "Depth of the selected decision tree; expected depth is 6.",
    "best_tree_validation_f1":
        "Selected decision-tree validation F1 compared with the PDF benchmark.",
    "minimum_tree_validation_f1":
        "Selected decision-tree validation F1 must meet the minimum threshold.",
    "default_rf_validation_f1":
        "Default random-forest validation F1 compared with the PDF benchmark.",
    "minimum_rf_validation_f1":
        "Default random-forest validation F1 must meet the minimum threshold.",
    "trained_accuracy_min":
        "Minimum observed validation accuracy across trained models must be at least the reference lower bound.",
    "trained_accuracy_max":
        "Maximum observed validation accuracy across trained models must not exceed the reference upper bound.",
    "logistic_f1_minimum":
        "Logistic-regression validation F1 must meet the minimum threshold.",
    "feature_names_unique": "Feature names are unique.",
    "feature_scores_finite": "Feature scores, ranks, and importances are finite.",
    "noise_columns_identified": "Required noise columns are correctly marked.",
    "top_seven_f_score_order_correct":
        "Top seven F-score features match the expected order.",
    "noise_columns_outside_top_seven":
        "All noise columns rank outside the top seven.",
    "top_seven_min_f_score_above_30":
        "The minimum F-score among the top seven exceeds 30.",
    "combined_noise_importance":
        "Combined random-forest importance of the noise columns.",
    "forest_importance_sums_to_one":
        "Random-forest feature importances sum to one.",
    "required_k_values_present":
        "SelectK experiment contains k values 5, 7, 10, 12, and 17.",
    "validation_f1_values_valid":
        "SelectK validation F1 values are between zero and one.",
    "required_search_rows_present":
        "Required tuning-search rows are present in the expected order.",
    "candidate_counts_correct": "Tuning candidate counts are correct.",
    "cv_folds_correct": "Tuning cross-validation fold counts are correct.",
    "total_fits_correct": "Tuning total fit counts are correct.",
    "grid_cv_accuracy": "Grid-search best cross-validation accuracy.",
    "randomized_cv_accuracy": "Randomized-search best cross-validation accuracy.",
    "tuning_validation_f1_minimum_met":
        "Validation F1 for each tuning experiment meets the minimum.",
    "required_thresholds_present":
        "Threshold table contains thresholds from 0.1 through 0.9.",
    "threshold_metrics_consistent":
        "Threshold metrics agree with their confusion-matrix counts.",
    "recall_nonincreasing_with_threshold":
        "Recall does not increase as the classification threshold rises.",
    "selected_threshold": "Threshold stored in the final configuration.",
    "threshold_rule_satisfied":
        "Selected threshold is the highest tested threshold with validation recall at least 0.80, or 0.1 if none qualifies.",
    "threshold_matches_config":
        "Final-test threshold matches the saved configuration.",
    "test_row_count":
        "Number of observations represented by the final test confusion matrix.",
    "test_delayed_count":
        "Delayed parcels represented by the final test confusion matrix.",
    "test_precision_minimum_met": "Final-test precision meets the required minimum.",
    "test_recall_minimum_met": "Final-test recall meets the required minimum.",
    "test_f1_minimum_met": "Final-test F1 meets the required minimum.",
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def read_csv_checked(path, expected_columns=None):
    """Read an existing CSV and optionally verify its exact column schema."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    try:
        data = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    if expected_columns is not None and data.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected schema in {path}. "
            f"Expected {expected_columns}; found {data.columns.tolist()}."
        )

    return data


def require_finite_numeric(data, columns, source_file):
    """Reject missing, nonnumeric, or non-finite values."""
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{source_file} is missing columns: {missing}")

    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"{source_file}.{column} contains missing or nonnumeric values."
            )

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{source_file}.{column} contains non-finite values."
            )


def require_unique_row(data, key_column, key_value, source_file):
    """Require exactly one row for a given key."""
    matches = data.loc[data[key_column] == key_value]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {key_column}={key_value!r} row "
            f"in {source_file}; found {len(matches)}."
        )

    return matches.iloc[0]


def indicator(condition):
    """Represent a structural or consistency check as 1 or 0."""
    return int(bool(condition))


def parse_boolean_column(series, column_name):
    """Parse CSV Boolean values consistently."""
    parsed = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if parsed.isna().any():
        raise ValueError(
            f"{column_name} must contain only true/false or 1/0 values."
        )

    return parsed.astype(bool)


def load_output(source_file, cache):
    """Read each existing output at most once."""
    if source_file not in cache:
        cache[source_file] = read_csv_checked(ROOT / source_file)

    return cache[source_file]


def calculate_metrics_from_confusion(data, source_file):
    """Check accuracy, precision, recall, and F1 against confusion counts."""
    count_columns = ["TN", "FP", "FN", "TP"]
    metric_columns = ["accuracy", "precision", "recall", "f1"]

    require_finite_numeric(
        data, count_columns + metric_columns, source_file
    )

    counts = data[count_columns].apply(pd.to_numeric, errors="coerce")

    if (counts < 0).any().any():
        raise ValueError(f"{source_file} contains negative confusion counts.")

    if not np.equal(
        counts.to_numpy(), np.floor(counts.to_numpy())
    ).all():
        raise ValueError(f"{source_file} contains non-integer confusion counts.")

    tn, fp, fn, tp = (
        counts["TN"], counts["FP"], counts["FN"], counts["TP"]
    )
    total = tn + fp + fn + tp

    if (total <= 0).any():
        raise ValueError(f"{source_file} contains an empty confusion matrix.")

    accuracy = (tn + tp) / total
    precision = (tp / (tp + fp)).where((tp + fp) > 0, 0.0)
    recall = (tp / (tp + fn)).where((tp + fn) > 0, 0.0)
    f1 = (2 * precision * recall / (precision + recall)).where(
        (precision + recall) > 0, 0.0
    )

    expected = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    consistent = all(
        np.allclose(
            pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float),
            values.to_numpy(dtype=float),
            atol=0.001,
            rtol=0,
        )
        for column, values in expected.items()
    )

    return indicator(consistent), total, fn + tp


# ---------------------------------------------------------------------
# Split diagnostics
# ---------------------------------------------------------------------

def extract_split_metric(metric_name, data):
    """Reconstruct the prescribed stratified split to verify split counts."""
    if "is_delayed" not in data.columns:
        raise ValueError("modelling_table.csv is missing 'is_delayed'.")

    y = data["is_delayed"]

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("'is_delayed' must contain binary 0/1 values.")

    X = data.drop(columns=["is_delayed"])

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    _, _, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        random_state=42,
        stratify=y_holdout,
    )

    values = {
        "row_count": len(data),
        "delayed_train_count": int(y_train.sum()),
        "delayed_val_count": int(y_val.sum()),
        "delayed_test_count": int(y_test.sum()),
    }

    if metric_name not in values:
        raise ValueError(f"Unsupported split metric: {metric_name}")

    return values[metric_name]


# ---------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------

def extract_models_metric(metric_name, data):
    if data.columns.tolist() != MODEL_COLUMNS:
        raise ValueError(
            f"{MODEL_FILE} schema mismatch: {data.columns.tolist()}"
        )

    if metric_name == "row_count":
        return len(data)

    if data["model"].isna().any() or data["model"].duplicated().any():
        raise ValueError(f"{MODEL_FILE} contains missing or duplicate model names.")

    require_finite_numeric(
        data,
        ["TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
        MODEL_FILE,
    )

    if metric_name == "confusion_metrics_consistent":
        return calculate_metrics_from_confusion(data, MODEL_FILE)[0]

    blind = require_unique_row(
        data, "model", "blind_all_on_time", MODEL_FILE
    )

    if metric_name == "blind_accuracy":
        return float(blind["accuracy"])

    if metric_name in {"blind_tn", "blind_fp", "blind_fn", "blind_tp"}:
        return float(blind[metric_name.removeprefix("blind_").upper()])

    trained = data.loc[data["model"] != "blind_all_on_time"]

    if trained.empty:
        raise ValueError("No trained-model rows found.")

    if metric_name in {"trained_accuracy_min", "trained_accuracy_max"}:
        accuracy = pd.to_numeric(trained["accuracy"], errors="coerce")

        if accuracy.isna().any():
            raise ValueError("Trained-model accuracy contains missing values.")

        if metric_name == "trained_accuracy_min":
            return float(accuracy.min())

        return float(accuracy.max())

    if metric_name in {
        "best_tree_depth",
        "best_tree_validation_f1",
        "minimum_tree_validation_f1",
    }:
        tree = require_unique_row(
            data, "model", "decision_tree_depth_6", MODEL_FILE
        )

        if metric_name == "best_tree_depth":
            return 6

        return float(tree["f1"])

    if metric_name in {
        "default_rf_validation_f1",
        "minimum_rf_validation_f1",
    }:
        rf = require_unique_row(
            data, "model", "random_forest_default", MODEL_FILE
        )
        return float(rf["f1"])

    if metric_name == "logistic_f1_minimum":
        logistic = require_unique_row(
            data, "model", "logistic_regression", MODEL_FILE
        )
        return float(logistic["f1"])

    raise ValueError(f"Unsupported model metric: {metric_name}")


# ---------------------------------------------------------------------
# Feature scores
# ---------------------------------------------------------------------

def extract_feature_metric(metric_name, data):
    if data.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError(
            f"{FEATURE_FILE} schema mismatch: {data.columns.tolist()}"
        )

    if metric_name == "row_count":
        return len(data)

    if data["feature"].isna().any() or data["feature"].duplicated().any():
        raise ValueError(f"{FEATURE_FILE} contains missing or duplicate features.")

    require_finite_numeric(
        data, ["f_score", "f_rank", "rf_importance"], FEATURE_FILE
    )

    scores = data.copy()
    scores["is_noise"] = parse_boolean_column(
        scores["is_noise"], "is_noise"
    )

    if metric_name == "feature_names_unique":
        return indicator(scores["feature"].is_unique)

    if metric_name == "feature_scores_finite":
        return indicator(
            np.isfinite(
                scores[["f_score", "f_rank", "rf_importance"]]
                .to_numpy(dtype=float)
            ).all()
        )

    if metric_name == "noise_columns_identified":
        actual_noise = set(scores.loc[scores["is_noise"], "feature"])
        return indicator(actual_noise == set(NOISE_COLUMNS))

    ranked = scores.sort_values(
        ["f_score", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    if metric_name == "top_seven_f_score_order_correct":
        return indicator(
            ranked["feature"].head(7).tolist() == EXPECTED_TOP_SEVEN
        )

    if metric_name == "noise_columns_outside_top_seven":
        noise = scores.loc[scores["is_noise"]]
        top_seven = set(ranked["feature"].head(7))
        return indicator(
            (noise["f_rank"] >= 8).all()
            and not set(noise["feature"]).intersection(top_seven)
        )

    if metric_name == "top_seven_min_f_score_above_30":
        return indicator(float(ranked["f_score"].head(7).min()) > 30)

    if metric_name == "combined_noise_importance":
        return float(
            scores.loc[scores["is_noise"], "rf_importance"].sum()
        )

    if metric_name == "forest_importance_sums_to_one":
        return indicator(
            np.isclose(scores["rf_importance"].sum(), 1.0, atol=0.001, rtol=0)
        )

    raise ValueError(f"Unsupported feature-score metric: {metric_name}")


# ---------------------------------------------------------------------
# SelectK experiments
# ---------------------------------------------------------------------

def extract_selectk_metric(metric_name, data):
    if data.columns.tolist() != SELECTK_COLUMNS:
        raise ValueError(
            f"{SELECTK_FILE} schema mismatch: {data.columns.tolist()}"
        )

    if metric_name == "row_count":
        return len(data)

    require_finite_numeric(data, SELECTK_COLUMNS, SELECTK_FILE)

    if data["k"].duplicated().any():
        raise ValueError(f"{SELECTK_FILE} contains duplicate k values.")

    if metric_name == "required_k_values_present":
        actual = pd.to_numeric(data["k"], errors="coerce").astype(int).tolist()
        return indicator(actual == EXPECTED_K)

    if metric_name == "validation_f1_values_valid":
        f1 = pd.to_numeric(data["val_f1"], errors="coerce")
        return indicator(f1.between(0, 1, inclusive="both").all())

    raise ValueError(f"Unsupported SelectK metric: {metric_name}")


# ---------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------

def extract_tuning_metric(metric_name, data):
    if data.columns.tolist() != TUNING_COLUMNS:
        raise ValueError(
            f"{TUNING_FILE} schema mismatch: {data.columns.tolist()}"
        )

    if metric_name == "row_count":
        return len(data)

    if data["search"].isna().any() or data["search"].duplicated().any():
        raise ValueError(f"{TUNING_FILE} contains missing or duplicate searches.")

    if metric_name == "required_search_rows_present":
        return indicator(data["search"].tolist() == EXPECTED_SEARCH_NAMES)

    expected = {
        "candidate_counts_correct": ("candidates", [27, 9, 1]),
        "cv_folds_correct": ("cv_folds", [5, 5, 0]),
        "total_fits_correct": ("total_fits", [135, 45, 1]),
    }

    if metric_name in expected:
        column, expected_values = expected[metric_name]
        require_finite_numeric(data, [column], TUNING_FILE)
        actual = pd.to_numeric(data[column], errors="coerce").astype(int).tolist()
        return indicator(actual == expected_values)

    if metric_name in {"grid_cv_accuracy", "randomized_cv_accuracy"}:
        search_name = (
            "grid_search"
            if metric_name == "grid_cv_accuracy"
            else "randomized_search"
        )
        row = require_unique_row(data, "search", search_name, TUNING_FILE)

        value = pd.to_numeric(
            pd.Series([row["best_cv_accuracy"]]), errors="coerce"
        ).iloc[0]

        if pd.isna(value) or not np.isfinite(value):
            raise ValueError(f"{search_name} best_cv_accuracy is missing or invalid.")

        return float(value)

    if metric_name == "tuning_validation_f1_minimum_met":
        require_finite_numeric(data, ["val_f1"], TUNING_FILE)
        values = pd.to_numeric(data["val_f1"], errors="coerce")
        return indicator(values.between(0.65, 1.0, inclusive="both").all())

    raise ValueError(f"Unsupported tuning metric: {metric_name}")


# ---------------------------------------------------------------------
# Thresholds and final configuration
# ---------------------------------------------------------------------

def validate_config(config):
    if len(config) != 1:
        raise ValueError(f"{CONFIG_FILE} must contain exactly one row.")

    if "threshold" not in config.columns:
        raise ValueError(f"{CONFIG_FILE} is missing the threshold column.")

    threshold = pd.to_numeric(config["threshold"], errors="coerce").iloc[0]

    if pd.isna(threshold) or not np.isfinite(threshold):
        raise ValueError(f"{CONFIG_FILE} threshold is missing or invalid.")

    return float(threshold)


def extract_threshold_metric(metric_name, thresholds, config):
    if thresholds.columns.tolist() != THRESHOLD_COLUMNS:
        raise ValueError(
            f"{THRESHOLD_FILE} schema mismatch: {thresholds.columns.tolist()}"
        )

    if metric_name == "row_count":
        return len(thresholds)

    if thresholds["threshold"].duplicated().any():
        raise ValueError(f"{THRESHOLD_FILE} contains duplicate thresholds.")

    require_finite_numeric(thresholds, THRESHOLD_COLUMNS, THRESHOLD_FILE)

    config_threshold = validate_config(config)

    if metric_name == "required_thresholds_present":
        actual = np.round(
            pd.to_numeric(thresholds["threshold"], errors="coerce").to_numpy(),
            1,
        )
        return indicator(np.array_equal(actual, EXPECTED_THRESHOLDS))

    if metric_name == "threshold_metrics_consistent":
        consistent, totals, _ = calculate_metrics_from_confusion(
            thresholds, THRESHOLD_FILE
        )
        return indicator(consistent == 1 and (totals == 180).all())

    if metric_name == "recall_nonincreasing_with_threshold":
        ordered = thresholds.sort_values("threshold")
        recall = pd.to_numeric(ordered["recall"], errors="coerce")
        return indicator(recall.diff().dropna().le(0.001).all())

    if metric_name == "selected_threshold":
        return config_threshold

    if metric_name == "threshold_rule_satisfied":
        qualifying = thresholds.loc[thresholds["recall"] >= 0.80]
        expected = (
            float(qualifying["threshold"].max())
            if not qualifying.empty
            else 0.1
        )
        return indicator(
            np.isclose(config_threshold, expected, atol=0.001, rtol=0)
        )

    raise ValueError(f"Unsupported threshold metric: {metric_name}")


# ---------------------------------------------------------------------
# Final test
# ---------------------------------------------------------------------

def extract_final_test_metric(metric_name, final_test, config):
    if final_test.columns.tolist() != THRESHOLD_COLUMNS:
        raise ValueError(
            f"{FINAL_TEST_FILE} schema mismatch: {final_test.columns.tolist()}"
        )

    if len(final_test) != 1:
        raise ValueError(f"{FINAL_TEST_FILE} must contain exactly one row.")

    config_threshold = validate_config(config)
    row = final_test.iloc[0]

    consistent, totals, delayed = calculate_metrics_from_confusion(
        final_test, FINAL_TEST_FILE
    )

    if metric_name == "row_count":
        return len(final_test)

    if metric_name == "test_row_count":
        return int(totals.iloc[0])

    if metric_name == "test_delayed_count":
        return int(delayed.iloc[0])

    if metric_name == "threshold_matches_config":
        return indicator(
            np.isclose(
                float(row["threshold"]),
                config_threshold,
                atol=0.001,
                rtol=0,
            )
        )

    if metric_name == "confusion_metrics_consistent":
        return consistent

    # These retain the project's established final-test acceptance rules.
    if metric_name == "test_precision_minimum_met":
        return indicator(float(row["precision"]) >= 0.45)

    if metric_name == "test_recall_minimum_met":
        return indicator(float(row["recall"]) >= 0.55)

    if metric_name == "test_f1_minimum_met":
        return indicator(float(row["f1"]) >= 0.60)

    raise ValueError(f"Unsupported final-test metric: {metric_name}")


# ---------------------------------------------------------------------
# Metric routing
# ---------------------------------------------------------------------

def extract_metric(reference_row, cache):
    source = reference_row["source_file"]
    name = reference_row["metric_name"]

    if source == MODELLING_FILE:
        return extract_split_metric(name, load_output(source, cache))

    if source == MODEL_FILE:
        return extract_models_metric(name, load_output(source, cache))

    if source == FEATURE_FILE:
        return extract_feature_metric(name, load_output(source, cache))

    if source == SELECTK_FILE:
        return extract_selectk_metric(name, load_output(source, cache))

    if source == TUNING_FILE:
        return extract_tuning_metric(name, load_output(source, cache))

    if source == THRESHOLD_FILE:
        return extract_threshold_metric(
            name,
            load_output(source, cache),
            load_output(CONFIG_FILE, cache),
        )

    if source == CONFIG_FILE:
        config = load_output(source, cache)
        threshold = validate_config(config)

        if name == "selected_threshold":
            return threshold

        if name == "threshold_rule_satisfied":
            return extract_threshold_metric(
                name,
                load_output(THRESHOLD_FILE, cache),
                config,
            )

        raise ValueError(f"Unsupported config metric: {name}")

    if source == FINAL_TEST_FILE:
        return extract_final_test_metric(
            name,
            load_output(source, cache),
            load_output(CONFIG_FILE, cache),
        )

    raise ValueError(f"Unsupported reference source_file: {source}")


# ---------------------------------------------------------------------
# Comparison and report
# ---------------------------------------------------------------------

def compare_part_c_metric(
    source_file,
    metric_name,
    reference_value,
    code_value,
    tolerance,
):
    """
    Apply the comparison semantics for Part C.

    Exact benchmarks use the shared comparator. Minimum and accuracy
    range-bound checks use their explicitly defined comparison operators.
    """
    try:
        expected = float(reference_value)
        actual = float(code_value)
        allowed = float(tolerance)
    except (TypeError, ValueError):
        return error_row(
            source_file, metric_name, "Reference, observed value, or tolerance is nonnumeric."
        )

    if (
        not np.isfinite(expected)
        or not np.isfinite(actual)
        or not np.isfinite(allowed)
        or allowed < 0
    ):
        return error_row(
            source_file,
            metric_name,
            "Reference, observed value, or tolerance is invalid or non-finite.",
        )

    if metric_name in MINIMUM_METRICS:
        passed = actual >= expected - allowed
        operator = ">="

    elif metric_name in LOWER_BOUND_METRICS:
        passed = actual >= expected - allowed
        operator = ">="

    elif metric_name in UPPER_BOUND_METRICS:
        passed = actual <= expected + allowed
        operator = "<="

    else:
        return compare_numeric(
            source_file,
            metric_name,
            expected,
            actual,
            allowed,
        )

    status = "PASS" if passed else "MISMATCH"

    from validations.validations_common_utility import make_report_row

    return make_report_row(
        source_file=source_file,
        metric_name=metric_name,
        reference_value=expected,
        code_value=actual,
        tolerance=allowed,
        status=status,
        comment=(
            f"{METRIC_DESCRIPTIONS.get(metric_name, metric_name)} "
            f"Observed value {actual:.6g}; required {operator} "
            f"{expected:.6g}. "
            f"{'Requirement met.' if passed else 'Requirement not met.'}"
        ),
    )


def validate_part_c():
    """Validate existing Part C outputs and write the standardized report."""
    reference, reference_errors = load_reference(REFERENCE_PATH)
    report_rows = list(reference_errors)
    cache = {}

    if reference.empty and not reference_errors:
        report_rows.append(
            error_row(
                str(REFERENCE_PATH),
                "__reference__",
                "Reference file contains no usable validation metrics.",
            )
        )

    for _, ref in reference.iterrows():
        source = ref["source_file"]
        name = ref["metric_name"]
        expected = ref["metric_value"]
        tolerance = ref["tolerance"]

        try:
            actual = extract_metric(ref, cache)

            row = compare_part_c_metric(
                source_file=source,
                metric_name=name,
                reference_value=expected,
                code_value=actual,
                tolerance=tolerance,
            )

        except Exception as exc:
            row = error_row(
                source,
                name,
                f"{METRIC_DESCRIPTIONS.get(name, name)} "
                f"Could not evaluate metric: {exc}",
            )
            row["reference_value"] = expected
            row["tolerance"] = tolerance

        report_rows.append(row)

    return write_report(report_rows, REPORT_PATH)


if __name__ == "__main__":
    sys.exit(validate_part_c())