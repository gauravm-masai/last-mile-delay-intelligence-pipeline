from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.shipment_prep import NOISE_COLUMNS
from validations.validations_common_utility import (
    compare_acceptance,
    compare_numeric,
    error_row,
    load_reference,
    run_dual_validation,
    write_report,
)


# ============================================================
# Paths and constants
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = ROOT / "references" / "part_c_reference_metrics.csv"
INTERNAL_REFERENCE_PATH = ROOT / "references" / "part_c_internal_metrics.csv"

REFERENCE_REPORT_PATH = (
    ROOT / "validations" / "reports" / "part_c_validation_report.csv"
)
INTERNAL_REPORT_PATH = (
    ROOT / "validations" / "reports" / "part_c_internal_validation_report.csv"
)

MODELLING_FILE = "data/modelling_table.csv"
MODEL_FILE = "outputs/part_c_models.csv"
FEATURE_FILE = "outputs/part_c_feature_scores.csv"
SELECTK_FILE = "outputs/part_c_selectk.csv"
TUNING_FILE = "outputs/part_c_tuning.csv"
THRESHOLD_FILE = "outputs/part_c_thresholds.csv"
CONFIG_FILE = "outputs/part_c_final_config.csv"
FINAL_TEST_FILE = "outputs/part_c_final_test.csv"

REPORT_SUFFIXES = {
    "REFERENCE": "_reference",
    "ACCEPTANCE": "_acceptance",
}

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

MODEL_COLUMNS = [
    "model",
    "TN",
    "FP",
    "FN",
    "TP",
    "accuracy",
    "precision",
    "recall",
    "f1",
]

FEATURE_COLUMNS = [
    "feature",
    "f_score",
    "f_rank",
    "rf_importance",
    "is_noise",
]

SELECTK_COLUMNS = ["k", "val_f1"]

TUNING_COLUMNS = [
    "search",
    "candidates",
    "cv_folds",
    "total_fits",
    "best_params",
    "best_cv_accuracy",
    "val_f1",
]

THRESHOLD_COLUMNS = [
    "threshold",
    "TN",
    "FP",
    "FN",
    "TP",
    "accuracy",
    "precision",
    "recall",
    "f1",
]

CONFIG_COLUMNS = [
    "selector_k",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "threshold",
    "chosen_from",
]

FINAL_TEST_COLUMNS = THRESHOLD_COLUMNS.copy()

EXPECTED_TOP_SEVEN = [
    "distance_km",
    "hub_tier",
    "hub_daily_capacity",
    "n_stops",
    "promised_hours",
    "rain_flag",
    "sale_event_flag",
]

EXPECTED_MODELS = [
    "blind_all_on_time",
    "logistic_regression",
    "decision_tree_depth_6",
    "random_forest_default",
]

EXPECTED_SEARCHES = [
    "grid_search",
    "randomized_search",
    "default_forest_same_7_features",
]

EXPECTED_K_VALUES = [5, 7, 10, 12, 17]
EXPECTED_THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 1)

# This is the established minimum acceptance threshold for tuning F1.
MIN_TUNING_F1 = 0.65

# The PDF requires trained validation accuracies to lie in this interval.
TRAINED_ACCURACY_MIN = 0.883
TRAINED_ACCURACY_MAX = 0.911

# Strict boundary: noise-column F-scores must be LESS THAN 10.
NOISE_F_SCORE_STRICT_MAX = 10.0


# ============================================================
# Explicit acceptance rules
# ============================================================

# All acceptance operators are explicitly configured.
# They are never inferred from metric names or tolerances.
#
# Supported:
# exact, minimum, maximum, strict_maximum, range
#
# For minimum/maximum rules, the threshold comes from the
# corresponding ACCEPTANCE row's metric_value.

ACCEPTANCE_RULES = {
    # Counts and structural checks
    "row_count": {"operator": "exact"},
    "train_row_count": {"operator": "exact"},
    "val_row_count": {"operator": "exact"},
    "test_row_count": {"operator": "exact"},
    "delayed_train_count": {"operator": "exact"},
    "delayed_val_count": {"operator": "exact"},
    "delayed_test_count": {"operator": "exact"},

    "every_model_accuracy_matches_confusion_matrix": {
        "operator": "exact"
    },
    "blind_tn_val": {"operator": "exact"},
    "blind_fp_val": {"operator": "exact"},
    "blind_fn_val": {"operator": "exact"},
    "blind_tp_val": {"operator": "exact"},

    "best_tree_depth_reference": {"operator": "exact"},

    # Feature-signal checks
    "feature_names_unique": {"operator": "exact"},
    "feature_scores_finite": {"operator": "exact"},
    "noise_columns_identified": {"operator": "exact"},
    "top_seven_f_score_order_correct": {"operator": "exact"},

    "selected_feature_distance_km_position": {"operator": "exact"},
    "selected_feature_hub_tier_position": {"operator": "exact"},
    "selected_feature_hub_daily_capacity_position": {"operator": "exact"},
    "selected_feature_n_stops_position": {"operator": "exact"},
    "selected_feature_promised_hours_position": {"operator": "exact"},
    "selected_feature_rain_flag_position": {"operator": "exact"},
    "selected_feature_sale_event_flag_position": {"operator": "exact"},

    "sale_event_flag_f_score_minimum": {"operator": "minimum"},
    "noise_columns_outside_top_seven": {"operator": "exact"},
    "every_noise_column_f_rank_minimum": {"operator": "minimum"},
    "every_noise_column_f_score_strict_maximum": {
        "operator": "strict_maximum"
    },
    "forest_importance_sums_to_one": {"operator": "exact"},

    "combined_noise_importance_minimum": {"operator": "minimum"},
    "combined_noise_importance_maximum": {"operator": "maximum"},

    # SelectK checks
    "required_k_values_present": {"operator": "exact"},
    "validation_f1_values_valid": {"operator": "exact"},

    # Tuning checks
    "required_search_rows_present": {"operator": "exact"},
    "grid_best_cv_accuracy_at_least_randomized": {"operator": "exact"},

    "grid_search_candidates": {"operator": "exact"},
    "grid_search_cv_folds": {"operator": "exact"},
    "grid_search_total_fits": {"operator": "exact"},

    "randomized_search_candidates": {"operator": "exact"},
    "randomized_search_cv_folds": {"operator": "exact"},
    "randomized_search_total_fits": {"operator": "exact"},

    "default_forest_same_7_features_candidates": {"operator": "exact"},
    "default_forest_same_7_features_cv_folds": {"operator": "exact"},
    "default_forest_same_7_features_total_fits": {"operator": "exact"},

    "logistic_regression_f1_val_minimum": {"operator": "minimum"},
    "decision_tree_f1_val_minimum": {"operator": "minimum"},
    "default_rf_f1_val_minimum": {"operator": "minimum"},

    "grid_search_f1_val_minimum": {"operator": "minimum"},
    "randomized_search_f1_val_minimum": {"operator": "minimum"},
    "default_forest_same_7_features_f1_val_minimum": {
        "operator": "minimum"
    },

    "grid_search_best_cv_accuracy_minimum": {"operator": "minimum"},
    "grid_search_best_cv_accuracy_maximum": {"operator": "maximum"},
    "randomized_search_best_cv_accuracy_minimum": {"operator": "minimum"},
    "randomized_search_best_cv_accuracy_maximum": {"operator": "maximum"},

    # Model validation accuracy checks
    "logistic_regression_accuracy_val_minimum": {"operator": "minimum"},
    "logistic_regression_accuracy_val_maximum": {"operator": "maximum"},
    "decision_tree_accuracy_val_minimum": {"operator": "minimum"},
    "decision_tree_accuracy_val_maximum": {"operator": "maximum"},
    "default_rf_accuracy_val_minimum": {"operator": "minimum"},
    "default_rf_accuracy_val_maximum": {"operator": "maximum"},

    # Threshold and final configuration checks
    "required_thresholds_present": {"operator": "exact"},
    "threshold_metrics_consistent": {"operator": "exact"},
    "recall_nonincreasing_with_threshold": {"operator": "exact"},
    "selected_threshold": {"operator": "exact"},
    "threshold_rule_satisfied": {"operator": "exact"},
    "threshold_matches_config": {"operator": "exact"},

    # Final-test checks
    "test_precision_minimum": {"operator": "minimum"},
    "test_recall_minimum": {"operator": "minimum"},
    "test_f1_minimum": {"operator": "minimum"},
    "test_row_count": {"operator": "exact"},
    "test_delayed_count": {"operator": "exact"},
    "confusion_metrics_consistent": {"operator": "exact"},
}


# ============================================================
# Human-readable descriptions
# ============================================================

METRIC_DESCRIPTIONS = {
    "row_count": "Number of rows in the specified source or output.",
    "train_row_count": "Number of rows in the training split.",
    "val_row_count": "Number of rows in the validation split.",
    "test_row_count": "Number of rows in the test split.",
    "delayed_train_count": "Delayed parcels in the training split.",
    "delayed_val_count": "Delayed parcels in the validation split.",
    "delayed_test_count": "Delayed parcels in the test split.",

    "every_model_accuracy_matches_confusion_matrix":
        "Stored model accuracies are consistent with confusion counts.",
    "blind_accuracy_val_reference":
        "Validation accuracy of the all-on-time baseline.",
    "blind_tn_val": "True negatives for the all-on-time baseline.",
    "blind_fp_val": "False positives for the all-on-time baseline.",
    "blind_fn_val": "False negatives for the all-on-time baseline.",
    "blind_tp_val": "True positives for the all-on-time baseline.",

    "logistic_regression_f1_val_reference":
        "Logistic-regression validation F1 benchmark.",
    "decision_tree_f1_val_reference":
        "Decision-tree validation F1 benchmark.",
    "default_rf_f1_val_reference":
        "Default Random Forest validation F1 benchmark.",
    "logistic_regression_f1_val_minimum":
        "Logistic-regression validation F1 minimum acceptance.",
    "decision_tree_f1_val_minimum":
        "Decision-tree validation F1 minimum acceptance.",
    "default_rf_f1_val_minimum":
        "Default Random Forest validation F1 minimum acceptance.",

    "best_tree_depth_reference":
        "Required selected decision-tree depth.",
    "feature_names_unique": "Feature names are unique.",
    "feature_scores_finite": "Feature scores, ranks, and importances are finite.",
    "noise_columns_identified": "Noise columns match NOISE_COLUMNS.",
    "top_seven_f_score_order_correct":
        "Top seven features match the expected order.",
    "sale_event_flag_f_score_minimum":
        "F-score for sale_event_flag meets the minimum.",
    "noise_columns_outside_top_seven":
        "Noise columns are outside the top seven.",
    "every_noise_column_f_rank_minimum":
        "Lowest noise-column F-score rank meets the minimum.",
    "every_noise_column_f_score_strict_maximum":
        "Every noise-column F-score is strictly below the maximum.",
    "combined_noise_importance_reference":
        "Combined Random Forest importance of noise columns.",
    "combined_noise_importance_minimum":
        "Combined noise importance meets the lower bound.",
    "combined_noise_importance_maximum":
        "Combined noise importance meets the upper bound.",

    "required_k_values_present":
        "SelectK experiments contain k = 5, 7, 10, 12, 17.",
    "validation_f1_values_valid":
        "SelectK validation F1 values are within [0, 1].",

    "required_search_rows_present":
        "Required tuning experiments are present in the expected order.",
    "grid_best_cv_accuracy_at_least_randomized":
        "Grid-search CV accuracy is at least randomized-search CV accuracy.",
    "grid_search_best_cv_accuracy_reference":
        "Grid-search best cross-validation accuracy benchmark.",
    "randomized_search_best_cv_accuracy_reference":
        "Randomized-search best cross-validation accuracy benchmark.",
    "grid_search_f1_val_reference":
        "Grid-search validation F1 benchmark.",
    "randomized_search_f1_val_reference":
        "Randomized-search validation F1 benchmark.",
    "default_forest_same_7_features_f1_val_reference":
        "Seven-feature default forest validation F1 benchmark.",

    "required_thresholds_present":
        "Threshold table contains thresholds 0.1 through 0.9.",
    "threshold_metrics_consistent":
        "Threshold metrics agree with confusion-matrix counts.",
    "recall_nonincreasing_with_threshold":
        "Recall does not increase as the threshold rises.",
    "selected_threshold":
        "Selected alert threshold.",
    "threshold_rule_satisfied":
        "Threshold follows the highest-threshold-with-recall rule.",
    "threshold_matches_config":
        "Final-test threshold matches the saved configuration.",

    "test_precision_reference": "Final-test precision benchmark.",
    "test_recall_reference": "Final-test recall benchmark.",
    "test_f1_reference": "Final-test F1 benchmark.",
    "test_precision_minimum": "Final-test precision acceptance.",
    "test_recall_minimum": "Final-test recall acceptance.",
    "test_f1_minimum": "Final-test F1 acceptance.",
    "test_delayed_count": "Delayed parcels represented in the test confusion matrix.",
}


# ============================================================
# CSV loading and validation helpers
# ============================================================

def read_csv_checked(relative_path, expected_columns=None):
    """Read an existing CSV and validate its column schema."""
    path = ROOT / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {relative_path}")

    try:
        data = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {relative_path}: {exc}") from exc

    if expected_columns is not None and data.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected columns in {relative_path}. "
            f"Expected {expected_columns}; found {data.columns.tolist()}."
        )

    return data


def require_finite_numeric(data, columns, source_file):
    """Reject missing, nonnumeric, and non-finite values."""
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{source_file} is missing columns: {missing}")

    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"{source_file}.{column} contains missing/nonnumeric values."
            )

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{source_file}.{column} contains non-finite values."
            )


def require_unique_row(data, key_column, key_value, source_file):
    """Return exactly one row matching the specified key."""
    if key_column not in data.columns:
        raise ValueError(f"{source_file} is missing column {key_column!r}.")

    matches = data.loc[data[key_column] == key_value]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {key_column}={key_value!r} row in "
            f"{source_file}; found {len(matches)}."
        )

    return matches.iloc[0]


def indicator(condition):
    """Return a numeric Boolean indicator for an acceptance check."""
    return int(bool(condition))


def parse_boolean_column(series, column_name):
    """Parse true/false and 1/0 representations from CSV."""
    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if values.isna().any():
        raise ValueError(
            f"{column_name} must contain true/false or 1/0 values."
        )

    return values.astype(bool)


# ============================================================
# Confusion-matrix consistency
# ============================================================

def calculate_confusion_consistency(data, source_file):
    """Validate stored accuracy, precision, recall, and F1."""
    count_columns = ["TN", "FP", "FN", "TP"]
    metric_columns = ["accuracy", "precision", "recall", "f1"]

    require_finite_numeric(
        data, count_columns + metric_columns, source_file
    )

    counts = data[count_columns].apply(pd.to_numeric, errors="coerce")

    if (counts < 0).any().any():
        raise ValueError(f"{source_file} contains negative confusion counts.")

    count_array = counts.to_numpy(dtype=float)

    if not np.equal(count_array, np.floor(count_array)).all():
        raise ValueError(f"{source_file} contains non-integer confusion counts.")

    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

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

    total_rows = total.astype(int)
    delayed_rows = (fn + tp).astype(int)

    return indicator(consistent), total_rows, delayed_rows


# ============================================================
# Split diagnostics
# ============================================================

def extract_split_metric(metric_name):
    """
    Rebuild the prescribed stratified split for split diagnostics only.

    This does not fit or retrain a model and does not generate outputs.
    """
    data = read_csv_checked(MODELLING_FILE)

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
        "train_row_count": len(y_train),
        "val_row_count": len(y_val),
        "test_row_count": len(y_test),
        "delayed_train_count": int(y_train.sum()),
        "delayed_val_count": int(y_val.sum()),
        "delayed_test_count": int(y_test.sum()),
    }

    if metric_name not in values:
        raise ValueError(f"Unsupported split metric: {metric_name}")

    return values[metric_name]


# ============================================================
# Model results
# ============================================================

def extract_model_metric(metric_name):
    data = read_csv_checked(MODEL_FILE, MODEL_COLUMNS)

    if metric_name == "row_count":
        return len(data)

    if data["model"].isna().any() or data["model"].duplicated().any():
        raise ValueError(f"{MODEL_FILE} contains missing/duplicate model names.")

    require_finite_numeric(
        data,
        ["TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
        MODEL_FILE,
    )

    if metric_name in {
        "every_model_accuracy_matches_confusion_matrix",
        "confusion_metrics_consistent",
    }:
        return calculate_confusion_consistency(data, MODEL_FILE)[0]

    model_names = {
        "blind_accuracy_val_reference": "blind_all_on_time",
        "blind_tn_val": "blind_all_on_time",
        "blind_fp_val": "blind_all_on_time",
        "blind_fn_val": "blind_all_on_time",
        "blind_tp_val": "blind_all_on_time",
        "logistic_regression_f1_val_reference": "logistic_regression",
        "logistic_regression_f1_val_minimum": "logistic_regression",
        "logistic_regression_accuracy_val_minimum": "logistic_regression",
        "logistic_regression_accuracy_val_maximum": "logistic_regression",
        "decision_tree_f1_val_reference": "decision_tree_depth_6",
        "decision_tree_f1_val_minimum": "decision_tree_depth_6",
        "decision_tree_accuracy_val_minimum": "decision_tree_depth_6",
        "decision_tree_accuracy_val_maximum": "decision_tree_depth_6",
        "default_rf_f1_val_reference": "random_forest_default",
        "default_rf_f1_val_minimum": "random_forest_default",
        "default_rf_accuracy_val_minimum": "random_forest_default",
        "default_rf_accuracy_val_maximum": "random_forest_default",
    }

    if metric_name == "best_tree_depth_reference":
        # The saved model table contains the selected tree only.
        # It does not contain F1 scores for all candidate depths.
        tree_rows = data.loc[
            data["model"].astype(str).str.fullmatch(
                r"decision_tree_depth_\d+"
            )
        ]

        if len(tree_rows) != 1:
            raise ValueError(
                "Cannot verify selected tree depth: expected exactly one "
                "saved decision_tree_depth_<d> model row."
            )

        selected_name = str(tree_rows.iloc[0]["model"])
        return int(selected_name.rsplit("_", 1)[1])

    if metric_name not in model_names:
        raise ValueError(f"Unsupported model metric: {metric_name}")

    model_name = model_names[metric_name]
    row = require_unique_row(data, "model", model_name, MODEL_FILE)

    if metric_name == "blind_accuracy_val_reference":
        return float(row["accuracy"])

    if metric_name in {
        "blind_tn_val",
        "blind_fp_val",
        "blind_fn_val",
        "blind_tp_val",
    }:
        column = {
            "blind_tn_val": "TN",
            "blind_fp_val": "FP",
            "blind_fn_val": "FN",
            "blind_tp_val": "TP",
        }[metric_name]
        return int(row[column])

    if "f1" in metric_name:
        return float(row["f1"])

    if "accuracy" in metric_name:
        return float(row["accuracy"])

    raise ValueError(f"Unsupported model metric: {metric_name}")


# ============================================================
# Feature-signal results
# ============================================================

def extract_feature_metric(metric_name):
    data = read_csv_checked(FEATURE_FILE, FEATURE_COLUMNS)

    if metric_name == "row_count":
        return len(data)

    if data["feature"].isna().any() or data["feature"].duplicated().any():
        raise ValueError(
            f"{FEATURE_FILE} contains missing/duplicate feature names."
        )

    require_finite_numeric(
        data,
        ["f_score", "f_rank", "rf_importance"],
        FEATURE_FILE,
    )

    data["is_noise"] = parse_boolean_column(data["is_noise"], "is_noise")

    if metric_name == "feature_names_unique":
        return indicator(data["feature"].is_unique)

    if metric_name == "feature_scores_finite":
        values = data[
            ["f_score", "f_rank", "rf_importance"]
        ].to_numpy(dtype=float)
        return indicator(np.isfinite(values).all())

    if metric_name == "noise_columns_identified":
        actual_noise = set(data.loc[data["is_noise"], "feature"])
        return indicator(actual_noise == set(NOISE_COLUMNS))

    ranked = data.sort_values(
        ["f_score", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if metric_name == "top_seven_f_score_order_correct":
        return indicator(
            ranked["feature"].head(7).tolist() == EXPECTED_TOP_SEVEN
        )

    if metric_name.startswith("selected_feature_") and metric_name.endswith(
        "_position"
    ):
        feature_name = metric_name[
            len("selected_feature_"):-len("_position")
        ]

        if feature_name not in ranked["feature"].tolist():
            raise ValueError(
                f"Feature {feature_name!r} is missing from {FEATURE_FILE}."
            )

        # Reference CSV positions are one-based.
        return ranked.index[ranked["feature"] == feature_name][0] + 1

    if metric_name == "sale_event_flag_f_score_minimum":
        row = require_unique_row(
            data, "feature", "sale_event_flag", FEATURE_FILE
        )
        return float(row["f_score"])

    if metric_name == "noise_columns_outside_top_seven":
        noise = data.loc[data["is_noise"]]
        top_seven = set(ranked["feature"].head(7))

        return indicator(
            not noise.empty
            and (noise["f_rank"] >= 8).all()
            and not set(noise["feature"]).intersection(top_seven)
        )

    if metric_name == "every_noise_column_f_rank_minimum":
        noise = data.loc[data["is_noise"]]

        if noise.empty:
            raise ValueError("No noise columns were identified.")

        return int(noise["f_rank"].min())

    if metric_name == "every_noise_column_f_score_strict_maximum":
        noise = data.loc[data["is_noise"]]

        if noise.empty:
            raise ValueError("No noise columns were identified.")

        return float(noise["f_score"].max())

    if metric_name == "combined_noise_importance_reference":
        return float(data.loc[data["is_noise"], "rf_importance"].sum())

    if metric_name in {
        "combined_noise_importance_minimum",
        "combined_noise_importance_maximum",
    }:
        return float(data.loc[data["is_noise"], "rf_importance"].sum())

    if metric_name == "forest_importance_sums_to_one":
        return indicator(
            np.isclose(
                data["rf_importance"].sum(),
                1.0,
                atol=0.001,
                rtol=0,
            )
        )

    raise ValueError(f"Unsupported feature metric: {metric_name}")


# ============================================================
# SelectK experiments
# ============================================================

def extract_selectk_metric(metric_name):
    data = read_csv_checked(SELECTK_FILE, SELECTK_COLUMNS)

    if metric_name == "row_count":
        return len(data)

    require_finite_numeric(data, SELECTK_COLUMNS, SELECTK_FILE)

    if data["k"].duplicated().any():
        raise ValueError(f"{SELECTK_FILE} contains duplicate k values.")

    if metric_name == "required_k_values_present":
        actual = pd.to_numeric(data["k"], errors="coerce").astype(int).tolist()
        return indicator(actual == EXPECTED_K_VALUES)

    if metric_name == "validation_f1_values_valid":
        f1 = pd.to_numeric(data["val_f1"], errors="coerce")
        return indicator(f1.between(0, 1, inclusive="both").all())

    raise ValueError(f"Unsupported SelectK metric: {metric_name}")


# ============================================================
# Tuning results
# ============================================================

def extract_tuning_metric(metric_name):
    """
    Extract and validate one metric from the existing Part C tuning output.

    Grid Search and Randomized Search must have finite CV accuracy values.
    The default forest trained on the selected seven features does not run
    cross-validation, so its best_cv_accuracy may legitimately be NaN.

    This function reads existing output only. It does not retrain models
    or regenerate outputs.
    """
    data = read_csv_checked(TUNING_FILE, TUNING_COLUMNS)

    if metric_name == "row_count":
        return len(data)

    if data["search"].isna().any():
        raise ValueError(f"{TUNING_FILE} contains missing search names.")

    if data["search"].duplicated().any():
        raise ValueError(f"{TUNING_FILE} contains duplicate search names.")

    # Validate exact experiment membership and order.
    if metric_name == "required_search_rows_present":
        return indicator(data["search"].tolist() == EXPECTED_SEARCHES)

    # All tuning metrics are extracted by named experiment, not by assuming
    # that a whole column is finite. This is important for best_cv_accuracy:
    # the seven-feature forest has cv_folds=0 and no CV accuracy.
    metric_specs = {
        "grid_search_candidates": ("grid_search", "candidates", "integer"),
        "grid_search_cv_folds": ("grid_search", "cv_folds", "integer"),
        "grid_search_total_fits": ("grid_search", "total_fits", "integer"),
        "grid_search_best_cv_accuracy_reference": (
            "grid_search", "best_cv_accuracy", "probability"
        ),
        "grid_search_best_cv_accuracy_minimum": (
            "grid_search", "best_cv_accuracy", "probability"
        ),
        "grid_search_best_cv_accuracy_maximum": (
            "grid_search", "best_cv_accuracy", "probability"
        ),
        "grid_search_f1_val_reference": ("grid_search", "val_f1", "probability"),
        "grid_search_f1_val_minimum": ("grid_search", "val_f1", "probability"),

        "randomized_search_candidates": (
            "randomized_search", "candidates", "integer"
        ),
        "randomized_search_cv_folds": (
            "randomized_search", "cv_folds", "integer"
        ),
        "randomized_search_total_fits": (
            "randomized_search", "total_fits", "integer"
        ),
        "randomized_search_best_cv_accuracy_reference": (
            "randomized_search", "best_cv_accuracy", "probability"
        ),
        "randomized_search_best_cv_accuracy_minimum": (
            "randomized_search", "best_cv_accuracy", "probability"
        ),
        "randomized_search_best_cv_accuracy_maximum": (
            "randomized_search", "best_cv_accuracy", "probability"
        ),
        "randomized_search_f1_val_reference": (
            "randomized_search", "val_f1", "probability"
        ),
        "randomized_search_f1_val_minimum": (
            "randomized_search", "val_f1", "probability"
        ),

        "default_forest_same_7_features_candidates": (
            "default_forest_same_7_features", "candidates", "integer"
        ),
        "default_forest_same_7_features_cv_folds": (
            "default_forest_same_7_features", "cv_folds", "integer"
        ),
        "default_forest_same_7_features_total_fits": (
            "default_forest_same_7_features", "total_fits", "integer"
        ),
        "default_forest_same_7_features_f1_val_reference": (
            "default_forest_same_7_features", "val_f1", "probability"
        ),
        "default_forest_same_7_features_f1_val_minimum": (
            "default_forest_same_7_features", "val_f1", "probability"
        ),
    }

    if metric_name == "grid_best_cv_accuracy_at_least_randomized":
        grid_row = require_unique_row(data, "search", "grid_search", TUNING_FILE)
        randomized_row = require_unique_row(
            data, "search", "randomized_search", TUNING_FILE
        )
        grid_value = pd.to_numeric(
            pd.Series([grid_row["best_cv_accuracy"]]), errors="coerce"
        ).iloc[0]
        randomized_value = pd.to_numeric(
            pd.Series([randomized_row["best_cv_accuracy"]]), errors="coerce"
        ).iloc[0]

        if (
            pd.isna(grid_value)
            or pd.isna(randomized_value)
            or not np.isfinite(float(grid_value))
            or not np.isfinite(float(randomized_value))
        ):
            raise ValueError(
                f"{TUNING_FILE}: Grid Search and Randomized Search "
                "best_cv_accuracy must both be finite."
            )

        return indicator(float(grid_value) >= float(randomized_value))

    if metric_name not in metric_specs:
        raise ValueError(f"Unsupported tuning metric: {metric_name}")

    search_name, column, value_kind = metric_specs[metric_name]
    row = require_unique_row(data, "search", search_name, TUNING_FILE)

    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]

    if pd.isna(value) or not np.isfinite(float(value)):
        raise ValueError(
            f"{TUNING_FILE}: {search_name} {column} is missing, "
            "nonnumeric, or non-finite."
        )

    value = float(value)

    if value_kind == "integer":
        if not value.is_integer():
            raise ValueError(
                f"{TUNING_FILE}: {search_name} {column} must be an integer."
            )
        return int(value)

    if value_kind == "probability" and not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{TUNING_FILE}: {search_name} {column} must be within [0, 1]."
        )

    return value


# ============================================================
# Threshold analysis
# ============================================================

def extract_threshold_metric(metric_name):
    data = read_csv_checked(THRESHOLD_FILE, THRESHOLD_COLUMNS)

    if metric_name == "row_count":
        return len(data)

    require_finite_numeric(data, THRESHOLD_COLUMNS, THRESHOLD_FILE)

    thresholds = pd.to_numeric(data["threshold"], errors="coerce").to_numpy()

    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError(f"{THRESHOLD_FILE} contains duplicate thresholds.")

    if metric_name == "required_thresholds_present":
        return indicator(
            np.array_equal(
                np.sort(thresholds),
                EXPECTED_THRESHOLDS,
            )
        )

    if metric_name == "threshold_metrics_consistent":
        return calculate_confusion_consistency(data, THRESHOLD_FILE)[0]

    if metric_name == "recall_nonincreasing_with_threshold":
        ordered = data.sort_values("threshold")
        recall = pd.to_numeric(ordered["recall"], errors="coerce").to_numpy()

        # Allow only tiny floating-point noise.
        return indicator(np.all(np.diff(recall) <= 0.001))

    raise ValueError(f"Unsupported threshold metric: {metric_name}")


# ============================================================
# Final configuration
# ============================================================

def extract_config_metric(metric_name):
    data = read_csv_checked(CONFIG_FILE, CONFIG_COLUMNS)

    if len(data) != 1:
        raise ValueError(f"{CONFIG_FILE} must contain exactly one row.")

    row = data.iloc[0]

    if metric_name == "selected_threshold":
        value = pd.to_numeric(
            pd.Series([row["threshold"]]), errors="coerce"
        ).iloc[0]

        if pd.isna(value) or not np.isfinite(value):
            raise ValueError("Final configuration threshold is missing/invalid.")

        return float(value)

    if metric_name == "threshold_rule_satisfied":
        thresholds = read_csv_checked(THRESHOLD_FILE, THRESHOLD_COLUMNS)
        require_finite_numeric(
            thresholds, ["threshold", "recall"], THRESHOLD_FILE
        )

        qualifying = thresholds.loc[
            thresholds["recall"] >= 0.80, "threshold"
        ]

        expected = float(qualifying.max()) if not qualifying.empty else 0.1
        actual = float(row["threshold"])

        return indicator(
            np.isclose(actual, expected, atol=0, rtol=0)
        )

    raise ValueError(f"Unsupported configuration metric: {metric_name}")


# ============================================================
# Final-test results
# ============================================================

def extract_final_test_metric(metric_name):
    data = read_csv_checked(FINAL_TEST_FILE, FINAL_TEST_COLUMNS)

    if len(data) != 1:
        raise ValueError(f"{FINAL_TEST_FILE} must contain exactly one row.")

    row = data.iloc[0]

    consistency, test_rows, delayed_rows = calculate_confusion_consistency(
        data, FINAL_TEST_FILE
    )

    if metric_name == "row_count":
        return len(data)

    if metric_name == "test_row_count":
        return int(test_rows.iloc[0])

    if metric_name == "test_delayed_count":
        return int(delayed_rows.iloc[0])

    if metric_name == "threshold_matches_config":
        config = read_csv_checked(CONFIG_FILE, CONFIG_COLUMNS)

        if len(config) != 1:
            raise ValueError(f"{CONFIG_FILE} must contain exactly one row.")

        return indicator(
            np.isclose(
                float(row["threshold"]),
                float(config.iloc[0]["threshold"]),
                atol=0,
                rtol=0,
            )
        )

    if metric_name == "confusion_metrics_consistent":
        return consistency

    if metric_name in {
        "selected_threshold",
        "test_precision_reference",
        "test_precision_minimum",
        "test_recall_reference",
        "test_recall_minimum",
        "test_f1_reference",
        "test_f1_minimum",
    }:
        column = {
            "selected_threshold": "threshold",
            "test_precision_reference": "precision",
            "test_precision_minimum": "precision",
            "test_recall_reference": "recall",
            "test_recall_minimum": "recall",
            "test_f1_reference": "f1",
            "test_f1_minimum": "f1",
        }[metric_name]

        return float(row[column])

    raise ValueError(f"Unsupported final-test metric: {metric_name}")


# ============================================================
# Metric dispatch
# ============================================================

def extract_code_metric(source_file, metric_name):
    """Extract metrics from existing files only."""
    dispatch = {
        MODELLING_FILE: extract_split_metric,
        MODEL_FILE: extract_model_metric,
        FEATURE_FILE: extract_feature_metric,
        SELECTK_FILE: extract_selectk_metric,
        TUNING_FILE: extract_tuning_metric,
        THRESHOLD_FILE: extract_threshold_metric,
        CONFIG_FILE: extract_config_metric,
        FINAL_TEST_FILE: extract_final_test_metric,
    }

    if source_file not in dispatch:
        raise ValueError(
            f"Unsupported source_file in reference CSV: {source_file}"
        )

    return dispatch[source_file](metric_name)


# ============================================================
# Reference and acceptance validation
# ============================================================

def validate_part_c(
    reference_path=REFERENCE_PATH,
    report_path=REFERENCE_REPORT_PATH,
):
    """
    Validate Part C references and acceptance rules against existing outputs.

    REFERENCE mismatch -> MISMATCH (informational).
    Failed mandatory acceptance -> ERROR.
    Missing/uncomputable metrics -> ERROR.
    """
    report_rows = []

    references, reference_errors = load_reference(reference_path)
    report_rows.extend(reference_errors)

    if references.empty:
        if not reference_errors:
            report_rows.append(
                error_row(
                    str(reference_path),
                    "__reference__",
                    "Reference file contains no valid metrics.",
                )
            )
        return write_report(report_rows, report_path)

    if references.columns.tolist() != REFERENCE_COLUMNS:
        report_rows.append(
            error_row(
                str(reference_path),
                "__reference_schema__",
                f"Expected columns {REFERENCE_COLUMNS}; "
                f"found {references.columns.tolist()}.",
            )
        )
        return write_report(report_rows, report_path)

    valid_types = {"REFERENCE", "ACCEPTANCE"}

    invalid_types = ~references["value_type"].isin(valid_types)

    for _, row in references.loc[invalid_types].iterrows():
        report_rows.append(
            error_row(
                row["source_file"],
                row["metric_name"],
                f"Invalid value_type {row['value_type']!r}; "
                "expected REFERENCE or ACCEPTANCE.",
            )
        )

    duplicates = references.duplicated(
        ["source_file", "metric_name", "value_type"],
        keep=False,
    )

    for _, row in references.loc[duplicates].iterrows():
        report_rows.append(
            error_row(
                row["source_file"],
                row["metric_name"],
                "Duplicate source_file/metric_name/value_type reference row.",
            )
        )

    # Do not evaluate duplicate rows: they have already generated ERROR.
    references = references.loc[~duplicates & ~invalid_types].copy()

    for _, reference in references.iterrows():
        source_file = str(reference["source_file"]).strip()
        metric_name = str(reference["metric_name"]).strip()
        value_type = str(reference["value_type"]).strip()

        report_metric_name = (
            metric_name + REPORT_SUFFIXES[value_type]
        )

        try:
            code_value = extract_code_metric(source_file, metric_name)
        except Exception as exc:
            report_rows.append(
                error_row(
                    source_file,
                    report_metric_name,
                    f"Could not evaluate metric: {exc}",
                )
            )
            continue

        try:
            reference_value = float(reference["metric_value"])
            tolerance = float(reference["tolerance"])
        except (TypeError, ValueError, OverflowError):
            report_rows.append(
                error_row(
                    source_file,
                    report_metric_name,
                    "Reference metric_value or tolerance is nonnumeric.",
                )
            )
            continue

        if (
            not np.isfinite(reference_value)
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            report_rows.append(
                error_row(
                    source_file,
                    report_metric_name,
                    "Reference value must be finite and tolerance "
                    "must be finite and nonnegative.",
                )
            )
            continue

        description = METRIC_DESCRIPTIONS.get(
            metric_name,
            f"Validation of {metric_name}.",
        )

        if value_type == "REFERENCE":
            result = compare_numeric(
                source_file=source_file,
                metric_name=report_metric_name,
                reference_value=reference_value,
                code_value=code_value,
                tolerance=tolerance,
            )

            result["comment"] = (
                f"{description} {result.get('comment', '')}"
            )
            report_rows.append(result)
            continue

        rule = ACCEPTANCE_RULES.get(metric_name)

        if rule is None:
            report_rows.append(
                error_row(
                    source_file,
                    report_metric_name,
                    "No explicit acceptance rule is configured for this metric.",
                )
            )
            continue

        operator = rule["operator"]

        # Acceptance thresholds come from the ACCEPTANCE row's metric_value.
        threshold = reference_value

        # Enforce strict maximum separately. The common utility's maximum
        # operator is inclusive, while this requirement is strictly < 10.
        if operator == "strict_maximum":
            try:
                actual = float(code_value)
            except (TypeError, ValueError, OverflowError):
                report_rows.append(
                    error_row(
                        source_file,
                        report_metric_name,
                        "Strict maximum check requires a numeric code value.",
                    )
                )
                continue

            passed = np.isfinite(actual) and actual < threshold

            result = {
                "source_file": source_file,
                "metric_name": report_metric_name,
                "reference_value": threshold,
                "code_value": actual,
                "tolerance": 0.0,
                "status": "PASS" if passed else "ERROR",
                "comment": (
                    f"{description} "
                    f"Strict acceptance requires actual < {threshold}; "
                    f"actual={actual}."
                ),
            }
            report_rows.append(result)
            continue

        # Exact numeric comparisons must use exact equality, not tolerance.
        acceptance_code_value = code_value

        if operator == "exact":
            try:
                acceptance_code_value = float(code_value)
                threshold = float(threshold)
            except (TypeError, ValueError, OverflowError):
                report_rows.append(
                    error_row(
                        source_file,
                        report_metric_name,
                        "Exact acceptance requires numeric code and expected values.",
                    )
                )
                continue

            if (
                not np.isfinite(acceptance_code_value)
                or not np.isfinite(threshold)
            ):
                report_rows.append(
                    error_row(
                        source_file,
                        report_metric_name,
                        "Exact acceptance requires finite values.",
                    )
                )
                continue

        result = compare_acceptance(
            source_file=source_file,
            metric_name=report_metric_name,
            code_value=acceptance_code_value,
            operator=operator,
            threshold=threshold,
            tolerance=0.0,
        )

        result["comment"] = (
            f"{description} {result.get('comment', '')}"
        )
        report_rows.append(result)

    return write_report(report_rows, report_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = run_dual_validation(
        validation_runner=validate_part_c,
        reference_path=REFERENCE_PATH,
        internal_reference_path=INTERNAL_REFERENCE_PATH,
        report_path=REFERENCE_REPORT_PATH,
        internal_report_path=INTERNAL_REPORT_PATH,
    )

    # Both official and internal validation reports must be free of ERROR.
    # MISMATCH remains informational and does not fail the run by itself.
    sys.exit(
        0
        if (
            results["reference_exit_code"] == 0
            and results["internal_exit_code"] == 0
        )
        else 1
    )