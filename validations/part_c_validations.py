from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.shipment_prep import FEATURES, NOISE_COLUMNS
from validations.validations_common_utility import (
    compare_numeric,
    error_row,
    load_reference,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REFERENCES = ROOT / "references"
REPORTS = ROOT / "validations" / "reports"

REFERENCE_PATH = REFERENCES / "part_c_reference_metrics.csv"
REPORT_PATH = REPORTS / "part_c_validation_report.csv"

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

FEATURE_SCORE_COLUMNS = [
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

FINAL_TEST_COLUMNS = THRESHOLD_COLUMNS

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

EXPECTED_THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 1)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def read_csv_checked(path, expected_columns=None):
    """Read an existing CSV and optionally validate its exact schema."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    try:
        data = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    if expected_columns is not None:
        if data.columns.tolist() != expected_columns:
            raise ValueError(
                f"Unexpected schema in {path}. "
                f"Expected {expected_columns}; "
                f"found {data.columns.tolist()}."
            )

    return data


def require_finite_numeric(data, columns, source_file):
    """Require the specified columns to contain finite numeric values."""
    missing = sorted(set(columns) - set(data.columns))

    if missing:
        raise ValueError(
            f"{source_file} is missing required columns: {missing}"
        )

    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"{source_file}.{column} contains missing or "
                "nonnumeric values."
            )

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{source_file}.{column} contains non-finite values."
            )


def indicator(condition):
    """Convert a Boolean validation condition to numeric 1 or 0."""
    return int(bool(condition))


def parse_boolean_column(series, column_name):
    """Normalize Boolean values read from CSV."""
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if normalized.isna().any():
        raise ValueError(
            f"{column_name} must contain only Boolean values."
        )

    return normalized.astype(bool)


def calculate_metrics_from_confusion(data, source_file):
    """Check stored metrics against the confusion-matrix counts."""
    count_columns = ["TN", "FP", "FN", "TP"]
    metric_columns = ["accuracy", "precision", "recall", "f1"]

    require_finite_numeric(data, count_columns + metric_columns, source_file)

    counts = data[count_columns].apply(pd.to_numeric, errors="coerce")

    if (counts < 0).any().any():
        raise ValueError(
            f"{source_file} contains negative confusion-matrix counts."
        )

    if not np.equal(counts.to_numpy(), np.floor(counts.to_numpy())).all():
        raise ValueError(
            f"{source_file} contains non-integer confusion-matrix counts."
        )

    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]

    total = tn + fp + fn + tp

    if (total <= 0).any():
        raise ValueError(
            f"{source_file} contains an empty confusion matrix."
        )

    accuracy = (tn + tp) / total
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    precision = (tp / precision_denominator).where(
        precision_denominator > 0,
        0.0,
    )

    recall = (tp / recall_denominator).where(
        recall_denominator > 0,
        0.0,
    )

    f1_denominator = precision + recall

    f1 = (2 * precision * recall / f1_denominator).where(
        f1_denominator > 0,
        0.0,
    )

    expected_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    consistent = True

    for column, expected_values in expected_metrics.items():
        actual = pd.to_numeric(data[column], errors="coerce")

        if actual.isna().any():
            raise ValueError(
                f"{source_file}.{column} contains missing or "
                "nonnumeric values."
            )

        if not np.isfinite(actual.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{source_file}.{column} contains non-finite values."
            )

        if not np.allclose(
            actual.to_numpy(dtype=float),
            expected_values.to_numpy(dtype=float),
            atol=0.001,
            rtol=0,
        ):
            consistent = False

    return indicator(consistent), total, fn + tp


def require_unique_row(data, key_column, key_value, source_file):
    """Return exactly one row matching a key."""
    matches = data.loc[data[key_column] == key_value]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row where {key_column}={key_value!r} "
            f"in {source_file}; found {len(matches)}."
        )

    return matches.iloc[0]


# ---------------------------------------------------------------------
# Output loading and caching
# ---------------------------------------------------------------------

def load_output(source_file, output_cache):
    """Load each output at most once during a validation run."""
    if source_file not in output_cache:
        if source_file == "data/modelling_table.csv":
            path = ROOT / source_file
        else:
            path = ROOT / source_file

        output_cache[source_file] = read_csv_checked(path)

    return output_cache[source_file]


# ---------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------

def extract_split_metric(metric_name, data):
    """Extract reproducible stratified-split diagnostics."""
    if "is_delayed" not in data.columns:
        raise ValueError(
            "data/modelling_table.csv is missing 'is_delayed'."
        )

    y = data["is_delayed"]

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "'is_delayed' must contain nonmissing binary 0/1 values."
        )

    X = data.drop(columns=["is_delayed"])

    _, X_temp, _, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    _, _, y_train, y_val_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Match the project's prescribed second split of the 30% holdout.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    if metric_name == "row_count":
        return len(data)

    if metric_name == "delayed_train_count":
        return int(y_train.sum())

    if metric_name == "delayed_val_count":
        return int(y_val.sum())

    if metric_name == "delayed_test_count":
        return int(y_test.sum())

    raise ValueError(f"Unsupported split metric: {metric_name}")


def extract_models_metric(metric_name, data):
    source_file = "outputs/part_c_models.csv"

    if list(data.columns) != MODEL_COLUMNS:
        raise ValueError(
            f"{source_file} schema mismatch. "
            f"Expected {MODEL_COLUMNS}; found {list(data.columns)}."
        )

    if metric_name == "row_count":
        return len(data)

    if data["model"].duplicated().any():
        raise ValueError(f"{source_file} contains duplicate model names.")

    if metric_name == "blind_accuracy":
        row = require_unique_row(
            data, "model", "blind_all_on_time", source_file
        )
        return float(row["accuracy"])

    if metric_name in {"blind_tn", "blind_fp", "blind_fn", "blind_tp"}:
        column = metric_name.removeprefix("blind_")
        row = require_unique_row(
            data, "model", "blind_all_on_time", source_file
        )
        return float(row[column.upper()])

    if metric_name == "confusion_metrics_consistent":
        consistent, _, _ = calculate_metrics_from_confusion(
            data, source_file
        )
        return consistent

    if metric_name == "trained_accuracy_in_reference_range":
        require_finite_numeric(data, ["accuracy"], source_file)

        trained = data.loc[data["model"] != "blind_all_on_time"]
        if trained.empty:
            raise ValueError("No trained-model rows found.")

        accuracy = pd.to_numeric(trained["accuracy"], errors="coerce")

        return indicator(
            accuracy.between(0.883, 0.911, inclusive="both").all()
        )

    if metric_name in {
        "logistic_f1_minimum_met",
        "tree_f1_minimum_met",
        "random_forest_f1_minimum_met",
    }:
        if metric_name == "logistic_f1_minimum_met":
            model_name = "logistic_regression"
            minimum = 0.65
        elif metric_name == "tree_f1_minimum_met":
            tree_rows = data.loc[
                data["model"].astype(str).str.startswith(
                    "decision_tree_depth_"
                )
            ]
            if len(tree_rows) != 1:
                raise ValueError(
                    "Expected exactly one decision-tree result row."
                )
            return indicator(
                float(tree_rows.iloc[0]["f1"]) >= 0.60
            )
        else:
            model_name = "random_forest_default"
            minimum = 0.65

        row = require_unique_row(data, "model", model_name, source_file)
        return indicator(float(row["f1"]) >= minimum)

    if metric_name == "tree_depth_reference":
        tree_rows = data.loc[
            data["model"].astype(str).str.startswith(
                "decision_tree_depth_"
            )
        ]

        if len(tree_rows) != 1:
            raise ValueError(
                "Expected exactly one decision-tree result row."
            )

        model_name = str(tree_rows.iloc[0]["model"])
        prefix = "decision_tree_depth_"

        try:
            return int(model_name.removeprefix(prefix))
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse tree depth from {model_name!r}."
            ) from exc

    raise ValueError(f"Unsupported model metric: {metric_name}")


def extract_feature_scores_metric(metric_name, data):
    source_file = "outputs/part_c_feature_scores.csv"

    if list(data.columns) != FEATURE_SCORE_COLUMNS:
        raise ValueError(
            f"{source_file} schema mismatch. "
            f"Expected {FEATURE_SCORE_COLUMNS}; "
            f"found {list(data.columns)}."
        )

    if metric_name == "row_count":
        return len(data)

    require_finite_numeric(
        data,
        ["f_score", "f_rank", "rf_importance"],
        source_file,
    )

    if data["feature"].isna().any() or data["feature"].duplicated().any():
        raise ValueError(
            f"{source_file} contains missing or duplicate feature names."
        )

    noise_flags = parse_boolean_column(data["is_noise"], "is_noise")
    scores = data.copy()
    scores["is_noise"] = noise_flags

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
        ["f_score", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if metric_name == "top_seven_f_score_order_correct":
        return indicator(
            ranked["feature"].head(7).tolist() == EXPECTED_TOP_SEVEN
        )

    if metric_name == "noise_columns_outside_top_seven":
        noise_rows = scores.loc[scores["is_noise"]]
        return indicator(
            (noise_rows["f_rank"] >= 8).all()
            and not set(noise_rows["feature"]).intersection(
                set(ranked["feature"].head(7))
            )
        )

    if metric_name == "top_seven_min_f_score_above_30":
        return indicator(
            float(ranked["f_score"].head(7).min()) > 30
        )

    if metric_name == "combined_noise_importance":
        return float(
            scores.loc[scores["is_noise"], "rf_importance"].sum()
        )

    if metric_name == "forest_importance_sums_to_one":
        return indicator(
            np.isclose(
                scores["rf_importance"].sum(),
                1.0,
                atol=0.001,
                rtol=0,
            )
        )

    raise ValueError(f"Unsupported feature-score metric: {metric_name}")


def extract_selectk_metric(metric_name, data):
    source_file = "outputs/part_c_selectk.csv"

    if list(data.columns) != SELECTK_COLUMNS:
        raise ValueError(
            f"{source_file} schema mismatch. "
            f"Expected {SELECTK_COLUMNS}; found {list(data.columns)}."
        )

    if metric_name == "row_count":
        return len(data)

    require_finite_numeric(data, ["k", "val_f1"], source_file)

    if data["k"].duplicated().any():
        raise ValueError(f"{source_file} contains duplicate k values.")

    if metric_name == "required_k_values_present":
        actual = pd.to_numeric(data["k"], errors="coerce").tolist()
        return indicator(actual == [5, 7, 10, 12, 17])

    if metric_name == "validation_f1_values_valid":
        f1 = pd.to_numeric(data["val_f1"], errors="coerce")
        return indicator(f1.between(0, 1, inclusive="both").all())

    raise ValueError(f"Unsupported SelectK metric: {metric_name}")


def extract_tuning_metric(metric_name, data):
    source_file = "outputs/part_c_tuning.csv"

    if list(data.columns) != TUNING_COLUMNS:
        raise ValueError(
            f"{source_file} schema mismatch. "
            f"Expected {TUNING_COLUMNS}; found {list(data.columns)}."
        )

    if metric_name == "row_count":
        return len(data)

    if data["search"].duplicated().any():
        raise ValueError(f"{source_file} contains duplicate search rows.")

    if metric_name == "required_search_rows_present":
        return indicator(data["search"].tolist() == EXPECTED_SEARCH_NAMES)

    if metric_name in {
        "candidate_counts_correct",
        "cv_folds_correct",
        "total_fits_correct",
    }:
        expected_by_metric = {
            "candidate_counts_correct": [27, 9, 1],
            "cv_folds_correct": [5, 5, 0],
            "total_fits_correct": [135, 45, 1],
        }

        column_by_metric = {
            "candidate_counts_correct": "candidates",
            "cv_folds_correct": "cv_folds",
            "total_fits_correct": "total_fits",
        }

        column = column_by_metric[metric_name]
        actual = pd.to_numeric(data[column], errors="coerce")

        if actual.isna().any():
            raise ValueError(f"{source_file}.{column} is not fully numeric.")

        return indicator(
            actual.astype(int).tolist() == expected_by_metric[metric_name]
        )

    if metric_name in {"grid_cv_accuracy", "randomized_cv_accuracy"}:
        search_name = (
            "grid_search"
            if metric_name == "grid_cv_accuracy"
            else "randomized_search"
        )
        row = require_unique_row(data, "search", search_name, source_file)

        if pd.isna(row["best_cv_accuracy"]):
            raise ValueError(
                f"{search_name} best_cv_accuracy is missing."
            )

        return float(row["best_cv_accuracy"])

    if metric_name == "tuning_validation_f1_minimum_met":
        require_finite_numeric(data, ["val_f1"], source_file)
        return indicator(
            pd.to_numeric(data["val_f1"], errors="coerce")
            .between(0.65, 1.0, inclusive="both")
            .all()
        )

    raise ValueError(f"Unsupported tuning metric: {metric_name}")


def extract_thresholds_metric(metric_name, thresholds, config):
    source_file = "outputs/part_c_thresholds.csv"

    if list(thresholds.columns) != THRESHOLD_COLUMNS:
        raise ValueError(
            f"{source_file} schema mismatch. "
            f"Expected {THRESHOLD_COLUMNS}; "
            f"found {list(thresholds.columns)}."
        )

    if len(config) != 1 or "threshold" not in config.columns:
        raise ValueError(
            "part_c_final_config.csv must contain exactly one row "
            "with a threshold column."
        )

    if metric_name == "row_count":
        return len(thresholds)

    if thresholds["threshold"].duplicated().any():
        raise ValueError(f"{source_file} contains duplicate thresholds.")

    require_finite_numeric(
        thresholds,
        THRESHOLD_COLUMNS,
        source_file,
    )

    if metric_name == "required_thresholds_present":
        actual = np.round(
            pd.to_numeric(thresholds["threshold"], errors="coerce")
            .to_numpy(dtype=float),
            1,
        )
        return indicator(
            np.array_equal(actual, EXPECTED_THRESHOLDS)
        )

    if metric_name == "threshold_metrics_consistent":
        consistent, totals, _ = calculate_metrics_from_confusion(
            thresholds, source_file
        )
        return indicator(
            consistent == 1
            and (totals == 180).all()
        )

    if metric_name == "recall_nonincreasing_with_threshold":
        ordered = thresholds.sort_values("threshold")
        recall = pd.to_numeric(ordered["recall"], errors="coerce")
        return indicator(recall.diff().dropna().le(0.001).all())

    if metric_name == "selected_threshold":
        return float(config.iloc[0]["threshold"])

    if metric_name == "threshold_rule_satisfied":
        qualifying = thresholds.loc[thresholds["recall"] >= 0.80]

        expected = (
            float(qualifying["threshold"].max())
            if not qualifying.empty
            else 0.1
        )

        actual = float(config.iloc[0]["threshold"])

        return indicator(np.isclose(actual, expected, atol=0.001, rtol=0))

    raise ValueError(f"Unsupported threshold metric: {metric_name}")


def extract_final_test_metric(metric_name, final_test, config):
    source_file = "outputs/part_c_final_test.csv"

    if list(final_test.columns) != FINAL_TEST_COLUMNS:
        raise ValueError(
            f"{source_file} schema mismatch. "
            f"Expected {FINAL_TEST_COLUMNS}; "
            f"found {list(final_test.columns)}."
        )

    if len(final_test) != 1:
        raise ValueError(
            f"{source_file} must contain exactly one row; "
            f"found {len(final_test)}."
        )

    if len(config) != 1 or "threshold" not in config.columns:
        raise ValueError(
            "part_c_final_config.csv must contain exactly one row "
            "with a threshold column."
        )

    row = final_test.iloc[0]

    consistent, totals, delayed = calculate_metrics_from_confusion(
        final_test, source_file
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
                float(config.iloc[0]["threshold"]),
                atol=0.001,
                rtol=0,
            )
        )

    if metric_name == "confusion_metrics_consistent":
        return consistent

    if metric_name == "test_precision_minimum_met":
        return indicator(float(row["precision"]) >= 0.45)

    if metric_name == "test_recall_minimum_met":
        return indicator(float(row["recall"]) >= 0.55)

    if metric_name == "test_f1_minimum_met":
        return indicator(float(row["f1"]) >= 0.60)

    raise ValueError(f"Unsupported final-test metric: {metric_name}")


def extract_metric(reference_row, output_cache):
    """Extract one metric named in the reference CSV."""
    source_file = reference_row["source_file"]
    metric_name = reference_row["metric_name"]

    if source_file == "data/modelling_table.csv":
        data = load_output(source_file, output_cache)
        return extract_split_metric(metric_name, data)

    if source_file == "outputs/part_c_models.csv":
        data = load_output(source_file, output_cache)
        return extract_models_metric(metric_name, data)

    if source_file == "outputs/part_c_feature_scores.csv":
        data = load_output(source_file, output_cache)
        return extract_feature_scores_metric(metric_name, data)

    if source_file == "outputs/part_c_selectk.csv":
        data = load_output(source_file, output_cache)
        return extract_selectk_metric(metric_name, data)

    if source_file == "outputs/part_c_tuning.csv":
        data = load_output(source_file, output_cache)
        return extract_tuning_metric(metric_name, data)

    if source_file == "outputs/part_c_thresholds.csv":
        thresholds = load_output(source_file, output_cache)
        config = load_output(
            "outputs/part_c_final_config.csv",
            output_cache,
        )
        return extract_thresholds_metric(metric_name, thresholds, config)

    if source_file == "outputs/part_c_final_config.csv":
        config = load_output(source_file, output_cache)

        if len(config) != 1 or "threshold" not in config.columns:
            raise ValueError(
                "part_c_final_config.csv must contain exactly one row "
                "with a threshold column."
            )

        if metric_name == "selected_threshold":
            return float(config.iloc[0]["threshold"])

        if metric_name == "threshold_rule_satisfied":
            thresholds = load_output(
                "outputs/part_c_thresholds.csv",
                output_cache,
            )
            return extract_thresholds_metric(
                metric_name,
                thresholds,
                config,
            )

        raise ValueError(
            f"Unsupported final-config metric: {metric_name}"
        )

    if source_file == "outputs/part_c_final_test.csv":
        final_test = load_output(source_file, output_cache)
        config = load_output(
            "outputs/part_c_final_config.csv",
            output_cache,
        )
        return extract_final_test_metric(metric_name, final_test, config)

    raise ValueError(
        f"Unsupported Part C source_file in reference: {source_file}"
    )


# ---------------------------------------------------------------------
# Descriptions and report generation
# ---------------------------------------------------------------------

METRIC_DESCRIPTIONS = {
    "row_count": "Number of rows in the existing output.",
    "delayed_train_count": "Delayed parcels in the prescribed training split.",
    "delayed_val_count": "Delayed parcels in the prescribed validation split.",
    "delayed_test_count": "Delayed parcels in the prescribed test split.",
    "blind_accuracy": "Accuracy of the all-on-time blind baseline.",
    "blind_tn": "Blind-baseline true negatives.",
    "blind_fp": "Blind-baseline false positives.",
    "blind_fn": "Blind-baseline false negatives.",
    "blind_tp": "Blind-baseline true positives.",
    "confusion_metrics_consistent": "Stored metrics agree with confusion-matrix counts.",
    "trained_accuracy_in_reference_range": "Every trained model's validation accuracy is within the established range.",
    "logistic_f1_minimum_met": "Logistic-regression validation F1 meets the minimum.",
    "tree_depth_reference": "Selected decision-tree depth matches the established benchmark.",
    "tree_f1_minimum_met": "Decision-tree validation F1 meets the minimum.",
    "random_forest_f1_minimum_met": "Default random-forest validation F1 meets the minimum.",
    "feature_names_unique": "Feature names are unique.",
    "feature_scores_finite": "Feature scores, ranks, and importances are finite.",
    "noise_columns_identified": "All required noise columns are correctly marked.",
    "top_seven_f_score_order_correct": "Top seven F-score features match the established ordering.",
    "noise_columns_outside_top_seven": "All five noise columns rank outside the top seven.",
    "top_seven_min_f_score_above_30": "Minimum F-score among the top seven exceeds 30.",
    "combined_noise_importance": "Combined random-forest importance of the five noise columns.",
    "forest_importance_sums_to_one": "Random-forest feature importances sum to one within tolerance.",
    "required_k_values_present": "The SelectK experiment contains k values 5, 7, 10, 12, and 17.",
    "validation_f1_values_valid": "All SelectK validation F1 scores are in [0, 1].",
    "required_search_rows_present": "Grid search, randomized search, and default forest rows are present in order.",
    "candidate_counts_correct": "Tuning candidate counts match the required search spaces.",
    "cv_folds_correct": "Tuning cross-validation fold counts are correct.",
    "total_fits_correct": "Total tuning fit counts match candidates and folds.",
    "grid_cv_accuracy": "Grid-search best cross-validation accuracy.",
    "randomized_cv_accuracy": "Randomized-search best cross-validation accuracy.",
    "tuning_validation_f1_minimum_met": "Every tuning result's validation F1 meets the minimum.",
    "required_thresholds_present": "Threshold table contains values from 0.1 through 0.9.",
    "threshold_metrics_consistent": "Threshold-table metrics agree with confusion-matrix counts.",
    "recall_nonincreasing_with_threshold": "Recall does not increase as the classification threshold rises.",
    "selected_threshold": "Threshold saved in the final configuration.",
    "threshold_rule_satisfied": "Saved threshold is the highest tested threshold with validation recall at least 0.80, or 0.1 if none qualifies.",
    "test_row_count": "Number of observations represented by the final test confusion matrix.",
    "test_delayed_count": "Number of delayed parcels represented by the final test confusion matrix.",
    "threshold_matches_config": "Final-test threshold matches the saved configuration.",
    "test_precision_minimum_met": "Final-test precision meets the required minimum.",
    "test_recall_minimum_met": "Final-test recall meets the required minimum.",
    "test_f1_minimum_met": "Final-test F1 meets the required minimum.",
}


def describe_metric(metric_name):
    """Return a readable explanation for a report row."""
    return METRIC_DESCRIPTIONS.get(
        metric_name,
        f"Part C validation check: {metric_name}.",
    )


def validate_part_c():
    """Validate all metrics and write the standardized report."""
    reference, reference_errors = load_reference(REFERENCE_PATH)
    report_rows = list(reference_errors)
    output_cache = {}

    if not reference.empty:
        # Each reference row must be evaluated independently so a failure
        # in one metric does not hide failures in other metrics.
        for _, reference_row in reference.iterrows():
            source_file = reference_row["source_file"]
            metric_name = reference_row["metric_name"]
            expected = reference_row["metric_value"]
            tolerance = reference_row["tolerance"]

            try:
                actual = extract_metric(reference_row, output_cache)

                row = compare_numeric(
                    source_file=source_file,
                    metric_name=metric_name,
                    reference_value=expected,
                    code_value=actual,
                    tolerance=tolerance,
                )

                row["comment"] = (
                    f"{describe_metric(metric_name)} {row['comment']}"
                )

            except Exception as exc:
                row = error_row(
                    source_file,
                    metric_name,
                    f"{describe_metric(metric_name)} "
                    f"Could not evaluate check: {exc}",
                )
                row["reference_value"] = expected
                row["tolerance"] = tolerance

            report_rows.append(row)

    return write_report(report_rows, REPORT_PATH)


if __name__ == "__main__":
    sys.exit(validate_part_c())