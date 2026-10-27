from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from validations.validations_common_utility import (
    compare_numeric,
    error_row,
    load_reference,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = ROOT / "references" / "part_b_reference_metrics.csv"
REPORT_PATH = ROOT / "validations" / "reports" / "part_b_validation_report.csv"

EXPECTED_SCHEMAS = {
    "outputs/part_b_checks.csv": ["metric", "value"],
    "outputs/part_b_metrics.csv": [
        "model", "split", "MAE", "MSE", "RMSE", "R2"
    ],
    "outputs/part_b_tree_depth.csv": [
        "max_depth", "train_rmse", "val_rmse"
    ],
    "outputs/part_b_gb_learning_rate.csv": [
        "learning_rate", "val_rmse", "val_r2"
    ],
    "outputs/part_b_cv.csv": ["model", "mean_r2", "std_r2"],
}

METRIC_DESCRIPTIONS = {
    "row_count": "Number of rows in the generated output.",
    "column_count": "Number of columns in the generated output.",
    "schema_matches": "Whether the output columns match the required schema and order.",
    "train_rows": "Number of training rows in the established data split.",
    "val_rows": "Number of validation rows in the established data split.",
    "test_rows": "Number of test rows in the established data split.",
    "train_scaled_max_abs_mean": (
        "Largest absolute training-feature mean after training-only scaling."
    ),
    "test_scaled_max_abs_mean": (
        "Largest absolute test-feature mean after applying the training-fitted scaler."
    ),
    "train_scaled_mean_below_1e_9": (
        "Whether the recorded training scaled mean is below 1e-9."
    ),
    "test_scaled_mean_exceeds_0_01": (
        "Whether the recorded test scaled mean exceeds 0.01."
    ),
    "rmse_consistency": "Whether RMSE equals sqrt(MSE) for every metrics row within 0.001.",
    "tree_best_depth": "Tree depth with the lowest validation RMSE.",
    "tree_best_val_rmse": "Lowest validation RMSE among the tree-depth experiments.",
    "tree_train_rmse_nonincreasing": (
        "Whether training RMSE does not increase as tree depth increases."
    ),
    "tree_depth_1_gap": (
        "Depth-1 validation RMSE minus training RMSE; evidence for the high-bias check."
    ),
    "tree_depth_1_vs_best_gap": (
        "Depth-1 validation RMSE minus the best tree validation RMSE."
    ),
    "tree_unlimited_train_rmse": "Training RMSE for the unlimited-depth tree.",
    "tree_unlimited_gap": (
        "Unlimited-depth validation RMSE minus training RMSE; evidence for the high-variance check."
    ),
    "tree_decision_tree_rmse_matches_best": (
        "Whether the decision-tree validation RMSE matches the best depth experiment within 0.001."
    ),
    "gb_r2_ranking_ok": (
        "Whether validation R² ranks learning rates as 0.1 > 1.0 > 0.01."
    ),
    "gb_r2_0_1_minimum_met": "Whether the 0.1 learning-rate validation R² is at least 0.90.",
    "gb_r2_0_01_maximum_met": "Whether the 0.01 learning-rate validation R² is at most 0.80.",
    "cv_all_means_at_least_0_90": "Whether both cross-validation mean R² values are at least 0.90.",
    "cv_all_stds_at_most_0_03": (
        "Whether both cross-validation R² standard deviations are at most 0.03."
    ),
}


def load_output(source_file, cache):
    """Read an existing generated CSV once. Never generate or repair outputs."""
    if source_file in cache:
        return cache[source_file]

    path = ROOT / source_file

    if not path.is_file():
        cache[source_file] = (None, f"Expected output file is missing: {source_file}")
        return cache[source_file]

    try:
        data = pd.read_csv(path)
        cache[source_file] = (data, None)
    except Exception as exc:
        cache[source_file] = (
            None,
            f"Could not read output file {source_file}: {exc}",
        )

    return cache[source_file]


def require_columns(data, columns, source_file):
    """Raise a clear error when a required output schema is absent."""
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"{source_file} is missing required columns: {missing}"
        )


def extract_indicator(condition):
    """Represent a structural or Boolean check as numeric 1 or 0."""
    return int(bool(condition))


def extract_checks_metric(metric_name, data):
    require_columns(data, ["metric", "value"], "outputs/part_b_checks.csv")

    if data["metric"].duplicated().any():
        raise ValueError("part_b_checks.csv contains duplicate metric names.")

    if metric_name == "row_count":
        return len(data)

    if metric_name == "column_count":
        return len(data.columns)

    if metric_name == "schema_matches":
        return extract_indicator(
            data.columns.tolist() == EXPECTED_SCHEMAS["outputs/part_b_checks.csv"]
        )

    if metric_name in {"train_scaled_mean_below_1e_9", "test_scaled_mean_exceeds_0_01"}:
        underlying_metric = (
            "train_scaled_max_abs_mean"
            if metric_name == "train_scaled_mean_below_1e_9"
            else "test_scaled_max_abs_mean"
        )
        value = extract_check_value(underlying_metric, data)

        if metric_name == "train_scaled_mean_below_1e_9":
            return extract_indicator(value < 1e-9)
        return extract_indicator(value > 0.01)

    return extract_check_value(metric_name, data)


def extract_check_value(metric_name, data):
    matches = data.loc[data["metric"] == metric_name, "value"]

    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one '{metric_name}' entry in part_b_checks.csv; "
            f"found {len(matches)}."
        )

    return matches.iloc[0]


def extract_metrics_metric(metric_name, data):
    required = EXPECTED_SCHEMAS["outputs/part_b_metrics.csv"]
    require_columns(data, required, "outputs/part_b_metrics.csv")

    if metric_name == "row_count":
        return len(data)
    if metric_name == "column_count":
        return len(data.columns)
    if metric_name == "schema_matches":
        return extract_indicator(data.columns.tolist() == required)
    if metric_name == "val_row_count":
        return int((data["split"] == "val").sum())
    if metric_name == "test_row_count":
        return int((data["split"] == "test").sum())

    if metric_name == "rmse_consistency":
        mse = pd.to_numeric(data["MSE"], errors="coerce")
        rmse = pd.to_numeric(data["RMSE"], errors="coerce")

        if mse.isna().any() or rmse.isna().any():
            raise ValueError("MSE or RMSE contains missing or nonnumeric values.")

        expected_rmse = np.sqrt(mse)
        return extract_indicator(
            np.isclose(rmse, expected_rmse, atol=0.001, rtol=0).all()
        )

    match = re.fullmatch(r"model_present__(val|test)__(.+)", metric_name)
    if match:
        split, model_name = match.groups()
        rows = data.loc[
            (data["split"] == split) & (data["model"] == model_name)
        ]
        return extract_indicator(len(rows) == 1)

    match = re.fullmatch(r"(val|test)_r2__(.+)", metric_name)
    if match:
        split, model_name = match.groups()
        return extract_metric_row_value(
            data, split, model_name, "R2", "part_b_metrics.csv"
        )

    match = re.fullmatch(r"(val|test)_rmse__(.+)", metric_name)
    if match:
        split, model_name = match.groups()
        return extract_metric_row_value(
            data, split, model_name, "RMSE", "part_b_metrics.csv"
        )

    raise ValueError(f"Unsupported Part B metrics metric: {metric_name}")


def extract_metric_row_value(data, split, model_name, column, filename):
    rows = data.loc[
        (data["split"] == split) & (data["model"] == model_name)
    ]

    if len(rows) != 1:
        raise KeyError(
            f"Expected one row for model='{model_name}', split='{split}' "
            f"in {filename}; found {len(rows)}."
        )

    return rows.iloc[0][column]


def normalize_depth(value):
    text = str(value).strip().lower()
    if text in {"unlimited", "none", "nan"}:
        return "unlimited"
    return str(int(float(text)))


def extract_tree_metric(metric_name, data, metrics_data):
    required = EXPECTED_SCHEMAS["outputs/part_b_tree_depth.csv"]
    require_columns(data, required, "outputs/part_b_tree_depth.csv")

    if metric_name == "row_count":
        return len(data)
    if metric_name == "column_count":
        return len(data.columns)

    depths = data["max_depth"].map(normalize_depth)

    if metric_name.startswith("tree_depth_present__"):
        requested = metric_name.split("__", maxsplit=1)[1]
        requested = normalize_depth(requested)
        return extract_indicator((depths == requested).sum() == 1)

    if depths.duplicated().any():
        raise ValueError("part_b_tree_depth.csv contains duplicate depth values.")

    if metric_name in {"tree_best_depth", "tree_best_val_rmse"}:
        best_index = pd.to_numeric(data["val_rmse"], errors="coerce").idxmin()
        if pd.isna(best_index):
            raise ValueError("Tree validation RMSE values are missing or nonnumeric.")
        best_row = data.loc[best_index]

        if metric_name == "tree_best_depth":
            return (
                np.inf
                if normalize_depth(best_row["max_depth"]) == "unlimited"
                else float(normalize_depth(best_row["max_depth"]))
            )
        return best_row["val_rmse"]

    numeric_train = pd.to_numeric(data["train_rmse"], errors="coerce")
    numeric_val = pd.to_numeric(data["val_rmse"], errors="coerce")

    if numeric_train.isna().any() or numeric_val.isna().any():
        raise ValueError("Tree RMSE columns contain missing or nonnumeric values.")

    if metric_name == "tree_train_rmse_nonincreasing":
        ordered = data.assign(
            _depth=depths.map(
                lambda value: np.inf if value == "unlimited" else float(value)
            ),
            _train=numeric_train,
        ).sort_values("_depth")
        differences = ordered["_train"].diff().dropna()
        return extract_indicator((differences <= 0.001).all())

    depth_1 = data.loc[depths == "1"]
    unlimited = data.loc[depths == "unlimited"]

    if metric_name in {"tree_depth_1_gap", "tree_depth_1_vs_best_gap"}:
        if len(depth_1) != 1:
            raise KeyError("Expected exactly one depth-1 row.")
        depth_1_gap = float(depth_1.iloc[0]["val_rmse"]) - float(
            depth_1.iloc[0]["train_rmse"]
        )
        if metric_name == "tree_depth_1_gap":
            return depth_1_gap
        return depth_1_gap + float(depth_1.iloc[0]["train_rmse"]) - float(
            numeric_val.min()
        )

    if metric_name in {"tree_unlimited_train_rmse", "tree_unlimited_gap"}:
        if len(unlimited) != 1:
            raise KeyError("Expected exactly one unlimited-depth row.")
        row = unlimited.iloc[0]
        if metric_name == "tree_unlimited_train_rmse":
            return row["train_rmse"]
        return float(row["val_rmse"]) - float(row["train_rmse"])

    if metric_name == "tree_decision_tree_rmse_matches_best":
        if metrics_data is None:
            raise ValueError("Metrics output is unavailable for the tree RMSE cross-check.")

        tree_metric_rows = metrics_data.loc[
            (metrics_data["model"] == "decision_tree")
            & (metrics_data["split"] == "val")
        ]
        if len(tree_metric_rows) != 1:
            raise KeyError("Expected one decision_tree validation row in part_b_metrics.csv.")

        best_rmse = float(numeric_val.min())
        model_rmse = float(tree_metric_rows.iloc[0]["RMSE"])
        return extract_indicator(abs(model_rmse - best_rmse) <= 0.001)

    raise ValueError(f"Unsupported tree-depth metric: {metric_name}")


def extract_gb_metric(metric_name, data):
    required = EXPECTED_SCHEMAS["outputs/part_b_gb_learning_rate.csv"]
    require_columns(data, required, "outputs/part_b_gb_learning_rate.csv")

    if metric_name == "row_count":
        return len(data)
    if metric_name == "column_count":
        return len(data.columns)

    if data["learning_rate"].duplicated().any():
        raise ValueError("part_b_gb_learning_rate.csv contains duplicate learning rates.")

    rates = pd.to_numeric(data["learning_rate"], errors="coerce")
    scores = pd.to_numeric(data["val_r2"], errors="coerce")

    if rates.isna().any() or scores.isna().any():
        raise ValueError("Learning rates or validation R² values are missing or nonnumeric.")

    if metric_name.startswith("gb_learning_rate_present__"):
        requested = float(metric_name.split("__", maxsplit=1)[1])
        return extract_indicator((np.isclose(rates, requested, atol=1e-8)).sum() == 1)

    match = re.fullmatch(r"gb_val_r2__lr_(.+)", metric_name)
    if match:
        requested = float(match.group(1))
        rows = data.loc[np.isclose(rates, requested, atol=1e-8)]
        if len(rows) != 1:
            raise KeyError(f"Expected one row for learning rate {requested}.")
        return rows.iloc[0]["val_r2"]

    score_by_rate = {
        float(row["learning_rate"]): float(row["val_r2"])
        for _, row in data.iterrows()
    }

    if metric_name == "gb_r2_ranking_ok":
        return extract_indicator(
            score_by_rate[0.1] > score_by_rate[1.0] > score_by_rate[0.01]
        )
    if metric_name == "gb_r2_0_1_minimum_met":
        return extract_indicator(score_by_rate[0.1] >= 0.90)
    if metric_name == "gb_r2_0_01_maximum_met":
        return extract_indicator(score_by_rate[0.01] <= 0.80)

    raise ValueError(f"Unsupported learning-rate metric: {metric_name}")


def extract_cv_metric(metric_name, data):
    required = EXPECTED_SCHEMAS["outputs/part_b_cv.csv"]
    require_columns(data, required, "outputs/part_b_cv.csv")

    if metric_name == "row_count":
        return len(data)
    if metric_name == "column_count":
        return len(data.columns)

    if data["model"].duplicated().any():
        raise ValueError("part_b_cv.csv contains duplicate model rows.")

    match = re.fullmatch(r"cv_model_present__(.+)", metric_name)
    if match:
        model_name = match.group(1)
        return extract_indicator((data["model"] == model_name).sum() == 1)

    match = re.fullmatch(r"cv_(mean_r2|std_r2)__(.+)", metric_name)
    if match:
        column, model_name = match.groups()
        rows = data.loc[data["model"] == model_name]
        if len(rows) != 1:
            raise KeyError(f"Expected one CV row for model '{model_name}'.")
        return rows.iloc[0][column]

    mean_r2 = pd.to_numeric(data["mean_r2"], errors="coerce")
    std_r2 = pd.to_numeric(data["std_r2"], errors="coerce")

    if mean_r2.isna().any() or std_r2.isna().any():
        raise ValueError("Cross-validation output contains missing or nonnumeric values.")

    if metric_name == "cv_all_means_at_least_0_90":
        return extract_indicator((mean_r2 >= 0.90).all())
    if metric_name == "cv_all_stds_at_most_0_03":
        return extract_indicator((std_r2 <= 0.03).all())

    raise ValueError(f"Unsupported cross-validation metric: {metric_name}")


def extract_metric(reference_row, output_cache):
    source_file = reference_row["source_file"]
    metric_name = reference_row["metric_name"]

    data, load_error = load_output(source_file, output_cache)
    if load_error:
        raise FileNotFoundError(load_error)

    if source_file not in EXPECTED_SCHEMAS:
        raise ValueError(f"Unsupported output source in reference: {source_file}")

    if source_file == "outputs/part_b_checks.csv":
        return extract_checks_metric(metric_name, data)

    if source_file == "outputs/part_b_metrics.csv":
        return extract_metrics_metric(metric_name, data)

    if source_file == "outputs/part_b_tree_depth.csv":
        metrics_data, metrics_error = load_output(
            "outputs/part_b_metrics.csv", output_cache
        )
        if metrics_error:
            metrics_data = None
        return extract_tree_metric(metric_name, data, metrics_data)

    if source_file == "outputs/part_b_gb_learning_rate.csv":
        return extract_gb_metric(metric_name, data)

    if source_file == "outputs/part_b_cv.csv":
        return extract_cv_metric(metric_name, data)

    raise ValueError(f"No extraction logic exists for {source_file}")


def describe_metric(metric_name):
    if metric_name in METRIC_DESCRIPTIONS:
        return METRIC_DESCRIPTIONS[metric_name]

    if metric_name.startswith("model_present__"):
        _, split, model = metric_name.split("__", maxsplit=2)
        return f"Whether exactly one {model} row exists for the {split} split."

    if metric_name.startswith("val_r2__"):
        return f"Validation R² for model '{metric_name.split('__', 1)[1]}'."

    if metric_name.startswith("test_r2__"):
        return f"Final test R² for model '{metric_name.split('__', 1)[1]}'."

    if metric_name.startswith("tree_depth_present__"):
        return f"Whether the required tree-depth configuration '{metric_name.split('__', 1)[1]}' appears exactly once."

    if metric_name.startswith("gb_learning_rate_present__"):
        return f"Whether the required learning rate {metric_name.split('__', 1)[1]} appears exactly once."

    if metric_name.startswith("gb_val_r2__"):
        return f"Validation R² for gradient boosting learning-rate experiment '{metric_name.split('__lr_', 1)[1]}'."

    if metric_name.startswith("cv_model_present__"):
        return f"Whether model '{metric_name.split('__', 1)[1]}' appears exactly once in cross-validation results."

    if metric_name.startswith("cv_mean_r2__"):
        return f"Mean five-fold cross-validation R² for model '{metric_name.split('__', 1)[1]}'."

    if metric_name.startswith("cv_std_r2__"):
        return f"Standard deviation of five-fold cross-validation R² for model '{metric_name.split('__', 1)[1]}'."

    if metric_name.startswith("val_rmse__"):
        return f"Validation RMSE for model '{metric_name.split('__', 1)[1]}'."

    return "Part B output metric or acceptance criterion."


def validate_part_b():
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

            row = compare_numeric(
                source_file=source_file,
                metric_name=metric_name,
                reference_value=expected,
                code_value=actual,
                tolerance=tolerance,
            )

            description = describe_metric(metric_name)
            comparison_summary = row["comment"]
            row["comment"] = f"{description} {comparison_summary}"

        except Exception as exc:
            row = error_row(source_file, metric_name, str(exc))
            row["reference_value"] = expected
            row["tolerance"] = tolerance

        report_rows.append(row)

    return write_report(report_rows, REPORT_PATH)


if __name__ == "__main__":
    sys.exit(validate_part_b())