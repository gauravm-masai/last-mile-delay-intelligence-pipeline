from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from src.shipment_prep import FEATURES, NOISE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REFERENCES = ROOT / "references"
REPORTS = ROOT / "validations/reports"

EXPECTED_TOP_SEVEN = [
    "distance_km",
    "hub_tier",
    "hub_daily_capacity",
    "n_stops",
    "promised_hours",
    "rain_flag",
    "sale_event_flag"
]

def validate_part_c_split():
    reference = pd.read_csv(REFERENCES / "part_c_reference_metrics.csv")
    data = pd.read_csv(DATA / "modelling_table.csv")

    # Read expected delayed counts from the reference CSV.
    split_reference = reference.loc[
        reference["Item"].str.strip()
        == "Delayed parcels in train / validation / test",
        "Reference"
    ]

    assert len(split_reference) == 1, (
        "Split reference row missing or duplicated in "
        "part_c_reference_metrics.csv"
    )

    expected_delayed = [
        int(value.strip())
        for value in split_reference.iloc[0].split("/")
    ]

    # Reproduce the required two-stage stratified split.
    X = data.drop(columns=["is_delayed"])
    y = data["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    actual_sizes = [
        len(X_train),
        len(X_val),
        len(X_test)
    ]

    actual_delayed = [
        int(y_train.sum()),
        int(y_val.sum()),
        int(y_test.sum())
    ]

    # Split sizes are specified in the project task instructions.
    expected_sizes = [840, 180, 180]

    assert actual_sizes == expected_sizes, (
        f"Split sizes mismatch: expected {expected_sizes}, "
        f"got {actual_sizes}"
    )

    assert actual_delayed == expected_delayed, (
        f"Delayed counts mismatch: expected {expected_delayed}, "
        f"got {actual_delayed}"
    )

    print("PASS: Split sizes match the project requirements.")
    print("PASS: Delayed counts match part_c_reference_metrics.csv.")
    print(f"Split sizes: {actual_sizes}")
    print(f"Delayed counts: {actual_delayed}")

def validate_part_c_models():
    reference = pd.read_csv(REFERENCES / "part_c_reference_metrics.csv")
    results = pd.read_csv(OUTPUTS / "part_c_models.csv")

    # Read reference values from the reference CSV.
    def get_reference(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Reference"
        ]
        assert len(row) == 1, (
            f"Reference item missing or duplicated: {item}"
        )
        return row.iloc[0]

    def get_pass_band(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Pass band"
        ]
        assert len(row) == 1, (
            f"Pass band missing or duplicated: {item}"
        )
        return row.iloc[0]

    # Validate output structure.
    expected_columns = [
        "model", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1"
    ]
    assert list(results.columns) == expected_columns
    assert len(results) == 4

    # Validate model names.
    assert results["model"].iloc[0] == "blind_all_on_time"
    assert results["model"].iloc[1] == "logistic_regression"
    assert results["model"].iloc[3] == "random_forest_default"

    tree_row = results.iloc[2]
    assert tree_row["model"].startswith("decision_tree_depth_")

    # Validate confusion-matrix counts and recomputed metrics.
    count_columns = ["TN", "FP", "FN", "TP"]
    assert (results[count_columns] >= 0).all().all()

    totals = results[count_columns].sum(axis=1)
    assert (totals == 180).all()

    expected_accuracy = (
        results["TN"] + results["TP"]
    ) / totals

    denominator_precision = results["TP"] + results["FP"]
    denominator_recall = results["TP"] + results["FN"]

    expected_precision = (
        (results["TP"] / denominator_precision)
        .where(denominator_precision > 0, 0)
    )

    expected_recall = (
        (results["TP"] / denominator_recall)
        .where(denominator_recall > 0, 0)
    )

    expected_f1 = (
        (2 * expected_precision * expected_recall)
        / (expected_precision + expected_recall)
    ).where(
        (expected_precision + expected_recall) > 0,
        0
    )

    for column, expected_values in {
        "accuracy": expected_accuracy,
        "precision": expected_precision,
        "recall": expected_recall,
        "f1": expected_f1
    }.items():
        assert (
            (results[column] - expected_values).abs() <= 0.001
        ).all(), f"{column} does not match the confusion matrix."

    print("PASS: Confusion-matrix metrics are internally consistent.")

    # Validate the blind baseline using the reference CSV.
    blind = results.loc[
        results["model"] == "blind_all_on_time"
    ].iloc[0]

    expected_blind = {
        "TN": 138,
        "FP": 0,
        "FN": 42,
        "TP": 0
    }

    for column, value in expected_blind.items():
        assert blind[column] == value

    baseline_accuracy = float(
        get_reference("Blind baseline accuracy (validation)")
    )
    baseline_band = get_pass_band(
        "Blind baseline accuracy (validation)"
    )

    assert abs(blind["accuracy"] - baseline_accuracy) <= 0.001
    assert "0.001" in baseline_band
    assert blind["precision"] == 0

    print("PASS: Blind baseline matches the reference CSV.")

    # Validate the accuracy range for trained models.
    accuracy_reference = get_reference(
        "Accuracy of every trained model (validation)"
    )
    accuracy_band = get_pass_band(
        "Accuracy of every trained model (validation)"
    )

    lower, upper = [
        float(value.strip())
        for value in accuracy_reference.split("–")
    ]

    trained = results.loc[
        results["model"] != "blind_all_on_time"
    ]

    assert trained["accuracy"].between(
        lower, upper, inclusive="both"
    ).all(), (
        f"Trained-model accuracy is outside the reference range: "
        f"{accuracy_reference}"
    )

    assert "above the blind baseline" in accuracy_band

    print("PASS: Trained-model accuracies meet the reference range.")

    tree = results.loc[
        results["model"].str.startswith("decision_tree_depth_")
    ].iloc[0]

    # Validate minimum F1 scores using the reference pass bands.
    f1_checks = [
        (
            "logistic_regression",
            "Logistic regression: F1 (validation)"
        ),
        (
            tree["model"],
            "Best decision tree: depth / F1"
        ),
        (
            "random_forest_default",
            "Default random forest F1 (validation)"
        )
    ]

    for model_name, reference_item in f1_checks:
        model_row = results.loc[
            results["model"] == model_name
        ]
        assert len(model_row) == 1, (
            f"Expected one result row for {model_name}"
        )

        pass_band = get_pass_band(reference_item)

        # Extract the minimum F1 value from criteria such as "≥ 0.65".
        minimum_f1 = float(
            re.search(r"≥\s*(\d+(?:\.\d+)?)", pass_band).group(1)
        )

        assert model_row.iloc[0]["f1"] >= minimum_f1, (
            f"{model_name} F1 is below the reference minimum "
            f"of {minimum_f1}"
        )

    # The tree's depth reference is the first value in "6 / 0.747".
    tree_reference = get_reference("Best decision tree: depth / F1")
    expected_depth = int(tree_reference.split("/")[0].strip())

    actual_depth = int(
        tree["model"].replace("decision_tree_depth_", "")
    )

    assert actual_depth == expected_depth, (
        f"Expected decision-tree depth {expected_depth}, "
        f"got {actual_depth}"
    )

    print("PASS: Model F1 scores meet the reference pass bands.")
    print("PASS: Decision-tree depth matches the reference.")
    print("PASS: All Part C model validations passed.")


def validate_part_c_feature_scores():
    reference = pd.read_csv(REFERENCES / "part_c_reference_metrics.csv")
    scores = pd.read_csv(OUTPUTS / "part_c_feature_scores.csv")

    # Read reference values and pass bands from the reference CSV.
    def get_reference(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Reference"
        ]
        assert len(row) == 1, (
            f"Reference item missing or duplicated: {item}"
        )
        return row.iloc[0]

    def get_pass_band(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Pass band"
        ]
        assert len(row) == 1, (
            f"Pass band missing or duplicated: {item}"
        )
        return row.iloc[0]

    # Validate output structure.
    expected_columns = [
        "feature", "f_score", "f_rank", "rf_importance", "is_noise"
    ]
    assert list(scores.columns) == expected_columns
    assert len(scores) == 17
    assert scores["feature"].is_unique
    assert scores["f_score"].notna().all()
    assert scores["f_rank"].notna().all()
    assert scores["rf_importance"].notna().all()

    print("PASS: Feature-score output has 17 unique features.")

    # Noise columns specified in the Part C project instructions.
    noise_columns = {
        "label_print_batch",
        "scanner_build_no",
        "tape_roll_no",
        "sticker_serial_mod",
        "shelf_slot_no"
    }

    # Normalize is_noise in case pandas reads it as text.
    noise_flags = (
        scores["is_noise"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    assert noise_flags.notna().all(), (
        "is_noise must contain True/False values."
    )

    scores = scores.copy()
    scores["is_noise"] = noise_flags.astype(bool)

    actual_noise = set(
        scores.loc[scores["is_noise"], "feature"]
    )

    assert actual_noise == noise_columns, (
        f"Noise-column mismatch. Expected {sorted(noise_columns)}, "
        f"got {sorted(actual_noise)}"
    )

    print("PASS: All five noise columns are correctly identified.")

    # Check F-score ranking.
    ranked = scores.sort_values(
        ["f_score", "feature"],
        ascending=[False, True]
    ).reset_index(drop=True)

    expected_top_seven = [
        "distance_km",
        "hub_tier",
        "hub_daily_capacity",
        "n_stops",
        "promised_hours",
        "rain_flag",
        "sale_event_flag"
    ]

    actual_top_seven = ranked["feature"].head(7).tolist()

    assert actual_top_seven == expected_top_seven, (
        f"Unexpected top-seven F-score features: {actual_top_seven}"
    )

    noise_scores = scores.loc[scores["is_noise"]]

    assert (noise_scores["f_rank"] >= 8).all(), (
        "A noise column appears in the top seven F-score ranks."
    )

    assert (noise_scores["f_score"] < 10).all(), (
        "A noise column has an F-score of 10 or higher."
    )

    assert ranked["f_score"].head(7).min() > 30, (
        "The minimum F-score among the top seven is not above 30."
    )

    print("PASS: Top-seven F-score ranking matches project requirements.")
    print("PASS: All noise columns rank outside the top seven.")

    # Validate combined noise importance against the reference pass band.
    importance_item = (
        "Noise columns' share of forest importance (17 columns)"
    )

    reference_share = float(get_reference(importance_item))
    pass_band = get_pass_band(importance_item)

    # Parse a range such as "0.15 to 0.30".
    import re

    limits = re.findall(r"\d+(?:\.\d+)?", pass_band)
    assert len(limits) == 2, (
        f"Cannot parse importance pass band: {pass_band}"
    )

    lower_limit, upper_limit = map(float, limits)

    noise_importance = scores.loc[
        scores["is_noise"], "rf_importance"
    ].sum()

    assert lower_limit <= noise_importance <= upper_limit, (
        f"Combined noise importance {noise_importance:.3f} is outside "
        f"the reference pass band {pass_band}."
    )

    assert abs(scores["rf_importance"].sum() - 1.0) <= 0.001, (
        "Random Forest importances should sum to approximately 1."
    )

    print(
        f"PASS: Combined noise importance = {noise_importance:.3f}; "
        f"reference value = {reference_share:.3f}; "
        f"pass band = {pass_band}."
    )

    print("PASS: All Part C feature-score validations passed.")


def validate_part_c_selectk():
    reference = pd.read_csv(REFERENCES / "part_c_reference_metrics.csv")
    results = pd.read_csv(OUTPUTS / "part_c_selectk.csv")

    # Validate output structure.
    expected_columns = ["k", "val_f1"]
    assert list(results.columns) == expected_columns
    assert len(results) == 5
    assert results["k"].tolist() == [5, 7, 10, 12, 17]
    assert results["k"].is_unique
    assert results["val_f1"].notna().all()
    assert results["val_f1"].between(0, 1).all()

    print("PASS: SelectK output contains the five required k values.")

    # Read the relevant reference pass band from the reference CSV.
    # The project reference table does not specify exact F1 values for
    # each k, so validate the required structure and score ranges here.
    assert results["val_f1"].ge(0).all()
    assert results["val_f1"].le(1).all()

    print("PASS: All SelectK validation F1 scores are within [0, 1].")
    print(results.to_string(index=False))
    print("PASS: All Part C SelectK validations passed.")


def validate_part_c_tuning():
    reference = pd.read_csv(REFERENCES / "part_c_reference_metrics.csv")
    results = pd.read_csv(OUTPUTS / "part_c_tuning.csv")

    # Read reference values and pass bands from the reference CSV.
    def get_reference(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Reference"
        ]
        assert len(row) == 1, (
            f"Reference item missing or duplicated: {item}"
        )
        return row.iloc[0]

    def get_pass_band(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Pass band"
        ]
        assert len(row) == 1, (
            f"Pass band missing or duplicated: {item}"
        )
        return row.iloc[0]

    # Validate output structure.
    expected_columns = [
        "search",
        "candidates",
        "cv_folds",
        "total_fits",
        "best_params",
        "best_cv_accuracy",
        "val_f1"
    ]

    assert list(results.columns) == expected_columns
    assert len(results) == 3
    assert results["search"].tolist() == [
        "grid_search",
        "randomized_search",
        "default_forest_same_7_features"
    ]

    print("PASS: Tuning output contains the three required rows.")

    # Validate candidate counts and cross-validation fits.
    expected_candidates = [27, 9, 1]
    expected_folds = [5, 5, 0]
    expected_fits = [135, 45, 1]

    assert results["candidates"].tolist() == expected_candidates
    assert results["cv_folds"].tolist() == expected_folds
    assert results["total_fits"].tolist() == expected_fits

    assert (
        results["candidates"] * results["cv_folds"]
    ).iloc[:2].tolist() == results["total_fits"].iloc[:2].tolist()

    assert results.iloc[2]["best_params"] == "defaults"
    assert pd.isna(results.iloc[2]["best_cv_accuracy"])

    print("PASS: Candidate counts, CV folds and total fits are correct.")

    # Read the CV accuracy reference and pass band.
    cv_item = "Grid / randomized best_cv_accuracy"
    cv_reference = [
        float(value.strip())
        for value in get_reference(cv_item).split("/")
    ]
    cv_band = get_pass_band(cv_item)

    import re

    limits = re.findall(r"\d+(?:\.\d+)?", cv_band)
    assert len(limits) == 2, (
        f"Cannot parse CV accuracy pass band: {cv_band}"
    )

    lower_limit, upper_limit = map(float, limits)

    grid_accuracy = float(results.loc[
        results["search"] == "grid_search",
        "best_cv_accuracy"
    ].iloc[0])

    random_accuracy = float(results.loc[
        results["search"] == "randomized_search",
        "best_cv_accuracy"
    ].iloc[0])

    assert lower_limit <= grid_accuracy <= upper_limit, (
        f"Grid CV accuracy {grid_accuracy:.4f} is outside "
        f"the reference pass band {cv_band}."
    )

    assert lower_limit <= random_accuracy <= upper_limit, (
        f"Randomized CV accuracy {random_accuracy:.4f} is outside "
        f"the reference pass band {cv_band}."
    )

    assert grid_accuracy >= random_accuracy, (
        "Grid search CV accuracy should be at least as high as "
        "randomized search CV accuracy."
    )

    print(
        f"PASS: Grid and randomized CV accuracies meet "
        f"the reference pass band {cv_band}."
    )

    # Validate validation F1 scores against the reference pass band.
    f1_item = "val_f1 of each row in part_c_tuning.csv"
    f1_reference = [
        float(value.strip())
        for value in get_reference(f1_item).split("/")
    ]
    f1_band = get_pass_band(f1_item)

    minimum_f1_match = re.search(
        r"≥\s*(\d+(?:\.\d+)?)",
        f1_band
    )
    assert minimum_f1_match, (
        f"Cannot parse F1 pass band: {f1_band}"
    )

    minimum_f1 = float(minimum_f1_match.group(1))

    assert results["val_f1"].notna().all()
    assert results["val_f1"].between(minimum_f1, 1).all(), (
        f"At least one validation F1 is below {minimum_f1}."
    )

    print(
        f"PASS: All tuning validation F1 scores meet "
        f"the reference minimum of {minimum_f1:.2f}."
    )

    print("Reference CV accuracies:", cv_reference)
    print("Reference validation F1 scores:", f1_reference)
    print(results.to_string(index=False))
    print("PASS: All Part C tuning validations passed.")


def validate_part_c_thresholds():
    reference = pd.read_csv(REFERENCES / "part_c_reference_metrics.csv")
    thresholds = pd.read_csv(OUTPUTS / "part_c_thresholds.csv")
    final_config = pd.read_csv(OUTPUTS / "part_c_final_config.csv")

    # Read reference values and pass bands from the reference CSV.
    def get_reference(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Reference"
        ]
        assert len(row) == 1, (
            f"Reference item missing or duplicated: {item}"
        )
        return row.iloc[0]

    def get_pass_band(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Pass band"
        ]
        assert len(row) == 1, (
            f"Pass band missing or duplicated: {item}"
        )
        return row.iloc[0]

    # Validate output structure.
    expected_columns = [
        "threshold", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1"
    ]

    assert list(thresholds.columns) == expected_columns
    assert len(thresholds) == 9
    assert len(final_config) == 1

    # Thresholds must be exact decimal values from 0.1 to 0.9.
    expected_thresholds = [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9
    ]

    assert thresholds["threshold"].round(1).tolist() == (
        expected_thresholds
    )

    print("PASS: Threshold table contains the nine required thresholds.")

    # Validate confusion-matrix metrics.
    count_columns = ["TN", "FP", "FN", "TP"]
    assert (thresholds[count_columns] >= 0).all().all()
    assert (thresholds[count_columns].sum(axis=1) == 180).all()

    expected_accuracy = (
        thresholds["TN"] + thresholds["TP"]
    ) / 180

    precision_denominator = thresholds["TP"] + thresholds["FP"]
    recall_denominator = thresholds["TP"] + thresholds["FN"]

    expected_precision = (
        thresholds["TP"] / precision_denominator
    ).where(precision_denominator > 0, 0)

    expected_recall = (
        thresholds["TP"] / recall_denominator
    ).where(recall_denominator > 0, 0)

    expected_f1 = (
        2 * expected_precision * expected_recall
        / (expected_precision + expected_recall)
    ).where(
        (expected_precision + expected_recall) > 0,
        0
    )

    for column, expected_values in {
        "accuracy": expected_accuracy,
        "precision": expected_precision,
        "recall": expected_recall,
        "f1": expected_f1
    }.items():
        assert (
            (thresholds[column] - expected_values).abs() <= 0.001
        ).all(), (
            f"{column} is inconsistent with the confusion matrix."
        )

    print("PASS: Threshold metrics match their confusion matrices.")

    # Recall must not increase as the threshold increases.
    assert thresholds["recall"].diff().dropna().le(0.001).all(), (
        "Recall increases as the threshold rises."
    )

    print("PASS: Recall does not increase as the threshold rises.")

    # Apply the operations rule from the project instructions:
    # choose the highest threshold with recall >= 0.80;
    # if none qualifies, choose 0.1.
    qualifying = thresholds.loc[thresholds["recall"] >= 0.80]

    expected_selected_threshold = (
        float(qualifying["threshold"].max())
        if not qualifying.empty
        else 0.1
    )

    actual_selected_threshold = float(
        final_config.iloc[0]["threshold"]
    )

    assert abs(
        actual_selected_threshold - expected_selected_threshold
    ) <= 0.001, (
        f"Selected threshold should be {expected_selected_threshold:.1f}, "
        f"got {actual_selected_threshold:.1f}."
    )

    print(
        "PASS: Selected threshold follows the recall >= 0.80 rule."
    )

    # The reference CSV provides a reference threshold and test metrics.
    # The threshold may differ if the implementation is valid, so check
    # consistency with the generated validation table rather than equality
    # to the reference value.
    reference_item = "Chosen threshold; test precision / recall / F1"
    reference_value = get_reference(reference_item)
    pass_band = get_pass_band(reference_item)

    print(f"Reference: {reference_value}")
    print(f"Acceptance criteria: {pass_band}")

    print("PASS: All Part C threshold validations passed.")

def validate_part_c_final_test():
    reference = pd.read_csv(
        REFERENCES / "part_c_reference_metrics.csv"
    )
    final_test = pd.read_csv(
        OUTPUTS / "part_c_final_test.csv"
    )
    final_config = pd.read_csv(
        OUTPUTS / "part_c_final_config.csv"
    )

    def get_reference(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Reference"
        ]

        assert len(row) == 1, (
            f"Reference item missing or duplicated: {item}"
        )

        return str(row.iloc[0]).strip()

    def get_pass_band(item):
        row = reference.loc[
            reference["Item"].str.strip() == item,
            "Pass band"
        ]

        assert len(row) == 1, (
            f"Pass band missing or duplicated: {item}"
        )

        return str(row.iloc[0]).strip()

    expected_columns = [
        "threshold", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1"
    ]

    assert list(final_test.columns) == expected_columns, (
        f"Unexpected columns in part_c_final_test.csv: "
        f"{list(final_test.columns)}"
    )

    assert len(final_test) == 1, (
        "part_c_final_test.csv must contain exactly one row."
    )

    assert len(final_config) == 1, (
        "part_c_final_config.csv must contain exactly one row."
    )

    test_row = final_test.iloc[0]
    config_row = final_config.iloc[0]

    # Validate that confusion-matrix counts are non-negative integers.
    count_columns = ["TN", "FP", "FN", "TP"]

    for column in count_columns:
        value = test_row[column]

        assert pd.notna(value) and float(value).is_integer(), (
            f"{column} must be a valid integer."
        )
        assert int(value) >= 0, (
            f"{column} cannot be negative."
        )

    tn = int(test_row["TN"])
    fp = int(test_row["FP"])
    fn = int(test_row["FN"])
    tp = int(test_row["TP"])

    # The test set must contain 180 observations.
    total = tn + fp + fn + tp

    assert total == 180, (
        f"Expected 180 test observations, got {total}."
    )

    # The reference specifies 41 delayed parcels in the test set.
    assert fn + tp == 41, (
        f"Expected 41 delayed parcels in the test set, "
        f"got {fn + tp}."
    )

    # The threshold used for final testing must match the saved config.
    assert pd.notna(test_row["threshold"]), (
        "The final test threshold is missing."
    )
    assert pd.notna(config_row["threshold"]), (
        "The configured threshold is missing."
    )

    assert abs(
        float(test_row["threshold"]) -
        float(config_row["threshold"])
    ) <= 0.001, (
        "The final test threshold does not match "
        "part_c_final_config.csv."
    )

    # Recalculate metrics from the confusion matrix.
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    calculated_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Stored metrics must agree with the confusion matrix.
    for metric, calculated_value in calculated_metrics.items():
        stored_value = test_row[metric]

        assert pd.notna(stored_value), (
            f"Stored {metric} is missing."
        )

        assert abs(
            float(stored_value) - calculated_value
        ) <= 0.001, (
            f"{metric} mismatch: stored={stored_value:.4f}, "
            f"calculated={calculated_value:.4f}."
        )

    # Read the final-test metric requirements from the reference CSV.
    reference_item = (
        "Chosen threshold; test precision / recall / F1"
    )
    reference_values = get_reference(reference_item)
    pass_band = get_pass_band(reference_item)

    f1_match = re.search(
        r"F1\s*≥\s*(\d+(?:\.\d+)?)",
        pass_band,
        flags=re.IGNORECASE
    )
    recall_match = re.search(
        r"recall\s*≥\s*(\d+(?:\.\d+)?)",
        pass_band,
        flags=re.IGNORECASE
    )
    precision_match = re.search(
        r"precision\s*≥\s*(\d+(?:\.\d+)?)",
        pass_band,
        flags=re.IGNORECASE
    )

    assert f1_match and recall_match and precision_match, (
        f"Could not parse final-test pass band: {pass_band}"
    )

    min_f1 = float(f1_match.group(1))
    min_recall = float(recall_match.group(1))
    min_precision = float(precision_match.group(1))

    assert f1 >= min_f1, (
        f"Test F1 below required minimum: "
        f"{f1:.3f} < {min_f1:.3f}."
    )
    assert recall >= min_recall, (
        f"Test recall below required minimum: "
        f"{recall:.3f} < {min_recall:.3f}."
    )
    assert precision >= min_precision, (
        f"Test precision below required minimum: "
        f"{precision:.3f} < {min_precision:.3f}."
    )

    print("PASS: Final test output has the expected structure.")
    print("PASS: Test set contains 180 observations and 41 delayed parcels.")
    print("PASS: Final test threshold matches the saved configuration.")
    print("PASS: Stored metrics match the confusion matrix.")
    print("PASS: Test precision, recall, and F1 meet the reference pass bands.")
    print(f"Reference values: {reference_values}")
    print(f"Pass band: {pass_band}")
    print(f"Threshold: {float(test_row['threshold']):.2f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

if __name__ == "__main__":
    validate_part_c_split()
    validate_part_c_models()
    validate_part_c_feature_scores()
    validate_part_c_selectk()
    validate_part_c_tuning()
    validate_part_c_thresholds()
    validate_part_c_final_test()