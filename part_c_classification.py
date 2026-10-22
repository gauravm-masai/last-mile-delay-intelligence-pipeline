import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS

# Load the modelling table
df = pd.read_csv("data/modelling_table.csv")

# Select features and target
X = df[FEATURES]
y = df["is_delayed"]

# Split into training (70%) and temporary (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Split the temporary data equally into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Validate split sizes and delayed counts
#print("Training rows:", len(X_train))
#print("Validation rows:", len(X_val))
#print("Test rows:", len(X_test))

#print("\nDelayed parcels:")
#print("Training:", int(y_train.sum()))
#print("Validation:", int(y_val.sum()))
#print("Test:", int(y_test.sum()))

def select_best_tree_depth():
    depths = [2, 3, 4, 5, 6, 8, 10]

    best_depth = None
    best_f1 = -1

    for depth in depths:
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        score = f1_score(y_val, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_depth = depth

    return best_depth

def evaluate_candidate_models():
    best_depth = select_best_tree_depth()

    models = {
        "blind_all_on_time": None,
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]),
        f"decision_tree_depth_{best_depth}": DecisionTreeClassifier(
            criterion="gini",
            max_depth=best_depth,
            random_state=42
        ),
        "random_forest_default": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []

    for model_name, model in models.items():

        if model is None:
            y_pred = [0] * len(y_val)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ).ravel()

        results.append({
            "model": model_name,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_val, y_pred, zero_division=0), 4)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "outputs/part_c_models.csv",
        index=False,
        float_format="%.4f"
    )

    print("Blind baseline and three classifiers evaluated and results saved to outputs/part_c_models.csv")

    return models, results_df

def analyze_feature_signal():
    # 1. Fit selector on training data
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_train, y_train)

    # 2. Transform training and validation data
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)

    # 3. Fit Random Forest on the transformed training data
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train_selected, y_train)

    # 4. Get F-scores and Random Forest feature importances
    feature_scores = pd.DataFrame({
        "feature": X_train.columns,
        "f_score": selector.scores_,
        "rf_importance": rf_model.feature_importances_
    })

    # 5. Rank features: rank 1 is the highest F-score
    feature_scores["f_rank"] = (
        feature_scores["f_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # 6. Identify noise columns
    feature_scores["is_noise"] = (
        feature_scores["feature"].isin(NOISE_COLUMNS)
    )

    # 7. Arrange columns and round decimal values
    feature_scores = feature_scores[
        ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
    ]

    feature_scores.to_csv(
        "outputs/part_c_feature_scores.csv",
        index=False,
        float_format="%.4f"
    )

    print("Feature signal evaluated and results saved to outputs/part_c_feature_scores.csv")

def evaluate_selectk():
    """Compare validation F1 using different numbers of selected features."""

    k_values = [5, 7, 10, 12, 17]
    results = []

    for k in k_values:
        model = Pipeline([
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        val_f1 = f1_score(
            y_val,
            y_pred,
            zero_division=0
        )

        results.append({
            "k": k,
            "val_f1": round(val_f1, 4)
        })

    selectk_results = pd.DataFrame(
        results,
        columns=["k", "val_f1"]
    )

    selectk_results.to_csv(
        "outputs/part_c_selectk.csv",
        index=False,
        float_format="%.4f"
    )

    print("Select K values evaluated and results saved to outputs/part_c_selectk.csv")

def tune_random_forest():
    # 1. Fit feature selector on training data only
    selector = SelectKBest(score_func=f_classif, k=7)
    selector.fit(X_train, y_train)

    # 2. Transform training and validation data
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)

    # 3. Define hyperparameter search space
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }

    # 4. Define candidate searches
    searches = {
        "grid_search": GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5
        ),
        "randomized_search": RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_grid,
            n_iter=9,
            cv=5,
            random_state=42
        )
    }

    results = []
    best_model = None
    best_f1 = -1
    chosen_from = None
    chosen_params = None

    # 5. Run each search and evaluate on validation data
    for search_name, search in searches.items():

        search.fit(X_train_selected, y_train)

        search_best_model = search.best_estimator_
        val_predictions = search_best_model.predict(X_val_selected)

        val_f1 = f1_score(y_val, val_predictions, zero_division=0)

        if val_f1 > best_f1:
            best_model = search_best_model
            best_f1 = val_f1
            chosen_from = search_name
            chosen_params = search.best_params_

        candidates = (len(search.cv_results_["params"]))

        results.append({
            "search": search_name,
            "candidates": candidates,
            "cv_folds": 5,
            "total_fits": candidates * 5,
            "best_params": str(search.best_params_),
            "best_cv_accuracy": round(search.best_score_, 4),
            "val_f1": round(val_f1, 4)
        })

    # 6. Evaluate the untuned forest on the same seven features
    default_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1
    )

    default_model.fit(X_train_selected, y_train)

    default_predictions = default_model.predict(X_val_selected)

    default_f1 = f1_score(y_val, default_predictions, zero_division=0)

    if default_f1 >= best_f1:
        best_model = default_model
        best_f1 = default_f1
        chosen_from = "default_forest_same_7_features"
        chosen_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1
        }

    results.append({
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": "",
        "val_f1": round(default_f1, 4)
    })

    # 7. Save results
    columns = [
        "search",
        "candidates",
        "cv_folds",
        "total_fits",
        "best_params",
        "best_cv_accuracy",
        "val_f1"
    ]

    results_df = pd.DataFrame(results, columns=columns)

    results_df.to_csv(
        "outputs/part_c_tuning.csv",
        index=False,
        float_format="%.4f"
    )

    print("Results for tuning Random Forest saved to outputs/part_c_tuning.csv")

    # 8. Fit the selected configuration on the original training data
    final_model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        ("model", RandomForestClassifier(
            random_state=42,
            **chosen_params
        ))
    ])

    final_model.fit(X_train, y_train)

    return final_model, chosen_from, chosen_params

def select_alert_threshold(final_model):
    # 1. Get validation probabilities for the positive class
    probabilities = final_model.predict_proba(X_val)[:, 1]

    # 2. Define exact threshold values
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    alert_threshold = 0.1

    # 3. Evaluate each threshold
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ).ravel()

        recall = recall_score(y_val, y_pred, zero_division=0)

        results.append({
            "threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
            "precision": round(
                precision_score(y_val, y_pred, zero_division=0), 4
            ),
            "recall": round(recall, 4),
            "f1": round(f1_score(y_val, y_pred, zero_division=0), 4)
        })

        # 4. Keep the highest threshold meeting the recall requirement
        if recall >= 0.80:
            alert_threshold = threshold

    # 5. Save threshold comparison results
    columns = [
        "threshold", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1"
    ]

    results_df = pd.DataFrame(results, columns=columns)

    results_df.to_csv(
        "outputs/part_c_thresholds.csv",
        index=False,
        float_format="%.4f"
    )

    print("Threshold results saved to outputs/part_c_thresholds.csv")
    #print(f"Selected alert threshold: {alert_threshold}")

    return alert_threshold

def evaluate_final_test(final_model, chosen_from, chosen_params, threshold):
    # 1. Predict test probabilities exactly once
    probabilities = final_model.predict_proba(X_test)[:, 1]

    # 2. Apply the selected threshold
    y_pred = (probabilities >= threshold).astype(int)

    # 3. Calculate test metrics
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    results = pd.DataFrame([{
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4)
    }])

    # 4. Save final test metrics
    results.to_csv(
        "outputs/part_c_final_test.csv",
        index=False,
        float_format="%.4f"
    )

    # 5. Save the selected model configuration
    config = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_params["n_estimators"],
        "max_depth": (
        "unlimited"
        if chosen_params["max_depth"] is None
        else chosen_params["max_depth"]
        ),
        "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": threshold,
        "chosen_from": chosen_from
    }])

    config.to_csv(
        "outputs/part_c_final_config.csv",
        index=False,
        float_format="%.4f"
    )

    print("Final test results saved to outputs/part_c_final_test.csv")
    print("Final configuration saved to outputs/part_c_final_config.csv")

if __name__ == "__main__":
    models, results_df = evaluate_candidate_models()

    analyze_feature_signal()
    evaluate_selectk()

    final_model, chosen_from, chosen_params = tune_random_forest()

    alert_threshold = select_alert_threshold(final_model)

    evaluate_final_test(final_model, chosen_from, chosen_params, alert_threshold)
