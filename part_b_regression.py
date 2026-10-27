import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES

def split_train_val_test_data():

    # Load the modelling table
    df = pd.read_csv("data/modelling_table.csv")

    # Select features in the pinned order and target
    X = df[FEATURES]
    y = df["delivery_hours"]

    # First split: 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # Second split: divide the temporary data equally
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def perform_scaling(X_train, X_test):
    # Scale features using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate the largest absolute column mean
    train_scaled_max_abs_mean = abs(
        X_train_scaled.mean(axis=0)
    ).max()

    test_scaled_max_abs_mean = abs(
        X_test_scaled.mean(axis=0)
    ).max()

    # Prepare validation checks
    part_b_checks = pd.DataFrame({
        "metric": [
            "train_rows",
            "val_rows",
            "test_rows",
            "train_scaled_max_abs_mean",
            "test_scaled_max_abs_mean",
        ],
        "value": [
            len(X_train),
            len(X_val),
            len(X_test),
            train_scaled_max_abs_mean,
            test_scaled_max_abs_mean,
        ]
    })

    return part_b_checks

def predict_baseline(y_train, y_val):
    # Predict the training target mean for every validation row
    baseline_prediction = np.full(
        shape=len(y_val),
        fill_value=y_train.mean()
    )

    return baseline_prediction

def evaluate_tree_depths(
        X_train,
        X_val,
        y_train,
        y_val
):
    # Candidate tree depths; None means unlimited depth
    depths = [1, 2, 3, 4, 6, 8, 12, None]

    tree_depth_results = []

    for depth in depths:
        tree = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )

        # Fit on training rows only
        tree.fit(X_train, y_train)

        # Predict on training and validation sets
        train_pred = tree.predict(X_train)
        val_pred = tree.predict(X_val)

        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        tree_depth_results.append({
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        })

    # Save results
    tree_depth_df = pd.DataFrame(tree_depth_results)

    # Find the row with the lowest validation RMSE
    best_tree_row = tree_depth_df.loc[
        tree_depth_df["val_rmse"].idxmin()
    ]

    best_depth = (
        None if best_tree_row["max_depth"] == "unlimited"
        else int(best_tree_row["max_depth"])
    )

    #print(f"Best tree depth - Validation metrics: {best_depth}")
    return tree_depth_df, best_depth

def evaluate_learning_rate(
        X_train,
        X_val,
        y_train,
        y_val
):
    # Learning-rate experiment
    learning_rates = [0.01, 0.1, 1.0]
    gb_results = []

    for lr in learning_rates:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=lr,
            random_state=42
        )

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)

        gb_results.append({
            "learning_rate": lr,
            "val_rmse": val_rmse,
            "val_r2": val_r2
        })

    # Save results
    gb_learning_rate_df = pd.DataFrame(gb_results)

    return gb_learning_rate_df

def compare_regressors(
        X_train,
        X_val,
        y_train,
        y_val,
        baseline_prediction,
        best_depth
):
    # Define candidate models
    models = {
        "baseline": None,
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "decision_tree": DecisionTreeRegressor(
            max_depth=best_depth,
            random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    metrics_rows = []

    # Fit and evaluate each candidate
    for name, model in models.items():
        if name == "baseline":
            val_pred = baseline_prediction
        else:
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, val_pred)

        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": mean_absolute_error(y_val, val_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_val, val_pred)
        })

    # Create and save the metrics table
    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
    )

    return models, metrics_df

def cross_validate_top_models(
        X_train,
        X_val,
        y_train,
        y_val,
        models,
        metrics_df
):
    # Combine training and validation data (1,020 rows)
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    # Rank the four candidate models by validation RMSE
    # Exclude baseline model from further analysis
    candidate_metrics = metrics_df.loc[
        metrics_df["model"] != "baseline"
        ].sort_values("RMSE", ascending=True)

    top_two_models = candidate_metrics["model"].head(2).tolist()

    # Create one shared KFold object
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Cross-validate the two best candidates using default R² scoring
    cv_results = []

    for name in top_two_models:
        scores = cross_val_score(
            models[name],
            X_train_val,
            y_train_val,
            cv=kfold
        )

        cv_results.append({
            "model": name,
            "mean_r2": scores.mean(),
            "std_r2": scores.std()
        })

    # Save cross-validation results
    cv_df = pd.DataFrame(
        cv_results,
        columns=["model", "mean_r2", "std_r2"]
    )

    best_model_name = candidate_metrics.iloc[0]["model"]

    return cv_df, best_model_name

def test_best_model(
        X_test,
        y_test,
        models,
        validation_metrics_df,
        best_model_name
):

    final_model = models[best_model_name]

    #As per standard practice, final model should be retrained on train + val sets
    #However, as mentioned in the assignment, this step needs to be avoided.
    #final_model.fit(X_train_val, y_train_val)

    # Evaluate the selected model on the test set exactly once
    test_pred = final_model.predict(X_test)

    test_mse = mean_squared_error(y_test, test_pred)

    test_metrics = {
        "model": best_model_name,
        "split": "test",
        "MAE": mean_absolute_error(y_test, test_pred),
        "MSE": test_mse,
        "RMSE": np.sqrt(test_mse),
        "R2": r2_score(y_test, test_pred)
    }

    # Add the final-test row to the existing validation metrics
    metrics_df = pd.concat(
        [validation_metrics_df, pd.DataFrame([test_metrics])],
        ignore_index=True
    )

    return metrics_df

def save_part_b_results(
        part_b_checks,
        tree_depth_df,
        gb_learning_rate_df,
        cv_df,
        metrics_df
):
    part_b_checks.to_csv(
        "outputs/part_b_checks.csv",
        index=False,
        float_format="%.4f"
    )

    print("Part b checks saved in outputs/part_b_checks.csv")

    tree_depth_df.to_csv(
        "outputs/part_b_tree_depth.csv",
        index=False,
        float_format="%.4f"
    )

    print("Tree depth diagnostics saved in outputs/part_b_tree_depth.csv")

    gb_learning_rate_df.to_csv(
        "outputs/part_b_gb_learning_rate.csv",
        index=False,
        float_format="%.4f"
    )

    print("Gradient boost learning rate experiment results saved in outputs/part_b_gb_learning_rate.csv")

    cv_df.to_csv(
        "outputs/part_b_cv.csv",
        index=False,
        float_format="%.4f"
    )

    print("Cross-validation metrics for best two models saved in outputs/part_b_cv.csv")

    metrics_df.to_csv(
        "outputs/part_b_metrics.csv",
        index=False,
        float_format="%.4f"
    )

    print("Performance metrics for candidate models saved in outputs/part_b_metrics.csv")

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_data()

    part_b_checks = perform_scaling(X_train, X_test)

    baseline_prediction = predict_baseline(y_train, y_val)

    tree_depth_df, best_tree_depth = evaluate_tree_depths(X_train, X_val, y_train, y_val)

    gb_learning_rate_df = evaluate_learning_rate(X_train, X_val, y_train, y_val)

    models, validation_metrics_df = compare_regressors(
        X_train,
        X_val,
        y_train,
        y_val,
        baseline_prediction,
        best_tree_depth
    )

    cv_df, best_model_name = cross_validate_top_models(
        X_train,
        X_val,
        y_train,
        y_val,
        models,
        validation_metrics_df
    )

    metrics_df = test_best_model(
        X_test,
        y_test,
        models,
        validation_metrics_df,
        best_model_name
    )

    save_part_b_results(
        part_b_checks,
        tree_depth_df,
        gb_learning_rate_df,
        cv_df,
        metrics_df
    )