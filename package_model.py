
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments

def package_model():

    # Load the modelling table, model config, and raw data (for three medians)
    df = pd.read_csv("data/modelling_table.csv")
    config_df = pd.read_csv("outputs/part_c_final_config.csv")
    raw = pd.read_csv("data/raw_shipments.csv")

    selector_k = config_df["selector_k"].iloc[0]
    n_estimators = int(config_df["n_estimators"].iloc[0])
    max_depth_value = config_df["max_depth"].iloc[0]
    max_depth = None if max_depth_value == "unlimited" else int(max_depth_value)
    min_samples_leaf = int(config_df["min_samples_leaf"].iloc[0])
    threshold = float(config_df["threshold"].iloc[0])

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

    # Fit Random Forest with configurations from Part C on the training data
    final_model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        ))
    ])

    final_model.fit(X_train, y_train)

    # Save model and metadata
    joblib.dump(final_model, "models/delay_model.pkl")

    pd.DataFrame({
        "threshold": [threshold]
    }).to_csv(
        "models/delay_threshold.csv",
        index=False,
        float_format="%.4f"
    )

    pd.DataFrame({
        "feature": FEATURES
    }).to_csv(
        "models/feature_order.csv",
        index=False,
        float_format="%.4f"
    )

    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    clean, medians = clean_shipments(raw)

    pd.DataFrame({
        "column": median_columns,
        "median": [medians[col] for col in median_columns]
    }).to_csv(
        "models/impute_values.csv",
        index=False,
        float_format="%.4f"
    )

    reloaded_pipeline = joblib.load("models/delay_model.pkl")

    memory_probabilities = final_model.predict_proba(X_test)[:, 1]

    reloaded_probabilities = reloaded_pipeline.predict_proba(X_test)[:, 1]

    max_abs_diff = np.max(
        np.abs(memory_probabilities - reloaded_probabilities)
    )

    reload_check = pd.DataFrame({
        "metric": [
            "test_rows_scored",
            "reload_max_abs_probability_diff",
            "threshold"
        ],
        "value": [
            len(X_test),
            max_abs_diff,
            threshold
        ]
    })

    reload_check.to_csv(
        "outputs/part_d_reload_check.csv",
        index=False,
        float_format="%.4f"
    )

    print("Memory and reload checks saved in outputs/part_d_reload_check.csv")

if __name__ == "__main__":
    package_model()