import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

def clean_new_shipments(new_shipments, medians):

    # Clean new shipments using saved medians
    clean, _ = clean_shipments(new_shipments, medians=medians)

    return clean

def load_features(hubs, clean):
    # Build features
    features_df = build_features(clean, hubs)

    # Arrange features in the exact saved order
    X_new = features_df[FEATURES]

    return X_new

def predict_delay(new_shipments, hub_segments, model, X_new, threshold):
    # Predict positive-class probabilities
    probabilities = model.predict_proba(X_new)[:, 1]

    # Apply the stored threshold
    flagged = (probabilities >= threshold).astype(int)

    # Attach hub segment
    results = new_shipments[["shipment_id", "hub_id"]].copy()

    results["delay_probability"] = probabilities
    results["flagged"] = flagged

    results = results.merge(
        hub_segments[["hub_id", "segment"]],
        on="hub_id",
        how="left",
        sort=False,
        validate="many_to_one"
    )

    return results

def assign_action(predictions):

    # Assign action
    predictions["action"] = np.select(
        [
            (predictions["flagged"] == 1)
            & (predictions["segment"] == "Escalate"),

            predictions["flagged"] == 1
        ],
        [
            "Escalate to hub manager",
            "Proactive customer call"
        ],
        default="No action"
    )

    return predictions

def save_part_d_results(actions):
    # Save output in the required format
    actions = actions[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action"
        ]
    ]

    actions.to_csv(
        "outputs/new_shipment_actions.csv",
        index=False,
        float_format="%.4f"
    )

    print("Saved shipment actions to outputs/new_shipment_actions.csv")

if __name__ == "__main__":
    # Load input data and saved model files
    new_shipments = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    model = joblib.load("models/delay_model.pkl")

    threshold_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(threshold_df["threshold"].iloc[0])

    feature_order_df = pd.read_csv("models/feature_order.csv")
    FEATURES = feature_order_df["feature"].tolist()

    impute_df = pd.read_csv("models/impute_values.csv")
    # Convert the saved median table into a dictionary
    medians = dict(
        zip(impute_df["column"], impute_df["median"])
    )

    clean_df = clean_new_shipments(new_shipments, medians)

    X_new = load_features(hubs, clean_df)

    predictions = predict_delay(clean_df, hub_segments, model, X_new, threshold)

    actions = assign_action(predictions)

    save_part_d_results(actions)