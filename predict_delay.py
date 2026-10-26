import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

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

# Clean new shipments using saved medians
clean, _ = clean_shipments(new_shipments, medians=medians)

# Build features
features_df = build_features(clean, hubs)

# Arrange features in the exact saved order
X_new = features_df[FEATURES]

# Predict positive-class probabilities
probabilities = model.predict_proba(X_new)[:, 1]

# Round probabilities to four decimal places for output.
rounded_probabilities = np.round(probabilities, 4)

# Apply the stored threshold to the unrounded probabilities.
flagged = (probabilities >= threshold).astype(int)

# Attach hub segment
results = new_shipments[["shipment_id", "hub_id"]].copy()

results["delay_probability"] = rounded_probabilities
results["flagged"] = flagged

results = results.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    sort=False,
    validate="many_to_one"
)

# Assign action
results["action"] = np.select(
    [
        (results["flagged"] == 1)
        & (results["segment"] == "Escalate"),

        results["flagged"] == 1
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call"
    ],
    default="No action"
)

# Save output in the required format
results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]

results.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False,
    float_format="%.4f"
    #lineterminator="\n"
)

print("Saved shipment actions to outputs/new_shipment_actions.csv")
