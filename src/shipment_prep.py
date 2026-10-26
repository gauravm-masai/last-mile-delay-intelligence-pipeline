import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    """Clean shipment data and fill missing numeric values with medians."""

    # 1. Drop exact duplicate rows
    clean_df = raw.drop_duplicates().copy()

    # 2. Standardize service_type
    clean_df["service_type"] = (
        clean_df["service_type"]
        .str.strip()
        .str.title()
    )

    # 3. Replace missing payment_mode values
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

    # 4. Convert declared_value to numeric INR
    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    clean_df = clean_df.drop(columns=["declared_value"])

    # 5. Convert dispatch_time to datetime
    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"],
        errors="coerce",
    )

    # 6. Treat scanner-fault weights as missing
    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100,
        "parcel_weight_kg",
    ] = np.nan

    # 7. Calculate or reuse the medians
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    if medians is None:
        medians = {
            column: float(clean_df[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    # Ensure every median can be used for imputation
    for column in median_columns:
        if not np.isfinite(medians[column]):
            raise ValueError(
                f"Cannot impute {column}: its median is missing or non-finite."
            )

    # Fill missing values using the selected medians
    for column in median_columns:
        clean_df[column] = (
            clean_df[column]
            .astype(float)
            .fillna(medians[column])
        )

    return clean_df, medians

def build_features(clean, hubs):
    """Attach hub information and create shipment features."""

    # 1. Attach hub information using a left merge
    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    features_df = clean.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # 2. Map service types to promised delivery hours
    features_df["promised_hours"] = (
        features_df["service_type"]
        .map(PROMISED_HOURS)
        .astype("int")
    )

    # 3. Create COD indicator
    features_df["is_cod"] = np.where(
        features_df["payment_mode"] == "COD",1,0
    )

    # 4. Identify weekend dispatches (Saturday = 5, Sunday = 6)
    features_df["is_weekend"] = np.where(
        features_df["dispatch_time"].dt.dayofweek >= 5,1,0
    )

    # 5. Extract dispatch hour
    features_df["dispatch_hour"] = (
        features_df["dispatch_time"].dt.hour.astype("int")
    )

    # 6. Log-transform declared value
    features_df["log_declared_value"] = np.log(
        features_df["declared_value_inr"]
    )

    return features_df
