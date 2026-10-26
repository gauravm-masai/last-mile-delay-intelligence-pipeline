import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features

# Generate metrics before cleanup
def generate_metrics_before_cleanup(raw):
    before = raw.drop_duplicates().copy()

    metrics = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(before),
        "rows_after_dedup": len(before),
        "missing_distance_km_before": before["distance_km"].isna().sum(),
        "missing_parcel_weight_before": before["parcel_weight_kg"].isna().sum(),
        "sentinel_weight_rows": (before["parcel_weight_kg"] > 100).sum(),
        "missing_declared_value_before": before["declared_value"].isna().sum(),
        "missing_payment_mode_before": before["payment_mode"].isna().sum(),
        "service_type_spellings_before": before["service_type"].nunique(),
    }

    return metrics

# Clean data and create the modelling table
def clean_data(raw, hubs):
    clean, medians = clean_shipments(raw)
    modelling_table = build_features(clean, hubs)

    modelling_table["is_delayed"] = np.where(
        modelling_table["delivery_hours"] > modelling_table["promised_hours"],
        1,
        0,
    )

    modelling_table.to_csv(
        "data/modelling_table.csv",
        index=False,
        float_format="%.4f"
    )

    print("Modelling table saved to data/modelling_table.csv")

    return clean, medians, modelling_table

# Generate metrics after cleanup
def generate_metrics_after_cleanup(
    clean, medians, modelling_table, before_metrics
):
    metrics = before_metrics.copy()

    metrics.update({
        "service_type_spellings_after": clean["service_type"].nunique(),
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": len(modelling_table),
        "nulls_in_features_and_target": int(
            modelling_table[
                FEATURES + ["delivery_hours", "is_delayed"]
            ].isna().sum().sum()
        ),
        "delayed_count": int(modelling_table["is_delayed"].sum()),
        "delay_rate": modelling_table["is_delayed"].mean(),
    })

    # These four metrics use the median-filled modelling table.
    declared_values = modelling_table["declared_value_inr"]
    log_values = modelling_table["log_declared_value"]

    metrics.update({
        "declared_value_mean_inr": declared_values.mean(),
        "declared_value_median_inr": declared_values.median(),
        "log_declared_value_mean": log_values.mean(),
        "log_declared_value_median": log_values.median(),
    })

    # Use unrounded quartiles and bounds for the IQR calculation.
    hours = modelling_table["delivery_hours"]
    q1 = hours.quantile(0.25)
    q3 = hours.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    metrics.update({
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_lower_bound": lower_bound,
        "hours_upper_bound": upper_bound,
        "hours_outlier_count": int(
            ((hours < lower_bound) | (hours > upper_bound)).sum()
        ),
    })

    return metrics

# Function 4: Save actual metrics reports
def save_part_a_metrics(metrics):
    actual = pd.DataFrame(
        [{"metric": name, "value": value}
         for name, value in metrics.items()]
    )

    # Required checks file: exactly two columns.
    actual.to_csv(
        "outputs/part_a_checks.csv",
        index=False,
        float_format="%.4f",
    )

def generate_delay_evidence(modelling_table):
    # Define the required distance bands.
    distance_bins = [0, 200, 400, 600, 2000]
    distance_labels = ["0-200", "201-400", "401-600", "601+"]

    df = modelling_table.copy()

    # Right-inclusive bins: (0, 200], (200, 400], etc.
    df["distance_band"] = pd.cut(
        df["distance_km"],
        bins=distance_bins,
        labels=distance_labels,
        right=True,
        include_lowest=True,
    )

    evidence = []

    # Generate metrics for each required dimension.
    dimensions = [
        ("service_type", "service_type"),
        ("hub_tier", "hub_tier"),
        ("distance_band", "distance_band"),
    ]

    for dimension, column in dimensions:
        grouped = df.groupby(column, observed=True, sort=True)

        for group, group_df in grouped:
            evidence.append({
                "dimension": dimension,
                "group": group,
                "shipments": len(group_df),
                "delayed": int(group_df["is_delayed"].sum()),
                "delay_rate": group_df["is_delayed"].mean(),
                "mean_hours": group_df["delivery_hours"].mean(),
            })

    result = pd.DataFrame(
        evidence,
        columns=[
            "dimension",
            "group",
            "shipments",
            "delayed",
            "delay_rate",
            "mean_hours",
        ],
    )

    result.to_csv(
        "outputs/part_a_delay_by_group.csv",
        index=False,
        float_format="%.4f",
    )

    print("Delay evidence saved to outputs/part_a_delay_by_group.csv")

    return result

def generate_correlations(modelling_table):
    numeric_df = modelling_table.select_dtypes(include="number")

    correlations = numeric_df.corr()

    correlations.to_csv(
        "outputs/part_a_correlations.csv",
        index=True,
        index_label="",
        float_format="%.4f",
    )

    print("Correlation matrix saved to outputs/part_a_correlations.csv")

    return correlations

if __name__ == "__main__":
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    before_metrics = generate_metrics_before_cleanup(raw)

    clean, medians, modelling_table = clean_data(raw, hubs)

    after_metrics = generate_metrics_after_cleanup(
        clean,
        medians,
        modelling_table,
        before_metrics
    )

    # Generate the required Part A checks file.
    save_part_a_metrics(after_metrics)

    # Generate the delay evidence and correlation outputs.
    generate_delay_evidence(modelling_table)
    generate_correlations(modelling_table)