from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REFERENCES = ROOT / "references"
REPORTS = ROOT / "validations/reports"

def validate_metrics():
    actual = pd.read_csv(OUTPUTS / "part_a_checks.csv")
    reference = pd.read_csv(
        REFERENCES / "part_a_reference_metrics.csv"
    )

    if reference["metric"].duplicated().any():
        raise ValueError("The reference file contains duplicate metric names.")

    actual_values = actual.set_index("metric")["value"].to_dict()
    reference_values = reference.set_index("metric")["value"].to_dict()
    tolerances = reference.set_index("metric")["tolerance"].to_dict()

    report_rows = []

    # Compare every reference metric, including missing actual metrics.
    for metric, expected in reference_values.items():
        if metric not in actual_values:
            observed = np.nan
            tolerance = tolerances[metric]
            difference = np.nan
            status = "MISMATCH"
        else:
            observed = actual_values[metric]
            tolerance = tolerances[metric]
            difference = abs(observed - expected)
            status = (
                "PASS" if difference <= tolerance else "MISMATCH"
            )

        report_rows.append({
            "metric": metric,
            "actual_value": observed,
            "reference_value": expected,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "status": status,
        })

    # Report unexpected generated metrics too.
    for metric, observed in actual_values.items():
        if metric not in reference_values:
            report_rows.append({
                "metric": metric,
                "actual_value": observed,
                "reference_value": np.nan,
                "absolute_difference": np.nan,
                "tolerance": np.nan,
                "status": "UNEXPECTED",
            })

    report = pd.DataFrame(report_rows)
    report.to_csv(
        REPORTS / "part_a_validation_report.csv",
        index=False,
        float_format="%.4f",
    )

    mismatches = report[report["status"] != "PASS"]

    if mismatches.empty:
        print("All Part A metrics match the reference data")
    else:
        print("Part A metric differences found:")
        print(
            mismatches[
                [
                    "metric",
                    "actual_value",
                    "reference_value",
                    "absolute_difference",
                    "tolerance",
                    "status",
                ]
            ].to_string(index=False)
        )

        print(
            "Review part_a_validation_report.csv for the complete comparison"
        )

def validate_delay_evidence():
    actual = pd.read_csv(OUTPUTS / "part_a_delay_by_group.csv")
    reference = pd.read_csv(
        REFERENCES / "part_a_delay_by_group_reference.csv"
    )

    keys = ["dimension", "group"]

    # Normalize group labels so numeric hub tiers match consistently.
    for frame in (actual, reference):
        frame["dimension"] = frame["dimension"].astype(str).str.strip()
        frame["group"] = frame["group"].astype(str).str.strip()

    if actual.duplicated(keys).any():
        raise ValueError("Computed delay evidence has duplicate groups.")

    if reference.duplicated(keys).any():
        raise ValueError("Reference delay evidence has duplicate groups.")

    comparison = actual.merge(
        reference,
        on=keys,
        how="outer",
        suffixes=("_actual", "_reference"),
        indicator=True,
        validate="one_to_one",
    )

    comparison["counts_match"] = (
        comparison["shipments_actual"].eq(
            comparison["shipments_reference"]
        )
        & comparison["delayed_actual"].eq(
            comparison["delayed_reference"]
        )
    )

    comparison["rate_matches"] = (
        (
            comparison["delay_rate_actual"]
            - comparison["delay_rate_reference"]
        ).abs() <= 0.00005
    )

    comparison["status"] = "PASS"
    comparison.loc[
        (comparison["_merge"] != "both")
        | ~comparison["counts_match"]
        | ~comparison["rate_matches"],
        "status",
    ] = "FAIL"

    comparison.to_csv(
        REPORTS / "part_a_delay_validation_report.csv",
        index=False,
        float_format="%.4f",
    )

    #print(comparison[keys + ["status"]].to_string(index=False))

    if not comparison["status"].eq("PASS").all():
        raise ValueError(
            "Delay evidence validation failed. "
            "Check part_a_delay_validation_report.csv for the complete comparison"
        )
    else:
        print("Delay evidence matches the reference data")

def validate_correlations():
    correlations = pd.read_csv(
        OUTPUTS / "part_a_correlations.csv",
        index_col=0,
    )

    reference_checks = [
        ("delivery_hours", "promised_hours", 0.837),
        ("delivery_hours", "distance_km", 0.401),
        ("is_delayed", "distance_km", 0.546),
    ]

    rows = []

    for column_1, column_2, expected in reference_checks:
        actual = correlations.loc[column_1, column_2]
        difference = abs(actual - expected)

        rows.append({
            "column_1": column_1,
            "column_2": column_2,
            "actual_correlation": actual,
            "reference_correlation": expected,
            "absolute_difference": difference,
            "tolerance": 0.01,
            "status": "PASS" if difference <= 0.01 else "FAIL",
        })

    report = pd.DataFrame(rows)

    report.to_csv(
        REPORTS / "part_a_correlation_validation_report.csv",
        index=False,
        float_format="%.4f",
    )

    #print(report.to_string(index=False))

    if not report["status"].eq("PASS").all():
        raise ValueError(
            "Correlation validation failed. "
            "Check part_a_correlation_validation_report.csv for the complete comparison"
        )
    else:
        print("Correlations match the reference data")

if __name__ == "__main__":
    validate_metrics()
    validate_delay_evidence()
    validate_correlations()