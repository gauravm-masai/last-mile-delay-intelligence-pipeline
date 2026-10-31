from pathlib import Path
import ast
import inspect
import sys

import numpy as np
import pandas as pd

from validations.validations_common_utility import (
    compare_metric,
    error_row,
    load_reference,
    make_report_row,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = ROOT / "references" / "part_a_reference_metrics.csv"
REPORT_PATH = ROOT / "validations" / "reports" / "part_a_validation_report.csv"

CHECKS_FILE = "outputs/part_a_checks.csv"
GROUP_FILE = "outputs/part_a_delay_by_group.csv"
CORRELATIONS_FILE = "outputs/part_a_correlations.csv"
MODELLING_FILE = "data/modelling_table.csv"

FEATURES_FILE = ROOT / "src" / "shipment_prep.py"
EDA_FILE = ROOT / "part_a_clean_eda.py"

# These are the required Part A distance bands from the PDF.
EXPECTED_DISTANCE_BINS = [0, 200, 400, 600, 2000]
EXPECTED_DISTANCE_LABELS = ["0-200", "201-400", "401-600", "601+"]

EXPECTED_CHECK_COLUMNS = ["metric", "value"]

EXPECTED_GROUP_COLUMNS = [
    "dimension",
    "group",
    "shipments",
    "delayed",
    "delay_rate",
    "mean_hours",
]

EXPECTED_CORRELATION_CHECKS = [
    ("delivery_hours", "promised_hours"),
    ("delivery_hours", "distance_km"),
    ("is_delayed", "distance_km"),
]

TARGET_COLUMNS = ["delivery_hours", "is_delayed"]

EXPECTED_GROUPS = {
    "service_type": {"Economy", "Express", "Standard"},
    "hub_tier": {"1", "2", "3"},
    "distance_band": {"0-200", "201-400", "401-600", "601+"},
}

# Explicit comparison rules:
# - ACCEPTANCE metrics are mandatory. A failed comparison becomes ERROR.
# - REFERENCE metrics are informational. A failed comparison becomes MISMATCH.
#
# These operators are deliberately configured here; they are not inferred
# from metric names.
DEFAULT_NUMERIC_OPERATOR = "numeric"


def load_csv(source_file, cache):
    """Read an existing CSV output once and cache it."""
    if source_file in cache:
        return cache[source_file]

    path = ROOT / source_file

    if not path.is_file():
        cache[source_file] = (
            None,
            f"Required file does not exist: {source_file}",
        )
        return cache[source_file]

    try:
        data = pd.read_csv(path)
        cache[source_file] = (data, None)
    except Exception as exc:
        cache[source_file] = (
            None,
            f"Could not read {source_file}: {exc}",
        )

    return cache[source_file]


def load_correlations(cache):
    """Read the correlation matrix while retaining its row labels."""
    source_file = CORRELATIONS_FILE

    if source_file in cache:
        return cache[source_file]

    path = ROOT / source_file

    if not path.is_file():
        cache[source_file] = (
            None,
            f"Required file does not exist: {source_file}",
        )
        return cache[source_file]

    try:
        data = pd.read_csv(path, index_col=0)
        cache[source_file] = (data, None)
    except Exception as exc:
        cache[source_file] = (
            None,
            f"Could not read {source_file}: {exc}",
        )

    return cache[source_file]


def add_acceptance_result(
    rows,
    source_file,
    metric_name,
    actual,
    operator,
    threshold=None,
    lower_bound=None,
    upper_bound=None,
    tolerance=0,
    comment_context="",
):
    """Run an explicit mandatory acceptance check and append its report row."""
    from validations.validations_common_utility import compare_acceptance

    result = compare_acceptance(
        source_file=source_file,
        metric_name=metric_name,
        code_value=actual,
        operator=operator,
        threshold=threshold,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance=tolerance,
    )

    if comment_context:
        result["comment"] = (
            f"{result['comment']} {comment_context}"
        ).strip()

    rows.append(result)


def add_error(rows, source_file, metric_name, comment):
    """Append a standardized ERROR result."""
    rows.append(error_row(source_file, metric_name, comment))


def validate_reference_metrics(reference, cache, rows):
    """
    Compare every configured reference/acceptance metric against existing
    output values.

    Missing files, missing metrics, duplicate metrics, invalid values,
    and invalid comparisons become ERROR.
    """
    for source_file in reference["source_file"].unique():
        source_rows = reference.loc[
            reference["source_file"] == source_file
        ]

        data, load_error = (
            load_correlations(cache)
            if source_file == CORRELATIONS_FILE
            else load_csv(source_file, cache)
        )

        if load_error:
            for _, ref_row in source_rows.iterrows():
                add_error(
                    rows,
                    source_file,
                    ref_row["metric_name"],
                    load_error,
                )
            continue

        if source_file == CHECKS_FILE:
            if not set(EXPECTED_CHECK_COLUMNS).issubset(data.columns):
                for _, ref_row in source_rows.iterrows():
                    add_error(
                        rows,
                        source_file,
                        ref_row["metric_name"],
                        "Checks output must contain 'metric' and 'value'.",
                    )
                continue

            if data["metric"].astype(str).duplicated().any():
                for _, ref_row in source_rows.iterrows():
                    add_error(
                        rows,
                        source_file,
                        ref_row["metric_name"],
                        "Checks output contains duplicate metric names.",
                    )
                continue

        for _, ref_row in source_rows.iterrows():
            metric_name = ref_row["metric_name"]
            value_type = ref_row["value_type"]
            expected = ref_row["metric_value"]
            tolerance = ref_row["tolerance"]

            try:
                actual = extract_metric(
                    source_file,
                    metric_name,
                    data,
                )

                result = compare_metric(
                    source_file=source_file,
                    metric_name=metric_name,
                    code_value=actual,
                    value_type=value_type,
                    operator=DEFAULT_NUMERIC_OPERATOR,
                    reference_value=expected,
                    tolerance=tolerance,
                    threshold=expected,
                )

                # Explain the intended meaning of this comparison.
                if value_type == "ACCEPTANCE":
                    result["comment"] = (
                        "Mandatory Part A acceptance comparison. "
                        + result["comment"]
                    )
                else:
                    result["comment"] = (
                        "Informational reference comparison. "
                        + result["comment"]
                    )

                rows.append(result)

            except Exception as exc:
                add_error(
                    rows,
                    source_file,
                    metric_name,
                    str(exc),
                )


def extract_metric(source_file, metric_name, data):
    """Extract a configured metric from its source output."""
    if source_file == CHECKS_FILE:
        matches = data.loc[
            data["metric"].astype(str) == metric_name,
            "value",
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one '{metric_name}' metric in "
                f"{CHECKS_FILE}; found {len(matches)}."
            )

        return matches.iloc[0]

    if source_file == GROUP_FILE:
        parts = metric_name.split("__", maxsplit=2)

        if len(parts) != 3:
            raise ValueError(
                f"Invalid group metric name: {metric_name!r}."
            )

        metric, dimension, group = parts

        if metric not in {"shipments", "delayed", "delay_rate"}:
            raise ValueError(
                f"Unsupported group metric: {metric!r}."
            )

        required = set(EXPECTED_GROUP_COLUMNS)
        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"{GROUP_FILE} is missing columns: {sorted(missing)}."
            )

        dimension_values = data["dimension"].astype(str).str.strip()
        group_values = data["group"].astype(str).str.strip()

        matches = data.loc[
            (dimension_values == dimension)
            & (group_values == group),
            metric,
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {dimension}={group} row for "
                f"metric {metric}; found {len(matches)}."
            )

        actual = matches.iloc[0]

        # The PDF requires the shipment and delayed counts to match exactly.
        # Delay rates are written to four decimal places.
        if metric in {"shipments", "delayed"}:
            numeric_actual = float(actual)

            if not numeric_actual.is_integer():
                raise ValueError(
                    f"{metric_name} must be an integer; found {actual!r}."
                )

            return int(numeric_actual)

        return round(float(actual), 4)

    if source_file == CORRELATIONS_FILE:
        parts = metric_name.split("__", maxsplit=2)

        if len(parts) != 3 or parts[0] != "corr":
            raise ValueError(
                f"Invalid correlation metric name: {metric_name!r}."
            )

        _, row_name, column_name = parts

        if row_name not in data.index:
            raise KeyError(
                f"Correlation row {row_name!r} is missing."
            )

        if column_name not in data.columns:
            raise KeyError(
                f"Correlation column {column_name!r} is missing."
            )

        return data.loc[row_name, column_name]

    raise ValueError(
        f"Unsupported Part A reference source: {source_file!r}."
    )


def validate_check_metric_inventory(reference, cache, rows):
    """Require the checks output to contain exactly the referenced metrics."""
    data, load_error = load_csv(CHECKS_FILE, cache)

    if load_error:
        add_error(
            rows,
            CHECKS_FILE,
            "__metric_inventory__",
            load_error,
        )
        return

    if not set(EXPECTED_CHECK_COLUMNS).issubset(data.columns):
        add_error(
            rows,
            CHECKS_FILE,
            "__metric_inventory__",
            "Cannot validate metric inventory because the required "
            "'metric' and 'value' columns are missing.",
        )
        return

    expected_metrics = set(
        reference.loc[
            reference["source_file"] == CHECKS_FILE,
            "metric_name",
        ]
    )
    actual_metrics = set(data["metric"].astype(str))

    missing = sorted(expected_metrics - actual_metrics)
    unexpected = sorted(actual_metrics - expected_metrics)

    add_acceptance_result(
        rows,
        CHECKS_FILE,
        "__all_expected_metrics_present__",
        not missing,
        "boolean",
        threshold=True,
        comment_context=f"Missing metrics: {missing}.",
    )

    add_acceptance_result(
        rows,
        CHECKS_FILE,
        "__no_unreferenced_metrics__",
        not unexpected,
        "boolean",
        threshold=True,
        comment_context=f"Unreferenced metrics: {unexpected}.",
    )


def validate_group_structure(cache, rows):
    """Check group-table structure and exact group coverage."""
    data, load_error = load_csv(GROUP_FILE, cache)

    if load_error:
        add_error(rows, GROUP_FILE, "__structure__", load_error)
        return

    missing_columns = set(EXPECTED_GROUP_COLUMNS) - set(data.columns)

    add_acceptance_result(
        rows,
        GROUP_FILE,
        "__required_columns_present__",
        not missing_columns,
        "boolean",
        threshold=True,
        comment_context=f"Missing columns: {sorted(missing_columns)}.",
    )

    if missing_columns:
        return

    # A dimension/group pair must appear only once.
    duplicate_keys = data.duplicated(
        subset=["dimension", "group"],
        keep=False,
    )

    add_acceptance_result(
        rows,
        GROUP_FILE,
        "__unique_dimension_group_keys__",
        not duplicate_keys.any(),
        "boolean",
        threshold=True,
        comment_context=(
            "Each dimension/group pair must occur exactly once."
        ),
    )

    actual_dimensions = set(data["dimension"].astype(str))
    expected_dimensions = set(EXPECTED_GROUPS)

    add_acceptance_result(
        rows,
        GROUP_FILE,
        "__expected_dimensions__",
        actual_dimensions == expected_dimensions,
        "boolean",
        threshold=True,
        comment_context=(
            f"Expected {sorted(expected_dimensions)}; "
            f"found {sorted(actual_dimensions)}."
        ),
    )

    for dimension, expected_groups in EXPECTED_GROUPS.items():
        actual_groups = set(
            data.loc[
                data["dimension"].astype(str) == dimension,
                "group",
            ].astype(str)
        )

        add_acceptance_result(
            rows,
            GROUP_FILE,
            f"__groups__{dimension}",
            actual_groups == expected_groups,
            "boolean",
            threshold=True,
            comment_context=(
                f"Expected {sorted(expected_groups)}; "
                f"found {sorted(actual_groups)}."
            ),
        )

    # Delay rates must agree with delayed / shipments to four decimal places.
    valid_counts = (
        pd.to_numeric(data["shipments"], errors="coerce").notna()
        & pd.to_numeric(data["delayed"], errors="coerce").notna()
        & (pd.to_numeric(data["shipments"], errors="coerce") > 0)
    )

    calculated_rates = pd.Series(np.nan, index=data.index, dtype=float)
    calculated_rates.loc[valid_counts] = (
        pd.to_numeric(data.loc[valid_counts, "delayed"])
        / pd.to_numeric(data.loc[valid_counts, "shipments"])
    )

    actual_rates = pd.to_numeric(data["delay_rate"], errors="coerce")

    rates_valid = (
        calculated_rates.notna()
        & actual_rates.notna()
        & (
            (calculated_rates - actual_rates).abs()
            <= 0.00005
        )
    )

    add_acceptance_result(
        rows,
        GROUP_FILE,
        "__delay_rate_consistency__",
        bool(rates_valid.all()),
        "boolean",
        threshold=True,
        comment_context=(
            "Each delay rate must equal delayed / shipments to four "
            "decimal places. Invalid counts or rates fail this check."
        ),
    )


def validate_modelling_table(cache, rows):
    """
    Validate the existing modelling table:
    - 1,200 rows
    - all pinned FEATURES plus delivery_hours and is_delayed
    - no missing values in those 19 columns
    """
    data, load_error = load_csv(MODELLING_FILE, cache)

    if load_error:
        add_error(rows, MODELLING_FILE, "__modelling_table__", load_error)
        return

    # Import only the pinned feature definition. This does not run Part A.
    try:
        from src.shipment_prep import FEATURES
    except Exception as exc:
        add_error(
            rows,
            MODELLING_FILE,
            "__pinned_features__",
            f"Could not import pinned FEATURES from src/shipment_prep.py: {exc}",
        )
        return

    required_columns = list(FEATURES) + TARGET_COLUMNS
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    add_acceptance_result(
        rows,
        MODELLING_FILE,
        "__row_count__",
        len(data),
        "exact",
        threshold=1200,
        comment_context="The modelling table must contain exactly 1,200 rows.",
    )

    add_acceptance_result(
        rows,
        MODELLING_FILE,
        "__required_feature_and_target_columns__",
        not missing_columns,
        "boolean",
        threshold=True,
        comment_context=(
            f"Required columns: {required_columns}. "
            f"Missing columns: {missing_columns}."
        ),
    )

    if missing_columns:
        # We cannot evaluate completeness for columns that do not exist.
        add_error(
            rows,
            MODELLING_FILE,
            "__missing_values__",
            "Cannot evaluate missing-value acceptance because required "
            f"columns are absent: {missing_columns}.",
        )
        return

    missing_cells = int(data[required_columns].isna().sum().sum())

    add_acceptance_result(
        rows,
        MODELLING_FILE,
        "__missing_cells_in_features_and_target__",
        missing_cells,
        "exact",
        threshold=0,
        comment_context=(
            "No missing values are permitted in the 17 FEATURES columns, "
            "delivery_hours, or is_delayed."
        ),
    )


def validate_distance_band_definition(rows):
    """Validate the distance-band definition using static source inspection."""
    source_file = "part_a_clean_eda.py"
    source_path = ROOT / source_file

    if not source_path.is_file():
        add_error(
            rows,
            source_file,
            "__distance_band_definition__",
            "Part A source file does not exist.",
        )
        return

    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_error(
            rows,
            source_file,
            "__distance_band_definition__",
            f"Could not parse Part A source: {exc}",
        )
        return

    # Resolve simple assignments such as:
    # distance_bins = [0, 200, 400, 600, 2000]
    # distance_labels = ["0-200", "201-400", "401-600", "601+"]
    assignments = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value

    found_bins = False
    found_labels = False
    found_right = False
    found_include_lowest = False
    found_cut_call = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Attribute):
            continue

        if node.func.attr != "cut":
            continue

        found_cut_call = True
        keywords = {kw.arg: kw.value for kw in node.keywords}

        def resolve_argument(name):
            argument = keywords.get(name)

            if argument is None:
                return None

            if isinstance(argument, ast.Name):
                return assignments.get(argument.id)

            try:
                return ast.literal_eval(argument)
            except (ValueError, TypeError):
                return None

        found_bins = (
            resolve_argument("bins") == EXPECTED_DISTANCE_BINS
        )
        found_labels = (
            resolve_argument("labels") == EXPECTED_DISTANCE_LABELS
        )
        found_right = resolve_argument("right") is True
        found_include_lowest = (
            resolve_argument("include_lowest") is True
        )

        break

    add_acceptance_result(
        rows,
        source_file,
        "__distance_band_edges__",
        found_cut_call and found_bins,
        "boolean",
        threshold=True,
        comment_context=(
            f"Required edges: {EXPECTED_DISTANCE_BINS}."
        ),
    )

    add_acceptance_result(
        rows,
        source_file,
        "__distance_band_labels__",
        found_cut_call and found_labels,
        "boolean",
        threshold=True,
        comment_context=(
            f"Required labels: {EXPECTED_DISTANCE_LABELS}."
        ),
    )

    add_acceptance_result(
        rows,
        source_file,
        "__distance_band_right_inclusive__",
        found_cut_call and found_right,
        "boolean",
        threshold=True,
        comment_context=(
            "The bins must be right-inclusive."
        ),
    )

    add_acceptance_result(
        rows,
        source_file,
        "__distance_band_include_lowest__",
        found_cut_call and found_include_lowest,
        "boolean",
        threshold=True,
        comment_context=(
            "The lowest boundary must be included in the first bin."
        ),
    )


def validate_shared_functions_and_new_shipments(rows):
    """
    Validate the shared preprocessing API and exercise it in memory on
    data/new_shipments.csv using the saved medians.

    This does not write files or regenerate any Part A outputs.
    """
    shipment_prep_path = FEATURES_FILE

    if not shipment_prep_path.is_file():
        add_error(
            "src/shipment_prep.py",
            "__shared_preprocessing__",
            "Shared preprocessing source file does not exist.",
        )
        return

    try:
        from src.shipment_prep import (
            FEATURES,
            build_features,
            clean_shipments,
        )
    except Exception as exc:
        add_error(
            rows,
            "src/shipment_prep.py",
            "__shared_preprocessing__",
            f"Could not import shared preprocessing functions: {exc}",
        )
        return

    # Check function signatures explicitly.
    clean_signature = str(inspect.signature(clean_shipments))
    build_signature = str(inspect.signature(build_features))

    add_acceptance_result(
        rows,
        "src/shipment_prep.py",
        "__clean_shipments_signature__",
        clean_signature == "(raw, medians=None)",
        "boolean",
        threshold=True,
        comment_context=f"Found signature: {clean_signature}.",
    )

    add_acceptance_result(
        rows,
        "src/shipment_prep.py",
        "__build_features_signature__",
        build_signature == "(clean, hubs)",
        "boolean",
        threshold=True,
        comment_context=f"Found signature: {build_signature}.",
    )

    new_shipments_path = ROOT / "data" / "new_shipments.csv"
    hubs_path = ROOT / "data" / "hubs.csv"
    medians_path = ROOT / "models" / "impute_values.csv"

    for path, metric_name in [
        (new_shipments_path, "__new_shipments_input__"),
        (hubs_path, "__hub_input__"),
        (medians_path, "__stored_medians__"),
    ]:
        if not path.is_file():
            add_error(
                rows,
                str(path.relative_to(ROOT)),
                metric_name,
                "Required input file for the preprocessing acceptance "
                "check does not exist.",
            )
            return

    try:
        new_shipments = pd.read_csv(new_shipments_path)
        hubs = pd.read_csv(hubs_path)
        medians_frame = pd.read_csv(medians_path)

        if not {"column", "median"}.issubset(medians_frame.columns):
            raise ValueError(
                "Stored medians file must contain 'column' and 'median'."
            )

        if medians_frame["column"].duplicated().any():
            raise ValueError(
                "Stored medians file contains duplicate column names."
            )

        medians = dict(
            zip(
                medians_frame["column"].astype(str),
                pd.to_numeric(medians_frame["median"], errors="raise"),
            )
        )

        expected_median_columns = {
            "distance_km",
            "parcel_weight_kg",
            "declared_value_inr",
        }

        if set(medians) != expected_median_columns:
            raise ValueError(
                "Stored medians must contain exactly "
                f"{sorted(expected_median_columns)}; found {sorted(medians)}."
            )

        clean_result = clean_shipments(new_shipments, medians)

        if not isinstance(clean_result, tuple) or len(clean_result) != 2:
            raise TypeError(
                "clean_shipments must return a two-item tuple: "
                "(clean_dataframe, medians_dictionary)."
            )

        clean, returned_medians = clean_result

        if not isinstance(clean, pd.DataFrame):
            raise TypeError(
                "The first clean_shipments return value must be a DataFrame."
            )

        if not isinstance(returned_medians, dict):
            raise TypeError(
                "The second clean_shipments return value must be a dictionary."
            )

        features = build_features(clean, hubs)

        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "build_features must return a pandas DataFrame."
            )

        missing_features = [
            column for column in FEATURES
            if column not in features.columns
        ]

        add_acceptance_result(
            rows,
            "data/new_shipments.csv",
            "__all_pinned_features_created__",
            not missing_features,
            "boolean",
            threshold=True,
            comment_context=(
                f"Missing features: {missing_features}."
            ),
        )

        if missing_features:
            add_error(
                rows,
                "data/new_shipments.csv",
                "__new_shipment_feature_nulls__",
                "Cannot evaluate feature completeness because some "
                f"required features are absent: {missing_features}.",
            )
            return

        missing_feature_cells = int(
            features[FEATURES].isna().sum().sum()
        )

        add_acceptance_result(
            rows,
            "data/new_shipments.csv",
            "__no_missing_new_shipment_features__",
            missing_feature_cells,
            "exact",
            threshold=0,
            comment_context=(
                "The 17 pinned feature columns must contain no missing values "
                "after cleaning and feature engineering."
            ),
        )

        add_acceptance_result(
            rows,
            "data/new_shipments.csv",
            "__clean_shipments_returned_medians__",
            set(returned_medians) == expected_median_columns,
            "boolean",
            threshold=True,
            comment_context=(
                "Returned median keys must be distance_km, "
                "parcel_weight_kg, and declared_value_inr."
            ),
        )

    except Exception as exc:
        add_error(
            rows,
            "data/new_shipments.csv",
            "__shared_preprocessing_execution__",
            "Preprocessing acceptance check failed in memory: "
            f"{type(exc).__name__}: {exc}",
        )


def validate_readme_findings(cache, rows):
    """
    Check that the README has a Part A Data Findings section and that the
    required finding categories are mentioned.

    The check does not modify README.md.
    """
    readme_path = ROOT / "README.md"

    if not readme_path.is_file():
        add_error(rows, "README.md", "__part_a_findings__", "README.md is missing.")
        return

    text = readme_path.read_text(encoding="utf-8")
    lowered = text.lower()

    required_topics = {
        "service type delay comparison": [
            "service type",
            "express",
            "delay",
        ],
        "hub tier delay comparison": [
            "tier 1",
            "tier 3",
        ],
        "distance-band delay comparison": [
            "distance band",
            "601+",
        ],
        "delivery-hours correlation findings": [
            "correlation",
            "delivery_hours",
        ],
        "declared-value log transformation explanation": [
            "declared_value_inr",
            "log",
        ],
    }

    # Find the Part A data findings section rather than accepting unrelated
    # Part B, C, or D mentions as evidence.
    start = lowered.find("## part a")
    if start == -1:
        start = lowered.find("## part a -")
    if start == -1:
        start = 0

    section = lowered[start:]
    next_section = section.find("\n## part b")
    if next_section != -1:
        section = section[:next_section]

    has_findings_heading = (
        "data findings" in section
        or "data findings" in lowered
    )

    add_acceptance_result(
        rows,
        "README.md",
        "__part_a_data_findings_heading__",
        has_findings_heading,
        "boolean",
        threshold=True,
        comment_context=(
            "The Part A README section must contain a Data Findings heading."
        ),
    )

    for topic, phrases in required_topics.items():
        present = all(phrase in section for phrase in phrases)

        add_acceptance_result(
            rows,
            "README.md",
            f"__finding_topic__{topic.replace(' ', '_')}",
            present,
            "boolean",
            threshold=True,
            comment_context=(
                f"Required topic phrases: {phrases}. "
                "This is a presence check, not a full semantic review."
            ),
        )


def validate_part_a():
    """Run Part A validation against existing outputs and source definitions."""
    reference, reference_errors = load_reference(REFERENCE_PATH)

    rows = list(reference_errors)
    cache = {}

    if reference.empty:
        # load_reference reports schema/file problems. A valid header-only
        # official reference file would not provide checks to execute.
        add_error(
            rows,
            str(REFERENCE_PATH.relative_to(ROOT)),
            "__empty_reference__",
            "The official Part A reference CSV contains no metric rows.",
        )
        return write_report(rows, REPORT_PATH)

    # Numeric reference/acceptance metrics from the approved CSV.
    validate_reference_metrics(reference, cache, rows)

    # Structural and non-scalar acceptance criteria from the PDF.
    validate_check_metric_inventory(reference, cache, rows)
    validate_group_structure(cache, rows)
    validate_modelling_table(cache, rows)
    validate_distance_band_definition(rows)
    validate_shared_functions_and_new_shipments(rows)
    validate_readme_findings(cache, rows)

    return write_report(rows, REPORT_PATH)


if __name__ == "__main__":
    sys.exit(validate_part_a())