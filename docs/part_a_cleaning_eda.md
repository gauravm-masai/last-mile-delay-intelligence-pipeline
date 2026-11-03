[← Back to Project Overview](../README.md)

# Part A: Data Cleaning and Exploratory Data Analysis

## Objectives
Part A prepares the shipment dataset for downstream analysis and modelling. The main objectives are to clean and standardize shipment data, handle missing and invalid values, combine shipment records with hub information, engineer the required features, and define whether each shipment was delayed based on its promised delivery time.

## High Level Approach
1. **Data cleaning:** Remove duplicate shipment records, standardize service types, handle missing payment modes, convert declared values to numeric format, treat invalid parcel weights as missing, and impute missing numeric values using historical medians.
2. **Feature engineering and hub integration:** Merge shipment records with hub information and create the required features, including promised delivery hours, log-transformed declared value, weekend and COD indicators, and dispatch hour.
3. **Delay labelling:** Compare actual delivery hours against the promised delivery hours for each service type to create the binary `is_delayed` target.
4. **Validation and quality checks:** Calculate data-quality metrics and compare them against reference values to verify that the cleaning and feature engineering steps produce the expected results.
5. **Exploratory analysis:** Generate delay-rate summaries by service type, hub tier, and distance band, along with a numeric correlation matrix to examine relationships between features and delivery hours. Distance bands are defined using right-inclusive bins with edges `[0, 200, 400, 600, 2000]` and labels `0-200`, `201-400`, `401-600`, and `601+`. The lowest boundary is included in the first bin.

The outputs include the cleaned modelling table, validation reports, delay summaries, and correlation matrix. These outputs provide a consistent, validated dataset and an initial understanding of the factors associated with delivery delays before proceeding to subsequent modelling tasks.

## Generated Outputs

Part A generates the following files:

- `data/modelling_table.csv` - prepared modelling dataset with engineered features and the delay target.
- `outputs/part_a_checks.csv` - data-quality and summary metrics.
- `outputs/part_a_delay_by_group.csv` - shipment counts, delayed counts, and delay rates by service type, hub tier, and distance band.
- `outputs/part_a_correlations.csv` - correlation matrix used to examine relationships among the modelling columns.

## Data findings

**Data quality and preparation**

- The raw dataset contained **1,224 rows**. Removing **24 duplicate rows** left **1,200 shipments**.
- Before cleaning, there were 36 missing distance values, 24 missing parcel-weight values, 5 sentinel parcel-weight values, 12 missing declared-value values, and 18 missing payment-mode values.
- Service-type spellings were standardized from 12 observed spellings to 3 categories: Economy, Express, and Standard.
- The hub merge retained all **1,200 shipments**. The required feature columns and target contained no missing values in the final modelling table.
- There were **278 delayed shipments**, giving an overall delay rate of **23.17%**.

**Delay rate by service type**

Express shipments had the highest delay rate at **48.97%** (119 delayed out of 243). Standard shipments had a delay rate of **20.56%** (132 out of 642), while Economy shipments had the lowest rate at **8.57%** (27 out of 315).

**Delay rate by hub tier**

The delay rate increased with hub tier: **4.00% for Tier 1**, **21.50% for Tier 2**, and **44.00% for Tier 3**. These results show a clear association between hub tier and delay rate in this synthetic dataset.

**Delay rate by distance band**

Delay rates rose as distance increased:

- **0-200 km**: 2.19%
- **201-400 km**: 13.68%
- **401-600 km**: 43.64%
- **601+ km**: 85.23%

The highest distance band had the highest observed delay rate, with 75 of 88 shipments delayed.

**Correlation with delivery time**

Excluding `delivery_hours` itself, the three columns with the strongest absolute correlations with `delivery_hours` were:

- promised_hours: **+0.8372**
- distance_km: **+0.4013**
- n_stops: **+0.3247**

These are correlations, not evidence of causation. The correlation-validation rows in the table below also include a separate check for `is_delayed` against `distance_km`; that check is not one of the three strongest correlations with `delivery_hours`.

**Log transformation of declared value**

After cleaning and filling missing declared values, the mean declared value was **INR 1,533.05** and the median was **INR 1,121.50**. The corresponding mean and median of `log_declared_value` were **7.028** and **7.022**, respectively. The difference between the original mean and median indicates right-skew in declared values. The log transformation reduces the influence of large values and makes the distribution less skewed.

## Validations

### Acceptance criteria and results

The Part A validation checks cover data quality, duplicate removal, missing-value and sentinel counts, service-type standardization, imputation statistics, hub-merge row counts, missing values in features and target, delay statistics, group-level delay metrics, and selected correlation checks. The validations consist checks for 26 quality metrics, 30 group-level metrics, and 3 correlations.

The modelling table contains 1,200 rows, preserves the required feature and target data without missing values, and retains all rows through the hub merge. The detailed reference-versus-code comparisons are below.

### Reference metrics compared with generated outputs

| Source file | Metric | PDF reference value | Code-generated value | Tolerance | Status | Comments |
|---|---|---:|---:|---|---|---|
| `outputs/part_a_checks.csv` | `raw_rows` | 1224 | 1224 | Exact | PASS | Confirms the raw shipment dataset contains the expected 1,224 rows. |
| `outputs/part_a_checks.csv` | `duplicate_rows_removed` | 24 | 24 | Exact | PASS | Confirms that the expected 24 duplicate rows were removed. |
| `outputs/part_a_checks.csv` | `rows_after_dedup` | 1200 | 1200 | Exact | PASS | Confirms 1,200 shipment rows remain after deduplication. |
| `outputs/part_a_checks.csv` | `missing_distance_km_before` | 36 | 36 | Exact | PASS | Matches the expected number of missing distance values before imputation. |
| `outputs/part_a_checks.csv` | `missing_parcel_weight_before` | 24 | 24 | Exact | PASS | Matches the expected number of missing parcel-weight values before cleaning. |
| `outputs/part_a_checks.csv` | `sentinel_weight_rows` | 5 | 5 | Exact | PASS | Confirms five parcel-weight records contain sentinel values that must be treated as missing. |
| `outputs/part_a_checks.csv` | `missing_declared_value_before` | 12 | 12 | Exact | PASS | Matches the expected number of missing declared-value records before imputation. |
| `outputs/part_a_checks.csv` | `missing_payment_mode_before` | 18 | 18 | Exact | PASS | Matches the expected number of missing payment-mode values before replacement. |
| `outputs/part_a_checks.csv` | `service_type_spellings_before` | 12 | 12 | Exact | PASS | Confirms the expected number of distinct service-type spellings before standardization. |
| `outputs/part_a_checks.csv` | `service_type_spellings_after` | 3 | 3 | Exact | PASS | Confirms service-type standardization reduces the distinct spellings to three. |
| `outputs/part_a_checks.csv` | `declared_value_mean_inr` | 1533.05 | 1533.0542 | ±0.01 | PASS | The generated mean declared value is within the reference tolerance. |
| `outputs/part_a_checks.csv` | `declared_value_median_inr` | 1121.5 | 1121.5000 | ±0.01 | PASS | The generated median declared value matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `log_declared_value_mean` | 7.028 | 7.0277 | ±0.01 | PASS | The mean log-transformed declared value matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `log_declared_value_median` | 7.022 | 7.0224 | ±0.01 | PASS | The median log-transformed declared value matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `median_distance_km` | 314.95 | 314.9500 | ±0.01 | PASS | The generated median distance matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `median_parcel_weight_kg` | 1.83 | 1.8300 | ±0.01 | PASS | The generated median parcel weight matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `median_declared_value_inr` | 1121.5 | 1121.5000 | ±0.01 | PASS | Confirms the declared-value median used for imputation matches the reference. |
| `outputs/part_a_checks.csv` | `rows_after_hub_merge` | 1200 | 1200 | Exact | PASS | Confirms the hub merge preserves all 1,200 shipment rows. |
| `outputs/part_a_checks.csv` | `nulls_in_features_and_target` | 0 | 0 | Exact | PASS | Confirms there are no null values in the specified modelling features and target. |
| `outputs/part_a_checks.csv` | `delayed_count` | 278 | 278 | Exact | PASS | Matches the expected count of 278 delayed shipments. |
| `outputs/part_a_checks.csv` | `delay_rate` | 0.2317 | 0.2317 | ±0.0001 | PASS | The overall delay rate matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `hours_q1` | 43.26 | 43.2550 | ±0.01 | PASS | The first quartile of delivery hours matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `hours_q3` | 72.56 | 72.5625 | ±0.01 | PASS | The third quartile of delivery hours matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `hours_lower_bound` | -0.71 | -0.7063 | ±0.01 | PASS | The calculated lower outlier bound matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `hours_upper_bound` | 116.52 | 116.5238 | ±0.01 | PASS | The calculated upper outlier bound matches the reference within tolerance. |
| `outputs/part_a_checks.csv` | `hours_outlier_count` | 1 | 1 | Exact | PASS | Confirms the expected number of delivery-hour outliers. |
| `outputs/part_a_delay_by_group.csv` | Service type - Economy: shipments; delayed; delay rate | 315; 27; 0.0857 | 315; 27; 0.0857 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for Economy shipments. |
| `outputs/part_a_delay_by_group.csv` | Service type - Express: shipments; delayed; delay rate | 243; 119; 0.4897 | 243; 119; 0.4897 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for Express shipments. |
| `outputs/part_a_delay_by_group.csv` | Service type - Standard: shipments; delayed; delay rate | 642; 132; 0.2056 | 642; 132; 0.2056 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for Standard shipments. |
| `outputs/part_a_delay_by_group.csv` | Hub tier - Tier 1: shipments; delayed; delay rate | 400; 16; 0.0400 | 400; 16; 0.0400 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for Tier 1 hubs. |
| `outputs/part_a_delay_by_group.csv` | Hub tier - Tier 2: shipments; delayed; delay rate | 400; 86; 0.2150 | 400; 86; 0.2150 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for Tier 2 hubs. |
| `outputs/part_a_delay_by_group.csv` | Hub tier - Tier 3: shipments; delayed; delay rate | 400; 176; 0.4400 | 400; 176; 0.4400 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for Tier 3 hubs. |
| `outputs/part_a_delay_by_group.csv` | Distance band - 0-200 km: shipments; delayed; delay rate | 274; 6; 0.0219 | 274; 6; 0.0219 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for the 0-200 km band. |
| `outputs/part_a_delay_by_group.csv` | Distance band - 201-400 km: shipments; delayed; delay rate | 563; 77; 0.1368 | 563; 77; 0.1368 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for the 201-400 km band. |
| `outputs/part_a_delay_by_group.csv` | Distance band - 401-600 km: shipments; delayed; delay rate | 275; 120; 0.4364 | 275; 120; 0.4364 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for the 401-600 km band. |
| `outputs/part_a_delay_by_group.csv` | Distance band - 601+ km: shipments; delayed; delay rate | 88; 75; 0.8523 | 88; 75; 0.8523 | Counts: Exact; rate: ±0.00005 | PASS | Shipment count, delayed count, and delay rate match the reference for shipments travelling 601 km or more. |
| `outputs/part_a_correlations.csv` | `corr__delivery_hours__promised_hours` | 0.837 | 0.8372 | ±0.01 | PASS | The correlation between delivery hours and promised hours matches the reference within tolerance. |
| `outputs/part_a_correlations.csv` | `corr__delivery_hours__distance_km` | 0.401 | 0.4013 | ±0.01 | PASS | The correlation between delivery hours and distance matches the reference within tolerance. |
| `outputs/part_a_correlations.csv` | `corr__is_delayed__distance_km` | 0.546 | 0.5455 | ±0.01 | PASS | The correlation between delay status and distance matches the reference within tolerance. |

**Validation conclusion:** All metrics for Part A pass against their reference values within the configured tolerances. No benchmark values have been changed to obtain these results.

[← Back to Project Overview](../README.md)