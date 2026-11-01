# Part A validation report

Run: 2026-11-02 05:16 · **61/61 passed** · PASS 61 · FAIL 0 · ERROR 0 · WARN 0

FAIL = graded criterion not met · ERROR = value could not be computed · WARN = differs from the pinned-configuration reference (graded on the band, not this number)

| Metric | Type | Reference | Generated | Status | Comment |
|---|---|---|---|---|---|
| raw_rows | ACCEPTANCE | 1224 | 1224 | PASS | Exact match |
| duplicate_rows_removed | ACCEPTANCE | 24 | 24 | PASS | Exact match |
| rows_after_dedup | ACCEPTANCE | 1200 | 1200 | PASS | Exact match |
| missing_distance_km_before | ACCEPTANCE | 36 | 36 | PASS | Exact match |
| missing_parcel_weight_before | ACCEPTANCE | 24 | 24 | PASS | Exact match |
| sentinel_weight_rows | ACCEPTANCE | 5 | 5 | PASS | Exact match |
| missing_declared_value_before | ACCEPTANCE | 12 | 12 | PASS | Exact match |
| missing_payment_mode_before | ACCEPTANCE | 18 | 18 | PASS | Exact match |
| service_type_spellings_before | ACCEPTANCE | 12 | 12 | PASS | Exact match |
| service_type_spellings_after | ACCEPTANCE | 3 | 3 | PASS | Exact match |
| declared_value_mean_inr | ACCEPTANCE | 1533.05 ± 0.01 | 1533.05 | PASS | Within ±0.01 (diff +0.0042) |
| declared_value_median_inr | ACCEPTANCE | 1121.5 ± 0.01 | 1121.5 | PASS | Within ±0.01 (diff +0) |
| log_declared_value_mean | ACCEPTANCE | 7.028 ± 0.01 | 7.0277 | PASS | Within ±0.01 (diff -0.0003) |
| log_declared_value_median | ACCEPTANCE | 7.022 ± 0.01 | 7.0224 | PASS | Within ±0.01 (diff +0.0004) |
| median_distance_km | ACCEPTANCE | 314.95 ± 0.01 | 314.95 | PASS | Within ±0.01 (diff +0) |
| median_parcel_weight_kg | ACCEPTANCE | 1.83 ± 0.01 | 1.83 | PASS | Within ±0.01 (diff +0) |
| median_declared_value_inr | ACCEPTANCE | 1121.5 ± 0.01 | 1121.5 | PASS | Within ±0.01 (diff +0) |
| rows_after_hub_merge | ACCEPTANCE | 1200 | 1200 | PASS | Exact match |
| nulls_in_features_and_target | ACCEPTANCE | 0 | 0 | PASS | Exact match |
| delayed_count | ACCEPTANCE | 278 | 278 | PASS | Exact match |
| delay_rate | ACCEPTANCE | 0.2317 ± 0.0001 | 0.2317 | PASS | Within ±0.0001 (diff +0) |
| hours_q1 | ACCEPTANCE | 43.26 ± 0.01 | 43.255 | PASS | Within ±0.01 (diff -0.005) |
| hours_q3 | ACCEPTANCE | 72.56 ± 0.01 | 72.5625 | PASS | Within ±0.01 (diff +0.0025) |
| hours_lower_bound | ACCEPTANCE | -0.71 ± 0.01 | -0.7063 | PASS | Within ±0.01 (diff +0.0037) |
| hours_upper_bound | ACCEPTANCE | 116.52 ± 0.01 | 116.524 | PASS | Within ±0.01 (diff +0.0038) |
| hours_outlier_count | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| shipments__service_type__Economy | ACCEPTANCE | 315 | 315 | PASS | Exact match |
| delayed__service_type__Economy | ACCEPTANCE | 27 | 27 | PASS | Exact match |
| delay_rate__service_type__Economy | ACCEPTANCE | 0.0857 ± 0.00005 | 0.0857 | PASS | Within ±0.00005 (diff +0) |
| shipments__service_type__Express | ACCEPTANCE | 243 | 243 | PASS | Exact match |
| delayed__service_type__Express | ACCEPTANCE | 119 | 119 | PASS | Exact match |
| delay_rate__service_type__Express | ACCEPTANCE | 0.4897 ± 0.00005 | 0.4897 | PASS | Within ±0.00005 (diff +0) |
| shipments__service_type__Standard | ACCEPTANCE | 642 | 642 | PASS | Exact match |
| delayed__service_type__Standard | ACCEPTANCE | 132 | 132 | PASS | Exact match |
| delay_rate__service_type__Standard | ACCEPTANCE | 0.2056 ± 0.00005 | 0.2056 | PASS | Within ±0.00005 (diff +0) |
| shipments__hub_tier__1 | ACCEPTANCE | 400 | 400 | PASS | Exact match |
| delayed__hub_tier__1 | ACCEPTANCE | 16 | 16 | PASS | Exact match |
| delay_rate__hub_tier__1 | ACCEPTANCE | 0.0400 ± 0.00005 | 0.04 | PASS | Within ±0.00005 (diff +0) |
| shipments__hub_tier__2 | ACCEPTANCE | 400 | 400 | PASS | Exact match |
| delayed__hub_tier__2 | ACCEPTANCE | 86 | 86 | PASS | Exact match |
| delay_rate__hub_tier__2 | ACCEPTANCE | 0.2150 ± 0.00005 | 0.215 | PASS | Within ±0.00005 (diff +0) |
| shipments__hub_tier__3 | ACCEPTANCE | 400 | 400 | PASS | Exact match |
| delayed__hub_tier__3 | ACCEPTANCE | 176 | 176 | PASS | Exact match |
| delay_rate__hub_tier__3 | ACCEPTANCE | 0.4400 ± 0.00005 | 0.44 | PASS | Within ±0.00005 (diff +0) |
| shipments__distance_band__0-200 | ACCEPTANCE | 274 | 274 | PASS | Exact match |
| delayed__distance_band__0-200 | ACCEPTANCE | 6 | 6 | PASS | Exact match |
| delay_rate__distance_band__0-200 | ACCEPTANCE | 0.0219 ± 0.00005 | 0.0219 | PASS | Within ±0.00005 (diff +0) |
| shipments__distance_band__201-400 | ACCEPTANCE | 563 | 563 | PASS | Exact match |
| delayed__distance_band__201-400 | ACCEPTANCE | 77 | 77 | PASS | Exact match |
| delay_rate__distance_band__201-400 | ACCEPTANCE | 0.1368 ± 0.00005 | 0.1368 | PASS | Within ±0.00005 (diff +0) |
| shipments__distance_band__401-600 | ACCEPTANCE | 275 | 275 | PASS | Exact match |
| delayed__distance_band__401-600 | ACCEPTANCE | 120 | 120 | PASS | Exact match |
| delay_rate__distance_band__401-600 | ACCEPTANCE | 0.4364 ± 0.00005 | 0.4364 | PASS | Within ±0.00005 (diff +0) |
| shipments__distance_band__601+ | ACCEPTANCE | 88 | 88 | PASS | Exact match |
| delayed__distance_band__601+ | ACCEPTANCE | 75 | 75 | PASS | Exact match |
| delay_rate__distance_band__601+ | ACCEPTANCE | 0.8523 ± 0.00005 | 0.8523 | PASS | Within ±0.00005 (diff +0) |
| promised_hours__delivery_hours | ACCEPTANCE | 0.837 ± 0.01 | 0.8372 | PASS | Within ±0.01 (diff +0.0002) |
| distance_km__delivery_hours | ACCEPTANCE | 0.401 ± 0.01 | 0.4013 | PASS | Within ±0.01 (diff +0.0003) |
| distance_km__is_delayed | ACCEPTANCE | 0.546 ± 0.01 | 0.5455 | PASS | Within ±0.01 (diff -0.0005) |
| ROW_COUNT | ACCEPTANCE | 1200 | 1200 | PASS | Exact match |
| ROW_COUNT | ACCEPTANCE | 10 | 10 | PASS | Exact match |
