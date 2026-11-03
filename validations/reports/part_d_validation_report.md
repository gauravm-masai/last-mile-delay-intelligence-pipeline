# Part D validation report

Run: 2026-11-04 02:47 · **102/102 passed** · PASS 102 · FAIL 0 · ERROR 0 · WARN 0

FAIL = graded criterion not met · ERROR = value could not be computed · WARN = differs from the pinned-configuration reference (graded on the band, not this number)

| Metric | Type | Reference | Generated | Status | Comment |
|---|---|---|---|---|---|
| hub_profile_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| row_count | ACCEPTANCE | 24 | 24 | PASS | Exact match |
| sorted_by_hub_id | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| every_hub_has_50_shipments | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| HUB01.mean_delivery_hours | ACCEPTANCE | 45.685 ± 0.01 | 45.685 | PASS | Within ±0.01 (diff +0) |
| HUB01.delay_rate | ACCEPTANCE | 0.02 ± 0.01 | 0.02 | PASS | Within ±0.01 (diff +0) |
| HUB01.mean_distance_km | ACCEPTANCE | 185.061 ± 0.01 | 185.061 | PASS | Within ±0.01 (diff +0) |
| HUB01.mean_n_stops | ACCEPTANCE | 54.7 ± 0.01 | 54.7 | PASS | Within ±0.01 (diff +0) |
| HUB13.mean_delivery_hours | ACCEPTANCE | 66.074 ± 0.01 | 66.0742 | PASS | Within ±0.01 (diff +0.0002) |
| HUB13.delay_rate | ACCEPTANCE | 0.46 ± 0.01 | 0.46 | PASS | Within ±0.01 (diff +0) |
| HUB13.mean_distance_km | ACCEPTANCE | 473.32 ± 0.01 | 473.32 | PASS | Within ±0.01 (diff +0) |
| HUB13.mean_n_stops | ACCEPTANCE | 83.92 ± 0.01 | 83.92 | PASS | Within ±0.01 (diff +0) |
| HUB17.mean_delivery_hours | ACCEPTANCE | 60.851 ± 0.01 | 60.8514 | PASS | Within ±0.01 (diff +0.0004) |
| HUB17.delay_rate | ACCEPTANCE | 0.46 ± 0.01 | 0.46 | PASS | Within ±0.01 (diff +0) |
| HUB17.mean_distance_km | ACCEPTANCE | 443.285 ± 0.01 | 443.285 | PASS | Within ±0.01 (diff +0) |
| HUB17.mean_n_stops | ACCEPTANCE | 84.18 ± 0.01 | 84.18 | PASS | Within ±0.01 (diff +0) |
| hub_profile_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| separation_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| row_count | ACCEPTANCE | 4 | 4 | PASS | Exact match |
| k_values_exact | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| cluster_sizes_well_formed | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| K2.cluster_sizes | ACCEPTANCE | 15-9 | 15-9 | PASS | Matches |
| K3.cluster_sizes | ACCEPTANCE | 9-8-7 | 9-8-7 | PASS | Matches |
| K4.cluster_sizes | ACCEPTANCE | 8-7-5-4 | 8-7-5-4 | PASS | Matches |
| K5.cluster_sizes | ACCEPTANCE | 8-7-4-3-2 | 8-7-4-3-2 | PASS | Matches |
| K2.across_over_within | ACCEPTANCE | 2.894 ± 0.01 | 2.894 | PASS | Within ±0.01 (diff +0) |
| K3.across_over_within | ACCEPTANCE | 5.393 ± 0.01 | 5.3928 | PASS | Within ±0.01 (diff -0.0002) |
| K4.across_over_within | ACCEPTANCE | 6.305 ± 0.01 | 6.3047 | PASS | Within ±0.01 (diff -0.0003) |
| K5.across_over_within | ACCEPTANCE | 6.887 ± 0.01 | 6.8866 | PASS | Within ±0.01 (diff -0.0004) |
| k4_k5_have_cluster_under_6_hubs | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| separation_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| hub_segments_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| row_count | ACCEPTANCE | 24 | 24 | PASS | Exact match |
| segment_names_valid | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| segment_profile_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| row_count | ACCEPTANCE | 3 | 3 | PASS | Exact match |
| names_follow_delay_rate_order | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| hubs_match_hub_segments | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| Standard.hubs | ACCEPTANCE | 7 | 7 | PASS | Exact match |
| Standard.mean_delivery_hours | ACCEPTANCE | 47.24 ± 0.01 | 47.2396 | PASS | Within ±0.01 (diff -0.0004) |
| Standard.delay_rate | ACCEPTANCE | 0.0257 ± 0.01 | 0.0257 | PASS | Within ±0.01 (diff +0) |
| Standard.mean_distance_km | ACCEPTANCE | 188.93 ± 0.01 | 188.93 | PASS | Within ±0.01 (diff -0.0004) |
| Standard.mean_n_stops | ACCEPTANCE | 55.37 ± 0.01 | 55.3686 | PASS | Within ±0.01 (diff -0.0014) |
| Watch.hubs | ACCEPTANCE | 8 | 8 | PASS | Exact match |
| Watch.mean_delivery_hours | ACCEPTANCE | 56.81 ± 0.01 | 56.8137 | PASS | Within ±0.01 (diff +0.0037) |
| Watch.delay_rate | ACCEPTANCE | 0.155 ± 0.01 | 0.155 | PASS | Within ±0.01 (diff +0) |
| Watch.mean_distance_km | ACCEPTANCE | 314.18 ± 0.01 | 314.176 | PASS | Within ±0.01 (diff -0.0038) |
| Watch.mean_n_stops | ACCEPTANCE | 70.72 ± 0.01 | 70.7175 | PASS | Within ±0.01 (diff -0.0025) |
| Escalate.hubs | ACCEPTANCE | 9 | 9 | PASS | Exact match |
| Escalate.mean_delivery_hours | ACCEPTANCE | 64.65 ± 0.01 | 64.6504 | PASS | Within ±0.01 (diff +0.0004) |
| Escalate.delay_rate | ACCEPTANCE | 0.46 ± 0.01 | 0.46 | PASS | Within ±0.01 (diff +0) |
| Escalate.mean_distance_km | ACCEPTANCE | 470.48 ± 0.01 | 470.476 | PASS | Within ±0.01 (diff -0.0043) |
| Escalate.mean_n_stops | ACCEPTANCE | 84.34 ± 0.01 | 84.3444 | PASS | Within ±0.01 (diff +0.0044) |
| segment_profile_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| mismatch_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| row_count | ACCEPTANCE | 4 | 4 | PASS | Exact match |
| mismatch_hubs_exact | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| mismatch_follows_tier_rule | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| HUB07.hub_city | ACCEPTANCE | Kolkata | Kolkata | PASS | Matches |
| HUB07.hub_tier | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| HUB07.segment | ACCEPTANCE | Watch | Watch | PASS | Matches |
| HUB13.hub_city | ACCEPTANCE | Nagpur | Nagpur | PASS | Matches |
| HUB13.hub_tier | ACCEPTANCE | 2 | 2 | PASS | Exact match |
| HUB13.segment | ACCEPTANCE | Escalate | Escalate | PASS | Matches |
| HUB15.hub_city | ACCEPTANCE | Kochi | Kochi | PASS | Matches |
| HUB15.hub_tier | ACCEPTANCE | 2 | 2 | PASS | Exact match |
| HUB15.segment | ACCEPTANCE | Escalate | Escalate | PASS | Matches |
| HUB22.hub_city | ACCEPTANCE | Dehradun | Dehradun | PASS | Matches |
| HUB22.hub_tier | ACCEPTANCE | 3 | 3 | PASS | Exact match |
| HUB22.segment | ACCEPTANCE | Watch | Watch | PASS | Matches |
| row_count | ACCEPTANCE | 48 | 48 | PASS | Exact match |
| missing_distance_km | ACCEPTANCE | 3 | 3 | PASS | Exact match |
| missing_parcel_weight_kg | ACCEPTANCE | 2 | 2 | PASS | Exact match |
| missing_declared_value | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| missing_payment_mode | ACCEPTANCE | 2 | 2 | PASS | Exact match |
| service_type_unique_spellings | ACCEPTANCE | 8 | 8 | PASS | Exact match |
| model_pickle_exists | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| threshold_file_matches_config | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| feature_order_matches_pinned_list | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| impute_values_schema | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| impute_values_match_part_a_medians | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| delay_threshold_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| feature_order_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| impute_values_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| reload_check_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| test_rows_scored | ACCEPTANCE | 180 | 180 | PASS | Exact match |
| reload_max_abs_probability_diff | ACCEPTANCE | ≤ 0.000000001 | 0 | PASS | Meets ≤ 0.000000001 |
| threshold_matches_config | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| threshold | REFERENCE | 0.5 | 0.5 | PASS | Exact match |
| reload_check_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| actions_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| row_count | ACCEPTANCE | 48 | 48 | PASS | Exact match |
| unique_shipment_ids | ACCEPTANCE | 48 | 48 | PASS | Exact match |
| no_missing_values | ACCEPTANCE | 0 | 0 | PASS | Exact match |
| same_order_as_new_shipments | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| probabilities_valid | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| flagged_follows_threshold | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| segment_matches_hub_segments | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| action_follows_rule | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| flagged_count_reference | REFERENCE | 8 | 8 | PASS | Exact match |
| escalations_reference | REFERENCE | 6 | 6 | PASS | Exact match |
| actions_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
