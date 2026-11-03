# Part C validation report

Run: 2026-11-04 02:47 · **116/118 passed** · PASS 116 · FAIL 0 · ERROR 0 · WARN 2

FAIL = graded criterion not met · ERROR = value could not be computed · WARN = differs from the pinned-configuration reference (graded on the band, not this number)

| Metric | Type | Reference | Generated | Status | Comment |
|---|---|---|---|---|---|
| default_rf_f1_val_reference | REFERENCE | 0.8 ± 0.001 | 0.7848 | **WARN** | Off by -0.0152 (allowed ±0.001) |
| randomized_search_f1_val_reference | REFERENCE | 0.79 ± 0.001 | 0.7848 | **WARN** | Off by -0.0052 (allowed ±0.001) |
| row_count | ACCEPTANCE | 1200 | 1200 | PASS | Exact match |
| train_row_count | ACCEPTANCE | 840 | 840 | PASS | Exact match |
| val_row_count | ACCEPTANCE | 180 | 180 | PASS | Exact match |
| test_row_count | ACCEPTANCE | 180 | 180 | PASS | Exact match |
| delayed_train_count | ACCEPTANCE | 195 | 195 | PASS | Exact match |
| delayed_val_count | ACCEPTANCE | 42 | 42 | PASS | Exact match |
| delayed_test_count | ACCEPTANCE | 41 | 41 | PASS | Exact match |
| stratified_split_both_calls | ACCEPTANCE | ≥ 2 | 2 | PASS | Meets ≥ 2 |
| split_configuration | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| models_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| models_row_count | ACCEPTANCE | 4 | 4 | PASS | Exact match |
| models_rows_exact | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| every_model_confusion_consistent | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| blind_tn_val | ACCEPTANCE | 138 | 138 | PASS | Exact match |
| blind_fp_val | ACCEPTANCE | 0 | 0 | PASS | Exact match |
| blind_fn_val | ACCEPTANCE | 42 | 42 | PASS | Exact match |
| blind_tp_val | ACCEPTANCE | 0 | 0 | PASS | Exact match |
| blind_accuracy_val | ACCEPTANCE | 0.7667 ± 0.001 | 0.7667 | PASS | Within ±0.001 (diff +0) |
| blind_precision_written_as_zero | ACCEPTANCE | 0 | 0 | PASS | Exact match |
| logistic_regression_f1_val_reference | REFERENCE | 0.805 ± 0.001 | 0.8046 | PASS | Within ±0.001 (diff -0.0004) |
| logistic_regression_f1_val_minimum | ACCEPTANCE | ≥ 0.65 | 0.8046 | PASS | Meets ≥ 0.65 |
| best_tree_depth_reference | REFERENCE | 6 | 6 | PASS | Exact match |
| decision_tree_f1_val_reference | REFERENCE | 0.747 ± 0.001 | 0.747 | PASS | Within ±0.001 (diff +0) |
| decision_tree_f1_val_minimum | ACCEPTANCE | ≥ 0.6 | 0.747 | PASS | Meets ≥ 0.6 |
| default_rf_f1_val_minimum | ACCEPTANCE | ≥ 0.65 | 0.7848 | PASS | Meets ≥ 0.65 |
| trained_models_accuracy_above_blind | ACCEPTANCE | > 0.7667 | 0.8833 | PASS | Meets > 0.7667 |
| logistic_regression_accuracy_val_minimum | REFERENCE | ≥ 0.883 | 0.9056 | PASS | Meets ≥ 0.883 |
| logistic_regression_accuracy_val_maximum | REFERENCE | ≤ 0.911 | 0.9056 | PASS | Meets ≤ 0.911 |
| decision_tree_accuracy_val_minimum | REFERENCE | ≥ 0.883 | 0.8833 | PASS | Meets ≥ 0.883 |
| decision_tree_accuracy_val_maximum | REFERENCE | ≤ 0.911 | 0.8833 | PASS | Meets ≤ 0.911 |
| default_rf_accuracy_val_minimum | REFERENCE | ≥ 0.883 | 0.9056 | PASS | Meets ≥ 0.883 |
| default_rf_accuracy_val_maximum | REFERENCE | ≤ 0.911 | 0.9056 | PASS | Meets ≤ 0.911 |
| models_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| feature_scores_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| feature_scores_row_count | ACCEPTANCE | 17 | 17 | PASS | Exact match |
| feature_names_unique | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| feature_scores_finite | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| f_rank_follows_f_score | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| top_seven_features_exact | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| selected_feature_distance_km_position | REFERENCE | 1 | 1 | PASS | Exact match |
| selected_feature_hub_tier_position | REFERENCE | 2 | 2 | PASS | Exact match |
| selected_feature_hub_daily_capacity_position | REFERENCE | 3 | 3 | PASS | Exact match |
| selected_feature_n_stops_position | REFERENCE | 4 | 4 | PASS | Exact match |
| selected_feature_promised_hours_position | REFERENCE | 5 | 5 | PASS | Exact match |
| selected_feature_rain_flag_position | REFERENCE | 6 | 6 | PASS | Exact match |
| selected_feature_sale_event_flag_position | REFERENCE | 7 | 7 | PASS | Exact match |
| top_seven_min_f_score_minimum | ACCEPTANCE | > 30 | 36.7281 | PASS | Meets > 30 |
| noise_columns_identified | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| every_noise_column_f_rank_minimum | ACCEPTANCE | ≥ 8 | 8 | PASS | Meets ≥ 8 |
| every_noise_column_f_score_strict_maximum | ACCEPTANCE | < 10 | 4.5847 | PASS | Meets < 10 |
| combined_noise_importance_reference | REFERENCE | 0.225 ± 0.001 | 0.2245 | PASS | Within ±0.001 (diff -0.0005) |
| combined_noise_importance_minimum | ACCEPTANCE | ≥ 0.15 | 0.2245 | PASS | Meets ≥ 0.15 |
| combined_noise_importance_maximum | ACCEPTANCE | ≤ 0.3 | 0.2245 | PASS | Meets ≤ 0.3 |
| forest_importance_sums_to_one | ACCEPTANCE | 1 ± 0.002 | 1.0002 | PASS | Within ±0.002 (diff +0.0002) |
| feature_scores_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| selectk_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| selectk_row_count | ACCEPTANCE | 5 | 5 | PASS | Exact match |
| required_k_values_present | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| validation_f1_values_valid | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| selectk_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| tuning_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| tuning_row_count | ACCEPTANCE | 3 | 3 | PASS | Exact match |
| required_search_rows_present | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| grid_search_candidates | ACCEPTANCE | 27 | 27 | PASS | Exact match |
| grid_search_cv_folds | ACCEPTANCE | 5 | 5 | PASS | Exact match |
| grid_search_total_fits | ACCEPTANCE | 135 | 135 | PASS | Exact match |
| randomized_search_candidates | ACCEPTANCE | 9 | 9 | PASS | Exact match |
| randomized_search_cv_folds | ACCEPTANCE | 5 | 5 | PASS | Exact match |
| randomized_search_total_fits | ACCEPTANCE | 45 | 45 | PASS | Exact match |
| default_forest_same_7_features_candidates | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| default_forest_same_7_features_cv_folds | ACCEPTANCE | 0 | 0 | PASS | Exact match |
| default_forest_same_7_features_total_fits | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| default_forest_best_params_is_defaults | ACCEPTANCE | defaults | defaults | PASS | Matches |
| default_forest_best_cv_accuracy_blank | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| grid_search_best_cv_accuracy_reference | REFERENCE | 0.8952 ± 0.0001 | 0.8952 | PASS | Within ±0.0001 (diff +0) |
| grid_search_best_cv_accuracy_minimum | ACCEPTANCE | ≥ 0.86 | 0.8952 | PASS | Meets ≥ 0.86 |
| grid_search_best_cv_accuracy_maximum | ACCEPTANCE | ≤ 0.93 | 0.8952 | PASS | Meets ≤ 0.93 |
| randomized_search_best_cv_accuracy_reference | REFERENCE | 0.8905 ± 0.0001 | 0.8905 | PASS | Within ±0.0001 (diff +0) |
| randomized_search_best_cv_accuracy_minimum | ACCEPTANCE | ≥ 0.86 | 0.8905 | PASS | Meets ≥ 0.86 |
| randomized_search_best_cv_accuracy_maximum | ACCEPTANCE | ≤ 0.93 | 0.8905 | PASS | Meets ≤ 0.93 |
| grid_best_cv_accuracy_at_least_randomized | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| grid_search_f1_val_reference | REFERENCE | 0.775 ± 0.001 | 0.775 | PASS | Within ±0.001 (diff +0) |
| grid_search_f1_val_minimum | ACCEPTANCE | ≥ 0.65 | 0.775 | PASS | Meets ≥ 0.65 |
| randomized_search_f1_val_minimum | ACCEPTANCE | ≥ 0.65 | 0.7848 | PASS | Meets ≥ 0.65 |
| default_forest_same_7_features_f1_val_reference | REFERENCE | 0.833 ± 0.001 | 0.8333 | PASS | Within ±0.001 (diff +0.0003) |
| default_forest_same_7_features_f1_val_minimum | ACCEPTANCE | ≥ 0.65 | 0.8333 | PASS | Meets ≥ 0.65 |
| tuning_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| thresholds_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| thresholds_row_count | ACCEPTANCE | 9 | 9 | PASS | Exact match |
| required_thresholds_present | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| thresholds_cover_validation_rows | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| threshold_metrics_consistent | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| recall_nonincreasing_with_threshold | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| thresholds_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| thresholds_not_unrounded_arange | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| final_config_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| final_config_row_count | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| final_config_selector_k | ACCEPTANCE | 7 | 7 | PASS | Exact match |
| chosen_from_follows_tie_rule | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| final_params_match_chosen_row | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| chosen_from_reference | REFERENCE | default_forest_same_7_features | default_forest_same_7_features | PASS | Matches |
| threshold_rule_satisfied | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| selected_threshold | REFERENCE | 0.5 | 0.5 | PASS | Exact match |
| final_test_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| final_test_row_count | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| threshold_matches_config | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| confusion_metrics_consistent | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| test_precision_reference | REFERENCE | 0.6889 ± 0.0001 | 0.6889 | PASS | Within ±0.0001 (diff +0) |
| test_precision_minimum | ACCEPTANCE | ≥ 0.45 | 0.6889 | PASS | Meets ≥ 0.45 |
| test_recall_reference | REFERENCE | 0.7561 ± 0.0001 | 0.7561 | PASS | Within ±0.0001 (diff +0) |
| test_recall_minimum | ACCEPTANCE | ≥ 0.55 | 0.7561 | PASS | Meets ≥ 0.55 |
| test_f1_reference | REFERENCE | 0.7209 ± 0.0001 | 0.7209 | PASS | Within ±0.0001 (diff +0) |
| test_f1_minimum | ACCEPTANCE | ≥ 0.6 | 0.7209 | PASS | Meets ≥ 0.6 |
| final_test_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| no_fit_on_val_or_test_rows | ACCEPTANCE | none | none | PASS | Matches |
| test_rows_scored_once | ACCEPTANCE | 1 | 1 | PASS | Exact match |
