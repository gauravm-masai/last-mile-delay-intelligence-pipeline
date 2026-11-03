# Part B validation report

Run: 2026-11-04 02:47 · **70/70 passed** · PASS 70 · FAIL 0 · ERROR 0 · WARN 0

FAIL = graded criterion not met · ERROR = value could not be computed · WARN = differs from the pinned-configuration reference (graded on the band, not this number)

| Metric | Type | Reference | Generated | Status | Comment |
|---|---|---|---|---|---|
| checks_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| checks_row_count | ACCEPTANCE | 5 | 5 | PASS | Exact match |
| train_rows | ACCEPTANCE | 840 | 840 | PASS | Exact match |
| val_rows | ACCEPTANCE | 180 | 180 | PASS | Exact match |
| test_rows | ACCEPTANCE | 180 | 180 | PASS | Exact match |
| train_scaled_max_abs_mean_max | ACCEPTANCE | < 0.000000001 | 0 | PASS | Meets < 0.000000001 |
| test_scaled_max_abs_mean_min | ACCEPTANCE | > 0.01 | 0.2187 | PASS | Meets > 0.01 |
| checks_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| metrics_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| metrics_row_count | ACCEPTANCE | 6 | 6 | PASS | Exact match |
| metrics_val_row_count | ACCEPTANCE | 5 | 5 | PASS | Exact match |
| metrics_test_row_count | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| baseline_present_val | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| linear_regression_present_val | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| decision_tree_present_val | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| random_forest_present_val | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| gradient_boosting_present_val | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| final_model_is_lowest_val_rmse | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| rmse_mse_consistency | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| baseline_r2_val | REFERENCE | -0.021 ± 0.01 | -0.0212 | PASS | Within ±0.01 (diff -0.0002) |
| baseline_r2_val_minimum | ACCEPTANCE | ≥ -0.1 | -0.0212 | PASS | Meets ≥ -0.1 |
| baseline_r2_val_maximum | ACCEPTANCE | ≤ 0.02 | -0.0212 | PASS | Meets ≤ 0.02 |
| linear_regression_r2_val | REFERENCE | 0.931 ± 0.01 | 0.9313 | PASS | Within ±0.01 (diff +0.0003) |
| linear_regression_r2_val_minimum | ACCEPTANCE | ≥ 0.88 | 0.9313 | PASS | Meets ≥ 0.88 |
| decision_tree_r2_val | REFERENCE | 0.887 ± 0.01 | 0.8867 | PASS | Within ±0.01 (diff -0.0003) |
| decision_tree_r2_val_minimum | ACCEPTANCE | ≥ 0.82 | 0.8867 | PASS | Meets ≥ 0.82 |
| random_forest_r2_val | REFERENCE | 0.914 ± 0.01 | 0.9141 | PASS | Within ±0.01 (diff +0.0001) |
| random_forest_r2_val_minimum | ACCEPTANCE | ≥ 0.86 | 0.9141 | PASS | Meets ≥ 0.86 |
| gradient_boosting_r2_val | REFERENCE | 0.931 ± 0.01 | 0.9306 | PASS | Within ±0.01 (diff -0.0004) |
| gradient_boosting_r2_val_minimum | ACCEPTANCE | ≥ 0.88 | 0.9306 | PASS | Meets ≥ 0.88 |
| final_model_r2_test | REFERENCE | 0.949 ± 0.01 | 0.9493 | PASS | Within ±0.01 (diff +0.0003) |
| final_model_r2_test_minimum | ACCEPTANCE | ≥ 0.88 | 0.9493 | PASS | Meets ≥ 0.88 |
| decision_tree_rmse_val_best_depth_abs_diff_max | ACCEPTANCE | ≤ 0.001 | 0 | PASS | Meets ≤ 0.001 |
| metrics_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| tree_depth_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| tree_depth_row_count | ACCEPTANCE | 8 | 8 | PASS | Exact match |
| tree_depths_exact_order | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| decision_tree_train_rmse_nonincreasing_depths | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| decision_tree_best_depth_val | REFERENCE | 6 | 6 | PASS | Exact match |
| decision_tree_best_depth_val_minimum | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| decision_tree_best_rmse_val | REFERENCE | 6.28 ± 0.01 | 6.2796 | PASS | Within ±0.01 (diff -0.0004) |
| decision_tree_rmse_gap_depth_1 | REFERENCE | 0.5 ± 0.01 | 0.499 | PASS | Within ±0.01 (diff -0.001) |
| decision_tree_rmse_gap_depth_1_maximum | ACCEPTANCE | ≤ 2.5 | 0.499 | PASS | Meets ≤ 2.5 |
| decision_tree_rmse_gap_depth_1_vs_best | REFERENCE | 5.85 ± 0.01 | 5.8528 | PASS | Within ±0.01 (diff +0.0028) |
| decision_tree_rmse_gap_depth_1_vs_best_minimum | ACCEPTANCE | ≥ 4 | 5.8528 | PASS | Meets ≥ 4 |
| decision_tree_train_rmse_unlimited | REFERENCE | 0 ± 0.01 | 0 | PASS | Within ±0.01 (diff +0) |
| decision_tree_val_rmse_unlimited | REFERENCE | 7.79 ± 0.01 | 7.789 | PASS | Within ±0.01 (diff -0.001) |
| decision_tree_rmse_gap_unlimited | REFERENCE | 7.79 ± 0.01 | 7.789 | PASS | Within ±0.01 (diff -0.001) |
| decision_tree_rmse_gap_unlimited_minimum | ACCEPTANCE | ≥ 5 | 7.789 | PASS | Meets ≥ 5 |
| tree_depth_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| gb_learning_rate_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| gb_learning_rate_row_count | ACCEPTANCE | 3 | 3 | PASS | Exact match |
| gb_learning_rates_exact | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| gradient_boosting_r2_ranking_correct | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| gradient_boosting_r2_val_lr_0.01 | REFERENCE | 0.713 ± 0.01 | 0.7127 | PASS | Within ±0.01 (diff -0.0003) |
| gradient_boosting_r2_val_lr_0.01_maximum | ACCEPTANCE | ≤ 0.8 | 0.7127 | PASS | Meets ≤ 0.8 |
| gradient_boosting_r2_val_lr_0.1 | REFERENCE | 0.931 ± 0.01 | 0.9306 | PASS | Within ±0.01 (diff -0.0004) |
| gradient_boosting_r2_val_lr_0.1_minimum | ACCEPTANCE | ≥ 0.9 | 0.9306 | PASS | Meets ≥ 0.9 |
| gradient_boosting_r2_val_lr_1.0 | REFERENCE | 0.844 ± 0.01 | 0.844 | PASS | Within ±0.01 (diff +0) |
| gb_learning_rate_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| cv_schema_matches | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| cv_row_count | ACCEPTANCE | 2 | 2 | PASS | Exact match |
| cv_models_are_top2_by_val_rmse | ACCEPTANCE | 1 | 1 | PASS | Exact match |
| linear_regression_mean_r2_cv | REFERENCE | 0.926 ± 0.01 | 0.9261 | PASS | Within ±0.01 (diff +0.0001) |
| gradient_boosting_mean_r2_cv | REFERENCE | 0.935 ± 0.01 | 0.9352 | PASS | Within ±0.01 (diff +0.0002) |
| linear_regression_std_r2_cv | REFERENCE | 0.0048 ± 0.0001 | 0.0048 | PASS | Within ±0.0001 (diff +0) |
| gradient_boosting_std_r2_cv | REFERENCE | 0.0044 ± 0.0001 | 0.0044 | PASS | Within ±0.0001 (diff +0) |
| cv_mean_r2_minimum | ACCEPTANCE | ≥ 0.9 | 0.9261 | PASS | Meets ≥ 0.9 |
| cv_std_r2_maximum | ACCEPTANCE | ≤ 0.03 | 0.0048 | PASS | Meets ≤ 0.03 |
| cv_csv_clean | ACCEPTANCE | 1 | 1 | PASS | Exact match |
