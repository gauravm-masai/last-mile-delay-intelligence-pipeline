[← Back to Project Overview](../README.md)

# Part C - Delay Classifier and Alert Threshold

## Objectives

- Establish a blind baseline for predicting parcel delays.
- Compare Logistic Regression, Decision Tree and Random Forest classifiers using validation metrics.
- Evaluate feature importance, identify noise features and study the effect of feature selection.
- Compare default and tuned Random Forest models and select a classification threshold using validation results.
- Evaluate the final model on the held-out test set.

## High Level Approach

- Split the data into training, validation and test sets using two-stage stratified splitting.
- Evaluate the blind baseline and trained models using confusion matrices, accuracy, precision, recall and F1-score.
- Rank features using F-scores and Random Forest feature importance, including the combined importance of the five noise columns.
- Compare validation F1-scores for different values of `k` in the feature-selection experiment.
- Compare the default Random Forest with grid search and randomized search.
- Evaluate thresholds from 0.1 to 0.9 and select the highest threshold that achieves validation recall of at least 0.80, using 0.1 if none qualifies.
- Evaluate the selected configuration once on the held-out test set.

**Leakage prevention:** Feature selection is kept inside the modelling pipeline during cross-validation. This ensures that the selector is fitted separately on each cross-validation training fold rather than using information from that fold's validation data. Although this differs from fitting the selector once before the search, it preserves the independence of validation folds and avoids feature-selection leakage. 

## Generated outputs

Part C generates the following files:

## Data Findings

**Baseline and classifier comparison**

The prescribed stratified split contains 1,200 shipments: 840 training, 180 validation and 180 test rows. The delayed-parcel counts are 195, 42 and 41, respectively.

The blind baseline predicts every parcel as on time. It achieves accuracy of **0.7667**, but identifies no delayed parcels: its recall and F1-score for the delayed class are both zero. This demonstrates why accuracy alone is insufficient for the operational objective.

| Model | Validation accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree, depth 6 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Default Random Forest, 17 features | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The decision tree depth experiment selected depth 6. Logistic Regression achieved the highest F1-score among these three classifier results, while the default Random Forest achieved higher precision but lower recall than Logistic Regression.

**Feature importance and noise features**

The feature analysis uses F-scores from `SelectKBest(f_classif, k="all")` and feature importances from a default Random Forest with 100 estimators and `random_state=42`.

The seven highest-ranked features by F-score are:

1. `distance_km`
2. `hub_tier`
3. `hub_daily_capacity`
4. `n_stops`
5. `promised_hours`
6. `rain_flag`
7. `sale_event_flag`

The five noise columns collectively account for approximately **22.45%** of Random Forest feature importance. Each noise column ranks outside the top seven by F-score. This illustrates that a feature can receive some importance from a model without being among the strongest features under the univariate F-score ranking.

**Feature-selection experiment**

The validation F1-scores for the required feature counts were:

| Number of features (`k`) | Validation F1-score |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

The best result was achieved with **seven features**, with a validation F1-score of **0.8333**. This selected feature count was used in the subsequent default-forest comparison and final configuration.

**Hyperparameter tuning**

The grid search evaluated 27 candidate configurations across five folds, for 135 fits. The randomized search evaluated nine candidates across five folds, for 45 fits.

| Search configuration | Candidates | CV folds | Total fits | Best CV accuracy | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Grid Search | 27 | 5 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 5 | 45 | 0.8905 | 0.7848 |
| Default Random Forest using seven selected features | 1 | Not applicable | 1 | Not applicable | 0.8333 |

Grid Search and Randomized Search were evaluated using five-fold cross-validation. The default Random Forest using seven selected features was a separate experiment and did not run cross-validation.

Neither tuning search improved validation F1 over the default forest using the selected seven features. The results therefore support retaining the simpler default configuration for this experiment rather than assuming that hyperparameter tuning must improve performance.

**Reference benchmark discrepancy**

The PDF benchmark for `default_rf_validation_f1` is **0.800**, with tolerance **±0.001**. The generated value for `random_forest_default`, which uses all 17 features, is **0.7848**. This does not match the benchmark.

However, the generated value exceeds the separate minimum acceptance criterion of **0.65**. These are distinct checks: meeting the minimum does not mean the benchmark matched. The reference value remains unchanged, and the discrepancy is retained for transparent reporting.

The seven-feature default-forest result of **0.8333** is a separate experiment; it must not be substituted for the all-feature default Random Forest result when reporting the benchmark comparison.

**Alert threshold selection**

The threshold experiment evaluates values from 0.1 to 0.9. The selected threshold is **0.5**, the highest tested threshold whose validation recall is at least 0.80.

At threshold 0.5, validation recall is **0.8333**, with precision **0.7955**. At threshold 0.6, recall falls to **0.7619**, below the required 0.80. Thus, 0.5 satisfies the selection rule while 0.6 does not.

Lower thresholds generally identify more delayed parcels but generate more false positives. Higher thresholds can improve precision, but risk missing additional delays. The chosen threshold reflects the project's recall requirement rather than simply maximizing accuracy.

**Final held-out test evaluation**

The final configuration uses seven selected features, a Random Forest with 100 estimators, unlimited maximum depth and `min_samples_leaf=1`, with the alert threshold set to 0.5.

On the held-out test set of 180 shipments, including 41 delayed parcels, the model achieved:

| Metric | Test result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1-score | 0.7209 |

Test recall is approximately **7.72 percentage points lower** than the validation recall of 0.8333 at the selected threshold. The test set contains only 41 delayed parcels, so the result should be interpreted in that context. The test metrics are calculated from the saved test confusion matrix; the held-out test set is not used to choose the threshold.

## Validations

Part C validates the prescribed stratified data split, classification model performance, feature analysis, feature-selection experiments, hyperparameter-tuning results, alert-threshold selection, and final held-out test metrics against reference values and acceptance criteria.

### Reference benchmark discrepancy

The reference for the all-feature default Random Forest validation F1 score is **0.800 ± 0.001**. The generated value is **0.7848**, so the reference comparison is a `MISMATCH`.

The generated F1 score nevertheless exceeds the separate minimum acceptance criterion of **0.65**. Therefore, the acceptance check passes even though the reference benchmark does not match.  

### Reference metrics compared with generated outputs

| Source file | Metric | PDF reference value | Code-generated value | Tolerance | Status | Comments |
|---|---|---:|---:|---|---|---|
| `data/modelling_table.csv` | Split Sizes | 840 / 180 / 180 | 840 / 180 / 180 | Exact | PASS | Matches the prescribed stratified training, validation and test split. |
| `data/modelling_table.csv` | Delayed parcels | 195 / 42 / 41 | 195 / 42 / 41 | Exact | PASS | Matches the delayed parcel count in each split. |
| `outputs/part_c_models.csv` | Blind baseline accuracy | 0.7667 | 0.7667 | ±0.001 | PASS | The all-on-time baseline matches the reference within tolerance. |
| `outputs/part_c_models.csv` | Blind baseline confusion matrix (TN, FP, FN, TP) | 138, 0, 42, 0 | 138, 0, 42, 0 | Exact | PASS | The baseline predicts all 180 validation shipments as on time. |
| `outputs/part_c_models.csv` | Selected Decision Tree depth | 6 | 6 | Exact | PASS | Depth 6 is the selected tree configuration. |
| `outputs/part_c_models.csv` | Decision Tree validation F1 | 0.747 | 0.7470 | ±0.001 | PASS | The selected tree's validation F1 matches the reference within tolerance and exceeds the minimum acceptance criterion of 0.60. |
| `outputs/part_c_models.csv` | Default Random Forest validation F1 - 17 features | 0.800 | 0.7848 | ±0.001 | MISMATCH | The generated result does not match the given benchmark. The separate minimum acceptance criterion of 0.65 passes. |
| `outputs/part_c_models.csv` | Trained-model validation accuracy | 0.883 - 0.911 | 0.9056 / 0.8833 / 0.9056 | Within specified range | PASS | The validation accuracies for Logistic Regression / Decision Tree / Random Forest are within prescribed range. |
| `outputs/part_c_models.csv` | Logistic Regression validation F1 | 0.805 | 0.8046 | ≥ 0.65 | PASS | Logistic Regression exceeds the minimum F1 acceptance criterion. |
| `outputs/part_c_feature_scores.csv` | Number of feature-score records | 17 | 17 | Exact | PASS | Feature analysis covers all 17 model features. |
| `outputs/part_c_feature_scores.csv` | Top-seven feature ranking | Prescribed order | Matches prescribed order | Exact | PASS | The top seven are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`. |
| `outputs/part_c_feature_scores.csv` | Minimum F-score among the top seven | > 30 | 36.7281 | > 30 | PASS | The minimum F-score (36.7281 for `sale_event_flag`) exceeds 30. |
| `outputs/part_c_feature_scores.csv` | Noise-feature F-score and ranking | Each noise feature has F-score < 10 and ranks below the top seven | Meets the prescribed conditions | Exact condition | PASS | The noise features remain outside the top seven by F-score, with each F-score below 10. |
| `outputs/part_c_feature_scores.csv` | Combined noise-feature Random Forest importance | 0.225 | 0.2245 | ±0.0001 | PASS | The generated value matches the reference (after rounding to 3 decimal places) and is also within acceptance range of 0.15 - 0.30. |
| `outputs/part_c_selectk.csv` | Feature-selection experiment values of k | 5, 7, 10, 12, 17 | 5, 7, 10, 12, 17 | Exact | PASS | All five prescribed feature-selection experiments are present. |
| `outputs/part_c_selectk.csv` | Validation F1 for k = 7 | Maximum amongst val F1 for all K values | 0.8333 | Maximum amongst val F1 for all K values | PASS | Validation F1 for `k = 7` is the highest compared to validation F1 for all other values of k. |
| `outputs/part_c_tuning.csv` | Grid Search | 27 / 5 / 135 | 27 / 5 / 135 | Exact | PASS | The prescribed grid contains 27 candidate configurations, 5 cross-validation folds, and 135 total fits. |
| `outputs/part_c_tuning.csv` | Grid Search best CV accuracy | 0.8952 | 0.8952 | 0.86 to 0.93 | PASS | The generated cross-validation accuracy matches the reference and is within the prescribed range. |
| `outputs/part_c_tuning.csv` | Grid Search validation F1 acceptance | 0.775 | 0.7750 | ≥ 0.65 | PASS | The validation F1 exceeds the minimum acceptance criterion. |
| `outputs/part_c_tuning.csv` | Randomized Search candidates | 9 / 5 / 45 | 9 / 5 / 45 | Exact | PASS | The prescribed randomized search evaluates 9 candidate configurations, 5 cross-validation folds, and 45 total fits (one-third of fits done by grid search). |
| `outputs/part_c_tuning.csv` | Randomized Search best CV accuracy | 0.8905 | 0.8905 | 0.86 to 0.93 | PASS | The generated cross-validation accuracy matches the reference and is within the prescribed range. |
| `outputs/part_c_tuning.csv` | Randomized Search validation F1 | 0.790 | 0.7848 | 0.0001 | MISMATCH | The generated F1 differs from the reference beyond tolerance. The separate minimum acceptance criterion of 0.65 passes. |
| `outputs/part_c_tuning.csv` | Default Random Forest validation F1 - seven selected features | 0.833 | 0.8333 | 0.0001 | PASS | The generated value matches the reference (after rounding to 3 decimal places) and the separate minimum acceptance criterion also passes. |
| `outputs/part_c_thresholds.csv` | Thresholds evaluated | 0.1-0.9 | 0.1-0.9 | Exact | PASS | All nine prescribed alert thresholds are evaluated. |
| `outputs/part_c_final_config.csv` | Selected alert threshold | 0.5 | 0.5 | Exact | PASS | The saved threshold is 0.5. |
| `outputs/part_c_final_config.csv` | Threshold-selection rule | Highest tested threshold with validation recall ≥ 0.80; fallback 0.1 | 0.5; validation recall 0.8333 | Exact rule | PASS | Threshold 0.5 meets the recall requirement, while threshold 0.6 has validation recall 0.7619. |
| `outputs/part_c_final_test.csv` | Test precision | 0.6889 | 0.6889 | 0.0001 | PASS | Matches the reference and exceeds the minimum acceptance criterion of 0.45. |
| `outputs/part_c_final_test.csv` | Test recall | 0.7561 | 0.7561 | 0.0001 | PASS | Matches the reference and exceeds the minimum acceptance criterion of 0.55. |
| `outputs/part_c_final_test.csv` | Test F1 | 0.7209 | 0.7209 | 0.0001 | PASS | Matches the reference and exceeds the minimum acceptance criterion of 0.60. |

**Validation conclusion:** The Part C reference comparisons confirm that the prescribed split, feature-analysis structure, feature-selection experiments, tuning-search structure, threshold rule and final-test acceptance criteria are satisfied. The all-feature default Random Forest F1 benchmark remains a documented `MISMATCH`; its separate minimum criterion passes.

[← Back to Project Overview](../README.md)