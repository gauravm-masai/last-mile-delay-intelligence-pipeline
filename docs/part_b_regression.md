[← Back to Project Overview](../README.md)

# Part B: Delivery-Hours Regression

## Objectives
- Establish a baseline for predicting shipment delivery hours.
- Train and compare Linear Regression, Decision Tree, Random Forest, and Gradient Boosting models.
- Study decision tree depth and gradient boosting learning rate to understand underfitting, overfitting, and model performance.
- Use cross-validation to assess the stability of the best candidate models and evaluate the selected model on the test set.

## High Level Approach
- Split the modelling dataset into training (840 rows), validation (180 rows), and test (180 rows).
- Fit `StandardScaler` on the training data only and check the scaled feature means.
- Evaluate a mean-prediction baseline using validation data.
- Compare decision trees across eight depth configurations and gradient boosting across three learning rates.
- Evaluate four regression candidates using MAE, MSE, RMSE, and R².
- Run five-fold cross-validation on the two best candidates using the same `KFold` configuration.
- Select the model with the lowest validation RMSE and evaluate its final performance on the held-out test set.

## Generated Outputs

Part B generates the following files:

- `outputs/part_b_checks.csv` - split sizes and scaling checks.
- `outputs/part_b_metrics.csv` - baseline, validation model metrics, and the selected model's final test metrics.
- `outputs/part_b_tree_depth.csv` - training and validation RMSE for the eight tree-depth configurations.
- `outputs/part_b_gb_learning_rate.csv` - validation RMSE and R² for the three learning rates.
- `outputs/part_b_cv.csv` - five-fold cross-validation mean and standard deviation of R² for the two selected candidates.

## Data Findings

**Baseline and model comparison**

- **Baseline:** The mean-prediction baseline achieved a validation R² of **-0.0212**, close to zero. It provides little predictive value compared with the trained regression models.
- **Validation winner:** Linear Regression achieved the lowest validation RMSE at **4.8912 hours**, with an R² of **0.9313**. Gradient Boosting was close behind, with an RMSE of **4.9165 hours** and R² of **0.9306**.
- **Other models:** Random Forest achieved an R² of **0.9141**, while the Decision Tree achieved **0.8867**. All four trained candidates substantially outperformed the baseline.
- **Final test evaluation:** Linear Regression was evaluated on the held-out test set once, achieving an RMSE of **4.5235 hours** and an R² of **0.9493**.

**Decision tree depth and bias-variance trade-off**

- The best depth was **6**, with a validation RMSE of **6.2796 hours**.
- At depth 1, training RMSE was **11.6334** and validation RMSE was **12.1324**, a gap of **0.4990 hours**. Both errors are high relative to the best tree, indicating underfitting and high bias.
- At unlimited depth, training RMSE fell to **0.0000**, while validation RMSE rose to **7.7890**. The gap of **7.7890 hours** indicates overfitting and high variance.
- Depth 1's validation RMSE exceeded the best depth's RMSE by **5.8528 hours**. Increasing tree depth improves performance initially, but deeper trees eventually generalize less well.

**Gradient Boosting learning-rate experiment**

| Learning rate | Validation RMSE (hours) | Validation R² |
|---:|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of **0.1** performed best. At **0.01**, the model learns too slowly with only 100 estimators, resulting in underfitting. At **1.0**, the larger updates reduce performance compared with 0.1, consistent with overshooting a useful solution.

**Cross-validation and model selection**

- Five-fold cross-validation produced a mean R² of **0.9261** for Linear Regression, with a standard deviation of **0.0048**.
- Gradient Boosting achieved a slightly higher mean R² of **0.9352**, with a standard deviation of **0.0044**.
- The cross-validation ranking therefore favours Gradient Boosting, while validation RMSE favours Linear Regression. Both models show stable cross-validation scores, with low standard deviations.
- **Linear Regression remains the selected model** because the assignment specifies selecting the candidate with the lowest validation RMSE. The model was then evaluated once on the held-out test set.

## Validations

### Acceptance criteria and results

Part B checks the prescribed train-validation-test split, training-only scaling, regression metric consistency, decision-tree depth experiments, Gradient Boosting learning rates, cross-validation results, and the final test evaluation.

The generated outputs are compared with the reference values and acceptance criteria in the Capstone PDF. The reference benchmarks below are retained unchanged; generated values are reported separately.

### Reference metrics compared with generated outputs

| Source file | Metric | PDF reference value | Code-generated value | Tolerance | Status | Comments |
|---|---|---:|---:|---|---|---|
| `outputs/part_b_checks.csv` | Train / validation / test rows | 840 / 180 / 180 | 840 / 180 / 180 | Exact | PASS | Required split sizes match. |
| `outputs/part_b_checks.csv` | Maximum absolute training scaled-column mean | < 1e-9 | 0.0000 (saved to 4 decimals) | < 1e-9 | PASS | The scaler is fitted on training data only. |
| `outputs/part_b_checks.csv` | Maximum absolute test scaled-column mean | Clearly non-zero (> 0.01) | 0.2187 | > 0.01 | PASS | The test features are transformed using the training-fitted scaler. |
| `outputs/part_b_metrics.csv` | Baseline validation R² | -0.021 | -0.0212 | -0.10 to 0.02 | PASS | Baseline falls within the required pass band. |
| `outputs/part_b_metrics.csv` | Linear Regression validation R² | 0.931 | 0.9313 | ≥ 0.88 | PASS | Meets the reference acceptance criterion. |
| `outputs/part_b_metrics.csv` | Gradient Boosting validation R² | 0.931 | 0.9306 | ≥ 0.88 | PASS | Meets the reference acceptance criterion. |
| `outputs/part_b_metrics.csv` | Random Forest validation R² | 0.914 | 0.9141 | ≥ 0.86 | PASS | Meets the reference acceptance criterion. |
| `outputs/part_b_metrics.csv` | Decision Tree validation R² | 0.887 | 0.8867 | ≥ 0.82 | PASS | Meets the reference acceptance criterion. |
| `outputs/part_b_metrics.csv` | Final Linear Regression test R² | 0.949 | 0.9493 | ≥ 0.88 | PASS | One final test evaluation is recorded for the selected model. |
| `outputs/part_b_metrics.csv` | Number of rows / columns | 6 rows / 6 columns | 6 rows / 6 columns | Exact | PASS | Includes five validation results and one test result. |
| `outputs/part_b_metrics.csv` | RMSE consistency with √MSE | Every row consistent | Consistent within 0.001 | Absolute difference ≤ 0.001 | PASS | RMSE is consistent with MSE across the metrics table. |
| `outputs/part_b_tree_depth.csv` | Number of depth experiments | 8 | 8 | Exact | PASS | Includes depths 1, 2, 3, 4, 6, 8, 12, and unlimited. |
| `outputs/part_b_tree_depth.csv` | Best depth / validation RMSE | 6 / 6.28 | 6 / 6.2796 | Depth ≥ 4 or unlimited; RMSE within 0.5 of reference | PASS | Depth 6 has the lowest validation RMSE. |
| `outputs/part_b_tree_depth.csv` | Depth 1: validation RMSE - training RMSE | 0.50 | 0.4990 | ≤ 2.5 | PASS | The small gap, alongside high errors, indicates high bias. |
| `outputs/part_b_tree_depth.csv` | Unlimited-depth training RMSE / validation RMSE - training RMSE | 0.00 / 7.79 | 0.0000 / 7.7890 | Gap ≥ 5.0 | PASS | The large gap indicates high variance. |
| `outputs/part_b_tree_depth.csv` | Depth 1 validation RMSE minus best validation RMSE | 5.85 | 5.8528 | ≥ 4.0 | PASS | The shallow tree performs materially worse than the best depth. |
| `outputs/part_b_tree_depth.csv` | Training RMSE as depth increases | Never rises | Non-increasing | Non-increasing | PASS | Training error falls or stays level as tree depth increases. |
| `outputs/part_b_tree_depth.csv` | Decision Tree model RMSE versus best depth experiment | Match within 0.001 | Both 6.2796 | Absolute difference ≤ 0.001 | PASS | The model-comparison result agrees with the depth experiment. |
| `outputs/part_b_gb_learning_rate.csv` | Number of learning-rate experiments | 3 | 3 | Exact | PASS | Rates 0.01, 0.1, and 1.0 are present. |
| `outputs/part_b_gb_learning_rate.csv` | Validation R² at rates 0.01 / 0.1 / 1.0 | 0.713 / 0.931 / 0.844 | 0.7127 / 0.9306 / 0.8440 | ±0.02 per value | PASS | All three values match the PDF references within tolerance. |
| `outputs/part_b_gb_learning_rate.csv` | Validation R² ranking | 0.1 > 1.0 > 0.01 | 0.9306 > 0.8440 > 0.7127 | Required ranking | PASS | Learning rate 0.1 performs best. |
| `outputs/part_b_gb_learning_rate.csv` | R² acceptance thresholds | R²(0.1) ≥ 0.90; R²(0.01) ≤ 0.80 | 0.9306; 0.7127 | As specified | PASS | Both threshold requirements are satisfied. |
| `outputs/part_b_cv.csv` | Number of cross-validated models | 2 | 2 | Exact | PASS | Linear Regression and Gradient Boosting are evaluated. |
| `outputs/part_b_cv.csv` | Cross-validation mean R²: Linear Regression / Gradient Boosting | 0.926 / 0.935 | 0.9261 / 0.9352 | ≥ 0.90 for each | PASS | Both models exceed the required mean R². |
| `outputs/part_b_cv.csv` | Cross-validation standard deviation: Linear Regression / Gradient Boosting | ≤ 0.03 for each | 0.0048 / 0.0044 | ≤ 0.03 for each | PASS | Both models show stable cross-validation scores. |
| `part_b_regression.py` | Shared KFold configuration | Same KFold object for both models | One shared `KFold(n_splits=5, shuffle=True, random_state=42)` object | Exact configuration | PASS | Both cross-validation calls use the same object. |
| `outputs/part_b_metrics.csv` | Test-set evaluation count | One final test evaluation only | One test row, for Linear Regression | Exactly one test row | PASS | No other candidate is evaluated on the test set in the recorded results. |

**Validation conclusion:** The generated Part B metrics meet the stated acceptance criteria. Linear Regression is selected by validation RMSE, despite Gradient Boosting having the higher cross-validation mean R². The reference values have not been changed to make validation pass.

[← Back to Project Overview](../README.md)