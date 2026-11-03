[← Back to Project Overview](../README.md)

# Part D - Hub Segments, Packaging and Operations Playbook

## Objectives

- Group delivery hubs into three operational segments based on their delivery performance, delay rates, distance, and number of stops.
- Compare the resulting segments with the official hub tiers to identify hubs whose operational performance does not match their assigned tier.
- Package the final classification model from Part C so it can be loaded and used without retraining during prediction.
- Score new shipments, estimate their delay probabilities, flag shipments that meet the saved alert threshold, and recommend appropriate operational actions.

## High Level Approach

- **Hub profiling:** Aggregate historical shipment data by hub to calculate shipment count, mean delivery time, delay rate, mean distance, and mean number of stops.
- **Hub segmentation:** Standardize the four operational performance metrics and apply K-Means clustering. Evaluate cluster separation for K values from 2 to 5. Retain three clusters to balance separation quality with practical cluster sizes.
- **Segment naming:** Rank the clusters by their mean delay rate and assign the names Standard, Watch, and Escalate, from lowest to highest delay rate.
- **Tier comparison:** Compare each hub's operational segment with its official tier to identify mismatches that may require further investigation.
- **Model packaging:** Rebuild the final Part C classification pipeline using the prescribed data split and configuration. Save the fitted model, alert threshold, feature order, and imputation values for reuse.
- **New shipment scoring:** Load the saved model and supporting files, prepare new shipment data using the shared preprocessing and feature engineering functions, calculate delay probabilities, apply the saved threshold, join the hub segments, and assign recommended actions.

## Generated outputs

Part D generates the following files:

## Data Findings

**Hub-level operational performance**

The hub profiles show substantial differences in delivery performance across the network.

| Hub | Shipments | Mean delivery hours | Delay rate | Mean distance (km) | Mean stops |
|---|---:|---:|---:|---:|---:|
| HUB01 | 50 | 45.6850 | 0.0200 | 185.0610 | 54.70 |
| HUB13 | 50 | 66.0742 | 0.4600 | 473.3200 | 83.92 |
| HUB17 | 50 | 60.8514 | 0.4600 | 443.2850 | 84.18 |

HUB01 has a delay rate of 2.00%, while HUB13 and HUB17 each have a delay rate of 46.00%. These profiles illustrate why hub performance should be assessed using actual operational metrics rather than relying exclusively on administrative tier labels.

**Comparison of clustering solutions**

| Number of clusters (K) | Cluster sizes, largest first | Across-to-within distance ratio |
|---:|---|---:|
| 2 | 15-9 | 2.8940 |
| 3 | 9-8-7 | 5.3928 |
| 4 | 8-7-5-4 | 6.3047 |
| 5 | 8-7-4-3-2 | 6.8866 |

The separation ratio increases as the number of clusters increases, indicating greater separation between clusters relative to distances within clusters.

However, the K = 4 and K = 5 solutions each contain clusters with fewer than six hubs. These smaller groups would have limited operational coverage and could require separate playbooks for too few hubs.

Therefore, **K = 3 is retained as the business-defined solution**, balancing cluster separation with practical group sizes. The three segments contain 7, 8, and 9 hubs, respectively.

**Operational segment profiles**

The segment profiles are calculated from the hub-level metrics.

| Segment | Number of hubs | Mean delivery hours | Delay rate | Mean distance (km) | Mean stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 0.0257 | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 0.1550 | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 0.4600 | 470.4757 | 84.3444 |

The segment profiles show a clear progression in operational performance.

- **Standard:** The average delay rate is 2.57%, with shorter average delivery times and distances.
- **Watch:** The average delay rate increases to 15.50%, indicating a need for closer monitoring.
- **Escalate:** The average delay rate reaches 46.00%, with the highest mean delivery time, distance, and number of stops.

The Escalate segment should receive priority for operational investigation and corrective action. The Watch segment warrants closer monitoring, while Standard hubs provide a useful operational baseline.

**Tier-versus-segment mismatches**

Four hubs have operational segments that differ from the segments implied by their official tiers.

| Hub | City | Official tier | Observed segment | Expected segment |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | Standard |
| HUB13 | Nagpur | 2 | Escalate | Watch |
| HUB15 | Kochi | 2 | Escalate | Watch |
| HUB22 | Dehradun | 3 | Watch | Escalate |

HUB07 performs below the expectations associated with Tier 1, while HUB13 and HUB15 have operational performance that warrants more attention than their Tier 2 labels suggest. HUB22 performs better than its Tier 3 classification would imply.

These discrepancies indicate that tier labels should remain useful for administrative and capacity planning, but operational interventions should also consider observed performance. The mismatches should trigger further investigation rather than an automatic change to official tier assignments.

**Model packaging and reload consistency**

The final Part C configuration uses seven selected features, a Random Forest with 100 estimators, unlimited tree depth, 1 minimum sample per leaf, and an alert threshold of 0.5.

The packaged model was reloaded and evaluated on 180 test rows. The maximum absolute difference between the original and reloaded model's predicted probabilities was 0.0000, satisfying the required tolerance of \(10^{-9}\).

The model and its supporting files preserve the threshold, pinned feature order, and historical imputation values required for consistent scoring.

**New-shipment scoring results**

The scoring pipeline processed all 48 new shipments and produced one action record per shipment.

| Scoring outcome | Count |
|---|---:|
| Total shipments scored | 48 |
| Shipments flagged for intervention | 8 |
| Flagged shipments requiring hub-manager escalation | 6 |
| Flagged shipments requiring proactive customer calls | 2 |
| Shipments requiring no action | 40 |

Of the 48 shipments, 8 were flagged using the stored threshold of 0.5. Six flagged shipments belonged to Escalate hubs and were assigned to hub-manager escalation. The remaining two flagged shipments were assigned proactive customer calls. The other 40 shipments were not flagged and received no recommendation for proactive action.

These results demonstrate how the classifier and hub segmentation work together to turn model predictions into an operational action list.

## Validations

Part D validates the prescribed hub-level reference metrics, K-Means separation ratios and cluster sizes, segments and operational metrics, tier-versus-segment mismatches, reload consistency, new shipments and corresponding metrics, and the expected counts of flagged shipments and hub-manager escalations from new shipments.

### Reference metrics compared with generated outputs

| Source file | Metric | PDF reference value | Code-generated value | Tolerance | Status | Comments |
|---|---|---:|---:|---|---|---|
| `outputs/hub_profile.csv` | HUB01 mean delivery hours | 45.685 | 45.6850 | ±0.01 | PASS | Matches the reference. |
| `outputs/hub_profile.csv` | HUB01 delay rate | 0.0200 | 0.0200 | ±0.01 | PASS | Matches the reference. |
| `outputs/hub_profile.csv` | HUB13 mean delivery hours | 66.074 | 66.0742 | ±0.01 | PASS | Within the permitted deviation. |
| `outputs/hub_profile.csv` | HUB13 delay rate | 0.4600 | 0.4600 | ±0.01 | PASS | Matches the reference. |
| `outputs/hub_profile.csv` | HUB17 mean delivery hours | 60.851 | 60.8514 | ±0.01 | PASS | Within the permitted deviation. |
| `outputs/hub_profile.csv` | HUB17 delay rate | 0.4600 | 0.4600 | ±0.01 | PASS | Matches the reference. |
| `outputs/part_d_separation.csv` | K = 2 cluster sizes | 15-9 | 15-9 | Exact | PASS | Cluster sizes match. |
| `outputs/part_d_separation.csv` | K = 3 cluster sizes | 9-8-7 | 9-8-7 | Exact | PASS | Cluster sizes match. |
| `outputs/part_d_separation.csv` | K = 4 cluster sizes | 8-7-5-4 | 8-7-5-4 | Exact | PASS | Cluster sizes match. |
| `outputs/part_d_separation.csv` | K = 5 cluster sizes | 8-7-4-3-2 | 8-7-4-3-2 | Exact | PASS | Cluster sizes match. |
| `outputs/part_d_separation.csv` | Across-to-within ratio, K = 2 | 2.894 | 2.8940 | ±0.01 | PASS | Within tolerance. |
| `outputs/part_d_separation.csv` | Across-to-within ratio, K = 3 | 5.393 | 5.3928 | ±0.01 | PASS | Within tolerance. |
| `outputs/part_d_separation.csv` | Across-to-within ratio, K = 4 | 6.305 | 6.3047 | ±0.01 | PASS | Within tolerance. |
| `outputs/part_d_separation.csv` | Across-to-within ratio, K = 5 | 6.887 | 6.8866 | ±0.01 | PASS | Within tolerance. |
| `outputs/segment_profile.csv` | Standard: hubs / delay rate | 7 / 0.0257 | 7 / 0.0257 | ±0.01 for rate | PASS | Segment size and delay rate match. |
| `outputs/segment_profile.csv` | Watch: hubs / delay rate | 8 / 0.1550 | 8 / 0.1550 | ±0.01 for rate | PASS | Segment size and delay rate match. |
| `outputs/segment_profile.csv` | Escalate: hubs / delay rate | 9 / 0.4600 | 9 / 0.4600 | ±0.01 for rate | PASS | Segment size and delay rate match. |
| `outputs/tier_vs_segment_mismatch.csv` | Mismatch hubs | HUB07, HUB13, HUB15, HUB22 | HUB07, HUB13, HUB15, HUB22 | Exact | PASS | All four expected mismatches are present. |
| `outputs/part_d_reload_check.csv` | Test rows scored | 180 | 180 | Exact | PASS | Expected test-set size. |
| `outputs/part_d_reload_check.csv` | Maximum absolute probability difference | 0.0 | 0.0000 | ≤ 1e-9 | PASS | Reloaded predictions match the in-memory model within tolerance. |
| `outputs/part_d_reload_check.csv` | Stored threshold | 0.5 | 0.5000 | Exact | PASS | Matches the final Part C configuration. |
| `outputs/new_shipment_actions.csv` | Number of scored shipments | 48 | 48 | Exact | PASS | All new shipments are present. |
| `outputs/new_shipment_actions.csv` | Number of flagged shipments | 8 | 8 | Exact | PASS | Matches the verified reference. |
| `outputs/new_shipment_actions.csv` | Hub-manager escalations | 6 | 6 | Exact | PASS | Matches the verified reference. |

**Validation conclusion:** All metrics for Part D pass against their reference values within the configured tolerances. No benchmark values have been changed to obtain these results.

[← Back to Project Overview](../README.md)