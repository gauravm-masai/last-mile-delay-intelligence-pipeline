import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

HUB_PROFILE_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

def create_hub_profile():
    # Load the modelling table
    df = pd.read_csv("data/modelling_table.csv")

    # Calculate hub-level profile
    hub_profile = (
        df.groupby("hub_id")
        .agg(
            shipments=("hub_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean")
        )
        .reset_index()
        .sort_values(by="hub_id", ascending=True)
    )

    hub_profile.to_csv(
        "outputs/hub_profile.csv",
        index=False,
        float_format="%.4f"
    )

    print(f"\nHub profiles saved in outputs/hub_profile.csv with {len(hub_profile)} rows")

    return hub_profile

def segment_hub_profile(hub_profile):
    # Preserve the required hub order
    hub_profile = (
        hub_profile
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    X = hub_profile[HUB_PROFILE_FEATURES]

    # Combine scaling and clustering
    kmeans_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10,
        )),
    ])

    # Fit and assign clusters
    hub_profile["cluster"] = kmeans_pipeline.fit_predict(X)

    #print(hub_profile[["hub_id", "cluster"]])

    return hub_profile

def generate_separation_evidence(hub_profile):
    X = hub_profile[HUB_PROFILE_FEATURES]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Separation evidence for K = 2, 3, 4, 5
    results = []

    # All unordered pairs among the 24 hubs
    n_hubs = len(hub_profile)
    row_idx, col_idx = np.triu_indices(n_hubs, k=1)

    # Euclidean distances between all hub pairs on the SAME scaled matrix
    pair_distances = np.linalg.norm(
        X_scaled[row_idx] - X_scaled[col_idx],
        axis=1,
    )

    assert len(pair_distances) == 276, (
        f"Expected 276 hub pairs, got {len(pair_distances)}"
    )

    for k in [2, 3, 4, 5]:
        # Cluster the 24 hubs
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        # Determine which pairs belong to the same cluster
        same_cluster = labels[row_idx] == labels[col_idx]
        different_cluster = ~same_cluster

        # Mean distance within and across clusters
        within_mean = np.mean(same_cluster)
        across_mean = np.mean(different_cluster)

        # Ratio of across cluster distances to within cluster distances
        ratio = across_mean / within_mean

        # Cluster sizes, largest first
        sizes = sorted(
            np.bincount(labels).tolist(),
            reverse=True,
        )
        sizes_text = "-".join(map(str, sizes))

        results.append({
            "K": k,
            "across_over_within": round(ratio, 4),
            "cluster_sizes": sizes_text
        })

    # Create the required output table
    separation_df = pd.DataFrame(
        results,
        columns=["K", "across_over_within", "cluster_sizes"],
    )

    separation_df.to_csv(
        "outputs/part_d_separation.csv",
        index=False,
        float_format="%.4f"
    )

    print("Saved separation evidence in outputs/part_d_separation.csv")

def name_segments_check_tier(hub_profile):
    hubs = pd.read_csv("data/hubs.csv")

    # 1. Calculate mean delay rate for each cluster
    cluster_delay = hub_profile.groupby("cluster")["delay_rate"].mean()

    # 2. Order clusters from lowest to highest delay rate
    ordered_clusters = cluster_delay.sort_values().index.tolist()

    # 3. Assign business segment names in that order
    segment_names = {
        ordered_clusters[0]: "Standard",
        ordered_clusters[1]: "Watch",
        ordered_clusters[2]: "Escalate",
    }

    hub_profile["segment"] = hub_profile["cluster"].map(segment_names)

    # 4. Create hub_segments.csv
    hub_segments = hub_profile[["hub_id", "segment"]].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
    )

    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values("hub_id")

    hub_segments.to_csv(
        "outputs/hub_segments.csv",
        index=False,
        float_format="%.4f"
    )

    print("Hub segments saved in outputs/hub_segments.csv")

    # 5. Create segment_profile.csv using hub-level averages
    profile_columns = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]

    segment_profile = (
        hub_profile.groupby("segment")
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reset_index()
    )

    # Keep the business-defined segment order
    segment_order = ["Standard", "Watch", "Escalate"]
    segment_profile["segment"] = pd.Categorical(
        segment_profile["segment"],
        categories=segment_order,
        ordered=True,
    )
    segment_profile = segment_profile.sort_values("segment")
    segment_profile["segment"] = segment_profile["segment"].astype(str)

    segment_profile.to_csv(
        "outputs/segment_profile.csv",
        index=False,
        float_format="%.4f"
    )

    print("Segment profiles saved in outputs/segment_profile.csv")

    # 6. Check whether each hub's segment matches its tier
    expected_segment = {
        1: "Standard",
        2: "Watch",
        3: "Escalate",
    }

    hub_segments["expected_segment"] = hub_segments["hub_tier"].map(
        expected_segment
    )

    mismatches = hub_segments[
        hub_segments["segment"] != hub_segments["expected_segment"]
        ].drop(columns="expected_segment")

    mismatches.to_csv(
        "outputs/tier_vs_segment_mismatch.csv",
        index=False,
        float_format="%.4f"
    )

    print("Mismatches saved in outputs/tier_vs_segment_mismatch.csv")

if __name__ == "__main__":
    hub_profile = create_hub_profile()
    hub_profile = segment_hub_profile(hub_profile)
    generate_separation_evidence(hub_profile)
    name_segments_check_tier(hub_profile)