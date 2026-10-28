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

def create_hub_profile(df):

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

    for k in [2, 3, 4, 5]:
        # Cluster the 24 hubs
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        # Determine which pairs belong to the same cluster
        same_cluster = labels[row_idx] == labels[col_idx]
        different_cluster = ~same_cluster

        # Mean distance within and across clusters
        within_mean = pair_distances[same_cluster].mean()
        across_mean = pair_distances[different_cluster].mean()

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

    return separation_df

def create_hub_segments(hubs, hub_profile):
    # 1. Calculate mean delay rate for each cluster
    cluster_delay_rates = (
        hub_profile.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
    )

    # 2. Assign business segment names in that order
    segment_names = ["Standard", "Watch", "Escalate"]

    cluster_to_segment = {
        cluster_id: segment_name
        for cluster_id, segment_name in zip(
            cluster_delay_rates.index,
            segment_names,
        )
    }

    hub_profile["segment"] = hub_profile["cluster"].map(cluster_to_segment)

    # 4. Create hub_segments.csv
    hub_segments = hub_profile[["hub_id", "segment"]].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )

    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values("hub_id")

    return hub_segments

def create_segment_profiles(hub_profile):

    # Create segment_profile using hub-level averages
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

    return segment_profile

def match_segments_tier(hub_segments):

    # Check whether each hub's segment matches its tier
    expected_segment = {
        1: "Standard",
        2: "Watch",
        3: "Escalate",
    }

    hub_segments["expected_segment"] = (
        hub_segments["hub_tier"].map(expected_segment)
    )

    tier_vs_segment_mismatch = hub_segments.loc[
        hub_segments["segment"] != hub_segments["expected_segment"]
        ].copy()

    return tier_vs_segment_mismatch

def save_part_d_results(
        hub_profile,
        separation_df,
        hub_segments,
        segment_profile,
        mismatches
):
    hub_profile.to_csv(
        "outputs/hub_profile.csv",
        index=False,
        float_format="%.4f"
    )

    print(f"\nHub profiles saved in outputs/hub_profile.csv with {len(hub_profile)} rows")

    separation_df.to_csv(
        "outputs/part_d_separation.csv",
        index=False,
        float_format="%.4f"
    )

    print("Saved separation evidence in outputs/part_d_separation.csv")

    hub_segments.to_csv(
        "outputs/hub_segments.csv",
        index=False,
        float_format="%.4f"
    )

    print("Hub segments saved in outputs/hub_segments.csv")

    segment_profile.to_csv(
        "outputs/segment_profile.csv",
        index=False,
        float_format="%.4f"
    )

    print("Segment profiles saved in outputs/segment_profile.csv")

    mismatches.to_csv(
        "outputs/tier_vs_segment_mismatch.csv",
        index=False,
        float_format="%.4f"
    )

    print("Mismatches saved in outputs/tier_vs_segment_mismatch.csv")

if __name__ == "__main__":
    # Load hubs and  modelling table
    hubs = pd.read_csv("data/hubs.csv")
    df = pd.read_csv("data/modelling_table.csv")

    hub_profile = create_hub_profile(df)

    hub_profile = segment_hub_profile(hub_profile)

    separation_df = generate_separation_evidence(hub_profile)

    hub_segments = create_hub_segments(hubs, hub_profile)
    segment_profile = create_segment_profiles(hub_profile)
    mismatches = match_segments_tier(hub_segments)

    save_part_d_results(
        hub_profile,
        separation_df,
        hub_segments,
        segment_profile,
        mismatches
    )