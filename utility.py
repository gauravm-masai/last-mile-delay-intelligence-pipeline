import pandas as pd

correlations = pd.read_csv(
    "outputs/part_a_correlations.csv",
    index_col=0,
)

top_three = (
    correlations["delivery_hours"]
    .drop(labels=["delivery_hours"], errors="ignore")
    .dropna()
    .loc[lambda values: values.abs().sort_values(ascending=False).index]
    .head(3)
)

for column, value in top_three.items():
    print(f"{column}: {value:+.4f}")