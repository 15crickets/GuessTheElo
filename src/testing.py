import pickle
import statistics as stats
import pandas as pd


def load_records(path):
    records = []

    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break

    return records


records = load_records("../data/game_records.pkl")

rows = []

for rec in records:
    move_scores = rec["move_scores"]

    if len(move_scores) < 2:
        continue

    white_losses = [
        -move_scores[i]
        for i in range(0, len(move_scores), 2)
    ]

    black_losses = [
        -move_scores[i]
        for i in range(1, len(move_scores), 2)
    ]

    if not white_losses or not black_losses:
        continue

    white_mean_loss = stats.mean(white_losses)
    black_mean_loss = stats.mean(black_losses)

    elo_diff = rec["white_elo"] - rec["black_elo"]

    # Positive means Black lost more evaluation than White.
    # Positive elo_diff means White is higher rated.
    loss_diff = black_mean_loss - white_mean_loss

    rows.append({
        "white_elo": rec["white_elo"],
        "black_elo": rec["black_elo"],
        "elo_diff": elo_diff,
        "white_mean_loss": white_mean_loss,
        "black_mean_loss": black_mean_loss,
        "loss_diff": loss_diff,
    })

df = pd.DataFrame(rows)

print("=" * 50)
print("Dataset")
print("=" * 50)
print(f"Games analyzed: {len(df)}")

print("\nElo difference statistics:")
print(df["elo_diff"].describe())

print("\nLoss difference statistics:")
print(df["loss_diff"].describe())

print("\n" + "=" * 50)
print("Main Correlation Test")
print("=" * 50)

correlation = df["elo_diff"].corr(df["loss_diff"])

print(f"\nCorrelation between Elo difference and move-loss difference:")
print(f"{correlation:.6f}")

print("\n" + "=" * 50)
print("Individual Player Correlations")
print("=" * 50)

print(
    "White Elo vs White mean loss:",
    df["white_elo"].corr(df["white_mean_loss"])
)

print(
    "Black Elo vs Black mean loss:",
    df["black_elo"].corr(df["black_mean_loss"])
)

print("\n" + "=" * 50)
print("Elo Difference Buckets")
print("=" * 50)

df["elo_diff_bucket"] = pd.cut(
    df["elo_diff"],
    bins=[
        -float("inf"),
        -800,
        -400,
        -200,
        -100,
        0,
        100,
        200,
        400,
        800,
        float("inf"),
    ]
)

bucket_stats = (
    df.groupby("elo_diff_bucket", observed=True)
    .agg(
        games=("elo_diff", "count"),
        avg_elo_diff=("elo_diff", "mean"),
        avg_loss_diff=("loss_diff", "mean"),
        median_loss_diff=("loss_diff", "median"),
    )
)

print(bucket_stats)

print("\n" + "=" * 50)
print("Extreme Elo Mismatch Examples")
print("=" * 50)

extreme = df[abs(df["elo_diff"]) >= 500]

print(f"\nGames with >= 500 Elo difference: {len(extreme)}")

print(
    extreme[
        [
            "white_elo",
            "black_elo",
            "elo_diff",
            "white_mean_loss",
            "black_mean_loss",
            "loss_diff",
        ]
    ].head(20)
)
