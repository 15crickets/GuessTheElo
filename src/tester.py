import pickle
import matplotlib.pyplot as plt

def load_all(path):
    items = []
    with open(path, "rb") as f:
        while True:
            try:
                items.append(pickle.load(f))
            except EOFError:
                break
            except Exception as e:
                print(f"Stopped reading {path} early due to {e!r} "
                      f"({len(items)} good records kept)")
                break
    return items

ratings = load_all("../data/new_ratings.pkl")  # list of [white_elo, black_elo]

white = [r[0] for r in ratings]
black = [r[1] for r in ratings]
combined = white + black  # both colors pooled together

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Combined distribution
axes[0].hist(combined, bins=50, color="steelblue", edgecolor="black")
axes[0].set_title(f"Rating Distribution (n={len(combined)} player-games)")
axes[0].set_xlabel("Elo")
axes[0].set_ylabel("Count")

# White vs Black overlay
axes[1].hist(white, bins=50, alpha=0.6, label="White", color="lightgray", edgecolor="black")
axes[1].hist(black, bins=50, alpha=0.6, label="Black", color="dimgray", edgecolor="black")
axes[1].set_title("White vs Black Elo")
axes[1].set_xlabel("Elo")
axes[1].legend()

plt.tight_layout()
plt.savefig("../data/rating_distribution.png", dpi=150)
print("Saved to ../data/rating_distribution.png")
print(f"n={len(ratings)} games, mean={sum(combined)/len(combined):.1f}, "
      f"min={min(combined)}, max={max(combined)}")