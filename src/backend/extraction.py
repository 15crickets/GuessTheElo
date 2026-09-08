import pickle
import statistics as stats
import pandas as pd


BLUNDER_THRESHOLD = 3.0      # pawns
INACCURACY_THRESHOLD = 0.7  # pawns
OPENING_PLIES = 10            # first ~5 moves per color


def load_records(path):
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break
            except Exception as e:
                print(f"Stopped reading {path} early due to {e!r} "
                      f"({len(records)} good records kept)")
                break
    return records


def swing_stats(swings, prefix):
    """swings: list of non-negative pawn magnitudes for one color/phase."""
    n = len(swings)
    if n == 0:
        return {
            f"{prefix}_mean": None, f"{prefix}_median": None,
            f"{prefix}_std": None, f"{prefix}_max": None,
            f"{prefix}_blunder_count": 0, f"{prefix}_blunder_rate": None,
            f"{prefix}_inaccuracy_count": 0, f"{prefix}_inaccuracy_rate": None,
        }
    blunders = [s for s in swings if s > BLUNDER_THRESHOLD]
    inaccuracies = [s for s in swings if INACCURACY_THRESHOLD < s <= BLUNDER_THRESHOLD]
    return {
        f"{prefix}_mean": stats.mean(swings),
        f"{prefix}_median": stats.median(swings),
        f"{prefix}_std": stats.pstdev(swings) if n > 1 else 0.0,
        f"{prefix}_max": max(swings),
        f"{prefix}_blunder_count": len(blunders),
        f"{prefix}_blunder_rate": len(blunders) / n,
        f"{prefix}_inaccuracy_count": len(inaccuracies),
        f"{prefix}_inaccuracy_rate": len(inaccuracies) / n,
    }


def phase_slices(n_plies):
    opening_end = min(OPENING_PLIES, n_plies)
    remaining = n_plies - opening_end
    mid_end = opening_end + remaining // 2
    return (0, opening_end), (opening_end, mid_end), (mid_end, n_plies)


def first_blunder_move_number(color_swings_with_ply):
    for ply_index, mag in color_swings_with_ply:
        if mag > BLUNDER_THRESHOLD:
            return (ply_index // 2) + 1
    return None


def extract_color_features(move_scores, color, ply_count, result, termination, elo):
    offset = 0 if color == "white" else 1
    magnitudes = [(-move_scores[i]) for i in range(offset, len(move_scores), 2)]
    with_ply = [(offset + 2 * j, m) for j, m in enumerate(magnitudes)]

    row = {}
    row.update(swing_stats(magnitudes, "overall"))

    (o_start, o_end), (m_start, m_end), (e_start, e_end) = phase_slices(len(move_scores))
    for phase_name, start, end in [("opening", o_start, o_end),
                                    ("middlegame", m_start, m_end),
                                    ("endgame", e_start, e_end)]:
        phase_mags = [move_scores[i] * -1 for i in range(start, end)
                      if i % 2 == offset]
        row.update(swing_stats(phase_mags, phase_name))

    row["first_blunder_move"] = first_blunder_move_number(with_ply)
    row["ply_count"] = ply_count
    row["move_count"] = ply_count // 2 + (ply_count % 2 if color == "white" else 0)
    row["result"] = result
    row["termination"] = termination
    row["color"] = color
    row["elo"] = elo
    return row


def build_dataset(records):
    rows = []
    for rec in records:
        move_scores = rec["move_scores"]
        if not move_scores:
            continue
        ply_count = len(move_scores)

        rows.append(extract_color_features(
            move_scores, "white", ply_count,
            rec.get("result"), rec.get("termination"), rec["white_elo"],
        ))
        rows.append(extract_color_features(
            move_scores, "black", ply_count,
            rec.get("result"), rec.get("termination"), rec["black_elo"],
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    records = load_records("../data/game_records.pkl")
    print(f"Loaded {len(records)} games")

    df = build_dataset(records)
    print(df.shape)
    print(df.head())

    df.to_csv("../data/features.csv", index=False)
    df.to_pickle("../data/features.pkl")
    print("Saved to ../data/features.csv and ../data/features.pkl")