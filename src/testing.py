import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci(
   "/nfs/stak/users/vasudevv/hpc-share/personal/GuessTheElo/src/Stockfish-sf_18/src/stockfish"
)

board = chess.Board()

info = engine.analyse(board, chess.engine.Limit(depth=6))
print(info["score"])
"""

import torch
import pickle
from train import eloModel  # import your model class from wherever it's defined

# Load model
model = eloModel(input_dim=1, hidden=32)
model.load_state_dict(torch.load("../data/elo_model.pth", map_location="cpu"))
model.eval()

with open("../data/all_games.pkl", "rb") as f:
    for _ in range(900):
        pickle.load(f)
    single_game = pickle.load(f)

with open("../data/game_ratings.pkl", "rb") as f:
    for _ in range(900):
        pickle.load(f)
    single_rating = pickle.load(f)


white_array = []
black_array = []

for i in range (len(single_game)):
    if i % 2 == 0:
        white_array.append(single_game[i])
    else:
        black_array.append(single_game[i])



# Prepare a single game's move scores
# Assuming `game_moves` is a list of centipawn scores for one player, e.g.:

# Mirror your training preprocessing
move_tensor = torch.tensor(white_array, dtype=torch.float32)  # shape: (seq_len,)
move_tensor = move_tensor.unsqueeze(0)                        # shape: (1, seq_len) — fake batch dim

# Run inference
with torch.no_grad():
    raw_output = model(move_tensor)  # returns normalized value in ~[0, 1]

# Undo the normalization from your Dataset: rating = (raw - 300) / 2700
predicted_rating = raw_output.item() * 2700 + 300
print(f"Predicted Elo: {predicted_rating:.0f}")
print("ACTUAL ELO: ",  single_rating[0])

"""