from itertools import islice
from stockfish import Stockfish
import chess
import re
import pickle


with open("../data/lichess_db_standard_rated_2025-09.pgn") as f:
    lines = list(islice(f, 2000000))


def assemble_game_metadata(lines):
    """One pass over the PGN headers, keyed the same way as the original
    assemble_ratings (increment on [WhiteElo), so results/termination line
    up with the same games. Returns a list of dicts."""
    games = []
    iterator = -1
    for line in lines:
        line = line.strip()
        if line.startswith('[WhiteElo'):
            iterator += 1
            rating = int(re.search(r'\d+', line).group())
            games.append({"white_elo": rating, "black_elo": None,
                          "result": None, "termination": None})
        elif line.startswith('[BlackElo'):
            rating = int(re.search(r'\d+', line).group())
            games[iterator]["black_elo"] = rating
        elif line.startswith('[Result'):
            m = re.search(r'"([^"]+)"', line)
            if m:
                games[iterator]["result"] = m.group(1)
        elif line.startswith('[Termination'):
            m = re.search(r'"([^"]+)"', line)
            if m:
                games[iterator]["termination"] = m.group(1)
    return games


def assemble_moves(lines):
    moves = []
    for line in lines:
        line = line.strip()
        if line.startswith('1.'):
            matches = re.findall(r'\d+\.+\s*(\S+)', line)
            matches = [m for m in matches if not m.startswith('{') and not m.startswith('[')]
            moves.append(matches)
    return moves


def moves_to_uci(move_list):
    board = chess.Board()
    uci_moves = []
    for move in move_list:
        try:
            m = board.parse_san(move)
            uci_moves.append(m.uci())
            board.push(m)
        except Exception:
            break
    return uci_moves


def option_score(option):
    if option['Mate'] is not None:
        return 30 if option['Mate'] > 0 else -30
    return option['Centipawn'] / 100


def calculation_options(move_options):
    if len(move_options) == 0:
        return 0
    return option_score(move_options[0])


def stockfish_calcs_streaming(game_moves_array, stockfish, metadata):
    with open("../data/game_records.pkl", "ab") as f_out:

        for idx, array in enumerate(game_moves_array):
            print(f"Processing game {idx}")

            uci_moves = moves_to_uci(array)
            if not uci_moves:
                continue

            stockfish.set_fen_position(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            )
            positions = []

            options = stockfish.get_top_moves(1)
            if not options:
                continue
            positions.append(calculation_options(options))

            for move in uci_moves:
                try:
                    stockfish.make_moves_from_current_position([move])
                except Exception:
                    break

                options = stockfish.get_top_moves(1)
                if not options:
                    break

                positions.append(calculation_options(options))

            move_scores = []
            for i in range(len(positions) - 1):
                pre = positions[i]
                post = positions[i + 1]
                move_scores.append(-abs(post - pre))

            record = {
                "move_scores": move_scores,
                "white_elo": metadata[idx]["white_elo"],
                "black_elo": metadata[idx]["black_elo"],
                "result": metadata[idx]["result"],
                "termination": metadata[idx]["termination"],
            }

            # single dump = single atomic record, no cross-file alignment risk
            pickle.dump(record, f_out)
            f_out.flush()


stockfish = Stockfish(
    path="/nfs/stak/users/vasudevv/hpc-share/personal/GuessTheElo/src/Stockfish-sf_18/src/stockfish",
    depth=6,
    parameters={"Threads": 2, "Minimum Thinking Time": 30}
)

stockfish.set_turn_perspective(False)

game_moves_array = assemble_moves(lines)
game_metadata = assemble_game_metadata(lines)

stockfish_calcs_streaming(game_moves_array, stockfish, game_metadata)