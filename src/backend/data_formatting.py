import pickle
import chess.pgn

from stockfish import Stockfish


PGN_PATH = "../data/lichess_db_standard_rated_2025-09.pgn"
OUTPUT_PATH = "../data/game_records.pkl"

# Start small for validation. Set to None for the entire PGN.
MAX_GAMES = 50000


def load_games(pgn_path, max_games=None):
    """
    Load complete PGN games using python-chess.

    Metadata and moves are extracted from the same game object,
    so they cannot become misaligned.
    """
    games = []

    with open(pgn_path, encoding="utf-8", errors="replace") as pgn_file:
        count = 0

        while True:
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            headers = game.headers

            try:
                white_elo = int(headers.get("WhiteElo"))
                black_elo = int(headers.get("BlackElo"))
            except (TypeError, ValueError):
                continue

            result_raw = headers.get("Result")

            if result_raw == "1-0":
                result = "White"
            elif result_raw == "0-1":
                result = "Black"
            elif result_raw == "1/2-1/2":
                result = "Draw"
            else:
                result = None

            termination = headers.get("Termination")

            uci_moves = [
                move.uci()
                for move in game.mainline_moves()
            ]

            if not uci_moves:
                continue

            games.append({
                "uci_moves": uci_moves,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "result": result,
                "termination": termination,
            })

            count += 1

            if count % 1000 == 0:
                print(f"Loaded {count} games")

            if max_games is not None and count >= max_games:
                break

    return games


def option_score(option):
    """Convert a Stockfish evaluation into pawn units."""

    if option["Mate"] is not None:
        return 30 if option["Mate"] > 0 else -30

    return option["Centipawn"] / 100


def calculation_options(move_options):
    """Return the evaluation of Stockfish's top move."""

    if not move_options:
        return 0

    return option_score(move_options[0])


def stockfish_calcs_streaming(games, stockfish, output_path):
    """
    Evaluate every game with Stockfish and write records directly to disk.
    """

    saved_games = 0
    skipped_games = 0

    # "wb" overwrites old data rather than appending to it.
    with open(output_path, "wb") as f_out:

        for idx, game in enumerate(games):

            print(
                f"Processing game {idx + 1}/{len(games)} | "
                f"{game['white_elo']} vs {game['black_elo']}"
            )

            uci_moves = game["uci_moves"]

            stockfish.set_fen_position(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            )

            positions = []

            # Evaluate initial position.
            options = stockfish.get_top_moves(1)
            initial_score = calculation_options(options)

            if initial_score is None:
                print(f"Skipping game {idx}: no initial evaluation")
                skipped_games += 1
                continue

            positions.append(initial_score)

            successful = True

            # Play each actual game move and evaluate resulting position.
            for move in uci_moves:

                try:
                    stockfish.make_moves_from_current_position([move])

                except Exception as e:
                    print(
                        f"Skipping game {idx}: "
                        f"failed to make move {move}: {e}"
                    )
                    successful = False
                    break

                options = stockfish.get_top_moves(1)
                score = calculation_options(options)

                if score is None:
                    print(
                        f"Skipping game {idx}: "
                        f"no evaluation after move {move}"
                    )
                    successful = False
                    break

                positions.append(score)

            # Don't save partial games.
            if not successful:
                skipped_games += 1
                continue

            if len(positions) != len(uci_moves) + 1:
                print(
                    f"Skipping incomplete game {idx}: "
                    f"{len(positions) - 1}/{len(uci_moves)} "
                    f"of {len(uci_moves)} moves evaluated"
                )
                skipped_games += 1
                continue

            # Calculate evaluation loss for each move.
            move_scores = []

            for i in range(len(positions) - 1):
                pre = positions[i]
                post = positions[i + 1]

                loss = abs(post - pre)

                # Stored as negative to match your existing feature
                # extraction pipeline, which negates these values.
                move_scores.append(-loss)

            record = {
                "move_scores": move_scores,
                "white_elo": game["white_elo"],
                "black_elo": game["black_elo"],
                "result": game["result"],
                "termination": game["termination"],
            }

            pickle.dump(record, f_out)
            f_out.flush()

            saved_games += 1

    print("\n" + "=" * 50)
    print("Finished")
    print("=" * 50)
    print(f"Games saved:   {saved_games}")
    print(f"Games skipped: {skipped_games}")
    print(f"Output:        {output_path}")


def stockfish_move_scores(uci_moves, stockfish):

    stockfish.set_fen_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    positions = []

    # Initial evaluation.
    options = stockfish.get_top_moves(1)
    initial_score = calculation_options(options)

    if initial_score is None:
        return None

    positions.append(initial_score)

    # Evaluate after every move.
    for move in uci_moves:

        try:
            stockfish.make_moves_from_current_position([move])
        except Exception as e:
            print(f"Failed to make move {move}: {e}")
            return None

        options = stockfish.get_top_moves(1)
        score = calculation_options(options)

        if score is None:
            return None

        positions.append(score)

    # Convert position evaluations to move losses.
    move_scores = []

    for i in range(len(positions) - 1):
        pre = positions[i]
        post = positions[i + 1]

        loss = abs(post - pre)
        move_scores.append(-loss)

    return move_scores


if __name__ == "__main__":

    print("=" * 50)
    print("Loading PGN games")
    print("=" * 50)

    games = load_games(
        PGN_PATH,
        max_games=MAX_GAMES,
    )

    print(f"\nSuccessfully loaded {len(games)} games")

    if not games:
        raise RuntimeError("No valid games were loaded.")

    print("\nFirst game sanity check:")
    print(f"White Elo:   {games[0]['white_elo']}")
    print(f"Black Elo:   {games[0]['black_elo']}")
    print(f"Result:      {games[0]['result']}")
    print(f"Termination: {games[0]['termination']}")
    print(f"Move count:  {len(games[0]['uci_moves'])}")
    print(f"First moves: {games[0]['uci_moves'][:10]}")

    print("\n" + "=" * 50)
    print("Starting Stockfish")
    print("=" * 50)

    stockfish = Stockfish(
        path=(
            "/nfs/stak/users/vasudevv/"
            "hpc-share/personal/GuessTheElo/"
            "src/Stockfish-sf_18/src/stockfish"
        ),
        depth=6,
        parameters={
            "Threads": 1,
            "Minimum Thinking Time": 30,
        },
    )

    stockfish.set_turn_perspective(False)

    stockfish_calcs_streaming(
        games=games,
        stockfish=stockfish,
        output_path=OUTPUT_PATH,
    )