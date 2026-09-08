import io
import chess.pgn


def pgn_to_uci(pgn_string):
    pgn_file = io.StringIO(pgn_string)

    game = chess.pgn.read_game(pgn_file)

    if game is None:
        return []

    return [
        move.uci()
        for move in game.mainline_moves()
    ]


# Example
pgn = """
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6
"""

uci_moves = pgn_to_uci(pgn)

print(uci_moves)