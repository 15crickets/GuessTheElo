from itertools import islice



with open("../data/lichess_db_standard_rated_2025-09.pgn") as f:
    lines = list(islice(f, 1000))



def assemble_moves(lines):
    import re
    moves = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('1.'):
            matches = re.findall(r'\d+\.+\s*(\S+)', line)
            print("Matches: ")
            print(matches)
            white_moves = matches[0::2]
            print("White Moves: ")
            print(white_moves)
            black_moves = matches[1::2]
            print("Black Moves: ")
            print(black_moves)
            pairs = [w + b for w, b in zip(white_moves, black_moves)]
            moves.append(pairs)
    
    return moves


moves = assemble_moves(lines)
print(moves)



