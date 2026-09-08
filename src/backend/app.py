from flask import Flask, request, jsonify
from flask_cors import CORS
import data_formatting
import prediction
import extraction
import numpy as np
from stockfish import Stockfish
from xgboost import XGBRegressor
import pandas as pd
from pathlib import Path




BASE_DIR = Path(__file__).resolve().parent


model = XGBRegressor()

MODEL_PATH = BASE_DIR.parent.parent / "models" / "xgboost_model.json"

model.load_model(str(MODEL_PATH))

app = Flask(__name__)
CORS(app)  # allows your React dev server (different port) to call this API


# Initialize Stockfish ONCE when the server starts
stockfish = Stockfish(
    path="../../Stockfish-sf_18/src/stockfish",
    depth=6,
    parameters={
        "Threads": 1,
        "Minimum Thinking Time": 30,
    },
)

stockfish.set_turn_perspective(False)



@app.route("/api/guess_elo", methods=["GET"])
def guess_elo():
    pgn = request.args.get("pgn")
    color = request.args.get("color")
    if not pgn:
        return jsonify({"error": "Missing 'pgn' query parameter"})
    if not color:
        return jsonify({"error": "Missing 'color' query parameter"})
    if color != "White" and color != "Black":
        return jsonify({"error": "Invalid 'color' query parameter"})
    
    try:
        uci = prediction.pgn_to_uci(pgn)
        stockfish_evals = data_formatting.stockfish_move_scores(uci, stockfish)


        features = extraction.extract_color_features(
            move_scores=stockfish_evals,
            color=color.lower(),
            ply_count=len(stockfish_evals),
            result=None,
            termination="Normal",
            elo=None,
        )



        X_single = pd.DataFrame([features])

        drop_columns = [
            "elo",
            "termination",
            "result",
            "color"
        ]

        X_single = X_single.drop(columns=drop_columns)

        X_single["had_blunder"] = X_single["first_blunder_move"].notna().astype(int)

        X_single["first_blunder_move"] = X_single["first_blunder_move"].fillna(-1)



        prediction = model.predict(X_single)


        return jsonify({"elo": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


