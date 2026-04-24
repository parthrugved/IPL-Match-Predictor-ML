from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "columns.pkl"
DATA_PATH = BASE_DIR / "data" / "matches.csv"


app = Flask(__name__)


def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    model_columns = getattr(model, "feature_names_in_", None)
    if model_columns is not None:
        columns = list(model_columns)
    else:
        with COLUMNS_PATH.open("rb") as columns_file:
            columns = list(pickle.load(columns_file))

    history = pd.read_csv(DATA_PATH)
    history["city"] = history["city"].ffill()
    history = history.dropna(subset=["winner"])
    history = history[history["result"] == "win"]
    history = history.sort_values("date").reset_index(drop=True)

    return model, columns, history


MODEL, MODEL_COLUMNS, MATCH_HISTORY = load_artifacts()

TEAMS = sorted(set(MATCH_HISTORY["team1"]) | set(MATCH_HISTORY["team2"]))
CITIES = sorted(MATCH_HISTORY["city"].dropna().unique())
TOSS_DECISIONS = ["field", "bat"]


def recent_form(df, team, n=5):
    matches = df[(df["team1"] == team) | (df["team2"] == team)].tail(n)
    total = len(matches)
    wins = int((matches["winner"] == team).sum())
    return wins / total if total > 0 else 0.5


def head_to_head(df, team1, team2):
    matches = df[
        ((df["team1"] == team1) & (df["team2"] == team2))
        | ((df["team1"] == team2) & (df["team2"] == team1))
    ]
    total = len(matches)
    wins = int((matches["winner"] == team1).sum())
    return wins / total if total > 0 else 0.5


def build_feature_row(team1, team2, city, toss_winner, toss_decision):
    team1_form = recent_form(MATCH_HISTORY, team1)
    team2_form = recent_form(MATCH_HISTORY, team2)
    h2h = head_to_head(MATCH_HISTORY, team1, team2)

    raw_features = {
        "team1_form": team1_form,
        "team2_form": team2_form,
        "h2h": h2h,
        "form_diff": team1_form - team2_form,
        "h2h_diff": h2h - 0.5,
        "toss_match": int(toss_winner == team1),
        f"city_{city}": 1,
        f"team1_{team1}": 1,
        f"team2_{team2}": 1,
        f"toss_winner_{toss_winner}": 1,
        f"toss_decision_{toss_decision}": 1,
    }

    row = pd.DataFrame([[0.0] * len(MODEL_COLUMNS)], columns=MODEL_COLUMNS)
    for feature_name, feature_value in raw_features.items():
        if feature_name in row.columns:
            row.at[0, feature_name] = feature_value

    return row, {
        "team1_form": team1_form,
        "team2_form": team2_form,
        "h2h": h2h,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    selected = {
        "team1": "",
        "team2": "",
        "city": "",
        "toss_winner": "",
        "toss_decision": "field",
    }

    if request.method == "POST":
        selected = {
            "team1": request.form.get("team1", "").strip(),
            "team2": request.form.get("team2", "").strip(),
            "city": request.form.get("city", "").strip(),
            "toss_winner": request.form.get("toss_winner", "").strip(),
            "toss_decision": request.form.get("toss_decision", "field").strip(),
        }

        team1 = selected["team1"]
        team2 = selected["team2"]
        city = selected["city"]
        toss_winner = selected["toss_winner"]
        toss_decision = selected["toss_decision"]

        if not all(selected.values()):
            error = "Select all match inputs before running the prediction."
        elif team1 == team2:
            error = "Choose two different teams for the fixture."
        elif toss_winner not in {team1, team2}:
            error = "Toss winner must be one of the two selected teams."
        else:
            feature_row, metrics = build_feature_row(
                team1=team1,
                team2=team2,
                city=city,
                toss_winner=toss_winner,
                toss_decision=toss_decision,
            )

            team1_win_probability = float(MODEL.predict_proba(feature_row)[0][1])
            team2_win_probability = 1 - team1_win_probability

            winner = team1 if team1_win_probability >= team2_win_probability else team2
            confidence = max(team1_win_probability, team2_win_probability) * 100

            prediction = {
                "winner": winner,
                "team1": team1,
                "team2": team2,
                "team1_probability": round(team1_win_probability * 100, 2),
                "team2_probability": round(team2_win_probability * 100, 2),
                "confidence": round(confidence, 2),
                "team1_form": round(metrics["team1_form"] * 100, 1),
                "team2_form": round(metrics["team2_form"] * 100, 1),
                "h2h_team1": round(metrics["h2h"] * 100, 1),
                "h2h_team2": round((1 - metrics["h2h"]) * 100, 1),
            }

    return render_template(
        "index.html",
        teams=TEAMS,
        cities=CITIES,
        toss_decisions=TOSS_DECISIONS,
        prediction=prediction,
        error=error,
        selected=selected,
    )


if __name__ == "__main__":
    app.run(debug=True)
