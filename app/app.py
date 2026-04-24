from pathlib import Path
import pickle
import pandas as pd
from flask import Flask, render_template, request
from flask import jsonify

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "columns.pkl"
DATA_PATH = BASE_DIR / "data" / "matches.csv"

app = Flask(__name__)

# ---------------- LOAD ----------------
def load_artifacts():
    model = pickle.load(open(MODEL_PATH, "rb"))

    # safer column loading
    if hasattr(model, "feature_names_in_"):
        columns = list(model.feature_names_in_)
    else:
        columns = pickle.load(open(COLUMNS_PATH, "rb"))

    df = pd.read_csv(DATA_PATH)
    df["city"] = df["city"].ffill()
    df = df.dropna(subset=["winner"])
    df = df[df["result"] == "win"]
    df = df.sort_values("date").reset_index(drop=True)

    return model, columns, df

MODEL, MODEL_COLUMNS, MATCH_HISTORY = load_artifacts()

TEAMS = sorted(set(MATCH_HISTORY["team1"]) | set(MATCH_HISTORY["team2"]))
CITIES = sorted(MATCH_HISTORY["city"].dropna().unique())
TOSS_DECISIONS = ["field", "bat"]

# ---------------- FEATURES ----------------

def recent_form(team, n=5):
    matches = MATCH_HISTORY[
        (MATCH_HISTORY["team1"] == team) |
        (MATCH_HISTORY["team2"] == team)
    ].tail(n)

    total = len(matches)
    wins = (matches["winner"] == team).sum()

    return wins / total if total > 0 else 0.5


def head_to_head(team1, team2):
    matches = MATCH_HISTORY[
        ((MATCH_HISTORY["team1"] == team1) & (MATCH_HISTORY["team2"] == team2)) |
        ((MATCH_HISTORY["team1"] == team2) & (MATCH_HISTORY["team2"] == team1))
    ]

    total = len(matches)
    wins = (matches["winner"] == team1).sum()

    return wins / total if total > 0 else 0.5


def build_feature_row(team1, team2, city, toss_winner, toss_decision):
    team1_form = recent_form(team1)
    team2_form = recent_form(team2)
    h2h = head_to_head(team1, team2)

    # 🔥 important features
    features = {
        "team1_form": team1_form,
        "team2_form": team2_form,
        "h2h": h2h,
        "form_diff": team1_form - team2_form,
        "h2h_diff": h2h - 0.5,
        "toss_match": int(toss_winner == team1),
    }

    # one-hot encoded fields
    features[f"city_{city}"] = 1
    features[f"team1_{team1}"] = 1
    features[f"team2_{team2}"] = 1
    features[f"toss_winner_{toss_winner}"] = 1
    features[f"toss_decision_{toss_decision}"] = 1

    # align with training columns
    row = pd.DataFrame([[0.0] * len(MODEL_COLUMNS)], columns=MODEL_COLUMNS)

    for key, value in features.items():
        if key in row.columns:
            row.at[0, key] = value

    return row, team1_form, team2_form, h2h


# ---------------- GROUND MAPS ----------------

TEAM_HOME_GROUNDS = {
    "Mumbai Indians": ["Mumbai"],
    "Chennai Super Kings": ["Chennai"],
    "Royal Challengers Bengaluru": ["Bengaluru"],
    "Kolkata Knight Riders": ["Kolkata"],
    "Delhi Capitals": ["Delhi"],
    "Punjab Kings": ["Mohali", "Dharamsala"],
    "Rajasthan Royals": ["Jaipur"],
    "Sunrisers Hyderabad": ["Hyderabad"],
    "Gujarat Titans": ["Ahmedabad"],
    "Lucknow Super Giants": ["Lucknow"]
}

# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    selected = {
        "team1": "",
        "team2": "",
        "city": "",
        "toss_winner": "",
        "toss_decision": "field"
    }

    if request.method == "POST":

        selected = {
            "team1": request.form.get("team1", "").strip(),
            "team2": request.form.get("team2", "").strip(),
            "city": request.form.get("city", "").strip(),
            "toss_winner": request.form.get("toss_winner", "").strip(),
            "toss_decision": request.form.get("toss_decision", "field")
        }

        team1 = selected["team1"]
        team2 = selected["team2"]
        city = selected["city"]
        toss_winner = selected["toss_winner"]
        toss_decision = selected["toss_decision"]

        # -------- VALIDATION --------
        if not all(selected.values()):
            error = "Please fill all fields."
        elif team1 == team2:
            error = "Select different teams."
        elif toss_winner not in {team1, team2}:
            error = "Toss winner must be one of the selected teams."

        else:
            # -------- FEATURE BUILD --------
            row, f1, f2, h2h = build_feature_row(
                team1, team2, city, toss_winner, toss_decision
            )

            # -------- PREDICTION --------
            prob_team1 = MODEL.predict_proba(row)[0][1]
            prob_team2 = 1 - prob_team1

            winner = team1 if prob_team1 >= prob_team2 else team2
            confidence = max(prob_team1, prob_team2) * 100

            # -------- INSIGHT --------
            if abs(prob_team1 - prob_team2) < 0.05:
                insight = "Very close match ⚔️"
            elif prob_team1 > 0.65:
                insight = f"{team1} is strong favorite 🔥"
            elif prob_team2 > 0.65:
                insight = f"{team2} is strong favorite 🔥"
            else:
                insight = "Slight edge based on recent form"

            prediction = {
                "winner": winner,
                "confidence": round(confidence, 2),
                "team1": team1,
                "team2": team2,
                "team1_probability": round(prob_team1 * 100, 2),
                "team2_probability": round(prob_team2 * 100, 2),
                "team1_form": round(f1 * 100, 1),
                "team2_form": round(f2 * 100, 1),
                "h2h_team1": round(h2h * 100, 1),
                "h2h_team2": round((1 - h2h) * 100, 1),
                "insight": insight
            }

    return render_template(
        "index.html",
        teams=TEAMS,
        cities=CITIES,
        toss_decisions=TOSS_DECISIONS,
        prediction=prediction,
        error=error,
        selected=selected
    )


# ---------------- OPTIONS ROUTE AND FUNCTION ----------------
@app.route("/get-options")
def get_options():
    team1 = request.args.get("team1")
    team2 = request.args.get("team2")

    toss_options = [team1, team2] if team1 and team2 else []

    city_options = []
    if team1 in TEAM_HOME_GROUNDS:
        city_options.extend(TEAM_HOME_GROUNDS[team1])
    if team2 in TEAM_HOME_GROUNDS:
        city_options.extend(TEAM_HOME_GROUNDS[team2])

    city_options = list(set(city_options))  # remove duplicates

    return jsonify({
        "toss": toss_options,
        "cities": city_options
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)