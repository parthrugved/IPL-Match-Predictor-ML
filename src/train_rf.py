import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# ------------------- LOAD -------------------
df = pd.read_csv("data/matches.csv")

# ------------------- CLEAN -------------------
df["city"] = df["city"].ffill()
df = df.dropna(subset=["winner"])
df = df[df["result"] == "win"]

# Sort by date (VERY IMPORTANT for time-based features)
df = df.sort_values("date").reset_index(drop=True)

# ------------------- FEATURE ENGINEERING -------------------

# Recent form (fixed version)
def recent_form(df, team, idx, n=5):
    past = df.iloc[:idx]
    matches = past[(past["team1"] == team) | (past["team2"] == team)].tail(n)

    total = len(matches)
    wins = (matches["winner"] == team).sum()

    return wins / total if total > 0 else 0.5


# Head-to-head
def h2h(df, team1, team2, idx):
    past = df.iloc[:idx]
    matches = past[
        ((past["team1"] == team1) & (past["team2"] == team2)) |
        ((past["team1"] == team2) & (past["team2"] == team1))
    ]

    total = len(matches)
    wins = (matches["winner"] == team1).sum()

    return wins / total if total > 0 else 0.5


team1_form = []
team2_form = []
h2h_list = []

for i, row in df.iterrows():
    team1_form.append(recent_form(df, row["team1"], i))
    team2_form.append(recent_form(df, row["team2"], i))
    h2h_list.append(h2h(df, row["team1"], row["team2"], i))

df["team1_form"] = team1_form
df["team2_form"] = team2_form
df["h2h"] = h2h_list

# Toss advantage
df["toss_match"] = (df["toss_winner"] == df["team1"]).astype(int)

#  Difference features (boost accuracy)
df["form_diff"] = df["team1_form"] - df["team2_form"]
df["h2h_diff"] = df["h2h"] - 0.5

# ------------------- FEATURES -------------------
features = [
    "city",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "team1_form",
    "team2_form",
    "h2h",
    "form_diff",
    "h2h_diff",
    "toss_match"
]

X = df[features]

# Binary target: 1 = team1 wins, 0 = team2 wins
y = (df["winner"] == df["team1"]).astype(int)

# Keep metadata to convert predictions back to names
match_meta = df[["team1", "team2"]]

# ------------------- ENCODING -------------------
X = pd.get_dummies(X)

# ------------------- SPLIT -------------------
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, match_meta, test_size=0.2, random_state=42
)

# ------------------- MODEL -------------------
model = RandomForestClassifier(
    n_estimators=600,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# ------------------- CONVERT BACK TO TEAM NAMES -------------------
predicted_winner = meta_test["team1"].where(pred == 1, meta_test["team2"])
actual_winner = meta_test["team1"].where(y_test == 1, meta_test["team2"])

comparison_df = pd.DataFrame({
    "Actual Winner": actual_winner.values,
    "Predicted Winner": predicted_winner.values
})

# ------------------- OUTPUT -------------------
print("First 15 comparisons RFC:")
print(comparison_df.head(15))
print("\nAccuracy:", accuracy_score(y_test, pred) * 100)

# ------------------- SAVE MODEL -------------------
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)