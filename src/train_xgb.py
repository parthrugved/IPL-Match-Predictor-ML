import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("data/matches.csv")

# Cleaning
df["city"] = df["city"].ffill()
df = df.dropna(subset=["winner"])
df = df[df["result"] == "win"]

# sort by time so "past matches" makes sense
df = df.sort_values("date").reset_index(drop=True)


def recent_form(df, team, idx, n=5):
    past = df.iloc[:idx]
    matches = past[(past["team1"] == team) | (past["team2"] == team)].tail(n)
    wins = int((matches["winner"] == team).sum())
    return wins / n if n > 0 else 0


def h2h(df, team1, team2, idx):
    past = df.iloc[:idx]
    matches = past[
        ((past["team1"] == team1) & (past["team2"] == team2))
        | ((past["team1"] == team2) & (past["team2"] == team1))
    ]
    total = len(matches)
    wins = int((matches["winner"] == team1).sum())
    return wins / total if total > 0 else 0.5


team1_form = []
team2_form = []
h2h_list = []

for i, row in df.iterrows():
    team1_form.append(recent_form(df, row["team1"], i, n=5))
    team2_form.append(recent_form(df, row["team2"], i, n=5))
    h2h_list.append(h2h(df, row["team1"], row["team2"], i))

df["team1_form"] = team1_form
df["team2_form"] = team2_form
df["h2h"] = h2h_list
df["toss_match"] = (df["toss_winner"] == df["team1"]).astype(int)

features = [
    "city",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "team1_form",
    "team2_form",
    "h2h",
    "toss_match",
]

X = pd.get_dummies(df[features])
y = (df["winner"] == df["team1"]).astype(int)
match_meta = df[["team1", "team2"]]

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, match_meta, test_size=0.2, random_state=42
)

model_xg = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=189,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0.3,
    reg_lambda=4,
    random_state=42,
)

model_xg.fit(X_train, y_train)
pred = model_xg.predict(X_test)

predicted_winner = meta_test["team1"].where(pred == 1, meta_test["team2"])
actual_winner = meta_test["team1"].where(y_test == 1, meta_test["team2"])

comparison_df = pd.DataFrame(
    {"Actual Winner": actual_winner.values, "Predicted Winner": predicted_winner.values}
)

print("First 15 comparisons XGBOOST:")
print(comparison_df.head(15))
print("\nAccuracy:", accuracy_score(y_test, pred) * 100)

prob = model_xg.predict_proba(X_test)[0][1]
print("Probability of winning:", prob)

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model_xg, f)

# with open("encoder.pkl", "wb") as f:
#     pickle.dump(le, f)

# with open("columns_xg.pkl", "wb") as f:
#     pickle.dump(X.columns, f)