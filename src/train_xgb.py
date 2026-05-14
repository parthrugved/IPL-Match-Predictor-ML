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

valid_teams = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

df = df[
    (df["team1"].isin(valid_teams)) &
    (df["team2"].isin(valid_teams))
]

# sort by time so "past matches" makes sense
df = df.sort_values("date").reset_index(drop=True)

# -------- TEAM STRENGTH --------
team_wins = df["winner"].value_counts().to_dict()
team_matches = pd.concat([df["team1"], df["team2"]]).value_counts().to_dict()

def strength(team):
    return team_wins.get(team, 0) / team_matches.get(team, 1)

df["team1_strength"] = df["team1"].apply(strength)
df["team2_strength"] = df["team2"].apply(strength)

df["strength_diff"] = df["team1_strength"] - df["team2_strength"]


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
    
    wins = int((matches["winner"] == team1).sum())
    return wins / len(matches) if len(matches) > 0 else 0.5


team1_form = []
team2_form = []
h2h_list = []

for i, row in df.iterrows():
    team1_form.append(recent_form(df, row["team1"], i, n=8))
    team2_form.append(recent_form(df, row["team2"], i, n=8))
    h2h_list.append(h2h(df, row["team1"], row["team2"], i))

df["team1_form"] = team1_form
df["team2_form"] = team2_form
df["h2h"] = h2h_list
df["toss_match"] = (df["toss_winner"] == df["team1"]).astype(int)
df["home_advantage"] = (df["city"] == df["team1"]).astype(int)



def win_streak(df, team, idx):
    past = df.iloc[:idx]
    matches = past[(past["team1"] == team) | (past["team2"] == team)].tail(5)

    streak = 0
    for result in reversed(matches["winner"].tolist()):
        if result == team:
            streak += 1
        else:
            break

    return streak

team1_streak = []
team2_streak = []

for i, row in df.iterrows():
    team1_streak.append(win_streak(df, row["team1"], i))
    team2_streak.append(win_streak(df, row["team2"], i))

df["team1_streak"] = team1_streak
df["team2_streak"] = team2_streak

# Difference features
df["form_diff"] = df["team1_form"] - df["team2_form"]
df["h2h_diff"] = df["h2h"] - 0.5

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
    "toss_match",
    "team1_strength",
    "team2_strength",
    "strength_diff",
    "home_advantage",
    "team1_streak",
    "team2_streak"
]

X = pd.get_dummies(df[features])
y = (df["winner"] == df["team1"]).astype(int)
match_meta = df[["team1", "team2"]]

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, match_meta, test_size=0.2, random_state=42
)

# model_xg = XGBClassifier(
#     n_estimators=500,
#     max_depth=5,
#     learning_rate=0.03,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     min_child_weight=3,
#     gamma=0.1,
#     reg_alpha=0.3,
#     reg_lambda=5,
#     random_state=42
# )

model_xg = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.3,
    reg_alpha=1,
    reg_lambda=8,
    random_state=42
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
print("Probability of winning:", prob *100)

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model_xg, f)

# with open("encoder.pkl", "wb") as f:
#     pickle.dump(le, f)

# with open("columns_xg.pkl", "wb") as f:
#     pickle.dump(X.columns, f)