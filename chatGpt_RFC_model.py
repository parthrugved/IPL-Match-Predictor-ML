import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------- LOAD -------------------
df = pd.read_csv("matches.csv")

# ------------------- CLEAN -------------------
df["city"] = df["city"].ffill()
df = df.dropna(subset=["winner"])
df = df[df["result"] == "win"]

# IMPORTANT: sort by time so "past matches" makes sense
df = df.sort_values("date").reset_index(drop=True)

# ------------------- FEATURE ENGINEERING -------------------

# recent form (last 5 matches)
def recent_form(df, team, idx, n=5):
    past = df.iloc[:idx]
    team_matches = past[(past["team1"] == team) | (past["team2"] == team)].tail(n)
    wins = sum(team_matches["winner"] == team)
    return wins / n if n > 0 else 0

# head-to-head
def h2h(df, t1, t2, idx):
    past = df.iloc[:idx]
    matches = past[
        ((past["team1"] == t1) & (past["team2"] == t2)) |
        ((past["team1"] == t2) & (past["team2"] == t1))
    ]
    total = len(matches)
    wins = sum(matches["winner"] == t1)
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

# toss advantage
df["toss_match"] = (df["toss_winner"] == df["team1"]).astype(int)

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
    "toss_match"
]

X = df[features]
y = df["winner"]

# ------------------- ENCODING -------------------
X = pd.get_dummies(X)

le = LabelEncoder()
y = le.fit_transform(y)

# ------------------- SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------- MODEL -------------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# ------------------- OUTPUT -------------------
comparison_df = pd.DataFrame({
    "Actual Winner": le.inverse_transform(y_test),
    "Predicted Winner": le.inverse_transform(pred)
})

print(comparison_df.head(15))
print("\nAccuracy:", accuracy_score(y_test, pred) * 100)
