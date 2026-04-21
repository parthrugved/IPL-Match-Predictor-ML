import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("matches.csv")

# Cleaning
df["city"] = df["city"].ffill()
df = df.dropna(subset=["winner"])
df = df[df["result"] == "win"]


features = ["city", "team1", "team2", "toss_winner", "toss_decision"]
X = df[features]
y = df["winner"]

# Encoding
X = pd.get_dummies(X)

le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Actual Winner': le.inverse_transform(y_test),
    'Predicted Winner': le.inverse_transform(pred)
})

print("First 5 comparisons:")
print(comparison_df.head(15))
print("\nAccuracy:", accuracy_score(y_test, pred)*100)
