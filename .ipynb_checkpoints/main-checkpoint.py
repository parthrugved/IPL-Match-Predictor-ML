import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LinearRegression

df = pd.read_csv("matches.csv")

# # Data cleaning
# df['city'] = df['city'].ffill()
# df.dropna(subset=['winner'], inplace=True)

# # Feature selection: only use info available before/at the start of the match
# features = ['city', 'team1', 'team2', 'toss_winner', 'toss_decision']
# X = df[features]
# y = df['winner']

# # Encoding categorical variables
# X = pd.get_dummies(X, columns=features)
# y = pd.get_dummies(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)


# # Model Lin
# lin_model = LinearRegression()
# lin_model.fit(X_train,y_train)
# lin_pred = lin_model.predict(X_test)

# # Translate the number arrays back into actual team names
# predicted_team_indices_lin = np.argmax(lin_pred, axis=1)
# team_names_lin = y.columns
# predicted_winners_lin = [team_names_lin[i] for i in predicted_team_indices_lin]

# print(f"first 5 lin model predictions are: {predicted_winners_lin[:5]}")

print(df.tail(1))