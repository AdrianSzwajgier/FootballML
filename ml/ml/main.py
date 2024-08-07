import os
import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ml.ml.data_preprocessing import filter_clubs_by_min_matches

db_path = r'data\archive\games.csv'

if not os.path.exists(db_path):
    print("The database file does not exist at the specified path.")

df = pd.read_csv(db_path)
df1 = df[['home_club_name', 'game_id']].groupby('home_club_name').count().sort_values(by='game_id')
df2 = df[['away_club_name', 'game_id']].groupby('away_club_name').count().sort_values(by='game_id')

# import matplotlib.pyplot as plt
#
# match_counts = df['home_club_name'].value_counts() + df['away_club_name'].value_counts()
# plt.hist(match_counts, bins=50)
# plt.xlabel('Number of Matches Played')
# plt.ylabel('Number of Clubs')
# plt.title('Distribution of Matches Played by Clubs')
# plt.show()

# print(df['date'].max())
#
# print(df1, df2)

df = df[["home_club_name", "away_club_name", "home_club_goals", "away_club_goals", "game_id"]]

df.to_csv("test.csv")
df = filter_clubs_by_min_matches(df, "game_id", 'home_club_name', 'away_club_name', threshold=50)

# quit(3465)
# df = df.head(50000)


encoder = OneHotEncoder()
encoded_clubs = encoder.fit_transform(df[['home_club_name', 'away_club_name']]).toarray()
encoded_club_names = encoder.get_feature_names_out(['home_club_name', 'away_club_name'])


encoded_df = pd.DataFrame(encoded_clubs, columns=encoded_club_names)

# Combine the encoded features with the original DataFrame
df = pd.concat([df, encoded_df], axis=1) #TODO nans
df[df["home_club_goals"].isna()].to_csv("test.csv")
# quit()
# Drop the original categorical columns
df = df.drop(columns=['home_club_name', 'away_club_name', 'game_id'])

# Features and target variables
X = df.drop(columns=['home_club_goals', 'away_club_goals'])
y_home = df['home_club_goals']
y_away = df['away_club_goals']

print(y_home.unique())

# Split the data into training and testing sets
X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
    X, y_home, y_away, test_size=0.2, random_state=42
)

# Initialize and train the model for home goals
home_model = RandomForestRegressor(random_state=42)
home_model.fit(X_train, y_home_train)

# Initialize and train the model for away goals
away_model = RandomForestRegressor(random_state=42)
away_model.fit(X_train, y_away_train)

# Predict on the test set
y_home_pred = home_model.predict(X_test)
y_away_pred = away_model.predict(X_test)

# Evaluate the model
home_mse = mean_squared_error(y_home_test, y_home_pred)
away_mse = mean_squared_error(y_away_test, y_away_pred)

print(f'Home Goals Prediction MSE: {home_mse}')
print(f'Away Goals Prediction MSE: {away_mse}')

# Example input for prediction
input_data = {
    'home_club_name': ['FC Barcelona'],
    'away_club_name': ['Real Madrid']
}

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

# Encode the input data
encoded_input = encoder.transform(input_df).toarray()
encoded_input_df = pd.DataFrame(encoded_input, columns=encoded_club_names)

# Predict the goals
home_goals_pred = home_model.predict(encoded_input_df)
away_goals_pred = away_model.predict(encoded_input_df)

print(f'Predicted Home Goals: {home_goals_pred[0]}')
print(f'Predicted Away Goals: {away_goals_pred[0]}')
