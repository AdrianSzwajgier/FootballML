import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ml.ml.data_preprocessing import (
    filter_clubs_by_min_matches,
    calculate_weights, encode_features
)
from ml.ml.utils import load_configuration


def load_data(file_path: str = r'data\archive\games.csv') -> pd.DataFrame:
    """
    Reads and returns DataFrame from csv file in given path.

    Args:
        file_path (str, optional): path to csv file.

    Returns:
        df (pd.DataFrame): loaded dataframe

    Raises:
        FileNotFoundError: if file does not exist.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"File '{file_path}' not found."
        ) from error

    return df


def main():
    df = load_data()

    config = load_configuration()

    df = df[
        [
            "season",
            "home_club_name",
            "away_club_name",
            "home_club_goals",
            "away_club_goals",
            "game_id"
        ]
    ]

    df = filter_clubs_by_min_matches(
        df,
        "game_id",
        'home_club_name',
        'away_club_name',
        threshold=config["min_matches_played"]
    )

    weights = calculate_weights(df, alpha=config["decay_factor"])

    encoder = OneHotEncoder()
    df, encoded_club_names = encode_features(
        encoder,
        df,
        config["categorical_features"]
    )

    df = df.drop(columns=['home_club_name', 'away_club_name', 'game_id', "season"])

    # Features and target variables
    X = df.drop(columns=['home_club_goals', 'away_club_goals'])
    y_home = df['home_club_goals']
    y_away = df['away_club_goals']

    # Split the data into training and testing sets
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, w_train, w_test = train_test_split(
        X, y_home, y_away, weights, test_size=0.2, random_state=42
    )

    # Initialize and train the model for home goals
    home_model = RandomForestRegressor(random_state=42)
    home_model.fit(X_train, y_home_train, sample_weight=w_train)

    # Initialize and train the model for away goals
    away_model = RandomForestRegressor(random_state=42)
    away_model.fit(X_train, y_away_train, sample_weight=w_train)

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
        'home_club_name': ['Real Madrid'],
        'away_club_name': ['FC Barcelona']
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # Encode the input data
    encoded_input = encoder.transform(input_df).toarray()
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoded_club_names)

    # Predict the goals
    home_goals_pred = home_model.predict(encoded_input_df)
    away_goals_pred = away_model.predict(encoded_input_df)
    print(f"{home_goals_pred = }")
    print(f'Predicted {input_data['home_club_name'][0]}: {home_goals_pred[0]}')
    print(f'Predicted {input_data['away_club_name'][0]}: {away_goals_pred[0]}')


if __name__ == "__main__":
    main()
