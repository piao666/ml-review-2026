"""Beginner House Prices regression baseline.

Run from the repository root:
    python 02_house_prices/house_prices_baseline.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).parent / "data" / "train.csv"


def main() -> None:
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Download the House Prices train.csv file and place it in 02_house_prices/data/.")
        return

    houses = pd.read_csv(DATA_PATH)

    # Use a small feature set first. Add more only after this baseline runs.
    target_column = "SalePrice"
    numeric_features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "FullBath",
        "YearBuilt",
    ]
    categorical_features = [
        "Neighborhood",
        "HouseStyle",
        "ExterQual",
        "KitchenQual",
    ]
    feature_columns = numeric_features + categorical_features

    X = houses[feature_columns]
    y = houses[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("House Prices random forest baseline")
    print(f"Rows: {len(houses)}")
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R2:   {r2:.3f}")

    plt.scatter(y_test, predictions, alpha=0.6)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("House Prices Baseline: Actual vs Predicted")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

