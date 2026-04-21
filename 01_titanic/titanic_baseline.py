"""Beginner Titanic classification baseline.

Run from the repository root:
    python 01_titanic/titanic_baseline.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).parent / "data" / "train.csv"


def main() -> None:
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Download the Titanic train.csv file and place it in 01_titanic/data/.")
        return

    titanic = pd.read_csv(DATA_PATH)

    # Start with a small feature set that mixes numbers and categories.
    target_column = "Survived"
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]
    feature_columns = numeric_features + categorical_features

    X = titanic[feature_columns]
    y = titanic[target_column]

    # Keep a test set aside so evaluation uses rows the model did not train on.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Titanic logistic regression baseline")
    print(f"Rows: {len(titanic)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print()
    print(classification_report(y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title("Titanic Baseline Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

