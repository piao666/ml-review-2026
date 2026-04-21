# Titanic Baseline

This project reviews a beginner-friendly classification workflow using the
Titanic dataset.

## Dataset

Download the Titanic dataset from Kaggle or another class source and place the
training file here:

```text
01_titanic/data/train.csv
```

The baseline expects the common Kaggle columns, including:

- `Survived`
- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`

## What the Baseline Does

Both `titanic_baseline.ipynb` and `titanic_baseline.py` do the same core steps:

1. Load `data/train.csv` with pandas.
2. Select a small set of readable starter features.
3. Fill missing numeric values with the median.
4. Fill missing categorical values with the most common value.
5. One-hot encode categorical columns.
6. Train a logistic regression classifier.
7. Print accuracy and a classification report.
8. Plot a simple confusion matrix.

The notebook is best for learning cell by cell. The script is best for checking
that the whole baseline can run from start to finish.

## Run

From the repository root:

```bash
python 01_titanic/titanic_baseline.py
```

Or start Jupyter and open:

```text
01_titanic/titanic_baseline.ipynb
```

