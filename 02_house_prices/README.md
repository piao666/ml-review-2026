# House Prices Baseline

This project reviews a beginner-friendly regression workflow using the House
Prices dataset.

## Dataset

Download the House Prices dataset from Kaggle or another class source and place
the training file here:

```text
02_house_prices/data/train.csv
```

The baseline expects the common Kaggle columns, including:

- `SalePrice`
- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `GarageArea`
- `TotalBsmtSF`
- `FullBath`
- `YearBuilt`
- `Neighborhood`
- `HouseStyle`
- `ExterQual`
- `KitchenQual`

## What the Baseline Does

Both `house_prices_baseline.ipynb` and `house_prices_baseline.py` do the same
core steps:

1. Load `data/train.csv` with pandas.
2. Select a small starter set of numeric and categorical features.
3. Fill missing numeric values with the median.
4. Fill missing categorical values with the most common value.
5. One-hot encode categorical columns.
6. Train a random forest regression model.
7. Print MAE, RMSE, and R2 on a held-out test set.
8. Plot predicted prices against actual prices.

The notebook is best for learning cell by cell. The script is best for checking
that the whole baseline can run from start to finish.

## Run

From the repository root:

```bash
python 02_house_prices/house_prices_baseline.py
```

Or start Jupyter and open:

```text
02_house_prices/house_prices_baseline.ipynb
```

