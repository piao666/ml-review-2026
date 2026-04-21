# ML Review 2026

This repository is a simple 30-day review plan for classic machine learning.
It is intended for a student who is also learning deep learning in parallel and
wants to keep classic ML practical through small projects.

The goal is not to build a large framework. The goal is to practice the common
workflow: load data, inspect it, prepare features, train a baseline model,
evaluate it, write down what you learned, and improve one thing at a time.

## Recommended Workflow

For each project:

1. Read the project README.
2. Download the dataset and place the CSV files in that project's `data/`
   folder.
3. Run the baseline notebook first if you want an interactive walkthrough.
4. Run the matching Python script when you want a repeatable command-line run.
5. Record observations in `notes/weekly_notes_template.md` or your own notes.
6. Make one small improvement at a time, such as adding features, trying a new
   model, or improving validation.

Keep each experiment readable. Prefer clear feature lists, simple comments, and
short notes over clever code.

## Folder Structure

```text
ml-review-2026/
  README.md
  requirements.txt
  .gitignore
  01_titanic/
    README.md
    titanic_baseline.ipynb
    titanic_baseline.py
    data/
  02_house_prices/
    README.md
    house_prices_baseline.ipynb
    house_prices_baseline.py
    data/
  03_sklearn_mini_cases/
    README.md
  notes/
    weekly_notes_template.md
```

## Install Dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the small starter dependency set:

```bash
python -m pip install -r requirements.txt
```

## Run Notebooks and Scripts

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open one of these notebooks:

- `01_titanic/titanic_baseline.ipynb`
- `02_house_prices/house_prices_baseline.ipynb`

You can also run the scripts directly:

```bash
python 01_titanic/titanic_baseline.py
python 02_house_prices/house_prices_baseline.py
```

The scripts expect dataset files under each project `data/` folder. See each
subproject README for exact filenames.

## 30-Day Roadmap Summary

### Days 1-7: Classification Basics with Titanic

- Review train/test split, target variables, and leakage.
- Load the Titanic dataset with pandas.
- Build a simple logistic regression baseline.
- Learn basic preprocessing for numeric and categorical columns.
- Evaluate accuracy, precision, recall, and confusion matrix.
- Try small improvements: more features, different imputing, or another model.

### Days 8-14: Regression Basics with House Prices

- Review regression metrics such as MAE, RMSE, and R2.
- Load the House Prices dataset.
- Build a baseline regression model.
- Compare simple numeric features with mixed numeric and categorical features.
- Practice reading model errors and spotting outliers.
- Write down which features seem most useful and why.

### Days 15-21: Scikit-Learn Mini Cases

- Practice small built-in datasets from scikit-learn.
- Compare classification and regression workflows.
- Try logistic regression, decision trees, random forests, and k-nearest
  neighbors.
- Practice cross-validation and simple hyperparameter tuning.
- Keep notes on when each model type feels appropriate.

### Days 22-30: Review, Improve, and Connect to Deep Learning

- Revisit the Titanic and House Prices baselines.
- Improve validation and compare at least two models per project.
- Write a short project summary for each dataset.
- Connect classic ML ideas to deep learning: features, loss functions,
  overfitting, regularization, and evaluation.
- End with a small portfolio note: what you built, what worked, and what you
  would try next.

