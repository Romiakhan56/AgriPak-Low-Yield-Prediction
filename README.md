# AgriPak Farms – Low-Yield Risk Prediction

## Project Overview
This project develops a Machine Learning baseline model to predict low-yield risk in agricultural data.

## Dataset
- Total records: 50,000
- Total features: 9
- Target variable: `low_yield_risk`

## Data Cleaning
- Original records: 50,000
- Duplicate rows removed: 0
- Missing values before cleaning: 0
- Missing values after cleaning: 0
- Final records: 50,000

## Train-Test Split
- Training samples: 40,000 (80%)
- Testing samples: 10,000 (20%)
- Method: Stratified
- Random state: 42

## Baseline Model
**Algorithm:** Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 91.87% |
| Precision (Class 1) | 88.84% |
| Recall (Class 1) | 87.80% |
| F1-Score (Class 1) | 88.32% |

## Data Leakage Check
- Highest absolute correlation: `rainfall_mm` (-0.636409)
- No feature exceeded the absolute correlation threshold of 0.90.
- No obvious high-correlation leakage feature was identified.

Correlation alone cannot rule out every form of data leakage. Feature availability and collection timing should also be reviewed.

## Project Structure
```text
AgriPak-Low-Yield-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── clean_data.py
│   ├── split_data.py
│   └── train_model.py
├── notebooks/
├── README.md
├── LICENSE
└── .gitignore
```

## How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

Run scripts from the project root:

```bash
python src/clean_data.py
python src/split_data.py
python src/train_model.py
```

## Conclusion
The Logistic Regression baseline achieved 91.87% accuracy and an 88.32% F1-score for class 1. No feature exceeded the 0.90 absolute-correlation threshold in the leakage check.
