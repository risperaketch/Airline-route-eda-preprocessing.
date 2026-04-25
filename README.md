# **Airline Route Preprocessing & EDA — Southwest Airlines Case Study**

## Business Problem

An analytics consulting firm working for **Southwest Airlines** needs to prepare a dataset of airline routes for predictive modeling. The raw data contains missing values, mixed data types, and unscaled features that must be resolved before any machine learning model can be built.

---

## Dataset

`Airfares2.csv` — 638 airline routes, 18 features

| Feature Group | Variables |
|---|---|
| Route identifiers | S_CODE, S_CITY, E_CODE, E_CITY |
| Market characteristics | COUPON, NEW, HI, DISTANCE, PAX |
| City demographics | S_INCOME, E_INCOME, S_POP, E_POP |
| Binary market indicators | VACATION, SW, SLOT, GATE |
| Target variable | FARE (average route fare) |

---

## What This Notebook Covers

1. **Data Loading & Inspection** — shape, dtypes, missing value audit using `MissingIndicator`
2. **Variable Type Classification** — converting 8 object columns to proper `category` dtype
3. **Exploratory Data Analysis** — descriptive statistics, FARE distribution (right-skewed, skewness = +0.62), VACATION class imbalance (73% No / 27% Yes)
4. **Missing Value Imputation** — median for numerical columns, mode for categorical columns
5. **Min-Max Normalization** — all 10 numerical features scaled to [0, 1]
6. **One-Hot Encoding** — 8 categorical features encoded with `drop_first=True` to avoid multicollinearity

---
**Tools:** Python · pandas · scikit-learn · seaborn · matplotlib


## Key Findings

- **FARE** is positively skewed (skewness = +0.62) — most routes have lower to mid-range fares, with a few high-fare outliers
 <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/7ab86955-645f-4117-9d4e-197d6950c6f6" />

- **VACATION** is imbalanced — 73% of routes are non-vacation; this should be accounted for if used as a prediction target
 <img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/9649b88d-f34d-4fe3-9e88-1ff59bbc7318" />

- **Missing data** is minimal (< 1% per column) and resolved entirely through imputation with no row deletion
- **Final feature set** after encoding: fully numerical and ready for ML modeling

---
## Skills Demonstrated

`Data Wrangling` `Missing Value Imputation` `Feature Engineering` `EDA` `MinMaxScaler` `One-Hot Encoding` `pandas` `scikit-learn` `seaborn`

---

*Part of my data analytics portfolio — [View full portfolio](https://github.com/your-username)*
