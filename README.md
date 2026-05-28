# Consumer Credit Risk Modelling - Loan Default Prediction with Canadian Macroeconomic Stress Testing for Credit Union Decision Support

End-to-end machine learning project for predicting consumer loan default risk using Lending Club and with a Canadian macroeconomic overlay and an IFRS 9 expected-credit-loss (ECL) framework.

## Overview
- Predicts loan default using origination-time borrower features only (no leakage).
- Overlays Canadian macro indicators: overnight rate, unemployment, CPI, and debt service ratio.
- Frames results in an IFRS 9 ECL and stress-testing context.

## Pipeline
| Notebook | Stage |
|---|---|
| 01_data_preparation | Load, clean, target-label policy, leakage audit |
| 02_eda_loans_macros | Exploratory analysis of loans and macro series |
| 03_feature_engineering | Build origination-time feature matrix |
| 04_preprocessing | Encoding, scaling, train/validation/test split |
| 05_modeling | Logistic regression, XGBoost, LightGBM |
| 06_documentation | Results, SHAP attribution, ECL framework |

## Key EDA findings
- FICO score is the strongest protective feature; DTI is the strongest risk feature.
- Single-feature correlations with default are weak, motivating tree-based models.
- Unemployment is the primary cyclical default driver in the Canadian macro panel.
- The household debt service ratio fell during COVID-19 (rate cuts, payment deferrals, income support), then rebounded to record highs.

## Project structure
```
data/        raw (gitignored) + processed files
notebooks/   01-06 pipeline
report/      figures and outputs
```

## Setup
```
pip install -r requirements.txt
```

## Data
LendingClub loan data and a Canadian macro panel (Statistics Canada, Bank of Canada). Large data files are gitignored and not stored in the repo.
