# Consumer Credit Risk Modelling - Loan Default Prediction with Canadian Macroeconomic Stress Testing for Credit Union Decision Support

End-to-end credit risk modeling project using Lending Club data, Canadian macroeconomic indicators, and an IFRS 9 expected credit loss (ECL) framework to predict consumer loan defaults.

## Overview
- Predicts loan default using origination-time borrower features only (no leakage).
- Overlays Canadian macro indicators: overnight rate, unemployment, CPI, and debt service ratio.
- Frames results in an IFRS 9 ECL and stress-testing context.

## Pipeline
| Notebook | Stage |
|---|---|
| 01_data_preparation | Load, clean, target-label policy, leakage audit |
| 02_eda_loans_macros | Exploratory analysis of loans and macro series |
| 03_feature_engineering_preprocessing | Build origination-time feature matrix, Encoding, scaling, train/validation/test split |
| 04_modeling | Logistic regression, XGBoost, LightGBM |
| 05_model_explainability | Feature importance, SHAP Analysis (Global), Business Insights |
| 06_stress-testing_and_ecl | Macro forecasting & scenario, IFRS 9 ECL framework, Stress overlay  |
| 07_documentation | ECL framwork, Project Overview, Business Recommendations |

## Key EDA findings
- FICO score is the strongest protective feature; DTI is the strongest risk feature.
- Single-feature correlations with default are weak, motivating tree-based models.
- Unemployment is the primary cyclical default driver in the Canadian macro panel.
- The household debt service ratio fell during COVID-19 (rate cuts, payment deferrals, income support), then rebounded to record highs.

## Project structure
```
data/        raw  + processed files
notebooks/   01-06 pipeline
report/      figures and outputs
models/
```

## Setup
```
pip install -r requirements.txt
```

## Data
LendingClub loan data and a Canadian macro panel (Statistics Canada, Bank of Canada). Large data files are gitignored and not stored in the repo.
