# Model Metrics

**Project:** Consumer Credit Risk Modelling — Loan Default Prediction with Canadian Macroeconomic Stress Testing
**Author:** Michael Ikechukwu Jumbo | Springboard Data Science Career Track — Capstone Three
**Repository:** https://github.com/MIJUMBO/consumer-credit-risk-modelling

This file summarizes the final model: its features, parameters, hyperparameters, and performance metrics, as required by the Springboard Capstone documentation rubric.

---

## 1. Final Model

| Item | Value |
|---|---|
| **Champion algorithm** | XGBoost (`XGBClassifier`) |
| **Selection basis** | Tied LightGBM on all discrimination metrics; chosen on parsimony (230 vs 591 trees) and lower Brier |
| **Raw model artifact** | `models/champion_xgb_production.pkl` (used for SHAP and ranking) |
| **Calibrated artifact** | `models/champion_xgb_calibrated.pkl` (isotonic; used for IFRS 9 ECL) |
| **Calibration method** | Isotonic regression (`CalibratedClassifierCV`, `cv='prefit'`), fit on the 2016 validation vintage |
| **Benchmark model** | Logistic Regression (retained as interpretable baseline) |
| **Target** | `default_flag` (1 = Charged Off / Default, 0 = Fully Paid) |
| **Prediction task** | Binary probability of default (PD) at loan origination |

---

## 2. Features (15 origination-time predictors)

The leakage-free feature set. LendingClub's own risk-pricing fields (`grade`, `sub_grade`, `int_rate`) are **deliberately excluded** from modelling to avoid circular prediction.

| # | Feature | Type | Family | Notes |
|---|---|---|---|---|
| 1 | `loan_to_income` | continuous | Affordability | loan_amnt / annual_inc |
| 2 | `dti` | continuous | Affordability | 99th-pct capped, median-imputed |
| 3 | `fico_orig` | continuous | Credit quality | midpoint of FICO range |
| 4 | `revol_util` | continuous | Credit quality | median-imputed |
| 5 | `loan_amnt` | continuous | Affordability | also serves as EAD in ECL |
| 6 | `log_annual_inc` | continuous | Affordability | log1p, 99th-pct capped |
| 7 | `employment_years` | continuous | Employment | parsed from emp_length |
| 8 | `short_tenure_flag` | binary | Employment | 1 if < 1 year |
| 9 | `delinq_flag` | binary | Derogatory | 1 if delinq_2yrs > 0 |
| 10 | `pub_rec_flag` | binary | Derogatory | 1 if pub_rec > 0 |
| 11 | `open_acc_band` | ordinal | Credit quality | Few/Some/Many/VeryMany |
| 12 | `purpose_risk_tier` | ordinal | Loan context | Low/Med/High from EDA default rates |
| 13 | `home_ownership` | ordinal | Loan context | NONE…MORTGAGE |
| 14 | `vintage_year` | numeric | Vintage | from issue_d |
| 15 | `vintage_quarter` | numeric | Vintage | from issue_d |

### Categorical encoding (rubric: "create dummy features")
Three categoricals (`open_acc_band`, `purpose_risk_tier`, `home_ownership`) are converted to numeric with **`OrdinalEncoder`**, fit on the training split only and applied (never refit) to validation, test, and production. Ordinal encoding is used rather than one-hot dummies because (a) all three are genuinely ordered and (b) tree-based models split on thresholds, so one-hot expansion would add no information while inflating dimensionality. `handle_unknown='use_encoded_value'`, `unknown_value=-1`.

### Magnitude standardization (rubric: "magnitude standardized")
Features are standardized with **`StandardScaler`** for **Logistic Regression only**, whose gradient-based optimisation is scale-sensitive. XGBoost and LightGBM are trained on raw values because tree splits are invariant to monotonic feature scaling. The fitted scaler is saved as `models/lr_scaler.pkl`.

---

## 3. Data Splits (temporal / out-of-time)

Partitioned by `vintage_year` to simulate production scoring and prevent vintage leakage.

| Split | Vintages | Rows (100k dev) | Rows (500k prod) | Default rate |
|---|---|---|---|---|
| Train | ≤ 2015 | 61,442 | 307,208 | 18.43% |
| Validation | 2016 | 21,786 | — | 23.29% |
| Test (out-of-time) | 2017–2018 | 16,772 | 83,859 | 21.29% |

Class imbalance handled via `class_weight='balanced'` (LR) and `scale_pos_weight = 4.43` (boosters); no resampling.

---

## 4. Hyperparameters

### XGBoost — Champion (Optuna, seeded TPE, 30-min budget)
| Hyperparameter | Value |
|---|---|
| `max_depth` | 4 |
| `learning_rate` | 0.0401 |
| `subsample` | 0.7888 |
| `colsample_bytree` | 0.6361 |
| `min_child_weight` | 9 |
| `reg_lambda` | 1.4759 |
| `n_estimators` (effective) | 230 (early-stopping `best_iteration_`) |
| `objective` | `binary:logistic` |
| `eval_metric` | `auc` |
| `scale_pos_weight` | 4.43 |
| `random_state` | 42 |

### LightGBM — Runner-up (Optuna, seeded TPE, 30-min budget)
| Hyperparameter | Value |
|---|---|
| `num_leaves` | 15 |
| `min_child_samples` | 176 |
| `learning_rate` | 0.0121 |
| `subsample` | 0.5932 |
| `colsample_bytree` | 0.7269 |
| `reg_lambda` | 0.1693 |
| `n_estimators` (effective) | 591 |
| `max_depth` | −1 (unbounded) |

### Logistic Regression — Baseline
| Hyperparameter | Value |
|---|---|
| `class_weight` | `balanced` |
| `max_iter` | 1000 |
| `random_state` | 42 |
| preprocessing | `StandardScaler` |

---

## 5. Model Performance Comparison Table (Validation, 2016 vintage)

| Model | AUC | KS | Gini | Brier |
|---|---|---|---|---|
| **XGBoost (champion)** | **0.6684** | **0.2472** | 0.3367 | **0.2316** |
| LightGBM | 0.6684 | 0.2471 | 0.3369 | 0.2320 |
| Logistic Regression | 0.6616 | 0.2362 | 0.3232 | 0.2368 |

Base-rate Brier reference: **0.1786** (at 23.3% default rate).

**Key finding:** Two independent Optuna searches from different configurations both converged to AUC 0.6684, indicating a predictive ceiling set by the leakage-free feature set, not by model choice.

---

## 6. Final Model — Out-of-Time Test Performance (2017–2018, 500k production retrain)

| Metric | Value |
|---|---|
| **Test AUC (pooled 2017–2018)** | **0.6726** |
| Test AUC — 2017 vintage | 0.6725 |
| Test AUC — 2018 vintage | 0.6668 |
| Temporal drift | None material (within sampling noise) |
| Test loans scored | 83,859 (drift) / 16,772 (ECL portfolio) |

### Diagnostic operating threshold (dollar-weighted policy)
| Metric | Value |
|---|---|
| Operating threshold | 0.490 |
| Youden's J threshold | 0.519 |
| Default recall (count) | 69% (3,483 of 5,073) |
| Default precision | 32% |
| Dollar recall (default $ captured) | ≥ 75% |
| Accuracy | 58% |

---

## 7. Calibration Metrics (isotonic, on 500k test)

| Metric | Value |
|---|---|
| Base-rate Brier (to beat) | 0.1676 |
| Brier — before calibration (raw, class-weighted) | 0.2254 |
| **Brier — after isotonic calibration** | **0.1576** |
| Mean calibrated PD (ECL portfolio) | 0.2205 |
| Observed default rate (ECL portfolio) | 0.2129 |

Isotonic regression is monotonic → AUC / KS / Gini unchanged; only the probability scale is corrected, making the PDs valid for `ECL = PD × LGD × EAD`.

---

## 8. Top Feature Importances

**XGBoost gain (top 6):** fico_orig (18.7%), loan_to_income (13.9%), dti (8.2%), home_ownership (8.0%), purpose_risk_tier (6.7%), loan_amnt (5.8%).

**SHAP mean |value| (top 6):** fico_orig (0.320), loan_to_income (0.231), dti (0.158), loan_amnt (0.109), home_ownership (0.093), purpose_risk_tier (0.085).

Both methods agree on the core drivers — credit quality and affordability — confirming a stable, economically defensible model.

---

## 9. IFRS 9 ECL — Downstream Application of the Final Model

**Assumptions:** LGD = 45%; EAD = loan amount; SICR threshold = 80th-percentile baseline PD (0.3064); lifetime factor = 2.5×; PD elasticity = 10% relative per +1pp unemployment.

### Baseline allowance (test portfolio, EAD $242,116,700)
| Stage | Loans | ECL | Coverage |
|---|---|---|---|
| Stage 1 — Performing | 10,687 | $10,828,689 | 7.99% |
| Stage 2 — SICR | 2,514 | $20,708,922 | 41.05% |
| Stage 3 — Credit-impaired | 3,571 | $25,295,490 | 45.00% |
| **Total** | **16,772** | **$56,833,101** | **23.47%** |

Naive flat-rate ECL: $23,197,557 → model adds $33.6M of risk-differentiated provisioning.

### Stress scenarios
| Scenario | Unemp shock | Total ECL | Coverage | Stage 2 loans | ECL vs baseline |
|---|---|---|---|---|---|
| Baseline | +0.0pp | $56,833,101 | 23.47% | 2,514 | +0.0% |
| Adverse (GFC replay) | +2.7pp | $65,636,032 | 27.11% | 3,537 | +15.5% |
| Severely Adverse | +4.5pp | $73,333,368 | 30.29% | 5,022 | +29.0% |

---

## 10. Reproducibility

- All random operations seeded (`random_state=42`); Optuna uses seeded TPE samplers within a 30-minute budget per model.
- Pipeline: `01_data_preparation` → `02_eda_loans_macros` → `03_feature_engineering_preprocessing` → `04_modeling` → `05_model_explainability` → `06_stress_testing_and_ecl`.
- Feature governance exported to `data/processed/consistent_features.json`.
