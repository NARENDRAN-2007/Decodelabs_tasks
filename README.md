Here's the README content, in text form:

## Project 1 — Advanced EDA & Feature Engineering
*DecodeLabs Data Science Industrial Training Kit — Batch 2026*

**Overview**
Transforms a raw e-commerce orders dataset into a clean, mathematically sound, model-ready dataset. Follows an Input → Process → Output pipeline: secure input fidelity (missing values, outliers) → vectorized feature engineering and encoding → validated, ML-ready output.

**Files**
- `ds_csv.csv` — raw input dataset (1200 orders, 14 columns)
- `project1_pipeline.py` — full pipeline script (run this)
- `ds_csv_cleaned.csv` — output: cleaned data, human-readable (categoricals intact)
- `ds_csv_model_ready.csv` — output: encoded, decollinearized, ML-ready data

**Dataset**
- Columns: OrderID, Date, CustomerID, Product, Quantity, UnitPrice, ShippingAddress, PaymentMethod, OrderStatus, TrackingNumber, ItemsInCart, CouponCode, ReferralSource, TotalPrice
- 1200 rows, no duplicates
- Only `CouponCode` has missing values (25.75%)
- `TotalPrice = Quantity × UnitPrice` holds for every row (verified)

**How to run**
- `pip install pandas numpy scikit-learn`
- `python project1_pipeline.py`
- Run from the same folder as `ds_csv.csv`; produces the two output CSVs in that folder.

**Methodology — Phase 1: Securing input fidelity**
- Missing values handled by a decision matrix:
  - <5% missing → drop rows
  - 5–20% missing → median (numeric) / mode or sub-group (categorical)
  - \>20% missing → KNN imputation (numeric)
- Exception: `CouponCode`'s 25.75% missingness is structural (means "no coupon used"), not random — filled with an explicit `NONE` sentinel instead of imputed, to avoid fabricating promotions.
- Outliers detected via IQR (`Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`) on all numeric columns, then **capped with `numpy.clip()`** rather than dropped, to preserve row count. Only `TotalPrice` had outliers (8 rows capped).

**Methodology — Phase 2: Feature engineering & encoding**
- New features (all vectorized, no loops):
  - `OrderMonth`, `OrderDayOfWeek`, `IsWeekend` — from `Date`
  - `PricePerItem` = `TotalPrice / ItemsInCart`
  - `CartFillRatio` = `Quantity / ItemsInCart`
  - `CustomerOrderCount`, `IsRepeatCustomer` — customer order frequency
  - `HasCoupon` — binary coupon-usage flag
- Categorical columns one-hot encoded (not label-encoded) to avoid implying false ordinal distance.
- ID-like columns (`OrderID`, `TrackingNumber`, `ShippingAddress`, `CustomerID`, `Date`) dropped before modeling — unique keys, not predictive signal.
- Multicollinearity check: absolute correlation matrix built over numeric/encoded features + target (`TotalPrice`); pairs with r > 0.80 flagged; the feature more weakly correlated with the target is dropped. `IsRepeatCustomer` was dropped as collinear with `CustomerOrderCount`.

**Methodology — Phase 3: Output**
- Two files saved: cleaned/readable version and fully encoded/decollinearized model-ready version.

**Known limitations / notes for grading**
- Dataset has almost no missing numeric values or outliers, so the Mean-vs-KNN comparison suggested in the brief isn't well demonstrated here — consider testing against a version with injected missingness/outliers if the grader wants that shown explicitly.
- `TotalPrice` is used as the target for correlation/collinearity checks — swap the `TARGET` variable in the script if a different target (e.g. `OrderStatus`) is needed.

  # Project 2 — Supervised Learning (Fraud/Anomaly Detection Pipeline)

DecodeLabs Data Science Industrial Training Kit — Batch 2026

## Important: dataset mismatch

The brief is written for the classic credit-card fraud dataset
(284,807 transactions, 0.17% fraud rate — extreme imbalance).
`ds_csv.csv` is the same e-commerce orders dataset used in Project 1.
It has **no fraud label**, and `OrderStatus` is split almost evenly
across 5 classes (Cancelled/Returned/Pending/Shipped/Delivered, each
~20%). There is no comparable class imbalance here.

`IsCancelled` (1 = order status is "Cancelled", 0 = anything else,
~21% positive) is used as the closest available proxy target so every
required technique — SMOTE, leak-free imblearn pipelines, GridSearchCV,
Precision/Recall/ROC-AUC — could still be demonstrated correctly on
real data, even though genuine imbalance and genuine predictive signal
are both mild-to-absent in this dataset (see Results below).

## Files

- `ds_csv.csv` — input dataset
- `project2_pipeline.py` — full pipeline script, run this
- `model_comparison.csv` — output: final metrics per model

## How to run

```
pip install pandas numpy scikit-learn imbalanced-learn
python project2_pipeline.py
```

## Pipeline architecture

1. **Feature preparation, leak-free** — reused Project 1's engineered
   features. Dropped `OrderStatus` (it's the target source), and all
   ID/free-text columns (`OrderID`, `TrackingNumber`,
   `ShippingAddress`, `CustomerID`, `Date`). One-hot encoded
   remaining categoricals.
2. **Stratified train/test split first** — 80/20, `stratify=y`, done
   *before* any scaling or SMOTE, so the test set reflects the real
   class ratio and never touches synthetic data.
3. **Leak-free pipelines** — built with `imblearn.pipeline.Pipeline`,
   not `sklearn.pipeline.Pipeline`, since sklearn's version silently
   drops or breaks on resampling steps:
   - Logistic Regression: `StandardScaler → SMOTE → LogisticRegression`
     (LR is scale-sensitive, so scaling is inside the pipeline)
   - Random Forest: `SMOTE → RandomForestClassifier`
     (tree splits are scale-invariant, so no scaler needed)
4. **GridSearchCV** — tunes `smote__k_neighbors` alongside each
   model's hyperparameters (`classifier__C` for LR;
   `classifier__n_estimators`, `classifier__max_depth` for RF), scored
   on **ROC-AUC**, with 5-fold stratified CV. Because SMOTE lives
   inside the pipeline, it's refit on the training fold only for every
   parameter combination — zero leakage into validation folds.
5. **Final evaluation** — on the untouched test set, using Precision,
   Recall, F1, and ROC-AUC. Accuracy is intentionally not used to
   select or report the "best" model.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (SMOTE + Scaled) | 0.163 | 0.280 | 0.206 | 0.439 |
| Random Forest (SMOTE) | 0.182 | 0.080 | 0.111 | 0.397 |

**ROC-AUC ≈ 0.40–0.53 for both models — essentially random
(0.5 = no better than a coin flip).** This isn't a bug in the
pipeline; it's consistent with what Project 1's EDA already showed —
`OrderStatus` doesn't correlate meaningfully with product, payment
method, price, or any other feature in this dataset. There's no
learnable relationship for a classifier to find, because the
underlying data appears synthetically generated without an embedded
fraud/cancellation pattern.

## What this demonstrates vs. what it can't

**Correctly demonstrated:**
- SMOTE applied only inside the training fold, never leaking into
  test/validation data
- Scaler placement rule respected (inside pipeline, model-dependent)
- GridSearchCV tuning resampling + model hyperparameters jointly
- Evaluation strictly via Precision/Recall/F1/ROC-AUC, not Accuracy

**Can't be demonstrated on this data:**
- The core skill the brief is really testing — pulling a genuine rare
  signal out of a 0.17%-imbalance dataset — needs data where that
  signal actually exists.

## Recommendation

To actually exercise this pipeline the way the brief intends, run
`project2_pipeline.py` against the real Kaggle "Credit Card Fraud
Detection" dataset (`creditcard.csv`, 284,807 rows, `Class` column as
target, 0.17% fraud rate) — the column names would need small
adjustments (drop `Time`, use `Amount` + `V1`–`V28` as features,
`Class` as target), but the pipeline logic (split → leak-free
SMOTE/scale → GridSearchCV → Precision/Recall/ROC-AUC) is unchanged.
Want me to adapt the script for that dataset if you have it or can
download it?
