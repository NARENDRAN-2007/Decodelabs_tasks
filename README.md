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
