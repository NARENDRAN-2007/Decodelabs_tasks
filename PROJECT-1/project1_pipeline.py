"""
DecodeLabs — Data Science Project 1
Advanced EDA & Feature Engineering
------------------------------------
Dataset: ds_csv.csv (e-commerce order data)
Target used for correlation checks: TotalPrice

Pipeline:
  Phase 1 - Securing Input Fidelity   (missing values, outlier bounds)
  Phase 2 - Vectorized Computation    (feature engineering, encoding, collinearity)
  Phase 3 - Output                   (clean, ML-ready CSV)
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ==================================================================
# LOAD
# ==================================================================
df = pd.read_csv('ds_csv.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("Initial shape:", df.shape)
print("\nMissingness per column (%):\n", (df.isnull().mean() * 100).round(2))

# ==================================================================
# PHASE 1 — SECURING INPUT FIDELITY
# ==================================================================

# --------------------------------------------------------------
# 1a. Missing-data decision matrix
#     <5%     -> drop rows
#     5%-20%  -> statistical imputation (median for skewed numeric,
#                sub-group/mode for categorical)
#     >20%    -> KNN imputation for numeric columns.
#     Categorical exception: if missingness is structurally meaningful
#     (e.g. "no coupon used"), fill with an explicit sentinel instead
#     of imputing a value that was never actually random.
# --------------------------------------------------------------
missing_pct = df.isnull().mean() * 100

for col in df.columns[df.isnull().any()]:
    pct = missing_pct[col]

    if pd.api.types.is_numeric_dtype(df[col]):
        if pct < 5:
            df = df.dropna(subset=[col])
        elif pct <= 20:
            df[col] = df[col].fillna(df[col].median())          # robust to skew
        else:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            df[[col]] = imputer.fit_transform(df[[col]])
    else:
        # Categorical column.
        # CouponCode's ~26% missingness is not random — it means
        # "no coupon applied". Imputing a mode/most-frequent value
        # here would fabricate promotions that never happened, so we
        # use an explicit sentinel category instead of the generic
        # >20% -> KNN rule (KNN also doesn't apply cleanly to nominal
        # categorical data without pre-encoding).
        df[col] = df[col].fillna('NONE')

print("\nShape after missing-value handling:", df.shape)
print("Remaining nulls:", df.isnull().sum().sum())

# --------------------------------------------------------------
# 1b. Outlier detection via IQR, neutralized by winsorization
#     (numpy.clip), not deletion — preserves row count and any
#     downstream sequential/time integrity.
# --------------------------------------------------------------
numeric_cols = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']

outlier_report = {}
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = {'lower': lower, 'upper': upper, 'n_outliers': n_outliers}
    df[col] = np.clip(df[col], lower, upper)                    # vectorized, no loop

print("\nOutlier bounds & counts (before capping):")
for col, r in outlier_report.items():
    print(f"  {col}: [{r['lower']:.2f}, {r['upper']:.2f}]  -> {r['n_outliers']} capped")

# ==================================================================
# PHASE 2 — VECTORIZED FEATURE ENGINEERING
# ==================================================================

# --- New feature 1: date-derived seasonality signals ---
df['OrderMonth'] = df['Date'].dt.month
df['OrderDayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['OrderDayOfWeek'].isin([5, 6]).astype(int)

# --- New feature 2: price-derived ratios ---
df['PricePerItem'] = df['TotalPrice'] / df['ItemsInCart']
df['CartFillRatio'] = df['Quantity'] / df['ItemsInCart']

# --- New feature 3: customer behavior signal ---
cust_orders = df.groupby('CustomerID').size().rename('CustomerOrderCount')
df = df.merge(cust_orders, on='CustomerID')
df['IsRepeatCustomer'] = (df['CustomerOrderCount'] > 1).astype(int)

# --- New feature 4 (bonus): coupon usage flag ---
df['HasCoupon'] = (df['CouponCode'] != 'NONE').astype(int)

print("\nEngineered feature columns added:",
      ['OrderMonth', 'OrderDayOfWeek', 'IsWeekend', 'PricePerItem',
       'CartFillRatio', 'CustomerOrderCount', 'IsRepeatCustomer', 'HasCoupon'])

# --------------------------------------------------------------
# 2a. Categorical translation into coordinate space
#     One-hot encoding (not label encoding) — avoids implying a
#     false ordinal distance between nominal categories.
#     Identifier-like columns (OrderID, TrackingNumber, ShippingAddress,
#     CustomerID) are dropped: they're unique keys, not predictive
#     signal, and one-hot encoding them would explode dimensionality.
# --------------------------------------------------------------
id_like_cols = ['OrderID', 'TrackingNumber', 'ShippingAddress', 'CustomerID', 'Date']
categorical_cols = ['Product', 'PaymentMethod', 'OrderStatus', 'CouponCode', 'ReferralSource']

df_model = df.drop(columns=id_like_cols)
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)

print("\nShape after encoding:", df_model.shape)

# --------------------------------------------------------------
# 2b. Multicollinearity eradication
#     Build |correlation| matrix on numeric features, isolate the
#     upper triangle, flag pairs with r > 0.80, and drop whichever
#     of the pair correlates more weakly with the target (TotalPrice)
#     — never just the first one encountered.
# --------------------------------------------------------------
TARGET = 'TotalPrice'
feature_cols = [c for c in df_model.select_dtypes(include=[np.number]).columns if c != TARGET]

corr_matrix = df_model[feature_cols + [TARGET]].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = set()
for col in upper.columns:
    for row in upper.index:
        r = upper.loc[row, col]
        if pd.notna(r) and r > 0.80 and row != TARGET and col != TARGET:
            # keep whichever has the stronger relationship with TARGET
            weaker = row if corr_matrix.loc[row, TARGET] < corr_matrix.loc[col, TARGET] else col
            to_drop.add(weaker)

print("\nHighly collinear features dropped (r > 0.80):", to_drop)
df_model = df_model.drop(columns=list(to_drop))

print("\nFinal model-ready shape:", df_model.shape)

# ==================================================================
# PHASE 3 — OUTPUT
# ==================================================================
df.to_csv('ds_csv_cleaned.csv', index=False)          # human-readable, cleaned
df_model.to_csv('ds_csv_model_ready.csv', index=False)  # encoded, decollinearized

print("\nSaved: ds_csv_cleaned.csv, ds_csv_model_ready.csv")
