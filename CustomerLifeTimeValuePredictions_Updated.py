import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# =====================================================
# 1. LOAD DATA
# =====================================================

print("Loading data...")

df = pd.read_excel("Online Retail.xlsx")

# =====================================================
# 2. DATA CLEANING
# =====================================================

print("Cleaning data...")

# Remove missing customer IDs
df = df.dropna(subset=["CustomerID"])

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid quantities and prices
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Convert dates
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Revenue feature
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

print(f"Rows after cleaning: {len(df):,}")

# =====================================================
# 3. TIME-BASED TRAINING FRAMEWORK
# =====================================================

latest_date = df["InvoiceDate"].max()

# Last 3 months become future window
cutoff_date = latest_date - pd.DateOffset(months=3)

print(f"Latest Date : {latest_date}")
print(f"Cutoff Date : {cutoff_date}")

past_df = df[df["InvoiceDate"] < cutoff_date]
future_df = df[df["InvoiceDate"] >= cutoff_date]

# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

print("Engineering customer features...")

snapshot_date = past_df["InvoiceDate"].max() + pd.DateOffset(days=1)

customer_features = past_df.groupby("CustomerID").agg(
    Recency=("InvoiceDate",
             lambda x: (snapshot_date - x.max()).days),

    TenureDays=("InvoiceDate",
                lambda x: (x.max() - x.min()).days),

    Frequency=("InvoiceNo", "nunique"),

    MonetaryValue=("Revenue", "sum"),

    AOV=("Revenue", "mean")
).reset_index()

print(customer_features.head())

# =====================================================
# 5. FUTURE CLV TARGET
# =====================================================

print("Creating target variable...")

future_clv = (
    future_df
    .groupby("CustomerID")["Revenue"]
    .sum()
    .reset_index()
)

future_clv.columns = [
    "CustomerID",
    "FutureCLV"
]

model_df = customer_features.merge(
    future_clv,
    on="CustomerID",
    how="left"
)

model_df["FutureCLV"] = model_df["FutureCLV"].fillna(0)

# =====================================================
# 6. RETENTION ANALYSIS
# =====================================================

retention_rate = (
    (model_df["FutureCLV"] > 0)
    .mean()
    * 100
)

print(f"\nRetention Rate: {retention_rate:.2f}%")

# =====================================================
# 7. TARGET TRANSFORMATION
# =====================================================

model_df["FutureCLV_Log"] = np.log1p(
    model_df["FutureCLV"]
)

# =====================================================
# 8. MODEL DATA
# =====================================================

X = model_df[
    [
        "Recency",
        "Frequency",
        "MonetaryValue",
        "AOV",
        "TenureDays"
    ]
]

y = model_df["FutureCLV_Log"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# 9. LINEAR REGRESSION
# =====================================================

print("Training Linear Regression...")

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_preds_log = lr.predict(X_test)

lr_preds = np.expm1(lr_preds_log)

# =====================================================
# 10. RANDOM FOREST
# =====================================================

print("Training Random Forest...")

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_preds_log = rf.predict(X_test)

rf_preds = np.expm1(rf_preds_log)

# =====================================================
# 11. EVALUATION
# =====================================================

y_test_original = np.expm1(y_test)

def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


lr_mae, lr_rmse, lr_r2 = evaluate_model(
    y_test_original,
    lr_preds
)

rf_mae, rf_rmse, rf_r2 = evaluate_model(
    y_test_original,
    rf_preds
)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        lr_mae,
        rf_mae
    ],
    "RMSE": [
        lr_rmse,
        rf_rmse
    ],
    "R2": [
        lr_r2,
        rf_r2
    ]
})

print("\nModel Results")
print(results)

results.to_csv(
    "model_results.csv",
    index=False
)

# =====================================================
# 12. FEATURE IMPORTANCE
# =====================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance")
print(feature_importance)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

# Plot
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance.png"
)

plt.close()

# =====================================================
# 13. CUSTOMER SEGMENTATION
# =====================================================

print("Creating customer segments...")

model_df["PredictedCLV"] = np.expm1(
    rf.predict(X)
)

model_df["Segment"] = pd.qcut(
    model_df["PredictedCLV"],
    q=4,
    labels=[
        "Low",
        "Mid",
        "High",
        "VIP"
    ]
)

segment_summary = (
    model_df
    .groupby("Segment")
    .agg({
        "PredictedCLV": "mean",
        "Frequency": "mean",
        "MonetaryValue": "mean",
        "Recency": "mean"
    })
)

print("\nSegment Summary")
print(segment_summary)

segment_summary.to_csv(
    "segment_summary.csv"
)

# Segment Distribution Plot
plt.figure(figsize=(8, 5))

sns.countplot(
    data=model_df,
    x="Segment"
)

plt.title(
    "Customer Segment Distribution"
)

plt.tight_layout()

plt.savefig(
    "segment_distribution.png"
)

plt.close()

# =====================================================
# 14. SAVE OUTPUTS
# =====================================================

model_df.to_csv(
    "processed_customer_data.csv",
    index=False
)

joblib.dump(
    rf,
    "rf_model.pkl"
)

print("\nArtifacts Saved:")
print("- rf_model.pkl")
print("- processed_customer_data.csv")
print("- model_results.csv")
print("- feature_importance.csv")
print("- segment_summary.csv")
print("- feature_importance.png")
print("- segment_distribution.png")

print("\nProject completed successfully!")