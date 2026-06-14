import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# 1. Load Data
print("Loading data...")
df = pd.read_excel('Online Retail.xlsx')

# 2. Data Cleaning
print("Cleaning data...")
df = df.dropna(subset=['CustomerID'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 3. Time-Based Split (Cutoff Date)
# Latest date in dataset
latest_date = df['InvoiceDate'].max()
# Use last 3 months for future window
cutoff_date = latest_date - pd.DateOffset(months=3)

print(f"Cutoff Date: {cutoff_date}")

past_df = df[df['InvoiceDate'] < cutoff_date]
future_df = df[df['InvoiceDate'] >= cutoff_date]

# 4. Feature Engineering (Past Window)
print("Engineering features...")
snapshot_date = past_df['InvoiceDate'].max() + pd.DateOffset(days=1)

customer_features = past_df.groupby('CustomerID').agg({
    'InvoiceDate': [lambda x: (snapshot_date - x.max()).days,
                    lambda x: (x.max() - x.min()).days],
    'InvoiceNo': 'nunique',
    'Revenue': ['sum', 'mean']
})

customer_features.columns = ['Recency', 'TenureDays', 'Frequency', 'MonetaryValue', 'AOV']
customer_features = customer_features.reset_index()

# 5. Target Variable (Future Window)
future_clv = future_df.groupby('CustomerID')['Revenue'].sum().reset_index()
future_clv.columns = ['CustomerID', 'FutureCLV']

# Merge
model_df = customer_features.merge(future_clv, on='CustomerID', how='left').fillna(0)

# 6. Target Transformation
model_df['FutureCLV_Log'] = np.log1p(model_df['FutureCLV'])

# 7. Model Training
X = model_df[['Recency', 'Frequency', 'MonetaryValue', 'AOV', 'TenureDays']]
y = model_df['FutureCLV_Log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression (Baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr_log = lr.predict(X_test)
y_pred_lr = np.expm1(y_pred_lr_log)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf_log = rf.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf_log)

# Actual y_test for evaluation
y_test_orig = np.expm1(y_test)

# Evaluation
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

mae_lr, rmse_lr, r2_lr = evaluate(y_test_orig, y_pred_lr)
mae_rf, rmse_rf, r2_rf = evaluate(y_test_orig, y_pred_rf)

print(f"LR - MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}, R2: {r2_lr:.2f}")
print(f"RF - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}, R2: {r2_rf:.2f}")

# 8. Visualizations
print("Generating visualizations...")
# Feature Importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.title('Feature Importance (Random Forest)')
plt.savefig('feature_importance.png')
plt.close()

# Segmentation based on Predicted CLV
model_df['PredictedCLV'] = np.expm1(rf.predict(X))
model_df['Segment'] = pd.qcut(model_df['PredictedCLV'], 4, labels=['Low Value', 'Mid Value', 'High Value', 'VIP'])

plt.figure(figsize=(10, 6))
sns.countplot(data=model_df, x='Segment', palette='viridis')
plt.title('Customer Segmentation Distribution')
plt.savefig('segment_distribution.png')
plt.close()

# Save models and data for UI
print("Saving artifacts...")
joblib.dump(rf, 'rf_model.pkl')
model_df.to_csv('processed_customer_data.csv', index=False)

print("Done!")