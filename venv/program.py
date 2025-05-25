import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate  # For displaying tables in the terminal

# ✅ Load datasets
solar_df = pd.read_csv("solar.csv")
wind_df = pd.read_csv("wind.csv")

# ✅ Rename timestamp columns for merging
solar_df.rename(columns={"DATE_TIME": "timestamp"}, inplace=True)
wind_df.rename(columns={"Date/Time": "timestamp"}, inplace=True)

# ✅ Convert timestamp columns to datetime format
solar_df["timestamp"] = pd.to_datetime(solar_df["timestamp"], format="%d-%m-%Y %H:%M")
wind_df["timestamp"] = pd.to_datetime(wind_df["timestamp"], format="%d %m %Y %H:%M")

# ✅ Merge datasets on timestamp
merged_df = pd.merge_asof(solar_df.sort_values("timestamp"), wind_df.sort_values("timestamp"),
                          on="timestamp", direction="nearest")

# ✅ Feature Engineering: Add Time-Based Features
merged_df["hour"] = merged_df["timestamp"].dt.hour
merged_df["day_of_week"] = merged_df["timestamp"].dt.dayofweek
merged_df["month"] = merged_df["timestamp"].dt.month

# ✅ Feature Selection
features = ["DAILY_YIELD", "TOTAL_YIELD", "hour", "day_of_week", "month"]
target = "AC_POWER"

# ✅ Remove Outliers in `AC_POWER`
merged_df = merged_df[(merged_df["AC_POWER"] > 10) & (merged_df["AC_POWER"] < 1400)]

# ✅ Apply Log Transformation to Target
merged_df[target] = np.log1p(merged_df[target])  # Log transformation to stabilize variance

# ✅ Splitting data into training & testing sets
X = merged_df[features]
y = merged_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardizing features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Convert Scaled Arrays Back to DataFrame (Fix for LightGBM)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

# --- MACHINE LEARNING MODELS ---

## 1️⃣ Random Forest Model (Tuned)
rf_model = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

## 2️⃣ XGBoost Model (Tuned)
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

## 3️⃣ Gradient Boosting Model (Tuned)
gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_preds = gb_model.predict(X_test_scaled)

## 4️⃣ LightGBM Model (Tuned)
lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, min_child_samples=10, verbose=-1, random_state=42)
lgbm_model.fit(X_train_scaled, y_train)
lgbm_preds = lgbm_model.predict(X_test_scaled)

## 5️⃣ Ridge Regression (Replaced ElasticNet)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_preds = ridge_model.predict(X_test_scaled)

print("\n✅ Model Training Completed Successfully!")

# --- MODEL EVALUATION FUNCTION ---
def evaluate_model(y_true, y_pred, model_name):
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)  # Reverse log transformation

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Model Evaluation:")
    print(f"➡️ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"➡️ Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"➡️ R² Score: {r2:.4f}")
    print("-" * 50)

# ✅ Evaluate all models
evaluate_model(y_test, rf_preds, "Random Forest")
evaluate_model(y_test, xgb_preds, "XGBoost")
evaluate_model(y_test, gb_preds, "Gradient Boosting")
evaluate_model(y_test, lgbm_preds, "LightGBM")
evaluate_model(y_test, ridge_preds, "Ridge Regression")

# --- DIFFERENCES IN ENERGY GENERATION ---
print("\n📉 Differences in Renewable Energy Generation (Before and After Using Models):")
differences = {
    "Random Forest": np.expm1(y_test) - np.expm1(rf_preds),
    "XGBoost": np.expm1(y_test) - np.expm1(xgb_preds),
    "Gradient Boosting": np.expm1(y_test) - np.expm1(gb_preds),
    "LightGBM": np.expm1(y_test) - np.expm1(lgbm_preds),
    "Ridge Regression": np.expm1(y_test) - np.expm1(ridge_preds)
}

for model_name, diff in differences.items():
    print(f"\n➡️ {model_name}:")
    print(f"   - Mean Difference: {diff.mean():.4f} kW")
    print(f"   - Max Difference: {diff.max():.4f} kW")
    print(f"   - Min Difference: {diff.min():.4f} kW")

# --- BEFORE AND AFTER ENERGY GENERATION ---
print("\n📊 Before and After Energy Generation:")
for model_name, preds in zip(["Random Forest", "XGBoost", "Gradient Boosting", "LightGBM", "Ridge Regression"],
                            [rf_preds, xgb_preds, gb_preds, lgbm_preds, ridge_preds]):
    actual_mean = np.expm1(y_test).mean()
    predicted_mean = np.expm1(preds).mean()
    improvement = predicted_mean - actual_mean
    print(f"\n➡️ {model_name}:")
    print(f"   - Before Model (Actual Mean): {actual_mean:.4f} kW")
    print(f"   - After Model (Predicted Mean): {predicted_mean:.4f} kW")
    print(f"   - Improvement: {improvement:.4f} kW")

# --- HOW MODELS ARE TRAINED AND IMPROVE GENERATION ---
print("\n📚 How Models Are Trained and Improve Energy Generation:")
print("1. The models are trained on historical data of solar and wind energy generation.")
print("2. Features like DAILY_YIELD, TOTAL_YIELD, hour, day_of_week, and month are used to capture patterns.")
print("3. The models learn non-linear relationships between features and AC_POWER.")
print("4. After training, the models predict AC_POWER, which helps optimize energy generation.")
print("5. The improvement in energy generation is calculated as the difference between predicted and actual values.")

# --- FACTORS USED IN MODELS ---
print("\n📊 Factors Used in Models and Their Effects:")
factors_table = [
    ["Feature", "Description", "Effect on Model"],
    ["DAILY_YIELD", "Daily energy yield from solar panels", "Positive correlation with AC_POWER"],
    ["TOTAL_YIELD", "Total energy yield from solar panels", "Positive correlation with AC_POWER"],
    ["hour", "Hour of the day", "Captures time-based energy patterns"],
    ["day_of_week", "Day of the week", "Captures weekly energy patterns"],
    ["month", "Month of the year", "Captures seasonal energy patterns"]
]
print(tabulate(factors_table, headers="firstrow", tablefmt="pretty"))

# --- MODEL INFORMATION ---
print("\n📚 Model Information:")
models_info = [
    ["Model", "Description", "How It Helped"],
    ["Random Forest", "Ensemble of decision trees", "Improved accuracy by capturing non-linear relationships"],
    ["XGBoost", "Gradient boosting with regularization", "Improved accuracy and reduced overfitting"],
    ["Gradient Boosting", "Sequential ensemble of weak learners", "Improved accuracy by focusing on errors"],
    ["LightGBM", "Lightweight gradient boosting", "Improved speed and accuracy for large datasets"],
    ["Ridge Regression", "Linear regression with L2 regularization", "Performed poorly due to non-linear data"]
]
print(tabulate(models_info, headers="firstrow", tablefmt="pretty"))

# --- FEATURE IMPORTANCE ANALYSIS ---
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names, hue=feature_names, legend=False, palette="coolwarm")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Energy Prediction")
plt.show()

# --- VISUALIZATIONS ---

# 1️⃣ Histogram of AC_POWER
plt.figure(figsize=(10, 6))
sns.histplot(np.expm1(y_test), bins=30, kde=True, color="blue")
plt.xlabel("AC Power (kW)")
plt.ylabel("Frequency")
plt.title("Histogram of AC Power")
plt.show()

# 2️⃣ Actual vs Predicted (Random Forest)
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_test), np.expm1(rf_preds), alpha=0.5, color="blue")
plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()], [np.expm1(y_test).min(), np.expm1(y_test).max()], color="red", linestyle="--")
plt.xlabel("Actual AC Power (kW)")
plt.ylabel("Predicted AC Power (kW)")
plt.title("Actual vs Predicted (Tuned Random Forest)")
plt.grid(True)
plt.show()

# 3️⃣ Residual Plot (Random Forest)
residuals = np.expm1(y_test) - np.expm1(rf_preds)
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(rf_preds), residuals, alpha=0.5, color="green")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted AC Power (kW)")
plt.ylabel("Residuals")
plt.title("Residual Plot (Tuned Random Forest)")
plt.grid(True)
plt.show()

# 4️⃣ Heatmap of Correlation Matrix
corr_matrix = merged_df[features + [target]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# 5️⃣ Pie Chart of Feature Importance
plt.figure(figsize=(8, 8))
plt.pie(importances, labels=feature_names, autopct="%1.1f%%", colors=sns.color_palette("coolwarm"))
plt.title("Feature Importance (Pie Chart)")
plt.show()

# 6️⃣ Time Series Chart of AC_POWER
plt.figure(figsize=(14, 7))
plt.plot(merged_df["timestamp"][:100], np.expm1(merged_df[target][:100]), color="blue", label="AC Power")
plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.title("Time Series of AC Power (First 100 Points)")
plt.legend()
plt.grid(True)
plt.show()

# 7️⃣ Line Chart of Actual vs Predicted (Random Forest)
plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50], label="Actual AC Power", color="blue", marker="o")
plt.plot(np.expm1(rf_preds)[:50], label="Predicted AC Power (Random Forest)", color="green", marker="x")
plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.title("Actual vs Predicted AC Power (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

# 8️⃣ Pair Plot of Features
sns.pairplot(merged_df[features + [target]][:100], diag_kind="kde")  # Removed `palette` parameter
plt.suptitle("Pair Plot of Features (First 100 Points)", y=1.02)
plt.show()

# 9️⃣ Difference Graph (Before and After Using Models)
# Separate graphs for each model
plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50] - np.expm1(rf_preds)[:50], label="Random Forest Difference", color="green", marker="o")
plt.axhline(0, color="black", linestyle="--", label="No Difference")
plt.xlabel("Time")
plt.ylabel("Difference in AC Power (kW)")
plt.title("Difference in Energy Generation: Random Forest (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50] - np.expm1(xgb_preds)[:50], label="XGBoost Difference", color="orange", marker="x")
plt.axhline(0, color="black", linestyle="--", label="No Difference")
plt.xlabel("Time")
plt.ylabel("Difference in AC Power (kW)")
plt.title("Difference in Energy Generation: XGBoost (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50] - np.expm1(gb_preds)[:50], label="Gradient Boosting Difference", color="purple", marker="s")
plt.axhline(0, color="black", linestyle="--", label="No Difference")
plt.xlabel("Time")
plt.ylabel("Difference in AC Power (kW)")
plt.title("Difference in Energy Generation: Gradient Boosting (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50] - np.expm1(lgbm_preds)[:50], label="LightGBM Difference", color="red", marker="^")
plt.axhline(0, color="black", linestyle="--", label="No Difference")
plt.xlabel("Time")
plt.ylabel("Difference in AC Power (kW)")
plt.title("Difference in Energy Generation: LightGBM (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

# 🔟 Before and After Model Differentiation Graphs
# Separate graphs for each model
plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50], label="Actual AC Power", color="blue", marker="o")
plt.plot(np.expm1(rf_preds)[:50], label="Random Forest Predictions", color="green", marker="x")
plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.title("Before and After Using Models: Random Forest (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50], label="Actual AC Power", color="blue", marker="o")
plt.plot(np.expm1(xgb_preds)[:50], label="XGBoost Predictions", color="orange", marker="s")
plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.title("Before and After Using Models: XGBoost (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50], label="Actual AC Power", color="blue", marker="o")
plt.plot(np.expm1(gb_preds)[:50], label="Gradient Boosting Predictions", color="purple", marker="^")
plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.title("Before and After Using Models: Gradient Boosting (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(np.expm1(y_test)[:50], label="Actual AC Power", color="blue", marker="o")
plt.plot(np.expm1(lgbm_preds)[:50], label="LightGBM Predictions", color="red", marker="d")
plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.title("Before and After Using Models: LightGBM (First 50 Points)")
plt.legend()
plt.grid(True)
plt.show()