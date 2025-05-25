from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Corrected spelling
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and preprocess data once
def preprocess_data():
    global solar_df  # Make solar_df a global variable
    solar_df = pd.read_csv("final_solar.csv")  # Load the dataset
    wind_df = pd.read_csv("final_wind.csv")

    solar_df.rename(columns={"DATE_TIME": "timestamp"}, inplace=True)
    wind_df.rename(columns={"Date/Time": "timestamp"}, inplace=True)

    solar_df["timestamp"] = pd.to_datetime(solar_df["timestamp"], format="%d-%m-%Y %H:%M")
    wind_df["timestamp"] = pd.to_datetime(wind_df["timestamp"], format="%d %m %Y %H:%M")

    merged_df = pd.merge_asof(solar_df.sort_values("timestamp"), wind_df.sort_values("timestamp"),
                              on="timestamp", direction="nearest")

    merged_df["hour"] = merged_df["timestamp"].dt.hour
    merged_df["day_of_week"] = merged_df["timestamp"].dt.dayofweek
    merged_df["month"] = merged_df["timestamp"].dt.month

    features = ["DAILY_YIELD", "TOTAL_YIELD", "hour", "day_of_week", "month"]
    target = "AC_POWER"

    merged_df = merged_df[(merged_df[target] > 10) & (merged_df[target] < 1400)]
    merged_df[target] = np.log1p(merged_df[target])

    X = merged_df[features]
    y = merged_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

# Load data and train models once
X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = preprocess_data()

# Check if models are already saved
if not all(os.path.exists(f"{model_name}.pkl") for model_name in ["rf_model", "xgb_model", "gb_model", "lgbm_model", "ridge_model"]):
    # Train models
    rf_model = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=5, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=1, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, min_child_samples=10, verbose=-1, random_state=42)
    lgbm_model.fit(X_train_scaled, y_train)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    # Save models to disk
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(gb_model, "gb_model.pkl")
    joblib.dump(lgbm_model, "lgbm_model.pkl")
    joblib.dump(ridge_model, "ridge_model.pkl")
else:
    # Load models from disk
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    gb_model = joblib.load("gb_model.pkl")
    lgbm_model = joblib.load("lgbm_model.pkl")
    ridge_model = joblib.load("ridge_model.pkl")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Energy Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or any(feature not in data for feature in features):
            return jsonify({"error": "Missing or invalid input data"}), 400

        input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        # Make predictions
        rf_pred = float(np.expm1(rf_model.predict(input_scaled))[0])
        xgb_pred = float(np.expm1(xgb_model.predict(input_scaled))[0])
        gb_pred = float(np.expm1(gb_model.predict(input_scaled))[0])
        lgbm_pred = float(np.expm1(lgbm_model.predict(input_scaled))[0])
        ridge_pred = float(np.expm1(ridge_model.predict(input_scaled))[0])

        # Mock actual value (e.g., mean of the test dataset)
        actual_value = float(np.expm1(y_test.mean()))

        # Calculate errors
        predictions = {
            "Random Forest": rf_pred,
            "XGBoost": xgb_pred,
            "Gradient Boosting": gb_pred,
            "LightGBM": lgbm_pred,
            "Ridge Regression": ridge_pred
        }
        errors = {model: abs(pred - actual_value) for model, pred in predictions.items()}

        # Select the best model (lowest error)
        best_model = min(errors, key=errors.get)
        best_prediction = predictions[best_model]

        # Add best model to the response
        response = {
            "predictions": predictions,
            "best_model": best_model,
            "best_prediction": best_prediction  # Corrected spelling
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/get_sample_plants', methods=['GET'])
def get_sample_plants():
    try:
        df = pd.read_csv('final_wind.csv')  # replace with actual CSV path
        sample = df.sample(n=5).to_dict(orient='records')
        return jsonify(sample)
    except Exception as e:
        return jsonify({'error': str(e)})


# Power Plant Details route
@app.route("/power_plant_details", methods=["POST"])
def power_plant_details():
    try:
        data = request.json
        if not data or "power_plant_id" not in data:
            return jsonify({"error": "Missing or invalid input data"}), 400

        power_plant_id = str(data.get("power_plant_id")).strip()

        # Make sure PLANT_ID is treated as string
        solar_df["PLANT_ID"] = solar_df["PLANT_ID"].astype(str).str.strip()

        # Filter for plant
        plant_details = solar_df[solar_df["PLANT_ID"] == power_plant_id]

        if plant_details.empty:
            return jsonify({
                "error": f"Power Plant ID {power_plant_id} not found.",
                "power_plant_id": power_plant_id,
                "location": "Unknown",
                "energy_type": "Unknown",
                "predicted_generation": 0.00
            })

        plant_details = plant_details.iloc[0]

        input_data = np.array([[plant_details["DAILY_YIELD"], plant_details["TOTAL_YIELD"], 12, 3, 5]])
        input_scaled = scaler.transform(input_data)
        predicted_generation = float(np.expm1(ridge_model.predict(input_scaled))[0])

        response = {
            "power_plant_id": power_plant_id,
            "location": plant_details["Location"],
            "energy_type": "Solar",
            "predicted_generation": predicted_generation
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=False)  # Disable debug mode in production