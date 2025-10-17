import pandas as pd
import argparse
import os
import joblib
import yaml
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main(features, target, model_dir):
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    model_params = params["train"]

    # Load data
    X = pd.read_csv(features)
    y = pd.read_csv(target).values.ravel()  # flatten target to 1D

    # Initialize model
    model = RandomForestRegressor(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"]
    )

    # Train model
    model.fit(X, y)
    preds = model.predict(X)

    # Evaluate
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # Save model and metrics
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    with open("metrics/eval.json", "w") as f:
        json.dump({"mae": mae, "mse": mse, "r2": r2}, f, indent=4)

    print("✅ Model trained and saved successfully!")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()

    main(args.features, args.target, args.model_dir)
