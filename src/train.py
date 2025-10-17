import pandas as pd
import argparse
import os
import joblib
import yaml
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

def main(features, target, model_dir):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    model_params = params["train"]

    # Load data
    X = pd.read_csv(features)
    y = pd.read_csv(target).values.ravel()

    #  Encode categorical columns
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    #  Keep track of feature order
    feature_list = X.columns.tolist()

    #  Train model
    model = RandomForestRegressor(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"]
    )
    model.fit(X, y)
    preds = model.predict(X)

    #  Evaluate model
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    #  Save model + helpers
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(feature_list, os.path.join(model_dir, "model_features.pkl"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))

    # Optional field mapping (can be 1:1 for now)
    field_map = {feat: feat for feat in feature_list}
    joblib.dump(field_map, os.path.join(model_dir, "feature_field_map.pkl"))

    with open("metrics/eval.json", "w") as f:
        json.dump({"mae": mae, "mse": mse, "r2": r2}, f, indent=4)

    print(f"Model and preprocessing artifacts saved in '{model_dir}/'")
    print(f"Features: {len(feature_list)} | MAE: {mae:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()
    main(args.features, args.target, args.model_dir)
