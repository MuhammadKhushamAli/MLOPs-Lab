import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import yaml

def main(in_csv, out_dir, test_size, random_state):
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    target_col = params["data"]["target"]

    df = pd.read_csv(in_csv)

    # Ensure target exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    # Drop columns that are entirely text/object (non-numeric)
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # Sometimes price (target) might be read as object due to commas — fix that
    if target_col not in numeric_df.columns:
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        numeric_df[target_col] = df[target_col]

    # Split X, y
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "target.csv"), index=False)

    print(f"✅ Saved numeric features and target in {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args.in_csv, args.out_dir, args.test_size, args.random_state)
