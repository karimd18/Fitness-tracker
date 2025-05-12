# test_model.py

import pandas as pd
import joblib
from pathlib import Path

def main():
    # 1. Load the real-data pickle you just created
    real_pkl = Path(r"C:\Users\karim\Desktop\Artificial Intelligence\Fitness-Tracker\data\external\output\real_data_processed.pkl")
    df_real = pd.read_pickle(real_pkl)
    print("▶ Head of real data:")
    print(df_real.head(), "\n")

    # 2. Load your trained model (update the path if needed)
    model = joblib.load(r"C:\Users\karim\Desktop\Artificial Intelligence\Fitness-Tracker\models\best_model.pkl")

    # 3. Prepare features (drop metadata)
    X_real = df_real.drop(columns=["participant", "label", "category", "set"])

    # 4. Predict
    preds = model.predict(X_real)
    print("▶ Predictions on real data:")
    print(preds)

if __name__ == "__main__":
    main()
