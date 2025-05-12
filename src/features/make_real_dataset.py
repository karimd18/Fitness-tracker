import pandas as pd
from pathlib import Path
from datetime import datetime

def create_real_pickle(
    accel_csv: Path,
    gyro_csv: Path,
    pickle_path: Path
):
    """
    Read your already-converted CSVs, merge & resample like make_dataset.ipynb,
    and dump out a real_data_processed.pkl compatible with your model.
    """

    # 1. Read the converted CSVs
    acc = pd.read_csv(accel_csv)
    gyr = pd.read_csv(gyro_csv)

    # 2. Time-index by epoch (ms)
    acc.index = pd.to_datetime(acc["epoch (ms)"], unit="ms")
    gyr.index = pd.to_datetime(gyr["epoch (ms)"], unit="ms")

    # 3. Add dummy metadata columns to match training data format
    for df in (acc, gyr):
        df["participant"] = "real"
        df["label"]       = "real"
        df["category"]    = "real"
        df["set"]         = 1

    # 4. Merge accelerometer + gyroscope side by side
    merged = pd.concat([
        acc[["x-axis (g)", "y-axis (g)", "z-axis (g)", "participant", "label", "category", "set"]],
        gyr[["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"]]
    ], axis=1)

    # 5. Rename columns exactly as make_dataset did
    merged.columns = [
        "acc_x", "acc_y", "acc_z",
        "participant", "label", "category", "set",
        "gyr_x", "gyr_y", "gyr_z"
    ]

    # 6. Resample at 200 ms with identical aggregation rules
    sampling = {
        "acc_x": "mean", "acc_y": "mean", "acc_z": "mean",
        "gyr_x": "mean", "gyr_y": "mean", "gyr_z": "mean",
        "participant": "last", "label": "last", "category": "last", "set": "last"
    }
    processed = (
        merged
        .resample("200ms")
        .apply(sampling)
        .dropna()
    )
    processed["set"] = processed["set"].astype(int)

    # 7. Export to pickle
    processed.to_pickle(pickle_path)
    print(f"✔ Real data processed and pickled to: {pickle_path}")

if __name__ == "__main__":
    # Paths to your *already* converted CSVs:
    accel_csv = Path(r"C:\Users\karim\Desktop\Artificial Intelligence\Fitness-Tracker\data\external\acceleration_2025-05-12_22-55-54.csv")
    gyro_csv  = Path(r"C:\Users\karim\Desktop\Artificial Intelligence\Fitness-Tracker\data\external\gyroscope_2025-05-13_00-16-23.csv")
    pickle_p  = accel_csv.parent / "output\\real_data_processed.pkl"

    create_real_pickle(
        accel_csv=accel_csv,
        gyro_csv=gyro_csv,
        pickle_path=pickle_p
    )
