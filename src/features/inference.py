import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib

from data_converter import convert_sensor_data
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

def prepare_timeseries(accel_csv: Path, gyro_csv: Path, base_time: datetime) -> pd.DataFrame:
    out_dir = accel_csv.parent / "output"
    convert_sensor_data(
        accel_path=accel_csv,
        gyro_path=gyro_csv,
        output_dir=out_dir,
        base_time=base_time
    )

    acc = pd.read_csv(out_dir / "converted_accel.csv")
    gyr = pd.read_csv(out_dir / "converted_gyro.csv")

    # time‐index
    acc.index = pd.to_datetime(acc["epoch (ms)"], unit="ms")
    gyr.index = pd.to_datetime(gyr["epoch (ms)"], unit="ms")

    # drop duplicate timestamps
    acc = acc.loc[~acc.index.duplicated(keep="first")]
    gyr = gyr.loc[~gyr.index.duplicated(keep="first")]

    # merge & resample to 200 ms
    merged = pd.concat([
        acc[["x-axis (g)", "y-axis (g)", "z-axis (g)"]],
        gyr[["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"]]
    ], axis=1)
    sampling = {
        "x-axis (g)": "mean", "y-axis (g)": "mean", "z-axis (g)": "mean",
        "x-axis (deg/s)": "mean", "y-axis (deg/s)": "mean", "z-axis (deg/s)": "mean"
    }
    merged = merged.resample("200ms").apply(sampling).dropna()
    merged.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    return merged

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1) low-pass filter
    lp = LowPassFilter()
    fs = 1000 / 200
    cutoff = 1.2
    for col in ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]:
        df = lp.low_pass_filter(df, col, fs, cutoff, order=5)

    # 2) PCA (n_components=3)
    pca = PrincipalComponentAnalysis()
    df = pca.apply_pca(
        df,
        ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
        number_comp=3
    )

    # 3) vector magnitudes
    df["acc_r"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["gyr_r"] = np.sqrt(df["gyr_x"]**2 + df["gyr_y"]**2 + df["gyr_z"]**2)

    # 4) temporal abstraction (ws=5)
    na = NumericalAbstraction()
    ws = int(1000/200)
    cols = ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","acc_r","gyr_r"]
    for col in cols:
        df = na.abstract_numerical(df, [col], ws, "mean")
        df = na.abstract_numerical(df, [col], ws, "std")

    # 5) frequency abstraction (ws=10)
    fa = FourierTransformation()
    df_freq = fa.abstract_frequency(df.reset_index(), cols, window_size=10, sampling_rate=fs)
    df_freq = df_freq.set_index("epoch (ms)", drop=True).dropna().iloc[::2]

    # 6) **cluster** exactly as in build_features.ipynb
    kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
    df_freq["cluster"] = kmeans.fit_predict(df_freq[["acc_x","acc_y","acc_z"]])

    return df_freq

def load_pipeline(model_path: Path):
    raw = joblib.load(model_path)
    pipeline = raw.get("pipeline", None)
    if pipeline is None or not hasattr(pipeline, "predict"):
        raise ValueError("Expected a 'pipeline' entry in the model dict")
    return pipeline

def main():
    accel_csv = Path(r"C:\Users\karim\Desktop\Artificial Intelligence\Fitness-Tracker\data\external\acceleration_2025-05-12_22-55-54.csv")
    gyro_csv  = Path(r"C:\Users\karim\Desktop\Artificial Intelligence\Fitness-Tracker\data\external\gyroscope_2025-05-13_00-16-23.csv")
    base_time = datetime(2025, 5, 12, 22, 55, 54)

    # 1) load & align
    ts = prepare_timeseries(accel_csv, gyro_csv, base_time)

    # 2) feature engineering (incl. cluster)
    X_real = build_features(ts)

    # 3) load the sklearn Pipeline
    pipeline = load_pipeline(
        Path(r"C:\Users\karim\Desktop\Artificial Intelligence\Fitness-Tracker\models\best_model.pkl")
    )

    # 4) predict
    preds = pipeline.predict(X_real)
    print("Predictions for each 200 ms window:")
    print(preds)

if __name__ == "__main__":
    main()
