import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def convert_sensor_data(
    accel_path: Path,
    gyro_path: Path,
    output_dir: Path,
    base_time: datetime = None
):
    """
    Convert raw sensor data to model-ready format with accurate timestamps
    :param accel_path: Path to raw accelerometer CSV
    :param gyro_path: Path to raw gyroscope CSV
    :param output_dir: Directory to save converted files
    :param base_time: Optional base timestamp (default: current time)
    """
    # Set base time for synchronization
    base_time = base_time or datetime.now()
    
    # Process accelerometer data
    accel_df = process_file(
        accel_path,
        sensor_type='accel',
        base_time=base_time,
        value_cols=['ax', 'ay', 'az'],
        output_cols=['x-axis (g)', 'y-axis (g)', 'z-axis (g)']
    )
    
    # Process gyroscope data
    gyro_df = process_file(
        gyro_path,
        sensor_type='gyro',
        base_time=base_time,
        value_cols=['wx', 'wy', 'wz'],
        output_cols=['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']
    )
    
    # Save converted files
    accel_output = output_dir / "converted_accel.csv"
    gyro_output = output_dir / "converted_gyro.csv"
    
    accel_df.to_csv(accel_output, index=False)
    gyro_df.to_csv(gyro_output, index=False)
    
    print(f"Converted files saved to:\n- {accel_output}\n- {gyro_output}")

def process_file(
    input_path: Path,
    sensor_type: str,
    base_time: datetime,
    value_cols: list,
    output_cols: list
) -> pd.DataFrame:
    """Process individual sensor file with accurate timestamp calculations"""
    df = pd.read_csv(input_path)
    
    # Convert elapsed seconds to datetime
    df['datetime'] = df['time'].apply(
        lambda x: base_time + timedelta(seconds=x)
    )
    
    # Generate required columns
    df['epoch (ms)'] = df['datetime'].apply(
        lambda x: int(x.timestamp() * 1000)
    )
    df['time (01:00)'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    df['elapsed (s)'] = df['time']
    
    # Rename value columns
    df = df.rename(columns=dict(zip(value_cols, output_cols)))
    
    # Select and order columns
    keep_cols = ['epoch (ms)', 'time (01:00)', 'elapsed (s)'] + output_cols
    return df[keep_cols].sort_values('epoch (ms)')

def verify_timestamps(df: pd.DataFrame, sensor_type: str):
    """Validate timestamp consistency and ordering"""
    time_diff = df['epoch (ms)'].diff().dropna()
    
    if not df['epoch (ms)'].is_monotonic_increasing:
        raise ValueError(f"{sensor_type} timestamps are not monotonically increasing")
    
    avg_sample_rate = time_diff.mean()
    print(f"{sensor_type} average sample rate: {avg_sample_rate:.2f} ms")

if __name__ == "__main__":
    base_time = datetime(2025, 5, 12, 22, 55, 54)    
    output_dir = Path("C:\\Users\\karim\\Desktop\\Artificial Intelligence\\Fitness-Tracker\\data\\external\\output")
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_sensor_data(
        accel_path=Path("C:\\Users\\karim\\Desktop\\Artificial Intelligence\\Fitness-Tracker\\data\\external\\acceleration_2025-05-12_22-55-54.csv"),
        gyro_path=Path("C:\\Users\\karim\\Desktop\\Artificial Intelligence\\Fitness-Tracker\\data\\external\\gyroscope_2025-05-13_00-16-23.csv"),
        output_dir=output_dir,
        base_time=base_time
    )
