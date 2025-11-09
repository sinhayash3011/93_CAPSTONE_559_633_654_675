# robust_sampling_rate.py
import pandas as pd
import numpy as np
import sys
import os
from pandas.api import types as pdtypes

csv_path = "IMU.csv"   # <- replace with your CSV path if different

if not os.path.exists(csv_path):
    print(f"ERROR: file not found: {csv_path}")
    sys.exit(1)

# Read CSV (do not force parse_dates yet)
df = pd.read_csv(csv_path)

if 'time' not in df.columns:
    print("ERROR: 'time' column not found in CSV. Available columns:", df.columns.tolist())
    sys.exit(1)

# Try to parse robustly
# 1) If already datetime dtype, keep it. If string/object, parse.
try:
    if not pdtypes.is_datetime64_any_dtype(df['time']):
        # try parsing with various options
        df['time'] = pd.to_datetime(df['time'], errors='coerce', infer_datetime_format=True, utc=False)
except Exception as e:
    # fallback: force UTC parse then remove tz
    df['time'] = pd.to_datetime(df['time'], errors='coerce', infer_datetime_format=True, utc=True)

# If dtype is tz-aware, convert to UTC then make naive (drop tz)
if pdtypes.is_datetime64tz_dtype(df['time']):
    # convert to UTC then remove tz info
    df['time'] = df['time'].dt.tz_convert('UTC').dt.tz_localize(None)

# If still not datetime, try one more time forcing utc parsing
if not pdtypes.is_datetime64_any_dtype(df['time']):
    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.tz_localize(None)

# Drop rows where time could not be parsed
n_before = len(df)
df = df.dropna(subset=['time']).reset_index(drop=True)
n_after = len(df)
if n_after < n_before:
    print(f"Warning: dropped {n_before - n_after} rows because 'time' could not be parsed.")

# If there are fewer than 2 timestamps, cannot compute delta
if len(df) < 2:
    print("Not enough valid timestamps to compute sampling rate.")
    sys.exit(1)

# Compute time differences in seconds
deltas = df['time'].diff().dt.total_seconds().dropna()

# Filter out zero or negative deltas (if any)
deltas = deltas[deltas > 0]

if len(deltas) == 0:
    print("No positive time deltas found. Check timestamps for duplicates or ordering.")
    sys.exit(1)

median_dt = deltas.median()
mean_dt = deltas.mean()
std_dt = deltas.std()
min_dt = deltas.min()
max_dt = deltas.max()

print("Sampling interval stats (seconds):")
print(f"  count = {len(deltas)}")
print(f"  median = {median_dt:.6f} s")
print(f"  mean   = {mean_dt:.6f} s")
print(f"  std    = {std_dt:.6f} s")
print(f"  min    = {min_dt:.6f} s")
print(f"  max    = {max_dt:.6f} s")
print()
print("Estimated sampling frequency (Hz):")
print(f"  median-based estimate = {1.0/median_dt:.3f} Hz")
print(f"  mean-based estimate   = {1.0/mean_dt:.3f} Hz")
print()

# Show distribution summary
print("Delta quantiles (s):")
print(deltas.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

# Optionally save deltas to CSV for inspection
out_deltas = "time_deltas_seconds.csv"
deltas.to_frame(name='delta_s').to_csv(out_deltas, index=False)
print(f"\nPer-sample deltas saved to: {out_deltas}")
