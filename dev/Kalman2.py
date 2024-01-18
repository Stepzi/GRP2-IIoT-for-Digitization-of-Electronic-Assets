import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy

SRCdata = "pump_station_data"

df = pd.read_pickle(f"{SRCdata}_filtered.pkl")
c = df.head(10)
print(c)

df = df[["height", "outflow"]].loc['2023-02-01':'2023-02-02']

t = 1
area = 18
initial_Qin = df.loc[df.index[0], "outflow"]
initial_Qout = df.loc[df.index[0], "outflow"]
initial_h = df.loc[df.index[0], "height"]


def remove_outliers(df, column_name, window_size=150, lower_quantile=0.30, upper_quantile=0.55, threshold=1.5):
    """
    Remove outliers from a DataFrame column based on rolling window IQR method.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to process.
    window_size (int): Size of the rolling window.
    lower_quantile (float): Lower quantile for IQR calculation.
    upper_quantile (float): Upper quantile for IQR calculation.
    threshold (float): Threshold multiplier for IQR.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """

    # Function to detect outliers in a series
    def detect_outliers(series):
        Q1 = series.quantile(lower_quantile)
        Q3 = series.quantile(upper_quantile)
        IQR = Q3 - Q1
        return ~((series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR)))

    # Initialize an empty Series to store the outlier mask
    outlier_mask = pd.Series([False] * len(df), index=df.index)

    # Iterate over the rolling windows
    for start in tqdm(range(len(df))):
        end = min(start + window_size, len(df))
        window = df[column_name][start:end]
        mask = detect_outliers(window)
        outlier_mask.iloc[end-1] = mask.iloc[-1]

    # Filter out the outliers
    return df[outlier_mask].reset_index(drop=True)


# Usage example:
df_filtered = remove_outliers(df, 'outflow', window_size=150)
# Process / Estimation Errors
error_est_Qin = 0.01
error_est_h = 0.01
error_est_Qout = 0.01

error_obs_h = 0.001
error_obs_Qout = 0.001

# Define the state transition matrix (A)
A = np.array([[1, 0, 0], [t/area, 1, -t/area], [0, 0, 1]])

# Create a Kalman Filter instance, where dim_x is the state dimension, dim_z is the measurement dimension
kf = KalmanFilter(dim_x=3, dim_z=2)

# Initialize the state (x) and covariance (P) matrices
kf.x = np.array([initial_Qin, initial_h, initial_Qout])  # Initial state
# kf.P *= 2 # Initial covariance, you can customize this

# Define the state transition (F) and observation (H) matrices
kf.F = A
kf.H = np.array([[0, 1, 0], [0, 0, 1]])  # Assuming you are observing h and Qout
# Store estimates
estimates = []

# Initialize tqdm with the total number of iterations (total rows in df)
progress_bar = tqdm(total=len(df_filtered), desc="Processing")

for index, row in df_filtered.iterrows():
    kf.predict()
    measurement = np.array([row['height'], row['outflow']])
    kf.update(measurement)
    estimates.append(kf.x.copy())
    progress_bar.update(1)  # Update the progress bar

progress_bar.close()  # Close the progress bar when done
estimates = np.array(estimates)
# Time steps for x-axis
time_steps = range(len(estimates))

# Plot for Qin
plt.subplot(2, 2, 1)
plt.plot(time_steps, df_filtered["outflow"], label='Estimated Qin', color="black", linestyle='-')
plt.xlabel('Time Step')
plt.ylabel('Qin')
plt.title('Qin Estimates Over Time')
plt.legend(loc='upper right', fancybox=True, framealpha=0.7)
plt.grid()

# Plot for h
plt.subplot(2, 2, 2)
plt.plot(time_steps, df_filtered["height"], label='Measured h', color="red", linestyle='-')
plt.plot(time_steps, estimates[:, 1], label='Estimated h', color="black", linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('h')
plt.title('Height (h) Estimates Over Time')
plt.legend(loc='upper right', fancybox=True, framealpha=0.7)
plt.grid()

# Plot for Qout
plt.subplot(2, 2, 3)
plt.plot(time_steps, df_filtered["outflow"], label='Measured Qout', color="red", linestyle='-')
plt.plot(time_steps, estimates[:, 2], label='Estimated Qout', color="black", linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Qout')
plt.title('Qout Estimates Over Time')
plt.legend(loc='upper right', fancybox=True, framealpha=0.7)
plt.grid()

plt.show()
