import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

SRCdata = "pump_station_data"

df = pd.read_pickle(f"{SRCdata}_filtered.pkl")
df = df[["height", "outflow"]].loc["2023-02-01":"2023-02-02"]


# Define a function to remove outliers using z-score
def remove_outliers_zscore(data_frame, column, threshold=1.5):
    z_scores = stats.zscore(data_frame[column])

    # Create a mask to identify outliers based on the threshold
    mask = (abs(z_scores) < threshold)

    # Filter the data using the mask
    filtered_data = data_frame[mask]

    return filtered_data


# # Define a function to remove outliers using z-score for each column
# def remove_outliers_zscore(data_frame, threshold=1.9):
#     # Initialize an empty DataFrame to store the filtered data
#     filtered_data = pd.DataFrame()
#
#     # Iterate over each column in the original DataFrame
#     for column in data_frame.columns:
#         # Calculate z-scores for the current column
#         z_scores = stats.zscore(data_frame[column])
#
#         # Create a mask to identify outliers based on the threshold
#         mask = (abs(z_scores) < threshold)
#
#         # Filter the data using the mask and store it in the filtered DataFrame
#         filtered_data[column] = data_frame[column][mask]
#
#     return filtered_data
#
#
# # Remove outliers from each column using z-score
# df_filtered = remove_outliers_zscore(df)

# Remove outliers from the 'Value' column using z-score
df_filtered = remove_outliers_zscore(df, "outflow")

# # Save data to new pickle file
# df_filtered.to_pickle(f"{SRCdata}_no_outliers.pkl")

t = 1
area = 18
initial_Qin = df.loc[df.index[0], "outflow"]
initial_Qout = df.loc[df.index[0], "outflow"]
initial_h = df.loc[df.index[0], "height"]

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
