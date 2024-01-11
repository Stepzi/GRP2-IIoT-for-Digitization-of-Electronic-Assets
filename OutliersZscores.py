import pandas as pd
from scipy import stats

SRCdata = "pump_station_data"

df = pd.read_pickle(f"{SRCdata}_filtered.pkl")


# Define a function to remove outliers using z-score
def remove_outliers_zscore(data_frame, column, threshold=2.15):
    z_scores = stats.zscore(data_frame[column])

    # Create a mask to identify outliers based on the threshold
    outlier_mask = (abs(z_scores) < threshold)

    # Filter the data using the mask
    filtered_data = data_frame[outlier_mask]

    return filtered_data


# Remove outliers from the 'Value' column using z-score
df_filtered = remove_outliers_zscore(df, "outflow")

# Save data to new pickle file
df_filtered.to_pickle(f"{SRCdata}_no_outliers.pkl")
