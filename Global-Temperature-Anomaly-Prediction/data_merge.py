import os
import pandas as pd

# NOAA global surface temperature anomaly data from 1850 to 2025
file_path = os.path.dirname(os.path.dirname(__file__)) + "\datasets\global_temp_anomaly.csv" # Adjust path to point correct the CSV file
temp = pd.read_csv(file_path)

# CO2 data from Mauna Loa Observatory (MLO) from 1958 to 2024
file_path = os.path.dirname(os.path.dirname(__file__)) + "\datasets\co2_conc.csv" # Adjust path to point correct the CSV file
co2 = pd.read_csv(file_path)

# merge on Year
data = pd.merge(temp, co2, on="Year", how="inner")