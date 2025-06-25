import pandas as pd
import os

# directory path of data files kept in the same directory as this script
dir_path = os.path.dirname(os.path.abspath(__file__))

# NOAA global surface temperature anomaly data from 1850 to 2025
temp = pd.read_csv(dir_path + "/global_temp.csv")

# CO2 data from Mauna Loa Observatory (MLO) from 1958 to 2024
co2 = pd.read_csv(dir_path + "/co2_mlo.csv")

# merge on Year
data = pd.merge(temp, co2, on="Year", how="inner")