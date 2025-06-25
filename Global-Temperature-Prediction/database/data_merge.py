import pandas as pd
import os
# load datasets
dir_path = os.path.dirname(os.path.abspath(__file__))
temp = pd.read_csv(dir_path + "/global_temp.csv")
co2 = pd.read_csv(dir_path + "/co2_mlo.csv")

# merge on Year
data = pd.merge(temp, co2, on="Year", how="inner")