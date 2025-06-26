import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Load rainfall anomaly data (assume data with 'Year' and 'rainfall_anomaly' columns)
file_path = os.path.dirname(os.path.dirname(__file__)) + "\\datasets\\rainfall_anomaly.csv" # Adjust path to point to the correct CSV file
df = pd.read_csv(file_path)  # Must have columns: Year, CO2_ppm
df = df.sort_values("Year").dropna()
df.set_index("Year", inplace=True)

# Log transform to stabilize variance if needed (optional)
# df["log_rain"] = np.log(df["rainfall_anomaly"] + 1)

# Step 1: ARIMA to model mean
arima_model = ARIMA(df["rainfall_anomaly"], order=(1, 0, 1))
arima_result = arima_model.fit()
residuals = arima_result.resid

# Step 2: GARCH to model volatility of residuals
garch_model = arch_model(residuals, vol="Garch", p=1, q=1)
garch_result = garch_model.fit(disp="off")

# Forecast next 10 years
timesteps = 10
arima_forecast = arima_result.get_forecast(steps=timesteps)
arima_mean = arima_forecast.predicted_mean

# GARCH volatility forecast (sigma)
garch_forecast = garch_result.forecast(horizon=timesteps)
sigma_forecast = garch_forecast.variance.values[-1] ** 0.5

# Confidence bounds
upper = arima_mean + 1.96 * sigma_forecast
lower = arima_mean - 1.96 * sigma_forecast

# Forecast year range
forecast_years = np.arange(df.index[-1] + 1, df.index[-1] + timesteps + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["rainfall_anomaly"], label="Historical")
plt.plot(forecast_years, arima_mean, label="Forecast", linestyle="--", marker='o')
plt.fill_between(forecast_years, lower, upper, color="gray", alpha=0.3, label="95% CI")
plt.title("Indian Rainfall Anomaly Forecast (ARIMA + GARCH)")
plt.xlabel("Year")
plt.ylabel("Rainfall Anomaly (mm)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=True)
