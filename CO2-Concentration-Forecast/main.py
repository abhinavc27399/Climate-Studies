import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
import matplotlib.ticker as mticker


# Load CO2 data
file_path = os.path.dirname(os.path.dirname(__file__)) + "\datasets\co2_conc.csv" # Adjust path to point to the correct CSV file
df = pd.read_csv(file_path)  # Must have columns: Year, CO2_ppm
df = df.sort_values("Year").dropna()
df.set_index("Year", inplace=True)

# Fit ARIMA model
model = ARIMA(df["CO2_ppm"], order=(1, 1, 1))  # (p,d,q) can be tuned
model_fit = model.fit()

# Forecast to 2035
forecast_years = np.arange(df.index[-1] + 1, 2036)
n_steps = len(forecast_years)
forecast = model_fit.get_forecast(steps=n_steps)
pred = forecast.predicted_mean
ci = forecast.conf_int()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["CO2_ppm"], label="Historical CO₂")
plt.plot(forecast_years, pred, label="Forecast", linestyle='--', marker='o')
plt.fill_between(forecast_years, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, label="95% CI")
plt.title("CO₂ Concentration Forecast (ARIMA)")
plt.xlabel("Year")
plt.ylabel("CO₂ ppm")
plt.grid(True)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
plt.legend()
plt.tight_layout()
plt.show(block=True)
