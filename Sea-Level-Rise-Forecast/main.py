import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.ticker as mticker

# Load sample sea level data
# CSV should contain 'Year' and 'GMSL' (Global Mean Sea Level in mm)
file_path = os.path.dirname(os.path.dirname(__file__)) + "\datasets\sea_level_data.csv" # Adjust path to point to the correct CSV file
data = pd.read_csv(file_path)
data = data.sort_values("Year")

# Normalize GMSL for better training stability
scaler = MinMaxScaler()
data["GMSL_scaled"] = scaler.fit_transform(data[["GMSL"]])

# Create sequences for LSTM (window of 5 years to predict next year)
def create_sequences(values, window=5):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(data["GMSL_scaled"].values)
X = X.reshape((X.shape[0], X.shape[1], 1))  # shape: (N, window, 1)

# Split into train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

# Predict & evaluate
pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
years = data["Year"].values[-len(y_test):]

# --- Project forward from last 5 steps --- #
future_years = np.arange(2024, 2036)
window = 5
last_seq = data["GMSL_scaled"].values[-window:]
future_preds = []

for _ in future_years:
    input_seq = last_seq.reshape((1, window, 1))
    next_scaled = model.predict(input_seq, verbose=0)[0, 0]
    future_preds.append(next_scaled)
    last_seq = np.append(last_seq[1:], next_scaled)

future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(years, y_true, label="Actual GMSL")
plt.plot(years, pred, label="Predicted GMSL")
plt.plot(future_years, future_preds_inv, label="Forecast 2024–2035", linestyle='--', marker='o')
plt.title("Sea Level Forecast (Keras LSTM with Projection)")
plt.xlabel("Year")
plt.ylabel("Global Mean Sea Level (mm)")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()