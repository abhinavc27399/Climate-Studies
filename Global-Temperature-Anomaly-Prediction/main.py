import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from data_merge import data

# Preprocessing of merged data from data_merge.py 
features = data[["Year", "CO2_ppm"]]
target = data["Temperature_Anomaly_C"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test["Year"], y_test, color="blue", label="Actual")
plt.scatter(X_test["Year"], y_pred, color="red", label="Predicted")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Predicted vs Actual Temperature Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)
