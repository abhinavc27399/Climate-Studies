import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
file_path = os.path.dirname(os.path.dirname(__file__)) + "\\datasets\\forestfires.csv" # Adjust path to point to the correct CSV file
df = pd.read_csv(file_path)

# Encode month and day as integers
le_month = LabelEncoder()
le_day = LabelEncoder()
df["month"] = le_month.fit_transform(df["month"])
df["day"] = le_day.fit_transform(df["day"])

# Create fire risk categories from area burned
# low: < 0.1 ha, medium: 0.1–10 ha, high: >10 ha
df["fire_risk"] = pd.cut(df["area"], bins=[-0.1, 0.1, 10, 1000], labels=["low", "medium", "high"])

# Drop rows with NaN fire risk
df = df.dropna(subset=["fire_risk"])

# Encode fire risk as target labels
le_risk = LabelEncoder()
df["fire_risk_encoded"] = le_risk.fit_transform(df["fire_risk"])

# Features and labels
features = ["temp", "RH", "wind", "rain", "month", "day"]
X = df[features]
y = df["fire_risk_encoded"]

# Normalize features
X = StandardScaler().fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))

# Visualize fire risk by month
heat_data = pd.crosstab(index=le_month.inverse_transform(df["month"]), columns=df["fire_risk"], normalize='index')
plt.figure(figsize=(8, 5))
sns.heatmap(heat_data, annot=True, cmap="YlOrRd")
plt.title("Fire Risk Distribution by Month")
plt.xlabel("Fire Risk Level")
plt.ylabel("Month")
plt.tight_layout()
plt.show(block=True)
