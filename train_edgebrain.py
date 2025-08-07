# train_edgebrain.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1) Generate the same mock‐data you use in the app
def generate_mock_data(n=1000):
    np.random.seed(42)
    odds     = np.random.uniform(2, 10, n).round(2)
    win_val  = np.random.uniform(5, 30, n).round(1)
    place_val= (win_val * 0.6).round(1)
    X = pd.DataFrame({
        "Odds": odds,
        "Win_Value": win_val,
        "Place_Value": place_val
    })
    # 2) Create a synthetic “win” label: win_pct > threshold
    win_pct = 1 / X["Odds"] * 100
    y = (0.6 * win_pct + 0.4 * X["Win_Value"] > 20).astype(int)
    return X, y

# Generate data
X, y = generate_mock_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit a simple classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "edgebrain_model.pkl")
print("Model saved to edgebrain_model.pkl")
