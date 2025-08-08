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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# 1. Load your historical data
hist = pd.read_csv("historical.csv")

# 2. Binarize the target: 1 if Win, else 0
hist['Outcome_bin'] = (hist['Outcome'].str.lower() == 'win').astype(int)

# 3. Select features you already have in the app
features = ['Odds', 'Win_Value', 'Place_Value']
X = hist[features]
y = hist['Outcome_bin']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Fit logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate on hold-out set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save the trained model to disk
with open("edgebrain_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✔️  edgebrain_model.pkl saved")
