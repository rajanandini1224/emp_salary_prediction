# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("salary_data.csv")

# ✅ Print value counts to check class distribution
print("Class distribution:\n", df["income"].value_counts())

# Select relevant columns
features = ["age", "education", "occupation", "gender", "hours-per-week"]
target = "income"
df = df[features + [target]]

# Encode categorical features
encoders = {}
for col in ["education", "occupation", "gender", "income"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split data
X = df.drop("income", axis=1)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "income_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Model trained and saved successfully.")
