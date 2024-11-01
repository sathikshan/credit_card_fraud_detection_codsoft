# credit_card_fraud_detection_codsoft
# Import necessary libraries
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from flask import Flask, request, jsonify

# Load data with error handling
try:
    data = pd.read_csv("fraudTest.csv", on_bad_lines='skip')  # Ignores problematic rows
    print("Data loaded successfully, with problematic rows skipped.")
except Exception as e:
    print(f"Error loading data: {e}")

# Print basic info and check for skipped rows
print(f"Data shape: {data.shape}")
print(f"Data columns: {data.columns}")

# Drop unnecessary columns
data = data.drop(columns=["first", "last", "street", "dob", "trans_num", "unix_time"])

# Convert latitude and longitude columns to numeric
lat_columns = ['lat', 'merch_lat', 'long', 'merch_long']
for col in lat_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, coercing errors to NaN

# Drop rows with NaN values after conversion
data = data.dropna()

# Feature engineering: calculate distance between transaction and merchant locations
def calculate_distance(row):
    return np.sqrt((row['lat'] - row['merch_lat'])**2 + (row['long'] - row['merch_long'])**2)

data['distance'] = data.apply(calculate_distance, axis=1)

# Convert datetime and create time-based features
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'], errors='coerce')  # Coerce any parsing errors
data['hour'] = data['trans_date_trans_time'].dt.hour
data = data.drop(columns=['trans_date_trans_time', 'cc_num', 'merchant'])

# Drop rows with any NaN values (in case some rows were corrupted)
data = data.dropna()

# Separate features and target
X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

# Preprocess categorical and numerical data
numeric_features = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "distance", "hour"]
categorical_features = ["category", "gender", "city", "state", "job"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Define models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate each model
best_model = None
best_score = 0
for model_name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"Model: {model_name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    score = pipeline.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = pipeline

# Save the best model
joblib.dump(best_model, "fraud_detection_model.pkl")
print("Best model saved as 'fraud_detection_model.pkl'")

# Set up Flask API for real-time predictions
app = Flask(__name__)

# Load the pre-trained model
best_model = joblib.load("fraud_detection_model.pkl")

# Track fraud transactions by user (cc_num or other identifier if available)
fraud_users = set()  # A simple set to keep track of users with fraudulent transactions

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    
    # Predict whether the transaction is fraudulent
    transaction_prediction = best_model.predict(input_data)
    
    # Determine if the user has had a previous fraudulent transaction
    user_id = data.get("cc_num", None)  # Assume 'cc_num' is available in the request data
    if transaction_prediction[0] == 1:
        fraud_users.add(user_id)
        response = {
            "is_fraud": True,
            "fraud_type": "Fraudulent transaction detected",
            "user_status": "User flagged for multiple frauds" if user_id in fraud_users else "User involved in a fraud"
        }
    else:
        response = {
            "is_fraud": False,
            "fraud_type": "Legitimate transaction",
            "user_status": "User not flagged for fraud"
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
