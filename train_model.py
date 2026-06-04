import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json

# Load the dataset
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base_dir, "uploads", "aligned_dataset.csv"))

# Print column names for debugging
print("Dataset columns:", data.columns)

# Update categorical features to match the dataset
categorical_features = ['protocol', 'service', 'flag']  # Use correct column names
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Save the encoders for later use
joblib.dump(encoders, "encoders.pkl")

# Separate features and target
X = data.drop(columns=['label'])  # Replace 'label' with your target column
y = data['label']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")

# Save class mapping
class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
with open("class_mapping.json", "w") as f:
    json.dump(class_mapping, f)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Save feature names
with open("feature_names.json", "w") as f:
    json.dump(list(X.columns), f)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a model (e.g., Random Forest)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "malware_model.pkl")

print("✅ Model training and preprocessing completed. Artifacts saved.")