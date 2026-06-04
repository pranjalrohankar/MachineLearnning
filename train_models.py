import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Generate Realistic Synthetic Dataset
np.random.seed(42)  # For reproducibility
num_samples = 10000  # Number of samples in the dataset

# Features:
# 1. syn_rate: Number of SYN packets per second
# 2. src_ip_diversity: Number of unique source IPs
# 3. packet_size: Size of the packet in bytes
# 4. syn_ack_ratio: Ratio of SYN-ACK packets to SYN packets
X = np.zeros((num_samples, 4))  # Initialize feature matrix
y = np.zeros(num_samples)  # Initialize labels

# Generate normal traffic samples
num_normal = num_samples // 2
X[:num_normal, 0] = np.random.randint(0, 100, num_normal)  # syn_rate: 0–100
X[:num_normal, 1] = np.random.randint(10, 50, num_normal)  # src_ip_diversity: 10–50
X[:num_normal, 2] = np.random.randint(40, 1500, num_normal)  # packet_size: 40–1500
X[:num_normal, 3] = np.random.uniform(0.8, 1.0, num_normal)  # syn_ack_ratio: 0.8–1.0
y[:num_normal] = 0  # Label for normal traffic

# Generate SYN flood traffic samples
num_flood = num_samples - num_normal
X[num_normal:, 0] = np.random.randint(500, 1000, num_flood)  # syn_rate: 500–1000
X[num_normal:, 1] = np.random.randint(50, 100, num_flood)  # src_ip_diversity: 50–100
X[num_normal:, 2] = np.random.randint(40, 100, num_flood)  # packet_size: 40–100
X[num_normal:, 3] = np.random.uniform(0.0, 0.2, num_flood)  # syn_ack_ratio: 0.0–0.2
y[num_normal:] = 1  # Label for SYN flood traffic

# Step 2: Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the test data

# Step 4: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the Model
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Save the Model and Scaler
joblib.dump(model, "syn_flood_model.pkl")  # Save the trained model
joblib.dump(scaler, "syn_flood_scaler.pkl")  # Save the scaler

print("Model and scaler saved as 'syn_flood_model.pkl' and 'syn_flood_scaler.pkl'.")