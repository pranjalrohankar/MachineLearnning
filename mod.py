import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('C:/Users/Del/Desktop/cpp2/dataset_phishing.csv')

# Assuming 'status' is the target variable and the rest are features
X = df.drop(['status', 'url'], axis=1)  # Drop the 'url' column
y = df['status']

# Convert categorical target variable to numerical
y = y.map({'legitimate': 0, 'phishing': 1})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'phishing_model.pkl')