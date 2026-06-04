import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load and preprocess the data
def load_data(filepath):
    data = pd.read_csv(filepath)
    
    # Select relevant features
    features = [
        'protocol', 'ip_version', 'bidirectional_packets', 'bidirectional_bytes',
        'src2dst_packets', 'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes',
        'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
        'bidirectional_max_ps', 'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms',
        'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms'
    ]
    
    # Convert protocol to numerical
    data['protocol'] = data['protocol'].astype(int)
    
    # Convert label to binary (1 for arp_spoofing, 0 for normal)
    data['Label'] = data['Label'].apply(lambda x: 1 if x == 'arp_spoofing' else 0)
    
    X = data[features]
    y = data['Label']
    
    return X, y

# 2. Train and evaluate model
def train_model(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

# 3. Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

# 4. Main execution
if __name__ == "__main__":
    import os
    # Load data - UPDATE THIS PATH
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "CIC_MITM_ArpSpoofing_All_Labelled.csv")
    X, y = load_data(filepath)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, X_test, y_train, y_test)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save model
    joblib.dump(model, "arp_spoofing_model.pkl")
    print("\nModel saved as 'arp_spoofing_model.pkl'")