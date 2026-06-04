from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models, encoders, and scaler
models = {}
label_encoders = {}
scaler = None

def initialize_models():
    global models, label_encoders, scaler
    try:
        # Load and preprocess data
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(os.path.join(base_dir, 'fs_new validation project.csv'), header=None)
        columns = [
            'protocol', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
            'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'label'
        ]
        data.columns = columns

        # Encode categorical variables
        categorical_cols = ['protocol', 'service', 'flag']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        
        # Map attack types
        data['label'] = data['label'].apply(map_attack_type)

        # Split data and train models
        X = data.drop('label', axis=1)
        y = data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        models = {
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
        
        logger.info("Models initialized successfully.")
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
        raise

def map_attack_type(attack):
    dos_attacks = ['neptune', 'smurf', 'pod', 'teardrop', 'land', 'back', 'apache2', 'udpstorm', 'processtable', 'mailbomb']
    
   
    if attack in dos_attacks:
        return 'DoS'
    
    else:
        return 'Normal'

@app.route('/')
def home():
    try:
        return render_template('dos_detection.html')
    except Exception as e:
        logger.error(f"Template error: {str(e)}")
        return "Template loading error", 500

@app.route('/predict_dos', methods=['POST'])
def predict_dos():
    try:
        user_input = request.get_json()
        logger.debug(f"Received input: {user_input}")

        numeric_fields = ['src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                          'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                          'su_attempted', 'num_root', 'num_file_creations']

        for field in numeric_fields:
            user_input[field] = float(user_input.get(field, 0.0))

        sample_data = pd.DataFrame([user_input])

        for col, le in label_encoders.items():
            if sample_data[col].iloc[0] not in le.classes_:
                le.classes_ = np.append(le.classes_, sample_data[col].iloc[0])
            sample_data[col] = le.transform(sample_data[col])

        sample_data_scaled = scaler.transform(sample_data)
        results = {}

        for name, model in models.items():
            prediction = model.predict(sample_data_scaled)[0]
            results[name] = {'prediction': prediction}
            logger.debug(f"{name} prediction: {prediction}")

        # Generate HTML for prediction results
        result_html = "<h2>Prediction Results:</h2>"
        for model_name, result in results.items():
            result_html += f"<div class='model-prediction'><h3>{model_name}</h3>"
            result_html += f"<p>Prediction: <strong>{result['prediction']}</strong></p></div>"

        return jsonify({'status': 'success', 'result_html': result_html})

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'status': 'error', 'error': f"Prediction failed: {str(e)}"}), 500




def detect_attack_patterns(user_input):
    patterns = []
    thresholds = {
        'src_bytes': 1000000,
        'dst_bytes': 1000000,
        'num_failed_logins': 5,
        'num_compromised': 0,
        'root_shell': 1,
        'su_attempted': 1
    }
    
    for feature, threshold in thresholds.items():
        if user_input.get(feature, 0) > threshold:
            patterns.append(f"Suspicious {feature.replace('_', ' ')}: {user_input[feature]}")
    
    return patterns if patterns else ["No suspicious patterns detected"]

if __name__ == '__main__':
    try:
        initialize_models()
        logger.info("Starting server on http://127.0.0.1:8000")
        app.run(host='127.0.0.1', port=8000, debug=True)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")
