import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, roc_auc_score, precision_recall_curve, 
                           average_precision_score, roc_curve, f1_score, 
                           precision_score, recall_score, make_scorer)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import defaultdict
import time
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
import os
import glob

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Global variables to store encoders and feature names
feature_encoders = defaultdict(LabelEncoder)
categorical_cols = []
feature_names = []
numeric_cols = []

def load_data(filepath):
    """Load and preprocess the dataset with enhanced validation"""
    try:
        # Read the data with error handling
        df = pd.read_csv(filepath, header=None, low_memory=False)
        
        print(f"\nDataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Enhanced data cleaning
        df = df.dropna(how='all')  # Only drop rows that are all NA
        df = df.drop_duplicates()
        
        # Validate minimum data requirements
        if len(df) < 100:
            raise ValueError("Insufficient data after cleaning (min 100 samples required)")
        
        # Create column names (last column is the label)
        num_columns = df.shape[1]
        column_names = [f'feature_{i}' for i in range(num_columns - 1)] + ['attack_type']
        df.columns = column_names
        
        # Basic validation of attack types
        if 'attack_type' not in df.columns:
            raise ValueError("Target column 'attack_type' not found")
            
        if df['attack_type'].nunique() < 2:
            raise ValueError("Insufficient classes in target variable")
        
        print("\nColumns after naming:")
        print(df.columns)
        print("\nAttack type distribution:")
        print(df['attack_type'].value_counts())
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def engineer_features(df):
    """Enhanced feature engineering with more robust handling"""
    global categorical_cols, feature_encoders, feature_names, numeric_cols
    
    # Create target variable first
    df['is_portscan'] = df['attack_type'].apply(lambda x: 1 if str(x).lower() in ['portsweep', 'portscan'] else 0)
    
    # Identify feature types more robustly
    categorical_cols = []
    numeric_cols = []
    
    for col in df.columns[:-2]:  # Exclude target columns
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].nunique() < 20 and df[col].dtype in ['int64', 'float64']:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    
    print(f"\nCategorical columns identified: {categorical_cols}")
    print(f"Numeric columns identified: {numeric_cols}")
    
    # Advanced feature engineering
    if 'feature_3' in df.columns and 'feature_4' in df.columns:
        df['total_packets'] = df['feature_3'] + df['feature_4']  # src + dst packets
        df['packet_ratio'] = np.where(df['feature_4'] != 0, 
                                    df['feature_3'] / df['feature_4'], 
                                    0)
        # Update numeric columns if we added new features
        numeric_cols.extend(['total_packets', 'packet_ratio'])
    
    # Convert categorical features with more robust encoding
    for col in categorical_cols:
        if col in df.columns:
            # Handle NaN values in categorical columns
            df[col] = df[col].fillna('missing')
            
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            feature_encoders[col] = le
            joblib.dump(le, f'{col}_encoder.pkl')
    
    # Show class distribution
    print("\nClass distribution:")
    print(df['is_portscan'].value_counts(normalize=True))
    
    # Store feature names for later use
    feature_names = [col for col in df.columns if col not in ['attack_type', 'is_portscan']]
    
    return df

def preprocess_data(df, training=False):
    """Enhanced preprocessing with column transformer"""
    global categorical_cols, numeric_cols
    
    # Make a copy to avoid modifying original dataframe
    processed_df = df.copy()
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    if training:
        # Fit and transform for training
        processed_data = preprocessor.fit_transform(processed_df[feature_names])
        
        # Save preprocessor
        joblib.dump(preprocessor, 'preprocessor.pkl')
    else:
        # Load preprocessor for inference
        try:
            preprocessor = joblib.load('preprocessor.pkl')
            processed_data = preprocessor.transform(processed_df[feature_names])
        except Exception as e:
            print(f"Warning: Could not load preprocessor ({str(e)}), using default scaling")
            processed_data = processed_df[feature_names].values  # Fallback
    
    return processed_data

def handle_imbalance(X, y):
    """Enhanced imbalance handling with validation"""
    print("\nClass distribution before balancing:")
    print(pd.Series(y).value_counts())
    
    if len(np.unique(y)) < 2:
        print("Warning: Only one class present - skipping balancing")
        return X, y
    
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_res, y_res = smote.fit_resample(X, y)
    
    print("\nClass distribution after balancing:")
    print(pd.Series(y_res).value_counts())
    
    return X_res, y_res

def train_model(X_train, y_train):
    """Enhanced model training with multiple algorithms and better CV"""
    print("\nTraining model with enhanced pipeline...")
    
    # Create scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # Hyperparameter grid with multiple algorithms
    param_grid = [
        {
            'classifier': [RandomForestClassifier(random_state=42, class_weight='balanced')],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'feature_selection__k': [10, 15, 'all']
        },
        {
            'classifier': [GradientBoostingClassifier(random_state=42)],
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.1, 0.05],
            'classifier__max_depth': [3, 5],
            'feature_selection__k': [10, 15]
        }
    ]
    
    # Enhanced cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search with refit on ROC AUC
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring=scoring,
        refit='roc_auc',
        return_train_score=True
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    # Print all CV results
    results = pd.DataFrame(grid_search.cv_results_)
    print("\nTop 5 parameter combinations:")
    print(results.sort_values('rank_test_roc_auc').head(5)[[
        'param_classifier', 'mean_test_roc_auc', 'mean_test_f1', 'rank_test_roc_auc'
    ]])
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Enhanced model evaluation with more metrics and plots"""
    print("\nEnhanced model evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Enhanced classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Portscan'], digits=4))
    
    # Enhanced confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Portscan'],
                yticklabels=['Normal', 'Portscan'],
                annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # Calculate comprehensive metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Average Precision': average_precision_score(y_test, y_pred_proba),
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }
    
    print("\nPerformance Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Feature importance
    try:
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Get feature names (handling feature selection)
            selected_features = model.named_steps['feature_selection'].get_support()
            if hasattr(model.named_steps['feature_selection'], 'get_feature_names_out'):
                feature_names_selected = model.named_steps['feature_selection'].get_feature_names_out(feature_names)
            else:
                feature_names_selected = np.array(feature_names)[selected_features]
            
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importances")
            plt.bar(range(min(10, len(indices))), importances[indices][:10], align="center")
            plt.xticks(range(min(10, len(indices))), feature_names_selected[indices][:10], rotation=45)
            plt.xlim([-1, min(10, len(indices))])
            plt.tight_layout()
            plt.savefig('feature_importances.png', bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Could not plot feature importances: {str(e)}")
    
    return metrics

def save_artifacts(model):
    """Enhanced artifact saving with metadata"""
    print("\nSaving enhanced model artifacts...")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save model with versioning
    model_filename = f'portscan_model_{timestamp}.pkl'
    joblib.dump(model, model_filename)
    
    # Create symlink to latest model
    try:
        if os.path.exists('portscan_model_latest.pkl'):
            os.remove('portscan_model_latest.pkl')
        os.symlink(model_filename, 'portscan_model_latest.pkl')
    except Exception as e:
        print(f"Could not create symlink: {str(e)}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'model_type': str(model.named_steps['classifier'].__class__.__name__)
    }
    
    joblib.dump(metadata, f'model_metadata_{timestamp}.pkl')
    
    # For backward compatibility, also save the scaler and features separately
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        joblib.dump(scaler, 'portscan_scaler.pkl')
        joblib.dump(feature_names, 'portscan_features.pkl')
    except Exception as e:
        print(f"Could not save backward-compatible files: {str(e)}")
    
    print(f"Model artifacts saved as {model_filename}")

def main():
    """Enhanced main function with better error handling"""
    try:
        print("\nStarting enhanced port scan detection model training...")
        
        # Load data
        filepath = 'fs_new validation project.csv'
        df = load_data(filepath)
        
        if df is None:
            raise ValueError("Data loading failed")
        
        # Feature engineering
        df = engineer_features(df)
        
        # Split data
        X = df.drop(['is_portscan', 'attack_type'], axis=1)
        y = df['is_portscan']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Handle imbalance
        X_train_res, y_train_res = handle_imbalance(X_train, y_train)
        
        # Preprocess data
        X_train_processed = preprocess_data(pd.DataFrame(X_train_res, columns=feature_names), training=True)
        X_test_processed = preprocess_data(X_test, training=False)
        
        # Train model
        model = train_model(X_train_processed, y_train_res)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test_processed, y_test)
        
        # Save artifacts
        save_artifacts(model)
        
        print("\nModel training completed successfully!")
        return model
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        return None

if __name__ == "__main__":
    trained_model = main()