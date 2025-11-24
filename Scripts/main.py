"""
Supervised Learning - Final Project
Main training pipeline for predicting business outcomes
"""

import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def load_dataset(path: str) -> pl.DataFrame:
    """
    Load the dataset from parquet file
    
    Args:
        path: Path to the parquet file
        
    Returns:
        Polars DataFrame with the loaded data
    """
    print("Loading dataset...")
    df = pl.read_parquet(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df: pl.DataFrame):
    """
    Preprocess the dataset: handle missing values, encode categorical variables
    
    Args:
        df: Raw dataframe
        
    Returns:
        X: Features (numpy array)
        y: Target variable (numpy array)
        feature_names: List of feature names
    """
    print("\nPreprocessing data...")
    
    # Convert to pandas for sklearn compatibility
    df_pandas = df.to_pandas()
    
    # Define target variable (assuming Y_RN is the target)
    target_col = 'Y_RN'
    
    # Separate features and target
    y = df_pandas[target_col].values
    
    # Select features - remove target and identifier columns
    cols_to_drop = [target_col, 'siren', 'dateCreationUniteLegale']
    feature_cols = [col for col in df_pandas.columns if col not in cols_to_drop]
    
    X = df_pandas[feature_cols].copy()
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    print(f"Encoding {len(categorical_cols)} categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('missing')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Handle missing values in numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{np.bincount(y)}")
    
    return X.values, y, X.columns.tolist()


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train a simple baseline model (Logistic Regression)
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Trained model and metrics dictionary
    """
    print("\n" + "="*60)
    print("BASELINE MODEL: Logistic Regression")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, "Baseline")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, scaler, metrics


def train_improved_model(X_train, y_train, X_test, y_test):
    """
    Train an improved model (Random Forest)
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Trained model and metrics dictionary
    """
    print("\n" + "="*60)
    print("IMPROVED MODEL: Random Forest")
    print("="*60)
    
    # Standardize features (optional for RF but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, "Random Forest")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    print("\nTop 10 most important features:")
    feature_importance = sorted(
        zip(range(len(model.feature_importances_)), model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for idx, importance in feature_importance:
        print(f"  Feature {idx}: {importance:.4f}")
    
    return model, scaler, metrics


def compute_metrics(y_true, y_pred, model_name):
    """
    Compute and display classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
        
    Returns:
        Dictionary with computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    print(f"\n{model_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return metrics


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("SUPERVISED LEARNING - FINAL PROJECT")
    print("="*60)
    
    # 1. Load dataset
    df = load_dataset('Data/processed/sirene_final2.parquet')
    
    # 2. Preprocess data
    X, y, feature_names = preprocess_data(df)
    
    # 3. Train-test split (80-20 split, stratified)
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 4. Train baseline model
    baseline_model, baseline_scaler, baseline_metrics = train_baseline_model(
        X_train, y_train, X_test, y_test
    )
    
    # 5. Train improved model
    improved_model, improved_scaler, improved_metrics = train_improved_model(
        X_train, y_train, X_test, y_test
    )
    
    # 6. Compare results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<15} {'Baseline':<15} {'Random Forest':<15} {'Improvement':<15}")
    print("-"*60)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        baseline_val = baseline_metrics[metric]
        improved_val = improved_metrics[metric]
        improvement = ((improved_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        print(f"{metric.capitalize():<15} {baseline_val:<15.4f} {improved_val:<15.4f} {improvement:+.2f}%")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()