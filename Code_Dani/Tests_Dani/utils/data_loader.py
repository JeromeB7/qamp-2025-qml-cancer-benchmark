"""
Data loading and preprocessing utilities for Breast Cancer classification

Author: Quantum ML Implementation
Date: November 2025
"""

import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    """
    Load and preprocess the Breast Cancer Wisconsin dataset
    
    Steps:
    1. Download dataset from Kaggle
    2. Remove unnecessary columns (id, unnamed columns)
    3. Separate features (X) and target (y)
    4. Convert target labels: M (Malignant) -> 0, B (Benign) -> 1
    
    Returns:
        X: Feature matrix [n_samples, n_features]
        y: Label vector [n_samples]
        feature_names: List of feature names
    """
    print("="*80)
    print("LOADING BREAST CANCER WISCONSIN DATASET")
    print("="*80)
    
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
    print(f"Dataset path: {path}")
    
    # Read CSV file (first row contains headers by default)
    data = pd.read_csv(f"{path}/data.csv")
    
    # Display original column information
    original_columns = data.columns.tolist()
    print(f"\nOriginal columns ({len(original_columns)}): {original_columns[:5]}...")
    print(f"Dataset shape: {data.shape}")
    
    # Remove unnecessary columns:
    # - 'id': Patient identifier (not useful for classification)
    # - 'Unnamed: 32': Empty column from CSV formatting
    columns_to_drop = ['id', 'Unnamed: 32']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"\nAfter dropping unnecessary columns: {data.shape}")
    print(f"First few rows:")
    print(data.head())
    
    # Create pairplot and save in base Output directory
    sns.pairplot(data, hue="diagnosis", palette="tab10")
    plt.savefig('../Output/Pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Separate features and target
    # Target column: 'diagnosis' with values 'M' (Malignant) or 'B' (Benign)
    X = data.drop(columns=["diagnosis"])
    
    # Get feature names before converting to numpy array
    feature_names = X.columns.tolist()
    print(f"\nFeature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names[:10]):  # Show first 10
        print(f"  {i+1}. {name}")
    if len(feature_names) > 10:
        print(f"  ... and {len(feature_names) - 10} more")
    
    # Convert diagnosis to binary labels
    # M (Malignant) -> 0, B (Benign) -> 1
    y = data["diagnosis"].replace({"M": 0, "B": 1}).astype(int)
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"\nClass distribution:")
    print(f"  Malignant (0): {class_counts[0]} samples ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"  Benign (1): {class_counts[1]} samples ({class_counts[1]/len(y)*100:.1f}%)")
    
    # Convert to numpy arrays
    X = X.values
    y = y.values
    
    return X, y, feature_names
