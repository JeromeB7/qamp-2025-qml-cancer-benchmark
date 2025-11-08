"""
Dimensionality reduction utilities using PCA

Author: Quantum ML Implementation
Date: November 2025
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def dimensionality_reduction_auto(X, y, variance_threshold=0.90):
    """
    Apply PCA with automatic component selection to explain target variance
    
    Args:
        X: Original features [n_samples, n_features]
        y: Labels
        variance_threshold: Target cumulative variance (default: 90%)
    
    Returns:
        X_reduced: Reduced features [n_samples, n_components]
        scaler: Fitted StandardScaler
        pca: Fitted PCA object
        n_components: Number of components selected
    """
    print(f"\n{'='*80}")
    print(f"AUTOMATIC DIMENSIONALITY REDUCTION (Target: {variance_threshold*100:.0f}% variance)")
    print(f"{'='*80}")
    
    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # First, fit PCA with all components to analyze variance
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Find minimum components needed for target variance
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_var >= variance_threshold) + 1
    
    print(f"\nVariance Analysis:")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Components for {variance_threshold*100:.0f}% variance: {n_components}")
    
    # Apply PCA with selected components
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Print explained variance for selected components
    explained_var = pca.explained_variance_ratio_
    cumulative_var_selected = np.cumsum(explained_var)
    
    print(f"\nSelected Principal Components:")
    for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var_selected)):
        print(f"  PC{i+1}: {ev*100:.2f}% variance (cumulative: {cv*100:.2f}%)")
    
    print(f"\nTotal variance explained: {cumulative_var_selected[-1]*100:.2f}%")
    
    return X_reduced, scaler, pca, n_components


def create_2d_projection(X_train, X_test):
    """
    Create 2D projections for visualization
    
    Args:
        X_train: Training data
        X_test: Test data
    
    Returns:
        X_train_2d: 2D projection of training data
        X_test_2d: 2D projection of test data
    """
    # Use first 2 components if available
    X_train_2d = X_train[:, :2] if X_train.shape[1] >= 2 else X_train
    X_test_2d = X_test[:, :2] if X_test.shape[1] >= 2 else X_test
    
    return X_train_2d, X_test_2d
