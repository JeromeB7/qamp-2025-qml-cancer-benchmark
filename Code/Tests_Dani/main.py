"""
Breast Cancer Classification using Classical and Quantum Kernels
=================================================================

This script performs binary classification on the Wisconsin Breast Cancer dataset
using both classical and quantum kernel methods.

Dataset: UCI Breast Cancer Wisconsin (Diagnostic)
- 569 samples
- 30 features (mean, SE, worst values for 10 measurements)
- 2 classes: Malignant (M) and Benign (B)

Methods:
1. Classical SVM with linear kernel (baseline)
2. Quantum Kernel SVM with Fourier Feature Map
3. Quantum Kernel SVM with QAOA-based Feature Map

Author: Quantum ML Implementation
Date: November 2025
"""

import kagglehub
import pandas as pd
import numpy as np
import os
import os.path as osp
import seaborn as sns

import sys
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add path to custom modules
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

# Import custom modules for classical methods
import quantum_feature_map.moduqusvm as mdqsvm
import quantum_feature_map.modudata as mddt
from quantum_feature_map.moduqusvm import svm_models

# Import quantum kernel implementations
from quantum_kernels import (
    QuantumKernelSVM,
    compare_quantum_kernels,
    FourierFeatureMap,
    QAOABasedFeatureMap
)


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
    sns.pairplot(data, hue="diagnosis", palette="tab10")
    plt.savefig('Pairplot.png', dpi=300, bbox_inches='tight')

    print('Leyendo gráfica de pairplot...')
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


def dimensionality_reduction(X, y, n_components=4):
    """
    Apply PCA for dimensionality reduction
    
    Quantum computers have limited qubits, so we need to reduce features.
    PCA extracts the most important information into fewer dimensions.
    
    Args:
        X: Original features [n_samples, n_features]
        y: Labels
        n_components: Number of principal components to keep
    
    Returns:
        X_reduced: Reduced features [n_samples, n_components]
    """
    print(f"\n{'='*80}")
    print(f"DIMENSIONALITY REDUCTION: {X.shape[1]} -> {n_components} features")
    print(f"{'='*80}")
    
    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Print explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"\nPCA Analysis:")
    for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var)):
        print(f"  PC{i+1}: {ev*100:.2f}% variance (cumulative: {cv*100:.2f}%)")
    
    print(f"\nTotal variance explained by {n_components} components: {cumulative_var[-1]*100:.2f}%")
    
    return X_reduced, scaler, pca


def train_classical_svm(X_train, y_train, X_test, y_test, kernel='linear'):
    """
    Train classical SVM as baseline comparison
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        kernel: SVM kernel type ('linear', 'rbf', 'poly')
    
    Returns:
        Dictionary with model and results
    """
    print(f"\n{'='*80}")
    print(f"TRAINING CLASSICAL SVM (kernel={kernel})")
    print(f"{'='*80}")
    
    # Get model from predefined svm_models
    clf = svm_models[kernel]["model"]
    
    # Train the model
    print(f"Training on {len(X_train)} samples...")
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Malignant', 'Benign']))
    
    return {
        'model': clf,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }


def train_quantum_kernels(X_train, y_train, X_test, y_test, n_qubits, reps=2):
    """
    Train quantum kernel SVMs with both Fourier and QAOA feature maps
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_qubits: Number of qubits (should match n_features)
        reps: Circuit depth (repetitions)
    
    Returns:
        Dictionary with results for both quantum methods
    """
    print(f"\n{'='*80}")
    print(f"QUANTUM KERNEL SVM COMPARISON")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Circuit repetitions: {reps}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Use the comparison function from quantum_kernels module
    results = compare_quantum_kernels(
        X_train, y_train,
        X_test, y_test,
        n_features=n_qubits,
        reps=reps
    )
    
    return results


def visualize_results(classical_results, quantum_results, save_path='results.png'):
    """
    Create visualization comparing classical and quantum methods
    
    Args:
        classical_results: Results from classical SVM
        quantum_results: Results from quantum kernels
        save_path: Path to save figure
    """
    print(f"\n{'='*80}")
    print(f"VISUALIZATION")
    print(f"{'='*80}")
    
    # Extract accuracies
    methods = ['Classical\nLinear SVM', 
               'Quantum\nFourier', 
               'Quantum\nQAOA']
    accuracies = [
        classical_results['test_accuracy'],
        quantum_results['fourier']['accuracy'],
        quantum_results['qaoa']['accuracy']
    ]
    
    # Create bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax1 = axes[0]
    bars = ax1.bar(methods, accuracies, color=['blue', 'green', 'red'], alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Confusion matrices
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create text summary
    summary_text = "CONFUSION MATRICES\n\n"
    summary_text += "Classical Linear SVM:\n"
    summary_text += f"{classical_results['confusion_matrix']}\n\n"
    summary_text += "Quantum Fourier:\n"
    summary_text += f"{quantum_results['fourier']['confusion_matrix']}\n\n"
    summary_text += "Quantum QAOA:\n"
    summary_text += f"{quantum_results['qaoa']['confusion_matrix']}"
    
    ax2.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()


def main():
    """
    Main execution pipeline
    
    Steps:
    1. Load and preprocess data
    2. Apply dimensionality reduction (PCA)
    3. Split into train/test sets
    4. Train classical SVM (baseline)
    5. Train quantum kernel SVMs (Fourier + QAOA)
    6. Compare and visualize results
    """
    print("\n" + "="*80)
    print("BREAST CANCER CLASSIFICATION: CLASSICAL vs QUANTUM KERNELS")
    print("="*80 + "\n")
    
    # Step 1: Load data
    X, y, feature_names = load_and_preprocess_data()
    
    # Step 2: Dimensionality reduction
    # Reduce to 4 features (4 qubits) for quantum processing
    n_qubits = 4
    X_reduced, scaler, pca = dimensionality_reduction(X, y, n_components=n_qubits)
    
    # Step 3: Split data
    X_train, y_train, X_test, y_test, X_holdout, y_holdout = mddt.data_split(
        X_reduced, y
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Holdout: {len(X_holdout)} samples")
    
    # Step 4: Train classical SVM
    classical_results = train_classical_svm(X_train, y_train, X_test, y_test, 
                                           kernel='linear')
    
    # Step 5: Train quantum kernels
    quantum_results = train_quantum_kernels(X_train, y_train, X_test, y_test, 
                                           n_qubits=n_qubits, reps=2)
    
    # Step 6: Final comparison
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Classical Linear SVM:    {classical_results['test_accuracy']:.4f}")
    print(f"Quantum Fourier Map:     {quantum_results['fourier']['accuracy']:.4f}")
    print(f"Quantum QAOA-based Map:  {quantum_results['qaoa']['accuracy']:.4f}")
    
    # Determine best method
    all_accuracies = {
        'Classical': classical_results['test_accuracy'],
        'Fourier': quantum_results['fourier']['accuracy'],
        'QAOA': quantum_results['qaoa']['accuracy']
    }
    best_method = max(all_accuracies, key=all_accuracies.get)
    print(f"\nBest performing method: {best_method} ({all_accuracies[best_method]:.4f})")
    
    # Step 7: Visualization
    visualize_results(classical_results, quantum_results)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return classical_results, quantum_results


if __name__ == "__main__":
    # Execute main pipeline
    classical_results, quantum_results = main()