"""
Classical SVM training utilities

Author: Quantum ML Implementation
Date: November 2025
"""

import os.path as osp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Add path to custom modules
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../../"))

import quantum_feature_map.moduqusvm as mdqsvm
from quantum_feature_map.moduqusvm import svm_models
from .visualization import plot_decision_boundary, plot_kernel_matrix


def train_classical_svm(X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,  output_dir, kernel='linear'):
    """
    Train classical SVM as baseline comparison
    
    Args:
        X_train, y_train: Training data (full dimensions)
        X_test, y_test: Test data (full dimensions)
        X_train_2d, X_test_2d: 2D projections for visualization
        num_qubits: Number of features (for consistency)
        output_dir: Directory to save results
        kernel: SVM kernel type
    
    Returns:
        Dictionary with model and results
    """
    method_dir = osp.join(output_dir, 'Classical')
    
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
    
    # Save results
    print(f"\nSaving Classical SVM results to {method_dir}...")
    
    # Train 2D classifier for visualization
    clf_2d = svm_models[kernel]["model"]
    clf_2d.fit(X_train_2d, y_train)
    
    # Decision boundary
    plot_decision_boundary(X_test_2d, y_test, clf_2d, 'Classical Linear SVM',
                          osp.join(method_dir, 'decision_boundary.png'))
    
    # Kernel matrix (classical kernels don't need num_qubits parameter)
    K_train = svm_models[kernel]["kernel_matrix"](X_train)
    plot_kernel_matrix(K_train, y_train, 'Classical Linear',
                      osp.join(method_dir, 'kernel_matrix.png'))
    
    # Save comparison plot
    mdqsvm.compare_predict_and_real(X_test_2d, y_pred, y_test, X_test_2d)
    plt.savefig(osp.join(method_dir, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'model': clf,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }