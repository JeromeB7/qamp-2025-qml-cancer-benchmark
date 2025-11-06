"""
Quantum kernel SVM training utilities

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


def train_quantum_kernel_individual(X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
                                   n_qubits, reps, method_type, output_dir):
    """
    Train individual quantum kernel SVM
    
    Args:
        X_train, y_train: Training data (full dimensions)
        X_test, y_test: Test data (full dimensions)
        X_train_2d, X_test_2d: 2D projections for visualization
        n_qubits: Number of qubits
        reps: Circuit repetitions
        method_type: 'fourier' or 'qaoa'
        output_dir: Base output directory
    
    Returns:
        Dictionary with results
    """
    method_name = 'Fourier' if method_type == 'fourier' else 'QAOA'
    method_dir = osp.join(output_dir, method_name)
    
    print(f"\n{'='*80}")
    print(f"TRAINING QUANTUM KERNEL SVM - {method_name.upper()}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Circuit repetitions: {reps}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Select appropriate model
    if method_type == 'fourier':
        clf = svm_models["quantum"]["model"](n_qubits, reps=reps)
    else:  # qaoa
        clf = svm_models["qaoa"]["model"](n_qubits, reps=reps)
    
    # Train
    print(f"\nTraining {method_name} quantum kernel...")
    clf.fit(X_train, y_train)
    
    # Evaluate
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
    print(f"\nSaving {method_name} results to {method_dir}...")
    
    # Train 2D classifier for visualization
    if method_type == 'fourier':
        clf_2d = svm_models["quantum"]["model"](2, reps=reps)
    else:
        clf_2d = svm_models["qaoa"]["model"](2, reps=reps)
    
    clf_2d.fit(X_train_2d, y_train)
    
    # Decision boundary
    plot_decision_boundary(X_test_2d, y_test, clf_2d, f'Quantum {method_name}',
                          osp.join(method_dir, 'decision_boundary.png'))
    
    # Kernel matrix
    kernel_key = "quantum" if method_type == 'fourier' else "qaoa"
    K_train = svm_models[kernel_key]["kernel_matrix"](X_train, n_qubits, reps=reps)
    plot_kernel_matrix(K_train, y_train, f'Quantum {method_name}',
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
        'predictions': y_pred,
        'accuracy': test_accuracy
    }
