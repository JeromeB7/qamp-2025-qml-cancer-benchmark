"""
Utility functions for training quantum kernel models
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os.path as osp

# Import quantum kernel implementations
import sys
import os.path as osp
# Get the parent directory (Code_Dani/Tests_Dani)
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
PARENT_DIR = osp.dirname(SCRIPT_DIR)  # Go up from utils/ to Tests_Dani/
sys.path.insert(0, PARENT_DIR)

from quantum_kernels import QuantumKernelSVM
from utils.visualization import (
    plot_decision_boundary,
    plot_kernel_matrix)



def train_quantum_kernel_individual(X_train, y_train, X_test, y_test,
                                   X_train_2d, X_test_2d,
                                   n_qubits, reps, feature_map_type, output_dir):
    """
    Train a single quantum kernel SVM model
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        X_train_2d, X_test_2d: 2D projections for visualization
        n_qubits: Number of qubits (features)
        reps: Number of circuit repetitions
        feature_map_type: 'fourier' or 'qaoa'
        output_dir: Directory to save results
    
    Returns:
        Dictionary with results
    """
    
    print(f"\n{'='*80}")
    print(f"TRAINING QUANTUM KERNEL SVM - {feature_map_type.upper()}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Circuit repetitions: {reps}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Initialize quantum kernel SVM with correct parameters
    qksvm = QuantumKernelSVM(
        feature_map_type=feature_map_type,
        n_features=n_qubits,
        reps=reps,
        entanglement='linear',
        shots=1024
    )
    
    # Train the model
    qksvm.fit(X_train, y_train, C=1.0)
    
    # Evaluate on test set
    results = qksvm.evaluate(X_test, y_test)
    
    print(f"\n{feature_map_type.upper()} Quantum Kernel Results:")
    print(f"{'='*80}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Create output subfolder
    method_dir = osp.join(output_dir, feature_map_type.capitalize())
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    # 1. Decision boundary (using 2D projection)
    plot_decision_boundary(
        qksvm, X_train_2d, y_train, X_test_2d, y_test,
        title=f"{feature_map_type.upper()} Quantum Kernel - Decision Boundary",
        save_path=osp.join(method_dir, "decision_boundary.png")
    )
    
    # 2. Kernel matrix
    plot_kernel_matrix(
        qksvm, X_train, y_train,
        title=f"{feature_map_type.upper()} Quantum Kernel Matrix",
        save_path=osp.join(method_dir, "kernel_matrix.png")
    )
    

    
    print(f"Visualizations saved in: {method_dir}/")
    
    return {
        'model': qksvm,
        'accuracy': results['accuracy'],
        'confusion_matrix': results['confusion_matrix'],
        'predictions': results['predictions'],
        'classification_report': results['classification_report']
    }