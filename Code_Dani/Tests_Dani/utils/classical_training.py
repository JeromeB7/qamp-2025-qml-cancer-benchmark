"""
Classical SVM training utilities

Author: Quantum ML Implementation
Date: November 2025
"""

import os.path as osp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import sys
import numpy as np 
import os 

# Add path to custom modules
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../../"))

import quantum_feature_map.moduqusvm as mdqsvm
from quantum_feature_map.moduqusvm import svm_models
from .visualization import (
    plot_decision_boundary, 
    plot_kernel_matrix, 
    plot_confusion_matrix
)


def train_classical_svm(X_train, y_train, X_test, y_test, 
                        X_train_2d, X_test_2d,
                        output_dir, n_qubits, kernel='linear'):
    """
    Train classical SVM as baseline comparison
    
    Args:
        ... (args) ...
        kernel: SVM kernel type ('linear', 'rbf', 'quantum')
    
    Returns:
        Dictionary with model and results
    """
    method_dir = osp.join(output_dir, 'Classical')
    method_name = f'Classical {kernel.capitalize()} SVM'
    
    os.makedirs(method_dir, exist_ok=True)
    
    if kernel == 'quantum':
        method_name = 'Moduqusvm Quantum SVM'

    print(f"\n{'='*80}")
    print(f"TRAINING SVM (kernel={kernel}) DESDE MODUQUVSM")
    print(f"{'='*80}")
    
    print(f"Training on {len(X_train)} samples...")
    if kernel == 'quantum':
        clf = svm_models[kernel]["model"](n_qubits)
    else:
        clf = svm_models[kernel]["model"]
    
    clf.fit(X_train, y_train)
    
    # ... (Evaluación y prints) ...
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Malignant', 'Benign']))
    
    print(f"\nSaving {method_name} results to {method_dir}...")
    
    # --- Lógica de instanciación 2D ---
    if kernel == 'quantum':
        clf_2d = svm_models[kernel]["model"](2) # n_qubits=2 para 2D
    else:
        clf_2d = svm_models[kernel]["model"]
        
    clf_2d.fit(X_train_2d, y_train)
    
    # --- Guardar Matriz de Confusión ---
    cm_title = f'Confusion Matrix - {method_name}'
    cm_save_path = osp.join(method_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, labels=np.unique(y_test), 
                          title=cm_title, save_path=cm_save_path)
    
    # --- Decision boundary ---
    plot_decision_boundary(X_test_2d, y_test, clf_2d, method_name,
                          osp.join(method_dir, 'decision_boundary.png'))
    
    # --- LÓGICA DE KERNEL MATRIX (CORREGIDA) ---
    print("  Calculando kernel matrix...")

    if kernel == 'quantum':
        # Para 'quantum', el kernel se guarda en 'clf.K_train'
        if hasattr(clf, 'K_train'):
            K_train = clf.K_train
            plot_kernel_matrix(None, None, y_train, method_name,
                              osp.join(method_dir, 'kernel_matrix.png'),
                              precomputed_matrix=K_train)
        else:
            print(f"  ERROR: El clasificador 'quantum' no tiene atributo 'K_train'.")
            print("  Saltando plot de kernel matrix.")
    
    elif kernel in ['linear', 'rbf'] and svm_models[kernel]["kernel_matrix"] is not None:
        # Para 'linear' y 'rbf', llamamos a su lambda
        K_train = svm_models[kernel]["kernel_matrix"](X_train)
        plot_kernel_matrix(None, None, y_train, method_name,
                          osp.join(method_dir, 'kernel_matrix.png'),
                          precomputed_matrix=K_train)
    
    else:
        print(f"  Plot de Kernel matrix no aplica o no está definido para '{kernel}'.")

    # --- FIN DE LA CORRECCIÓN ---
    
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