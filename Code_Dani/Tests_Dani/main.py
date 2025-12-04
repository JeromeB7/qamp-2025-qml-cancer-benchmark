"""
Breast Cancer Classification using Classical and Quantum Kernels
=================================================================
Main execution script that orchestrates the complete workflow:
1. Data loading and preprocessing
2. Dimensionality reduction with PCA
3. Model Training (Selectable via Flags)
4. Results comparison and visualization

Author: Quantum ML Implementation
Date: November 2025
"""

import numpy as np
import os.path as osp
import sys
import os

# Add path to custom modules
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

import quantum_feature_map.modudata as mddt

# --- IMPORTS ---
from utils.file_utils import create_output_directories
from utils.data_loader import load_and_preprocess_data
from utils.dimensionality_reduction import (
    dimensionality_reduction_auto,
    create_2d_projection
)
from utils.classical_training import train_classical_svm
# Importamos SOLO la función unificada que maneja todo
from utils.quantum_training import train_quantum_kernel_individual 

from utils.visualization import (
    visualize_overall_results,
    plot_pca_variance_analysis
)

def print_final_summary(classical_lin_results, classical_rbf_results,
                       moduqusvm_quantum_results, qumo_results,
                       fourier_results, qaoa_results, 
                       reservoir_results, qft_results,
                       n_qubits, pca):
    """
    Print comprehensive summary of all results (Robust to skipped models)
    """
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Helper para obtener accuracy
    def get_acc(result_dict):
        if result_dict is None: return -1.0
        if 'test_accuracy' in result_dict:
            return result_dict['test_accuracy']
        elif 'accuracy' in result_dict:
            return result_dict['accuracy']
        return 0.0

    print(f"Number of features used (PCs): {n_qubits}")
    print(f"Variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # Diccionario con todos los posibles resultados
    all_models = {
        'QKernels QFT': qft_results,
        'Moduqusvm QuMO': qumo_results,
        'QKernels Reservoir': reservoir_results,
        'Classical Linear SVM': classical_lin_results,
        'Classical RBF SVM': classical_rbf_results,
        'Moduqusvm Quantum': moduqusvm_quantum_results,
        'QKernels Fourier Map': fourier_results,
        'QKernels QAOA-based Map': qaoa_results
    }

    print(f"\nTest Accuracies (Executed Models):")
    
    valid_accuracies = {}
    
    for name, res in all_models.items():
        if res is not None:
            acc = get_acc(res)
            print(f"  - {name:<25}: {acc:.4f}")
            valid_accuracies[name] = acc
            
    if valid_accuracies:
        best_method = max(valid_accuracies, key=valid_accuracies.get)
        print(f"\nBest performing method: {best_method} ({valid_accuracies[best_method]:.4f})")
    else:
        print("\nNo models were executed.")


def main():
    """
    Main execution pipeline with selectable models
    """
    # ==========================================
    # FLAGS DE CONFIGURACIÓN (CONTROL PANEL)
    # ==========================================
    
    RUN_QFT           = True
    RUN_QUMO          = True
    RUN_RESERVOIR     = True
    
    RUN_CLASSICAL_LIN = True  # Activado para tener baseline
    RUN_CLASSICAL_RBF = True  # Activado para tener baseline
    RUN_MODUQU_QUANT  = False
    
    RUN_FOURIER       = True
    RUN_QAOA          = True
    
    # ==========================================
    
    print("\n" + "="*80)
    print("BREAST CANCER CLASSIFICATION: SELECTIVE EXECUTION")
    print("="*80 + "\n")
    
    # --- Pasos 1-5: Load, PCA, Vis, Split ---
    output_dir = create_output_directories()
    X, y, feature_names = load_and_preprocess_data()
    X_reduced, scaler, pca, n_qubits = dimensionality_reduction_auto(
        X, y, variance_threshold=0.90
    )
    plot_pca_variance_analysis(pca, n_qubits, output_dir)
    
    X_train, y_train, X_test, y_test, X_holdout, y_holdout = mddt.data_split(
        X_reduced, y
    )
    print(f"\nData split: Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    X_train_2d, X_test_2d = create_2d_projection(X_train, X_test)
    
    # Inicializar resultados a None por defecto
    qft_results = None
    qumo_results = None
    reservoir_results = None
    classical_lin_results = None
    classical_rbf_results = None
    moduqusvm_quantum_results = None
    fourier_results = None
    qaoa_results = None

    reps = 3
    shots = 1024 # Definir shots globales

    # --- Step 6: Ejecución Condicional de modelos ---

    # --- 6a. QK_QFT ---
    if RUN_QFT:
        # Usamos train_quantum_kernel_individual pasando el string 'qft'
        qft_results = train_quantum_kernel_individual(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            n_qubits, reps, 'qft', output_dir, shots=shots
        )

    # --- 6b. QuMO ---
    if RUN_QUMO:
        # Usamos train_quantum_kernel_individual pasando el string 'qumo'
        qumo_results = train_quantum_kernel_individual(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            n_qubits, reps, 'qumo', output_dir, shots=shots
        )

    # --- 6c. QK_Reservoir ---
    if RUN_RESERVOIR:
        # Usamos train_quantum_kernel_individual pasando el string 'reservoir'
        reservoir_results = train_quantum_kernel_individual(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            n_qubits, reps, 'reservoir', output_dir, shots=shots
        )

    # --- 6d. Classical Linear ---
    if RUN_CLASSICAL_LIN:
        print("\n--- [MODELO] Training Classical Linear SVM ---")
        classical_lin_results = train_classical_svm(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            output_dir, n_qubits=n_qubits, kernel='linear'
        )
        # Rename output folder logic if necessary
        if osp.exists(osp.join(output_dir, "Classical")):
            os.rename(osp.join(output_dir, "Classical"), osp.join(output_dir, "Classical_Linear"))

    # --- 6e. Classical RBF ---
    if RUN_CLASSICAL_RBF:
        print("\n--- [MODELO] Training Classical RBF SVM ---")
        classical_rbf_results = train_classical_svm(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            output_dir, n_qubits=n_qubits, kernel='rbf'
        )
        if osp.exists(osp.join(output_dir, "Classical")):
            os.rename(osp.join(output_dir, "Classical"), osp.join(output_dir, "Classical_RBF"))
        
    # --- 6f. Moduqusvm Quantum Kernel (Legacy) ---
    if RUN_MODUQU_QUANT:
        print("\n--- [MODELO] Training Moduqusvm Quantum SVM ---")
        moduqusvm_quantum_results = train_classical_svm(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            output_dir, n_qubits=n_qubits, kernel='quantum'
        )
        if osp.exists(osp.join(output_dir, "Classical")):
            os.rename(osp.join(output_dir, "Classical"), osp.join(output_dir, "Moduqusvm_Quantum"))

    # --- 6g. QKernels Fourier ---
    if RUN_FOURIER:
        fourier_results = train_quantum_kernel_individual(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            n_qubits, reps, 'fourier', output_dir, shots=shots
        )
    
    # --- 6h. QKernels QAOA ---
    if RUN_QAOA:
        qaoa_results = train_quantum_kernel_individual(
            X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
            n_qubits, reps, 'qaoa', output_dir, shots=shots
        )
    
    # --- Step 8: Final summary ---
    print_final_summary(
        classical_lin_results, classical_rbf_results, 
        moduqusvm_quantum_results, qumo_results,
        fourier_results, qaoa_results, 
        reservoir_results, qft_results,
        n_qubits, pca
    )
    
    # --- Step 9: Overall visualization ---
    try:
        executed_count = sum([RUN_QFT, RUN_QUMO, RUN_RESERVOIR, RUN_CLASSICAL_LIN, 
                              RUN_CLASSICAL_RBF, RUN_MODUQU_QUANT, RUN_FOURIER, RUN_QAOA])
        
        if executed_count >= 1:
            print("\nGenerando visualización comparativa...")
            visualize_overall_results(
                classical_lin_results, classical_rbf_results,
                moduqusvm_quantum_results, qumo_results,
                fourier_results, qaoa_results,
                reservoir_results, qft_results,
                output_dir
            )
        else:
            print("\nSaltando visualización comparativa (ningún modelo ejecutado).")
            
    except Exception as e:
        print(f"\nNo se pudo generar la gráfica comparativa completa.")
        print(f"Error details: {e}")
    
    # Final message
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved in: {osp.abspath(output_dir)}/")
    
    return classical_lin_results, classical_rbf_results, \
           moduqusvm_quantum_results, qumo_results, \
           fourier_results, qaoa_results, \
           reservoir_results, qft_results


if __name__ == "__main__":
    main()