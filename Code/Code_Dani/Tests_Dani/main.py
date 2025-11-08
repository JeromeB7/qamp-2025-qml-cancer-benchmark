"""
Breast Cancer Classification using Classical and Quantum Kernels
=================================================================

Main execution script that orchestrates the complete workflow:
1. Data loading and preprocessing
2. Dimensionality reduction with PCA
3. Classical SVM training (baseline)
4. Quantum kernel SVM training (Fourier + QAOA)
5. Results comparison and visualization

Dataset: UCI Breast Cancer Wisconsin (Diagnostic)
- 569 samples
- 30 features (mean, SE, worst values for 10 measurements)
- 2 classes: Malignant (M) and Benign (B)

Author: Quantum ML Implementation
Date: November 2025
"""

import numpy as np
import os.path as osp
import sys

# Add path to custom modules
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

# Import custom modules for classical methods
import quantum_feature_map.modudata as mddt

# Import utility modules
from utils.file_utils import create_output_directories
from utils.data_loader import load_and_preprocess_data
from utils.dimensionality_reduction import (
    dimensionality_reduction_auto,
    create_2d_projection
)
from utils.classical_training import train_classical_svm
from utils.quantum_training import train_quantum_kernel_individual
from utils.visualization import (
    visualize_overall_results,
    plot_pca_variance_analysis  # New import
)


def print_final_summary(classical_results, fourier_results, qaoa_results, 
                       n_qubits, pca):
    """
    Print comprehensive summary of all results
    
    Args:
        classical_results: Results from classical SVM
        fourier_results: Results from Fourier quantum kernel
        qaoa_results: Results from QAOA quantum kernel
        n_qubits: Number of principal components used
        pca: Fitted PCA object
    """
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Number of features used (PCs): {n_qubits}")
    print(f"Variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"\nTest Accuracies:")
    print(f"  Classical Linear SVM:    {classical_results['test_accuracy']:.4f}")
    print(f"  Quantum Fourier Map:     {fourier_results['accuracy']:.4f}")
    print(f"  Quantum QAOA-based Map:  {qaoa_results['accuracy']:.4f}")
    
    # Determine best method
    all_accuracies = {
        'Classical': classical_results['test_accuracy'],
        'Fourier': fourier_results['accuracy'],
        'QAOA': qaoa_results['accuracy']
    }
    best_method = max(all_accuracies, key=all_accuracies.get)
    print(f"\nBest performing method: {best_method} ({all_accuracies[best_method]:.4f})")


def main():
    """
    Main execution pipeline with automatic PCA and organized results
    
    Pipeline Steps:
    1. Create output directory structure
    2. Load and preprocess data
    3. Apply automatic dimensionality reduction (90% variance)
    4. Visualize PCA variance analysis
    5. Split into train/test sets
    6. Train classical SVM (baseline)
    7. Train quantum kernel SVMs (Fourier + QAOA)
    8. Compare and visualize results in organized folders
    
    Returns:
        classical_results: Dictionary with classical SVM results
        fourier_results: Dictionary with Fourier quantum kernel results
        qaoa_results: Dictionary with QAOA quantum kernel results
    """
    print("\n" + "="*80)
    print("BREAST CANCER CLASSIFICATION: CLASSICAL vs QUANTUM KERNELS")
    print("WITH AUTOMATIC PCA AND ORGANIZED OUTPUT")
    print("="*80 + "\n")
    
    # Step 1: Create output directories
    output_dir = create_output_directories()
    
    # Step 2: Load data
    X, y, feature_names = load_and_preprocess_data()
    
    # Step 3: Automatic dimensionality reduction (90% variance)
    X_reduced, scaler, pca, n_qubits = dimensionality_reduction_auto(
        X, y, variance_threshold=0.90
    )
    
    # Step 4: Visualize PCA variance analysis
    plot_pca_variance_analysis(pca, n_qubits, output_dir)
    
    # Step 5: Split data
    X_train, y_train, X_test, y_test, X_holdout, y_holdout = mddt.data_split(
        X_reduced, y
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Holdout: {len(X_holdout)} samples")
    
    # Create 2D projection for visualization
    print(f"\nCreating 2D projections for visualization...")
    X_train_2d, X_test_2d = create_2d_projection(X_train, X_test)
    print(f"  2D projection shape: {X_train_2d.shape}")
    
    # Step 6: Train classical SVM
    classical_results = train_classical_svm(
        X_train, y_train, X_test, y_test,
        X_train_2d, X_test_2d,
        output_dir, kernel='linear'
    )
    
    # Step 7: Train quantum kernels
    reps = 2
    
    # Fourier quantum kernel
    fourier_results = train_quantum_kernel_individual(
        X_train, y_train, X_test, y_test,
        X_train_2d, X_test_2d,
        n_qubits, reps, 'fourier', output_dir
    )
    
    # QAOA quantum kernel
    qaoa_results = train_quantum_kernel_individual(
        X_train, y_train, X_test, y_test,
        X_train_2d, X_test_2d,
        n_qubits, reps, 'qaoa', output_dir
    )
    
    # Step 8: Final summary and visualization
    print_final_summary(classical_results, fourier_results, qaoa_results, 
                       n_qubits, pca)
    
    # Step 9: Overall visualization
    visualize_overall_results(
        classical_results, fourier_results, qaoa_results, output_dir
    )
    
    # Final message
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved in:")
    print(f"  {osp.abspath(output_dir)}/")
    print(f"\nGenerated files:")
    print(f"  - pca_variance_analysis.png (PCA variance explained)")
    print(f"  - Classical/decision_boundary.png")
    print(f"  - Classical/kernel_matrix.png")
    print(f"  - Classical/predictions_comparison.png")
    print(f"  - Fourier/decision_boundary.png")
    print(f"  - Fourier/kernel_matrix.png")
    print(f"  - Fourier/predictions_comparison.png")
    print(f"  - QAOA/decision_boundary.png")
    print(f"  - QAOA/kernel_matrix.png")
    print(f"  - QAOA/predictions_comparison.png")
    print(f"  - overall_comparison.png")
    print(f"\n{'='*80}\n")
    
    return classical_results, fourier_results, qaoa_results


if __name__ == "__main__":
    # Execute main pipeline
    classical_results, fourier_results, qaoa_results = main()