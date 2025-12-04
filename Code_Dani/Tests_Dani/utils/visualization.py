"""
Visualization utilities for classification results

Author: Quantum ML Implementation
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path as osp

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add path to custom modules
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../../"))

import quantum_feature_map.modudata as mddt


def make_meshgrid(x, y, h=0.02):
    """Create a mesh grid for decision boundary visualization"""
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot decision boundary contours"""
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_decision_boundary(X, y, clf, method_name, save_path):
    """
    Plot and save decision boundary
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    X0, X1 = X[:, 0], X[:, 1]
    
    grid_points = 50
    x_min, x_max = X0.min() - 0.5, X0.max() + 0.5
    y_min, y_max = X1.min() - 0.5, X1.max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points),
        np.linspace(y_min, y_max, grid_points)
    )
    
    print(f"\nGenerando contorno con {xx.ravel().shape[0]} puntos de predicción...")
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    print("Contorno generado.")
    
    scatter = ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', linewidth=1)
    
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title(f'Decision Boundary - {method_name}', fontsize=14, fontweight='bold')
    
    legend_labels = ['Malignant (0)', 'Benign (1)'] 
    handles, _ = scatter.legend_elements()
    if len(handles) == len(legend_labels):
        ax.legend(handles=handles, labels=legend_labels, loc='best')
    else:
        print(f"Warning: Mismatch in legend elements. Found {len(handles)} handles, expected {len(legend_labels)}.")

    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Decision boundary saved to: {save_path}")


def plot_kernel_matrix(kernel_object, X, y, title, save_path=None, precomputed_matrix=None):
    """
    Calcula, plotea y guarda la matriz del kernel.
    """
    print("  Calculando la matriz del kernel para la visualización...")
    
    if precomputed_matrix is not None:
        K = precomputed_matrix
        print("  Usando matriz de kernel pre-calculada.")
    else:
        try:
            K = kernel_object.calculate_kernel(X)
        except AttributeError:
            print(f"No se encontró el método 'calculate_kernel(X)' en {type(kernel_object)}.")
            if hasattr(kernel_object, 'kernel_matrix_'):
                print("... Usando 'kernel_object.kernel_matrix_' como alternativa.")
                K = kernel_object.kernel_matrix_
            else:
                print("No se puede generar la matriz del kernel. Abortando plot.")
                return
        except Exception as e:
            print(f"Error durante el cálculo del kernel: {e}")
            return

    print("  Ordenando la matriz del kernel...")
    K_sorted = mddt.sort_K(K, y)
    
    print("  Generando la gráfica de la matriz del kernel...")
    mddt.plot_kernel_matrix(K_sorted, title=title) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Kernel matrix saved to: {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true=None, y_pred=None, labels=None, title='Confusion Matrix', save_path=None, cm=None):
    """
    Plotea y guarda una matriz de confusión.
    """
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Debe proporcionar (y_true, y_pred) o cm.")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if labels is None:
        labels = np.arange(cm.shape[0])

    display_labels = ['Malignant', 'Benign'] 
    if len(labels) != 2:
        display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix saved to: {save_path}")
    else:
        plt.show()


def visualize_overall_results(classical_lin_results, classical_rbf_results, 
                              moduqusvm_quantum_results, qumo_results,
                              fourier_results, qaoa_results, 
                              reservoir_results, qft_results,
                              output_dir):
    """
    Create overall comparison visualization
    
    Args:
        ... (todos los 8 diccionarios de resultados) ...
        output_dir: Directory to save figure
    """
    print(f"\n{'='*80}")
    print(f"CREATING OVERALL COMPARISON VISUALIZATION (8 MODELS)")
    print(f"{'='*80}")
    
    # Helper para obtener accuracy
    def get_acc(result_dict):
        if 'test_accuracy' in result_dict:
            return result_dict['test_accuracy']
        elif 'accuracy' in result_dict:
            return result_dict['accuracy']
        return 0.0

    # --- LISTAS ACTUALIZADAS (8 MODELOS) ---
    results_list = [
        classical_lin_results, classical_rbf_results, 
        moduqusvm_quantum_results, qumo_results,
        fourier_results, qaoa_results, 
        reservoir_results, qft_results
    ]
    
    titles = [
        'Classical\nLinear', 'Classical\nRBF', 'Moduqusvm\nQuantum', 'Moduqusvm\nQuMO',
        'QKernels\nFourier', 'QKernels\nQAOA', 'QKernels\nReservoir', 'QKernels\nQFT'
    ]
    
    accuracies = [get_acc(res) for res in results_list]
    colors = ['blue', 'cyan', 'purple', 'magenta', 'green', 'red', 'orange', 'brown']
    
    # --- GRÁFICO ACTUALIZADO (8 COLUMNAS) ---
    fig = plt.figure(figsize=(32, 10)) # SÚPER ancha
    gs = fig.add_gridspec(2, 8, hspace=0.5, wspace=0.3)
    
    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(titles, accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Classification Accuracy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2-9. Confusion matrices
    for idx, (results, title) in enumerate(zip(results_list, titles)):
        ax = fig.add_subplot(gs[1, idx])
        
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
        else:
            print(f"Warning: No se encontró 'confusion_matrix' en {title}")
            cm = np.zeros((2,2)) 

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        classes = ['Malignant', 'Benign']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9, rotation=45, ha='right')
        ax.set_yticklabels(classes, fontsize=9)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    plt.suptitle('Breast Cancer Classification: Classical vs Quantum Kernels', 
                fontsize=18, fontweight='bold', y=1.02)
    
    save_path = osp.join(output_dir, 'overall_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overall comparison saved to: {save_path}")


def plot_pca_variance_analysis(pca, n_components_selected, output_dir):
    """
    Plot PCA variance analysis showing explained variance vs number of components
    """
    # ... (Esta función no cambia) ...
    n_components_total = len(pca.explained_variance_ratio_)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1
    ax1.bar(range(1, n_components_total + 1), 
            pca.explained_variance_ratio_ * 100,
            alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=n_components_selected, color='red', linestyle='--', 
                linewidth=2, label=f'Selected: {n_components_selected} PCs')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Variance Explained by Each PC', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, n_components_total + 1))
    
    # Plot 2
    ax2.plot(range(1, n_components_total + 1), 
             cumulative_variance * 100,
             marker='o', markersize=6, linewidth=2, 
             color='darkgreen', label='Cumulative Variance')
    ax2.axhline(y=90, color='orange', linestyle='--', 
                linewidth=2, label='90% Threshold')
    ax2.axvline(x=n_components_selected, color='red', linestyle='--', 
                linewidth=2, label=f'Selected: {n_components_selected} PCs')
    ax2.fill_between(range(1, n_components_total + 1), 
                      cumulative_variance * 100, 
                      alpha=0.2, color='darkgreen')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Variance vs Components', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, n_components_total + 1))
    ax2.set_ylim([0, 105])
    
    variance_at_selection = cumulative_variance[n_components_selected - 1] * 100
    ax2.annotate(f'{variance_at_selection:.2f}%',
                xy=(n_components_selected, variance_at_selection),
                xytext=(n_components_selected + 1, variance_at_selection - 10),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                               color='red', lw=2))
    
    plt.tight_layout()
    
    save_path = osp.join(output_dir, "pca_variance_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPCA variance analysis saved: {save_path}")
    plt.close()
    
    print(f"\n{'='*70}")
    print("PCA VARIANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Total components available: {n_components_total}")
    print(f"Components selected: {n_components_selected}")
    print(f"Variance explained by selected components: {variance_at_selection:.2f}%")
    print(f"\nVariance per component (first {min(10, n_components_total)}):")
    for i in range(min(10, n_components_total)):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}% "f"(Cumulative: {cumulative_variance[i]*100:.2f}%)")
