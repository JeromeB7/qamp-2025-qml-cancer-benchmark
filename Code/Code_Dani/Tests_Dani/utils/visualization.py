"""
Visualization utilities for classification results

Author: Quantum ML Implementation
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path as osp

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
    
    Args:
        X: 2D features for visualization
        y: Labels
        clf: Trained classifier
        method_name: Name of the method (for title)
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    scatter = ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', linewidth=1)
    
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title(f'Decision Boundary - {method_name}', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_labels = ['Malignant (0)', 'Benign (1)']
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Decision boundary saved to: {save_path}")


def plot_kernel_matrix(K, y, method_name, save_path):
    """
    Plot and save kernel matrix
    
    Args:
        K: Kernel matrix
        y: Labels for sorting
        method_name: Name of the method
        save_path: Path to save the figure
    """
    K_sorted = mddt.sort_K(K, y)
    mddt.plot_kernel_matrix(K_sorted, title=f'{method_name} Kernel Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Kernel matrix saved to: {save_path}")


def visualize_overall_results(classical_results, fourier_results, qaoa_results, 
                              output_dir):
    """
    Create overall comparison visualization
    
    Args:
        classical_results: Results from classical SVM
        fourier_results: Results from Fourier quantum kernel
        qaoa_results: Results from QAOA quantum kernel
        output_dir: Directory to save figure
    """
    print(f"\n{'='*80}")
    print(f"CREATING OVERALL COMPARISON VISUALIZATION")
    print(f"{'='*80}")
    
    # Extract accuracies
    methods = ['Classical\nLinear SVM', 'Quantum\nFourier', 'Quantum\nQAOA']
    accuracies = [
        classical_results['test_accuracy'],
        fourier_results['accuracy'],
        qaoa_results['accuracy']
    ]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy comparison (top left)
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(methods, accuracies, color=['blue', 'green', 'red'], alpha=0.7)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2-4. Confusion matrices (bottom row)
    results_list = [classical_results, fourier_results, qaoa_results]
    titles = ['Classical Linear SVM', 'Quantum Fourier', 'Quantum QAOA']
    
    for idx, (results, title) in enumerate(zip(results_list, titles)):
        ax = fig.add_subplot(gs[1, idx])
        cm = results['confusion_matrix']
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add labels
        classes = ['Malignant', 'Benign']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        
        # Add text annotations
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
                fontsize=16, fontweight='bold', y=0.98)
    
    save_path = osp.join(output_dir, 'overall_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overall comparison saved to: {save_path}")
