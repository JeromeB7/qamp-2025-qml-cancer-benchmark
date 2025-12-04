"""
utils/quantum_training.py
Utility functions for training quantum models.
FIXED: Now routes 'qft', 'qumo', etc. to their correct classes instead of forcing everything into QuantumKernelSVM.
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os.path as osp
import sys
import os

# Import path handling
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
PARENT_DIR = osp.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# IMPORTAR TODAS LAS CLASES DE TU ARCHIVO quantum_kernels.py
from quantum_kernels import QuantumKernelSVM, QK_QFT, QuMO, QK_Reservoir
from utils.visualization import (
    plot_decision_boundary,
    plot_kernel_matrix,
    plot_confusion_matrix
)

def train_quantum_kernel_individual(X_train, y_train, X_test, y_test,
                                   X_train_2d, X_test_2d,
                                   n_qubits, reps, kernel_type, output_dir, shots=1024):
    """
    Función Maestra de Entrenamiento.
    Actúa como un 'Dispatcher': selecciona la clase correcta basada en 'kernel_type'.
    """
    
    # 1. Normalizar el nombre del kernel/modelo
    k_type = kernel_type.lower()
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {k_type.upper()}")
    print(f"{'='*80}")
    
    model = None
    
    # ======================================================
    # SELECCIÓN DE CLASE (DISPATCHER LOGIC)
    # ======================================================
    
    # CASO 1: QFT (VQC)
    if k_type == 'qft':
        print(" -> Instantiating QK_QFT...")
        model = QK_QFT(n_features=n_qubits, reps=reps, shots=shots)
        
    # CASO 2: QuMO (Moiré + SVM)
    elif k_type == 'qumo':
        print(" -> Instantiating QuMO...")
        model = QuMO(n_features=n_qubits, reps=reps, shots=shots)
        
    # CASO 3: Reservoir Computing
    elif k_type == 'reservoir':
        print(" -> Instantiating QK_Reservoir...")
        model = QK_Reservoir(n_features=n_qubits, reps=reps, shots=shots)
        
    # CASO 4: Standard SVM Kernels (Fourier / QAOA)
    elif k_type in ['fourier', 'qaoa']:
        print(f" -> Instantiating QuantumKernelSVM ({k_type})...")
        model = QuantumKernelSVM(
            feature_map_type=k_type,
            n_features=n_qubits,
            reps=reps,
            entanglement='linear',
            shots=shots
        )
    else:
        raise ValueError(f"Unknown model/kernel type: {kernel_type}")

    # ======================================================
    # ENTRENAMIENTO
    # ======================================================
    
    # Todos nuestros modelos corregidos aceptan fit(X, y, **kwargs)
    print(" -> Fitting model...")
    model.fit(X_train, y_train, C=1.0)
    
    # ======================================================
    # EVALUACIÓN
    # ======================================================
    print(" -> Evaluating...")
    results = model.evaluate(X_test, y_test)
    
    print(f"\n{k_type.upper()} Results:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    # ======================================================
    # VISUALIZACIÓN
    # ======================================================
    method_dir = osp.join(output_dir, k_type.capitalize())
    os.makedirs(method_dir, exist_ok=True) 
    
    print(f"\nGenerating visualizations in {method_dir}...")
    
    # 1. Matriz de Confusión
    cm_title = f'Confusion Matrix - {k_type.upper()}'
    cm_save_path = osp.join(method_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels=np.unique(y_test), title=cm_title, 
                          save_path=cm_save_path, cm=results['confusion_matrix'])
    
    # 2. Decision Boundary (Requiere re-entrenar un modelo pequeño 2D)
    print(f" -> Training 2D visualization model...")
    
    # Instanciamos el modelo 2D usando la misma lógica
    if k_type == 'qft':
        model_2d = QK_QFT(n_features=2, reps=reps, shots=shots)
    elif k_type == 'qumo':
        model_2d = QuMO(n_features=2, reps=reps, shots=shots)
    elif k_type == 'reservoir':
        model_2d = QK_Reservoir(n_features=2, reps=reps, shots=shots)
    else: # Fourier/QAOA
        model_2d = QuantumKernelSVM(feature_map_type=k_type, n_features=2, reps=reps, shots=shots)
        
    model_2d.fit(X_train_2d, y_train, C=1.0)
    
    plot_decision_boundary(
        X=X_train_2d, y=y_train, clf=model_2d,
        method_name=f"{k_type.upper()} (2D Projection)",
        save_path=osp.join(method_dir, "decision_boundary.png")
    )
    
    # 3. Kernel Matrix (SOLO si es un modelo basado en Kernel explícito)
    # QuantumKernelSVM tiene .quantum_kernel
    # QuMO tiene .qsvm.quantum_kernel
    
    if k_type in ['fourier', 'qaoa']:
        print(" -> Plotting Kernel Matrix...")
        plot_kernel_matrix(
            model, X_train, y_train,
            title=f"{k_type.upper()} Kernel Matrix",
            save_path=osp.join(method_dir, "kernel_matrix.png")
        )
    elif k_type == 'qumo':
        # QuMO es especial, usa un kernel interno. Podemos intentar plotearlo accediendo a su qsvm
        try:
            print(" -> Plotting Kernel Matrix (QuMO Internal)...")
            plot_kernel_matrix(
                model.qsvm, model.qsvm.X_train_scaled, y_train, # Usamos los datos escalados internos
                title=f"QuMO Internal Kernel Matrix",
                save_path=osp.join(method_dir, "kernel_matrix.png")
            )
        except Exception as e:
            print(f"Could not plot QuMO kernel: {e}")
    else:
        print(f" -> Kernel matrix plot skipped (Not applicable for {k_type})")
    
    print(f"Visualizations saved.")
    
    return {
        'model': model,
        'accuracy': results['accuracy'], 
        'confusion_matrix': results['confusion_matrix'],
        'predictions': results['predictions'],
    }


# Mantenemos esta función por si algún script viejo la llama, 
# pero la redirigimos a la nueva lógica para evitar duplicar código.
def train_vqc_classifier(X_train, y_train, X_test, y_test, X_train_2d, X_test_2d, n_qubits, reps, output_dir):
    # Esta función solía ser solo para QFT/VQC.
    # Asumimos que si se llama a esto, se quiere un QK_QFT por defecto o se pasa la clase.
    # Para no romper nada, simplemente llamamos a la lógica de QFT.
    return train_quantum_kernel_individual(
        X_train, y_train, X_test, y_test, X_train_2d, X_test_2d,
        n_qubits, reps, 'qft', output_dir
    )