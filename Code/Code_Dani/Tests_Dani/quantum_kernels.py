"""
Quantum Kernel Implementation for Breast Cancer Classification
================================================================

This module implements quantum feature maps for Support Vector Machine classification,
specifically:
1. Fourier Feature Map (ZZFeatureMap) - Standard quantum embedding
2. QAOA-based Feature Map - Quantum Approximate Optimization Algorithm inspired encoding

Based on quantum kernel methods for medical image classification.

Author: Quantum ML Implementation
Date: November 2025
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FourierFeatureMap:
    """
    Fourier Feature Map using ZZFeatureMap from Qiskit
    
    This creates a quantum feature map based on Pauli-Z rotations and 
    ZZ entangling gates. It's one of the standard quantum embeddings.
    """
    def __init__(self, n_features: int, reps: int = 2, entanglement: str = 'linear'):
        """
        Initialize Fourier Feature Map
        
        Args:
            n_features: Number of classical features (qubits)
            reps: Number of repetitions (circuit depth)
            entanglement: Type of entanglement ('linear', 'full', 'circular')
        """
        self.n_features = n_features
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map = None
        self._build_circuit()

    def _build_circuit(self):
        """Build and decompose the feature map circuit"""
        fm = ZZFeatureMap(
            feature_dimension=self.n_features,
            reps=self.reps,
            entanglement=self.entanglement,
            insert_barriers=False
        )
        # Decompose to ensure no composite instructions
        self.feature_map = fm.decompose().decompose()

    def get_circuit(self) -> QuantumCircuit:
        """Return the decomposed quantum circuit"""
        return self.feature_map

    def visualize_circuit(self):
        """Print circuit structure"""
        print(f"\nFourier Feature Map Circuit ({self.n_features} qubits, {self.reps} reps):")
        print(self.feature_map)


class QAOABasedFeatureMap:
    """
    QAOA-Based Feature Map Implementation
    
    This feature map is inspired by the Quantum Approximate Optimization Algorithm (QAOA).
    QAOA alternates between:
    1. Problem Hamiltonian: Encodes the classical data
    2. Mixer Hamiltonian: Creates quantum superposition
    
    Structure:
    - Problem layer: RZ rotations encoding data and ZZ interactions
    - Mixer layer: RX rotations for quantum exploration
    - Multiple layers increase expressiveness
    
    This approach is particularly suited for optimization-inspired feature encoding,
    where the quantum state evolves through a series of parameterized unitaries.
    """
    
    def __init__(self, n_features: int, reps: int = 2, entanglement: str = 'linear'):
        """
        Initialize QAOA-based Feature Map
        
        Args:
            n_features: Number of classical features (qubits)
            reps: Number of QAOA layers (depth)
            entanglement: Connectivity pattern between qubits
        """
        self.n_features = n_features
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map = None
        self._build_circuit()
    
    def _build_circuit(self):
        """
        Construct QAOA-inspired quantum feature map
        
        Each QAOA layer consists of:
        1. Cost layer (Problem Hamiltonian): 
           - RZ gates with data encoding
           - ZZ entangling gates between connected qubits
        2. Mixer layer:
           - RX gates for quantum superposition
        """
        # Create parameter vector for data encoding
        x = ParameterVector('x', self.n_features)
        
        # Initialize quantum circuit
        qc = QuantumCircuit(self.n_features)
        
        # Initial superposition
        qc.h(range(self.n_features))
        
        # QAOA layers
        for rep in range(self.reps):
            # === COST LAYER (Problem Hamiltonian) ===
            # Single qubit rotations encoding classical data
            for i in range(self.n_features):
                qc.rz(2 * x[i], i)
            
            # Two-qubit interactions (entanglement)
            if self.entanglement == 'linear':
                # Connect adjacent qubits
                for i in range(self.n_features - 1):
                    qc.cx(i, i + 1)
                    qc.rz(2 * x[i] * x[i + 1], i + 1)
                    qc.cx(i, i + 1)
            
            elif self.entanglement == 'full':
                # Connect all qubit pairs
                for i in range(self.n_features):
                    for j in range(i + 1, self.n_features):
                        qc.cx(i, j)
                        qc.rz(2 * x[i] * x[j], j)
                        qc.cx(i, j)
            
            elif self.entanglement == 'circular':
                # Connect adjacent qubits + last to first
                for i in range(self.n_features):
                    j = (i + 1) % self.n_features
                    qc.cx(i, j)
                    qc.rz(2 * x[i] * x[j], j)
                    qc.cx(i, j)
            
            # === MIXER LAYER ===
            # Quantum exploration through X rotations
            for i in range(self.n_features):
                qc.rx(np.pi / 2, i)
        
        self.feature_map = qc
    
    def get_circuit(self) -> QuantumCircuit:
        """Return the quantum circuit"""
        return self.feature_map
    
    def visualize_circuit(self):
        """Print circuit for inspection"""
        print(f"\nQAOA-based Feature Map Circuit ({self.n_features} qubits, {self.reps} reps):")
        print(self.feature_map.decompose())


class QuantumKernelSVM:
    """
    Quantum Kernel Support Vector Machine
    
    This class wraps quantum feature maps with classical SVM training.
    The quantum kernel is computed as:
        K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
    
    Where φ(x) is the quantum feature map that encodes classical data into quantum states.
    The fidelity (overlap) between quantum states serves as a similarity measure.
    
    Workflow:
    1. Encode data into quantum states using feature map
    2. Compute kernel matrix via state fidelities
    3. Train classical SVM on quantum kernel
    4. Predict using quantum kernel evaluations
    """
    
    def __init__(self, feature_map_type: str = 'fourier', n_features: int = 4, 
                 reps: int = 2, entanglement: str = 'linear', shots: int = 1024):
        """
        Initialize Quantum Kernel SVM
        
        Args:
            feature_map_type: 'fourier' or 'qaoa' - type of quantum encoding
            n_features: Number of classical features
            reps: Circuit depth (repetitions)
            entanglement: Qubit connectivity pattern
            shots: Number of measurement shots (higher = more accurate)
        """
        self.feature_map_type = feature_map_type
        self.n_features = n_features
        self.reps = reps
        self.entanglement = entanglement
        self.shots = shots
        
        # Initialize quantum backend (simulator)
        from qiskit_aer.primitives import SamplerV2
        self.backend = SamplerV2()
        
        # Build feature map
        if feature_map_type == 'fourier':
            self.feature_mapper = FourierFeatureMap(n_features, reps, entanglement)
        elif feature_map_type == 'qaoa':
            self.feature_mapper = QAOABasedFeatureMap(n_features, reps, entanglement)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
        # Setup quantum kernel
        self._setup_kernel()
        
        # Classical SVM (will be trained on quantum kernel)
        self.svm = None
        self.X_train = None
    
    def _setup_kernel(self):
        """
        Setup quantum kernel for computing state fidelities
        
        Uses the ComputeUncompute method which:
        1. Prepares state |φ(x_i)⟩
        2. Prepares state |φ(x_j)⟩ and applies inverse
        3. Measures overlap (fidelity)
        """
        fidelity = ComputeUncompute(sampler=self.backend)
        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_mapper.get_circuit(),
            fidelity=fidelity
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0):
        """
        Train the quantum kernel SVM
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels [n_samples]
            C: SVM regularization parameter (higher = less regularization)
        
        Process:
        1. Store training data
        2. Compute quantum kernel matrix for training set
        3. Train classical SVM on precomputed kernel
        """
        self.X_train = X_train
        
        # Compute quantum kernel matrix K_train[i,j] = |⟨φ(x_i)|φ(x_j)⟩|²
        print(f"\n{'='*70}")
        print(f"Training Quantum Kernel SVM ({self.feature_map_type.upper()} feature map)")
        print(f"{'='*70}")
        print(f"Computing quantum kernel matrix for {len(X_train)} training samples...")
        
        K_train = self.quantum_kernel.evaluate(x_vec=X_train)
        
        print(f"Kernel matrix computed: shape {K_train.shape}")
        print(f"Kernel matrix statistics:")
        print(f"  - Min: {K_train.min():.4f}, Max: {K_train.max():.4f}")
        print(f"  - Mean: {K_train.mean():.4f}, Std: {K_train.std():.4f}")
        
        # Train SVM with precomputed quantum kernel
        self.svm = SVC(kernel='precomputed', C=C)
        self.svm.fit(K_train, y_train)
        
        print(f"SVM training complete!")
        print(f"  - Support vectors: {len(self.svm.support_)}/{len(X_train)}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test data
        
        Args:
            X_test: Test features [n_samples_test, n_features]
        
        Returns:
            Predicted labels [n_samples_test]
        
        Process:
        1. Compute quantum kernel between test and training data
        2. Use trained SVM to predict
        """
        if self.svm is None or self.X_train is None:
            raise ValueError("Model must be trained before prediction!")
        
        # Compute kernel between test and training data
        print(f"\nComputing quantum kernel for {len(X_test)} test samples...")
        K_test = self.quantum_kernel.evaluate(x_vec=X_test, y_vec=self.X_train)
        
        # Predict using trained SVM
        predictions = self.svm.predict(K_test)
        return predictions
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute accuracy on test set
        
        Args:
            X_test: Test features
            y_test: True test labels
        
        Returns:
            Accuracy score [0, 1]
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Comprehensive evaluation of model performance
        
        Args:
            X_test: Test features
            y_test: True labels
        
        Returns:
            Dictionary with accuracy, confusion matrix, and classification report
        """
        predictions = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'predictions': predictions
        }
        
        return results


def compare_quantum_kernels(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           n_features: int, reps: int = 2) -> dict:
    """
    Compare Fourier and QAOA-based quantum feature maps
    
    This function trains both quantum kernel types and compares their performance
    on the same dataset, providing insights into which encoding works better.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_features: Number of features to use
        reps: Circuit depth
    
    Returns:
        Dictionary with results for both methods
    """
    results = {}
    
    # Test both feature map types
    for fm_type in ['fourier', 'qaoa']:
        print(f"\n{'#'*70}")
        print(f"# Testing {fm_type.upper()} Feature Map")
        print(f"{'#'*70}")
        
        # Initialize and train model
        qksvm = QuantumKernelSVM(
            feature_map_type=fm_type,
            n_features=n_features,
            reps=reps,
            entanglement='linear'
        )
        
        qksvm.fit(X_train, y_train)
        
        # Evaluate on test set
        eval_results = qksvm.evaluate(X_test, y_test)
        
        print(f"\n{fm_type.upper()} Results:")
        print(f"{'='*70}")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"\nConfusion Matrix:")
        print(eval_results['confusion_matrix'])
        print(f"\nClassification Report:")
        print(eval_results['classification_report'])
        
        results[fm_type] = {
            'model': qksvm,
            'accuracy': eval_results['accuracy'],
            'confusion_matrix': eval_results['confusion_matrix'],
            'predictions': eval_results['predictions']
        }
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Fourier Feature Map Accuracy: {results['fourier']['accuracy']:.4f}")
    print(f"QAOA-based Feature Map Accuracy: {results['qaoa']['accuracy']:.4f}")
    
    best = 'fourier' if results['fourier']['accuracy'] > results['qaoa']['accuracy'] else 'qaoa'
    print(f"\nBest performing method: {best.upper()}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    """
    Demonstration of quantum kernel methods
    
    This example shows how to:
    1. Create synthetic data
    2. Build and visualize quantum feature maps
    3. Train quantum kernel SVMs
    4. Compare different quantum encodings
    """
    print("="*70)
    print("Quantum Kernel SVM - Feature Map Demonstration")
    print("="*70)
    
    # Create synthetic dataset for testing
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, n_informative=3,
                               n_redundant=0, n_classes=2, random_state=42)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Visualize circuits
    print("\n" + "="*70)
    print("QUANTUM CIRCUIT ARCHITECTURES")
    print("="*70)
    
    fourier_fm = FourierFeatureMap(n_features=4, reps=2)
    fourier_fm.visualize_circuit()
    
    qaoa_fm = QAOABasedFeatureMap(n_features=4, reps=2)
    qaoa_fm.visualize_circuit()
    
    # Compare both methods
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    results = compare_quantum_kernels(
        X_train, y_train, X_test, y_test,
        n_features=4, reps=2
    )
    
    print("\n" + "="*70)
    print("Quantum Kernel SVM demonstration complete!")
    print("="*70)