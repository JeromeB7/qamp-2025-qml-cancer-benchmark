"""
quantum_kernels.py
Definición de Clases de Modelos Cuánticos.
Corregido para:
1. Soportar argumento 'shots'.
2. Soportar argumento 'C' en .fit() (via **kwargs).
3. Solucionar Mode Collapse (class_weight='balanced').
4. QuMO estable (Kernel + Moiré).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, QFT, RealAmplitudes
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

# --- FEATURE MAPS ---
class FourierFeatureMap:
    def __init__(self, n_features: int, reps: int = 2, entanglement: str = 'linear'):
        self.feature_map = ZZFeatureMap(n_features, reps=reps, entanglement=entanglement, insert_barriers=False).decompose()
    def get_circuit(self): return self.feature_map

class QAOABasedFeatureMap:
    def __init__(self, n_features: int, reps: int = 2, entanglement: str = 'linear'):
        self.n_features = n_features; self.reps = reps; self.entanglement = entanglement
        self._build_circuit()
    def _build_circuit(self):
        x = ParameterVector('x', self.n_features)
        qc = QuantumCircuit(self.n_features); qc.h(range(self.n_features))
        for _ in range(self.reps):
            for i in range(self.n_features): qc.rz(2 * x[i], i)
            if self.entanglement == 'linear':
                for i in range(self.n_features - 1):
                    qc.cx(i, i+1); qc.rz(2 * x[i] * x[i+1], i+1); qc.cx(i, i+1)
            for i in range(self.n_features): qc.rx(np.pi/2, i)
        self.feature_map = qc
    def get_circuit(self): return self.feature_map

# --- CLASES PRINCIPALES ---

class QuantumKernelSVM:
    def __init__(self, feature_map_type: str = 'fourier', n_features: int = 4, 
                 reps: int = 2, entanglement: str = 'linear', batch_size: int = 50, 
                 shots: int = 1024):
        self.feature_map_type = feature_map_type
        self.n_features = n_features
        self.batch_size = batch_size 
        self.shots = shots
        self.scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
        self.sampler = SamplerV2()
        self.sampler.options.default_shots = self.shots 
        
        if feature_map_type == 'fourier': self.feature_mapper = FourierFeatureMap(n_features, reps, entanglement)
        elif feature_map_type == 'qaoa': self.feature_mapper = QAOABasedFeatureMap(n_features, reps, entanglement)
        else: raise ValueError(f"Unknown: {feature_map_type}")
        
        fidelity = ComputeUncompute(sampler=self.sampler)
        self.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_mapper.get_circuit(), fidelity=fidelity)
        # FIX: class_weight='balanced' para evitar que prediga siempre Benigno
        self.svm = SVC(kernel='precomputed', class_weight='balanced')
        self.X_train_scaled = None

    def _compute_kernel_matrix_batched(self, X1, X2=None):
        if X2 is None: X2 = X1
        n1, n2 = X1.shape[0], X2.shape[0]
        kernel_matrix = np.zeros((n1, n2))
        print(f"   > Kernel ({n1}x{n2}) Batched (Shots: {self.shots})...")
        for i in range(0, n1, self.batch_size):
            i_end = min(i + self.batch_size, n1)
            for j in range(0, n2, self.batch_size):
                j_end = min(j + self.batch_size, n2)
                kernel_matrix[i:i_end, j:j_end] = self.quantum_kernel.evaluate(X1[i:i_end], X2[j:j_end])
        return kernel_matrix

    def fit(self, X_train, y_train, **kwargs):
        # FIX: Aceptamos **kwargs para que el argumento 'C' no rompa nada
        if 'C' in kwargs: self.svm.C = kwargs['C']
        
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        K_train = self._compute_kernel_matrix_batched(self.X_train_scaled)
        self.svm.fit(K_train, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        K_test = self._compute_kernel_matrix_batched(X_test_scaled, self.X_train_scaled)
        return self.svm.predict(K_test)
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {'accuracy': accuracy_score(y_test, preds), 'confusion_matrix': confusion_matrix(y_test, preds), 'predictions': preds, 'classification_report': ""}

class QuMO:
    """ QuMO V2: Usa Moiré + Kernel SVM (Estable) """
    def __init__(self, n_features: int, reps: int = 2, entanglement: str = 'linear', batch_size: int = 50, shots: int = 1024):
        self.n_features = n_features
        self.freq1 = 0.8; self.freq2 = 3.5
        # Reutilizamos la lógica robusta de SVM
        self.qsvm = QuantumKernelSVM('fourier', n_features, reps, entanglement, batch_size, shots)
        self.moire_scaler = MinMaxScaler(feature_range=(0, np.pi)) 
        self.svm_readout = SVC(kernel='precomputed', class_weight='balanced', C=1.0)

    def _apply_moire(self, X):
        X_base = self.moire_scaler.fit_transform(X) if not hasattr(self.moire_scaler, 'n_samples_seen_') else self.moire_scaler.transform(X)
        return X_base + (np.sin(self.freq1 * X_base) * np.cos(self.freq2 * X_base))

    def fit(self, X, y, **kwargs):
        print(f"\n[QuMO] Moiré Transform & Training...")
        if 'C' in kwargs: self.svm_readout.C = kwargs['C']
        X_trans = self._apply_moire(X)
        
        self.qsvm.X_train_scaled = self.qsvm.scaler.fit_transform(X_trans)
        K_train = self.qsvm._compute_kernel_matrix_batched(self.qsvm.X_train_scaled)
        self.svm_readout.fit(K_train, y)

    def predict(self, X):
        X_trans = self._apply_moire(X)
        X_test_scaled = self.qsvm.scaler.transform(X_trans)
        K_test = self.qsvm._compute_kernel_matrix_batched(X_test_scaled, self.qsvm.X_train_scaled)
        return self.svm_readout.predict(K_test)

    def score(self, X, y): return accuracy_score(y, self.predict(X))
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {'accuracy': accuracy_score(y_test, preds), 'confusion_matrix': confusion_matrix(y_test, preds), 'predictions': preds}

class QK_Reservoir:
    """ Reservoir V2: Logistic Regression Balanceada """
    def __init__(self, n_features: int, reps: int = 2, entanglement: str = 'linear', reservoir_depth: int = 6, seed: int = 42, shots: int = 1024):
        self.rng = np.random.default_rng(seed)
        self.input_scaler = MinMaxScaler((0, 2*np.pi)); self.res_scaler = StandardScaler()
        self.encoding = ZZFeatureMap(n_features, reps=reps, entanglement=entanglement)
        self.reservoir = self._create_reservoir(n_features, reservoir_depth)
        self.full_circuit = self.encoding.compose(self.reservoir).decompose()
        # FIX: class_weight='balanced'
        self.readout = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.estimator = EstimatorV2()
        self.observables = [SparsePauliOp("I"*i+"Z"+"I"*(n_features-1-i)) for i in range(n_features)]

    def _create_reservoir(self, n, depth):
        qc = QuantumCircuit(n)
        for _ in range(depth):
            for i in range(n): qc.u(*self.rng.random(3)*2*np.pi, i)
            for i in range(n): 
                for j in range(i+1, n): 
                    if self.rng.random() > 0.5: qc.cx(i, j)
            qc.barrier()
        return qc

    def _get_states(self, X):
        all_states = []
        for i in range(0, len(X), 50):
            job = self.estimator.run([(self.full_circuit, self.observables, x) for x in X[i:i+50]])
            all_states.extend([res.data.evs for res in job.result()])
        return np.array(all_states)

    def fit(self, X, y, **kwargs):
        print(f"\n[Reservoir] Processing States...")
        states = self._get_states(self.input_scaler.fit_transform(X))
        self.readout.fit(self.res_scaler.fit_transform(states), y)

    def predict(self, X):
        states = self._get_states(self.input_scaler.transform(X))
        return self.readout.predict(self.res_scaler.transform(states))

    def score(self, X, y): return accuracy_score(y, self.predict(X))
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {'accuracy': accuracy_score(y_test, preds), 'confusion_matrix': confusion_matrix(y_test, preds), 'predictions': preds}

class QK_QFT:
    def __init__(self, n_features: int, reps: int = 2, entanglement: str = 'linear', ansatz_reps: int = 3, optimizer=None, shots=1024):
        self.scaler = MinMaxScaler((0, 2*np.pi))
        feature_map = ZZFeatureMap(n_features, reps=reps).compose(QFT(n_features, do_swaps=False))
        # FIX: Añadido método .score() abajo para evitar AttributeError
        self.vqc = VQC(feature_map=feature_map, ansatz=RealAmplitudes(n_features, reps=ansatz_reps), optimizer=COBYLA(maxiter=200), sampler=Sampler())

    def fit(self, X, y, **kwargs):
        self.vqc.fit(self.scaler.fit_transform(X), y)

    def predict(self, X): return self.vqc.predict(self.scaler.transform(X))
    def score(self, X, y): return self.vqc.score(self.scaler.transform(X), y)
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {'accuracy': accuracy_score(y_test, preds), 'confusion_matrix': confusion_matrix(y_test, preds), 'predictions': preds}