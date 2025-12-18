import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms.optimizers import SPSA

def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")
    X = df.drop(columns=["diagnosis"]).apply(pd.to_numeric, errors="coerce")
    y = df["diagnosis"].map({"M": 1, "B": 0}).values
    return X.values, y
    
def trainable_data_reuploading_map(n_features, reps=2):
    qc = QuantumCircuit(n_features)
    x_params = ParameterVector("x", n_features)
    theta_params = ParameterVector("θ", reps * n_features)
    idx = 0
    for r in range(reps):
        for i in range(n_features):
            qc.ry(x_params[i], i)
        for i in range(n_features):
            qc.rz(theta_params[idx], i)
            idx += 1
        for i in range(n_features):
            qc.cz(i, (i + 1) % n_features)
    return qc, theta_params


def build_trainable_kernel(n_features, reps=2):
    feature_map, training_parameters = trainable_data_reuploading_map(n_features, reps)

    return TrainableFidelityStatevectorKernel(
        feature_map=feature_map,
        training_parameters=training_parameters
    )


def run_trainable_qsvc(X, y, n_components=4, reps=2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = MinMaxScaler((0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use PCA with 4 components for quantum, but also fit a 2D PCA for visualization
    pca_quantum = PCA(n_components=n_components)
    X_train_pca = pca_quantum.fit_transform(X_train)
    X_test_pca = pca_quantum.transform(X_test)

    pca_2d = PCA(n_components=2)
    X_train_2d = pca_2d.fit_transform(X_train)
    X_test_2d = pca_2d.transform(X_test)

    qkernel = build_trainable_kernel(n_components, reps=reps)

    optimizer = SPSA(maxiter=20)

    print("Optimizing quantum kernel parameters (fast statevector method)...")
    trainer = QuantumKernelTrainer(
        quantum_kernel=qkernel,
        loss=SVCLoss(C=1.0),
        optimizer=optimizer,
        initial_point=np.random.uniform(0, 2 * np.pi, len(qkernel.training_parameters))
    )

    trainer_result = trainer.fit(X_train_pca, y_train)
    trained_kernel = trainer_result.quantum_kernel

    print("Kernel training completed.")

    qsvc = QSVC(quantum_kernel=trained_kernel)
    qsvc.fit(X_train_pca, y_train)

    preds = qsvc.predict(X_test_pca)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Classification Report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=["Benign (B)", "Malignant (M)"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # 2D Decision Boundary Plot (using first 2 PCA components)
    # Create a mesh grid on the 2D PCA space
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict on the grid using the full pipeline (impute/scale/PCA 4D → QSVC)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Inverse transform to original feature space approximately (using 2D PCA inverse)
    grid_original_approx = pca_2d.inverse_transform(grid_points)
    grid_scaled = scaler.transform(grid_original_approx)  # Scale
    grid_pca_full = pca_quantum.transform(grid_scaled)   # Project to 4D for quantum kernel
    Z = qsvc.predict(grid_pca_full)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Blues")
    scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap="Blues", edgecolors="k")
    wrong = (preds != y_test)
    plt.scatter(X_test_2d[wrong, 0], X_test_2d[wrong, 1], c=preds[wrong], cmap="Reds", marker="x", s=100, linewidths=2, label="Misclassified")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("QSVC Decision Boundary (2D PCA Projection) on Test Data")
    plt.legend(*scatter.legend_elements(), title="True Class")
    plt.tight_layout()
    plt.savefig("decision_boundary_2d.png")
    plt.show()

    return acc, f1

if __name__ == "__main__":
    if not os.path.exists("data.csv"):
        raise FileNotFoundError("Please upload 'data.csv' with WDBC data in the working directory.")

    X, y = load_and_process_data("data.csv")
    acc, f1 = run_trainable_qsvc(X, y, n_components=4, reps=2)

    print(f"\n✅ Trained QSVC Results:")
    print(f"Accuracy = {acc:.4f}")
    print(f"F1 Score = {f1:.4f}")
