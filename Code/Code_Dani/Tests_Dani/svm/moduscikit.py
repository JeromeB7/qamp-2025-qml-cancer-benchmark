import numpy as np
import pandas as pd
from sklearn import datasets, metrics, svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

pd.set_option("future.no_silent_downcasting", True)


# =======================
# data_split
# =======================
def data_split(X, Y):
    """
    Splits the data into training, testing, and holdout sets.

    Args:
        X (numpy.ndarray): Feature matrix.
        Y (numpy.ndarray): Target vector.

    Returns:
        tuple: A tuple containing:
            - X_train (numpy.ndarray): Training feature matrix.
            - Y_train (numpy.ndarray): Training target vector.
            - X_test (numpy.ndarray): Testing feature matrix.
            - Y_test (numpy.ndarray): Testing target vector.
            - X_holdout (numpy.ndarray): Holdout feature matrix.
            - Y_holdout (numpy.ndarray): Holdout target vector.
    """

    X_train, X_holdout, Y_train, Y_holdout = train_test_split(
        X, Y, test_size=0.02, random_state=1
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=1
    )
    return X_train, Y_train, X_test, Y_test, X_holdout, Y_holdout


# =======================
# dictionary for kernel choosing
# =======================
"""
Dictionary for selecting different kernel configurations for an SVM.

This dictionary provides predefined configurations of `SVC` models 
with different kernels, including linear, polynomial, radial basis function (RBF), 
and sigmoid. Each entry corresponds to an SVC model with specific hyperparameters.

Standard usage example:
    clf = SVC(kernel="poly", degree=2, coef0=0)

Dictionary structure:
    - "linear": Linear kernel with default parameters.
    - "poly": Polynomial kernel with degree=2 and coef0=0.
    - "rbf": RBF kernel with gamma="scale".
    - "sigmoid": Sigmoid kernel with coef0=0.

Attributes:
    svm_models (dict): A dictionary mapping kernel names to `SVC` objects.

Example:
    model = svm_models["rbf"]
    model.fit(X_train, y_train)
"""
svm_models = {
    "linear": SVC(kernel="linear", C=1.0),
    "poly": SVC(kernel="poly", degree=2, coef0=0, C=1.0),
    "rbf": SVC(kernel="rbf", gamma="scale", C=1.0),
    "sigmoid": SVC(kernel="sigmoid", coef0=0, C=1.0),
}
