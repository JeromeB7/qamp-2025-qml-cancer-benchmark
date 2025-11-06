# %%

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pd.set_option("future.no_silent_downcasting", True)

""" esto es para añadir el actual directorio y poder impotar módulos """
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import reload

import moduscikit as mdsck

mdsck = reload(mdsck)

from moduscikit import svm_models

# %%

# ====================================
# read data, split the data
# ====================================

column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
data = pd.read_csv("../data/diabetes.csv", names=column_names, header=0)

# check data size and the number of features
print(data.shape)

X = data.drop(columns=["Outcome"])
Y = data["Outcome"]

X = X.values
Y = Y.values

X_train, Y_train, X_test, Y_test, X_holdout, Y_holdout = mdsck.data_split(X, Y)

print("train shape =", X_train.shape)
print("test shape =", X_test.shape)
print("holdout shape =", X_holdout.shape)

# %%

selected_kernel = "rbf"
clf_svm = svm_models[selected_kernel]

# train:
clf_svm.fit(X_train, Y_train)


accuracy = clf_svm.score(X_test, Y_test)
print(f"Precisión: {accuracy}")

Y_pred = clf_svm.predict(X_test)

conf_matrix = confusion_matrix(Y_test, Y_pred)
print("confusion matrix:")
print(conf_matrix)


# %%

# pca to visualize
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# choose kernel
selected_kernel = "linear"
clf = svm.SVC(kernel=selected_kernel, C=1.0, gamma="scale")
clf.fit(X_train_pca, Y_train)


# Plot decision boundaries
x1_min, x1_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
x2_min, x2_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
num_points = 20  # Número de puntos deseados en cada eje
x1 = np.linspace(x1_min, x1_max, num_points)
x2 = np.linspace(x2_min, x2_max, num_points)
x1, x2 = np.meshgrid(x1, x2)
# we have created a grid within the train boundaries,at every point of the grid, we predict the class. we use the numpy function np.c_ that concatenates the data as colums.
Z = clf.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)


# Plot
plt.contourf(x1, x2, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=Y_train,
    cmap=plt.cm.coolwarm,
    edgecolor="k",
    s=20,
)
plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=Y_test,
    cmap=plt.cm.coolwarm,
    edgecolor="k",
    s=20,
    marker="x",
)
plt.title(f"SVM Decision Boundary with {selected_kernel} kernel")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Evaluate performance
accuracy = clf.score(X_test_pca, Y_test)
print(f"Accuracy: {accuracy}")
conf_matrix = confusion_matrix(Y_test, clf.predict(X_test_pca))
print("Confusion Matrix:")
print(conf_matrix)


# %%

# Seleccionar las características deseadas
selected_features = ["Glucose", "BMI"]
X_selected = data[selected_features].values

# Dividir los datos
X_train_selected, X_test_selected, Y_train_selected, Y_test_selected = train_test_split(
    X_selected, Y, test_size=0.3, random_state=42
)

# Entrenar con las características seleccionadas
clf = svm.SVC(kernel=selected_kernel, C=1.0, gamma="scale")
clf.fit(X_train_selected, Y_train_selected)


x1_min, x1_max = X_train_selected[:, 0].min() - 1, X_train_selected[:, 0].max() + 1
x2_min, x2_max = X_train_selected[:, 1].min() - 1, X_train_selected[:, 1].max() + 1
x1 = np.linspace(x1_min, x1_max, num_points)
x2 = np.linspace(x2_min, x2_max, num_points)
x1, x2 = np.meshgrid(x1, x2)
Z = clf.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)


plt.contourf(x1, x2, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(
    X_train_selected[:, 0],
    X_train_selected[:, 1],
    c=Y_train_selected,
    cmap=plt.cm.coolwarm,
    edgecolor="k",
    s=20,
)
plt.scatter(
    X_test_selected[:, 0],
    X_test_selected[:, 1],
    c=Y_test_selected,
    cmap=plt.cm.coolwarm,
    edgecolor="k",
    s=20,
    marker="x",
)
plt.title(f"SVM Decision Boundary with {selected_kernel} kernel (Selected Features)")
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.show()

# %%


# Definir el rango de hiperparámetros
param_grid = {
    "C": [0.1],
    "gamma": [5, 1],
    "degree": [2, 3],
    "kernel": ["linear", "rbf"],
}
 
# Configurar el GridSearchCV
grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring="accuracy",
    cv=100,
    verbose=2,
    n_jobs=-1,
)


grid_search.fit(X_train_selected, Y_train_selected)


print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor puntuación:", grid_search.best_score_)


best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_selected, Y_test_selected)
print("Precisión en el conjunto de prueba:", test_accuracy)
# %%
