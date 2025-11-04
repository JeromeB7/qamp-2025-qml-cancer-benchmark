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

pd.set_option("future.no_silent_downcasting", True)

""" add folder and load modules """
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import reload

import moduscikit as mdsck

mdsck = reload(mdsck)

from moduscikit import svm_models

# %%

# ====================================
# read data, split the data
# ====================================

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

data = pd.read_csv("../data/iris.data", names=column_names)

# check data size and the number of features
print(data.shape)

X = data.drop(columns=["class"])
Y = (
    data["class"]
    .replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    .astype(int)
)

X = X.values
Y = Y.values


X_train, Y_train, X_test, Y_test, X_holdout, Y_holdout = mdsck.data_split(X, Y)

print("train shape =", X_train.shape[1])
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
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
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
