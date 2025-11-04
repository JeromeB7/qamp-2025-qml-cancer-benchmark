import os
import pandas as pd
import os.path as osp
import sys
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))
import  quantum_feature_map.moduqusvm as mdqsvm
import  quantum_feature_map.modudata as mddt
from  quantum_feature_map.moduqusvm import svm_models



# %%

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
data = pd.read_csv("../data/iris.data", names=column_names)
# check data size and the number of features
print(data.shape)
X = data.drop(columns=["class"])
Y = (data["class"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}).astype(int))
X = X.values
Y = Y.values
mddt.plot_explained_variance(X)
necesary_pca=2
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X_train, Y_train, X_test, Y_test, X_holdout, Y_holdout = mddt.data_split(X, Y)


# %%

# "classical" vsm

selected_kernel = "linear"
clf = svm_models[selected_kernel]["model"]
# train:
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(f"Precisión: {accuracy}")
Y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("confusion matrix:")
print(conf_matrix)
# visualize kernels
K_train = svm_models[selected_kernel]["kernel_matrix"](X_train)
K_sorted = mddt.sort_K(K_train, Y_train)
print('kernel')
mddt.plot_kernel_matrix(K_sorted, title=selected_kernel)
mddt.plot_kernel_matrix(K_sorted, title=selected_kernel)
mddt.visualize_with_pca(X, Y,feature_names, selected_kernel,necesary_pca)




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# quantum svm
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_qubits = 4
# visualize quantum feature map
x = [0.1, 0.4, 0.6, 0.2]
mdqsvm.cir_ej(4, lambda: mdqsvm.ansatz(x, 4))
print('a')
adjoint_ansatz = qml.adjoint(mdqsvm.layer)(x, 4)
mdqsvm.cir_ej(4, lambda: adjoint_ansatz)

# %%
selected_kernel = "quantum"
clf = svm_models[selected_kernel]["model"](num_qubits)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(f"Precisión: {accuracy}")
Y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("confusion matrix:")
print(conf_matrix)
# %%
#DANI MOD
# Lista de colores personalizada (puedes añadir más)

mdqsvm.compare_predict_and_real(X_test, Y_test, Y_train)
print('AAAA')
#Clasificador cuantico !=QSVM
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
from qiskit_machine_learning.utils import algorithm_globals
num_features = X.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", style="clifford", fold=20)
from qiskit.circuit.library import RealAmplitudes

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
from qiskit_machine_learning.optimizers import COBYLA
maxiteractions=1000
optimizer = COBYLA(maxiter=maxiteractions)
from qiskit.primitives import StatevectorSampler as Sampler

sampler = Sampler()
from IPython.display import clear_output

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    if  len(objective_func_vals)==maxiteractions-3:
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(objective_func_vals)), objective_func_vals)
        plt.show()
    
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []
vqc.fit(X_train, Y_train)

train_score_q4 = vqc.score(X_train, Y_train)
test_score_q4 = vqc.score(X_test, Y_test)
print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
# visualize kernels
K_train = svm_models[selected_kernel]["kernel_matrix"](X_train, num_qubits)
K_sorted = mddt.sort_K(K_train, Y_train)
mddt.plot_kernel_matrix(K_sorted, title=selected_kernel)

# %%
