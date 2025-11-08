# 102778 Machine Learning and quantum computers: QUANTUM FEATURE MAP 

Leveraging the kernel trick of Support Vector Machines (SVMs), quantum feature maps can be implemented using a quantum computer [1, 2]. In this folder, we provide Python codes for implementing quantum algorithms for classification tasks within the kernel method using the SVM approach. Python codes utilize the Pennylane library for the quantum "parts".

[1] https://arxiv.org/abs/1804.11326

[2] https://arxiv.org/abs/1803.07128

This is the version 0.  Take care, it may contain errors. 
Course: 2024-2025 

## create conda environment,  activate and install:

* create:
```
conda create -n pennylane python=3.10
```

* activate
```
conda activate pennylane
```
* install python stuff
```
pip install numpy scipy jax matplotlib netket ipython ipykernel pandas seaborn

pip install ipympl

pip install tqdm
```
* install data stuff

````
pip install pandas
pip install seaborn

````


* install scikit-learn 
```
pip install scikit-learn
```
* and pennylane
```
pip install pennylane --upgrade
```
## Pennylane (in a nuthsell)

You may flex your muscles by taking a look at Pennylane, e.g. at the gates.

https://docs.pennylane.ai/en/stable/introduction/operations.html

In any case, you can google Pennylane and dive into its documentation. Have fun!





## What do you find here?

* Data folder:   the databases are in the main directory in data/


* Modules:

    - modudata.py: Functions for data manipulation (using pandas): reading, preparing for training, and data visualization functions.
    - moduqusvm.py: Quantum circuit(s) and quantum kernel construction.

* Python codes:

    - simple_pl_circuit.py: A simple circuit constructor (class exercise).
    - qSVM_v0.py: The quantum algorithm for classification. You will also find its classical version.


