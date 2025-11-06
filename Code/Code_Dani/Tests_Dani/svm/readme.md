## ENVIRONMENTS:

```
conda create -n svm python=3.12
```

```
conda activate svm
```

## PACKAGES:

```
conda install ipykernel

conda install numpy scikit-learn pandas matplotlib seaborn

conda install graphviz python-graphviz
```

## IF YOU PREFER NOTEBOOKS INSTEAD

automatic change from .py to .ipynb

```
pip install jupytext
```

```
jupytext --to notebook main.py
```

* for notebooks in colab:
    * you need to activate.  In the first cell:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

    * and  load the folder
    ```python 
    %cd  your path
    ```
    *   In the includes change in the .py
    ```python
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    ```

    by 

    ```python
    current_dir = os.getcwd()
    sys.path.append(current_dir)
    ```

