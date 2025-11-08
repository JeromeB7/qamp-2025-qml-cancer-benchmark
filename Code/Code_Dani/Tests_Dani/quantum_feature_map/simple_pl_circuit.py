# %%
import os
import sys


""" add the current folder for importing modules """
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import reload

import moduqusvm as mdqsvm

mdqsvm = reload(mdqsvm)

# %%

# draw and compute a simple q circuit.  It uses the function cir_ej (see its documentation in moduqusvm.py)
mdqsvm.cir_ej(2, mdqsvm.circuito)

# %%
