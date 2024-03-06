import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Usaremos a Mean-Squared-Error como função de perda
def mse(y_true, y_pred):
    return(np.mean((y_pred - y_true)**2))

# Derivada da função de perda
def mse_prime(y_true, y_pred):
    return(2*(y_pred - y_true) / y_true.size)