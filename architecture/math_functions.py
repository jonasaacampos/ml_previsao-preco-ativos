import numpy as np

# Usaremos a Mean-Squared-Error como função de perda
def mse(y_true, y_pred):
    return(np.mean((y_pred - y_true)**2))

# Derivada da função de perda
def mse_prime(y_true, y_pred):
    return(2*(y_pred - y_true) / y_true.size)