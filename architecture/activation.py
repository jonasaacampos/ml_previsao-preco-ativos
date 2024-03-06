import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Derivada da função de ativação
def relu_prime(x):  
    x[x > 0] = 1
    x[x <= 0] = 0  
    return x

# Função de ativação
def relu(x):  
    return(np.maximum(0, x))


# Classe da camada de ativação
class ActLayer:
    
    # Método construtor
    def __init__(self, act, act_prime):
        self.act = act
        self.act_prime = act_prime

    # Recebe a entrada (input) e retorna a saída da função de ativação
    def forward(self, input_data):
        self.input = input_data
        self.output = self.act(self.input)
        return(self.output)

    # Observe que não estamos atualizando nenhum parâmetro aqui
    # Usamos a taxa de aprendizagem como parâmetro porque definiremos o método de ajuste de uma forma 
    # que todas as camadas o exigirão.
    def backward(self, output_der, lr):
        return(self.act_prime(self.input) * output_der)