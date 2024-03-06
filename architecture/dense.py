import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Classe para a camada densa
class Dense:
    
    # Método construtor
    def __init__(self, feat_size, out_size):
        self.feat_size = feat_size
        self.out_size = out_size
        self.weights = (np.random.normal(0, 1, feat_size * out_size) * np.sqrt(2 / feat_size)).reshape(feat_size, out_size)
        self.bias = np.random.rand(1, out_size) - 0.5

    # Método da passada linear para frente
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return(self.output)

    # Método da passada de volta (backpropagation)
    def backward(self, output_der, lr): 
        input_der = np.dot(output_der, self.weights.T)
        weight_der = np.dot(self.input.T.reshape(-1, 1), output_der)
        self.weights -= lr * weight_der
        self.bias -= lr * output_der
        return(input_der)