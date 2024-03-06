import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Modelo


class Network:
    
    # Método construtor
    # Inicializa com a função de perda e sua derivada
    def __init__(self, loss, loss_prime):  
        self.layers = []  
        self.loss = loss
        self.loss_prime = loss_prime

    # Método para adicionar camadas ao grafo computacional
    def add(self, layer):
        self.layers.append(layer)

    # Implementando apenas forward-pass para predição
    def predict(self, input_data):
        
        # Lista para o resultado
        result = [] 

        for a in range(len(input_data)):
            
            # Camada de saída
            layer_output = input_data[a]
            
            # Loop pelas camadas
            for layer in self.layers:
                
                # Movendo vetores de camada para camada
                layer_output = layer.forward(layer_output)
                
            result.append(layer_output)

        return(result)

    # Método de treinamento
    def fit(self, X_train, y_train, epochs, lr):

        # Número de iterações
        for a in range(epochs):  
            
            # Inicializa a variável de cálculo do erro
            err = 0

            # Temos 1 passagem para a frente e para trás para cada ponto de dados 
            # Esse algoritmo de aprendizagem usa a Descida Estocástica do Gradiente
            for j in range(len(X_train)):
                
                # Camada de saída
                layer_output = X_train[j]
                
                # Loop pelas camadas
                for layer in self.layers:
                    layer_output = layer.forward(layer_output)

                # Vamos guardar o erro e mostrar durante o treinamento
                err += self.loss(y_train[j], layer_output)

                # Observe que fazemos o loop nas camadas em ordem reversa.
                # Inicialmente calculamos a derivada da perda com relação à previsão.
                # Em seguida, a camada de saída irá calcular a derivada em relação à sua entrada
                # e irá passar esta derivada de entrada para a camada anterior que corresponde à sua derivada de saída
                # e essa camada repetirá o mesmo processo, passando sua derivada de entrada para a camada anterior.

                # dL/dY_hat
                gradient = self.loss_prime(y_train[j], layer_output)  
                
                # Este loop é a razão de termos dado lr à camada de ativação como argumento
                for layer in reversed(self.layers):
                    
                    # Definindo gradiente para dY / dh_ {i + 1} da camada atual
                    gradient = layer.backward(gradient, lr)

            err /= len(X_train)
            
            print('Epoch %d/%d   Erro = %f' % (a + 1, epochs, err))