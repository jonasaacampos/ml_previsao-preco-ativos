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
    def fit(self, X_train, y_train, epochs, lr, verbose=False):
        from utils.log_operations import write_log_in_txt
        from utils.text_decorators import text_separator
        from time import time, strftime, localtime

        log = []
        msg = f'Ajuste de dados com {epochs} passadas.'
        log.append(msg)
        log.append(text_separator())
        
        print(text_separator())
        print(f'{msg} Aguarde...')
        time_starts = time()
        for epoch in range(epochs):  
            
            # Inicializa a variável de cálculo do erro
            err = 0

            # Temos 1 passagem para a frente e para trás para cada ponto de dados 
            # Esse algoritmo de aprendizagem usa a Descida Estocástica do Gradiente
            for i in range(len(X_train)):
                
                # Camada de saída
                layer_output = X_train[i]
                
                # Loop pelas camadas
                for layer in self.layers:
                    layer_output = layer.forward(layer_output)

                # Vamos guardar o erro e mostrar durante o treinamento
                err += self.loss(y_train[i], layer_output)

                # Observe que fazemos o loop nas camadas em ordem reversa.
                # Inicialmente calculamos a derivada da perda com relação à previsão.
                # Em seguida, a camada de saída irá calcular a derivada em relação à sua entrada
                # e irá passar esta derivada de entrada para a camada anterior que corresponde à sua derivada de saída
                # e essa camada repetirá o mesmo processo, passando sua derivada de entrada para a camada anterior.

                # dL/dY_hat
                gradient = self.loss_prime(y_train[i], layer_output)  
                
                # Este loop é a razão de termos dado lr à camada de ativação como argumento
                for layer in reversed(self.layers):
                    
                    # Definindo gradiente para dY / dh_ {i + 1} da camada atual
                    gradient = layer.backward(gradient, lr)

            err /= len(X_train)
            log.append(f'Epoch: {epoch + 1}/{epochs}        | Error = {err:.6f}')

            if verbose:
                print(log[epoch+2]) #linha de cabeçalho e linha de separador => duas linhas
                        
            
        time_ends = time()
        time_elapsed = time_ends - time_starts

        time_starts_format = strftime('%H:%M:%S', localtime(time_starts))
        time_ends_format = strftime('%H:%M:%S', localtime(time_ends))
        time_elapsed_format = "{:.2f}".format(time_elapsed)


        log.insert(1, f'Time Analysist = > Start:{time_starts_format} | Ends:{time_ends_format} || Elapsed:{time_elapsed_format}')
        print(log[1])
        write_log_in_txt(log, name_file=f'{epochs}_epochs')