import numpy as np

from architecture.dense import Dense

# Em vez de implementar a função de ativação dentro da camada densa, a implementação como camada separada simplifica o backpropagation. Esta camada não atualizará nenhum parâmetro, apenas retornará a derivada da função de perda em relação à função de ativação para a camada anterior totalmente conectada.
# Na passagem para a frente, a camada de ativação pegará a saída da camada densa e a transferirá após a aplicação da função ReLu.
from architecture.activation import relu, relu_prime

# Classe da camada de ativação
from architecture.activation import ActLayer

# ## Função de Perda e Derivada
from architecture.math_functions import mse, mse_prime

# Modelo
from architecture.model import Network

# %%
# Dados
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Ajuste dos dados
x_train = x_train.reshape(-1, 2)
y_train = y_train.reshape(-1, 1)

# Modelo
modelo_xor = Network(mse, mse_prime)
modelo_xor.add(Dense(2, 3))
modelo_xor.add(ActLayer(relu, relu_prime))
modelo_xor.add(Dense(3, 1))

# Treinamento
modelo_xor.fit(x_train, y_train, epochs = 2000, lr = 0.01)

# Teste
y_pred = modelo_xor.predict(x_train)

from utils.text_decorators import text_separator, print_predict

text_separator()
text_separator()
print_predict(y_train, y_pred)

