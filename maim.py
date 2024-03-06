from architecture.model import Network
from architecture.dense import Dense
from architecture.activation import ActLayer

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