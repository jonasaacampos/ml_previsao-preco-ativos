import numpy as np
import pandas as pd
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

from utils.text_decorators import text_separator, print_predict


def test_model():
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
    modelo_xor.fit(x_train, y_train, epochs=2000, lr=0.01)
    # Teste
    y_pred = modelo_xor.predict(x_train)

    text_separator()
    print_predict(y_train, y_pred)
    # todo: Insert log.txt save file with steps analysis


### TODO: Start Here

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Volatility (standard deviation), sigma de SDE e PDE
vol = 0.17
# Maturity
T = 1 / 2
#  Número de etapas que usamos para discretizar o processo acima
n = 1000
# O preço inicial do estoque corresponde a S_0 nas funções acima
s_0 = 100
#  Taxa livre de risco, termo que deriva de SDE -> r
r = 0.05
# Preço de exercício da opção -> K
k = 100


# ## Funções de Cálculo Para Geração de Dados
def calculate_spot(prev, sigma, r, step, random):
    return prev + (sigma * prev * random) + (r * prev * step)


def sim_spot(s0, r, steps, maturity, vol):
    delta_t = T / steps
    time = np.round(np.arange(0, maturity + delta_t, delta_t), 4)
    prices = [s0]
    normal_dist = np.random.normal(0, np.sqrt(delta_t), 10000)
    for a in range(steps):
        prices.append(calculate_spot(prices[-1], vol, r, delta_t, normal_dist[a]))
    return prices


# Gerando 5 caminhos diferentes para testar as funções
# Vamos usar apenas 1 caminho no treinamento de rede
sims = pd.DataFrame()
for a in range(5):
    sims[a] = sim_spot(s_0, r, n, T, vol)

# Valores para simulações
sims.columns = ["Sim_1", "Sim_2", "Sim_3", "Sim_4", "Sim_5"]
sims.index = np.round(np.arange(0, 0.5 + (0.5 / 1000), 0.5 / 1000), 4)

sns.set(style="whitegrid", font_scale=2.5)
plt.figure(figsize=(40, 18))
ax = sns.lineplot(data=sims, palette="bright", linewidth=2.7)
ax.set(xlabel="Passos", ylabel="Preços dos Ativos", title="Simulações")

# ## Preparando os Preços Finais


def d1(s, k, r, t, T, vol):
    if T != t:
        nomin = np.log(s / k) + (r + 0.5 * (vol**2)) * (T - t)
        denom = vol * np.sqrt((T - t))
        return nomin / denom
    else:
        None


def d2(s, k, r, t, T, vol):
    if T != t:
        nomin = np.log(s / k) + (r - 0.5 * (vol**2)) * (T - t)
        denom = vol * np.sqrt((T - t))
        return nomin / denom
    else:
        None


def call(d1, d2, k, r, T, t, s):
    return s * scipy.stats.norm.cdf(d1) - k * np.exp(
        -r * (T - t)
    ) * scipy.stats.norm.cdf(d2)


call_prices = []
maturity = []
for a, b in zip(sims["Sim_1"], sims.index):
    if b != T:
        d1_ = d1(a, k, r, b, T, vol)
        d2_ = d2(a, k, r, b, T, vol)
        call_prices.append(call(d1_, d2_, k, r, T, b, a))
        maturity.append((T - b))
    else:
        call_prices.append(max(a - k, 0))
        maturity.append(0)

# Dataframe dos preços
opt_price = pd.DataFrame(call_prices, sims.index)
opt_price = opt_price.rename(columns={0: "Sim_1_Call"})
opt_price = pd.concat([opt_price, sims["Sim_1"]], axis=1)
min_max = MinMaxScaler(feature_range=(min(call_prices), max(call_prices)))
opt_price["Sim_1_scaled"] = min_max.fit_transform(
    opt_price["Sim_1"].values.reshape(-1, 1)
)
opt_price.index = pd.date_range(start="01/01/2018", end="06/01/2018", periods=1001)

sns.set(style="whitegrid", font_scale=2.5)
plt.figure(figsize=(40, 18))
ax = sns.lineplot(
    data=opt_price[["Sim_1_scaled", "Sim_1_Call"]], palette="bright", linewidth=2.7
)
ax.set(
    xlabel="Data",
    ylabel="Preços dos Ativos",
    title="Preço da Ação - Valor de Venda da Ação",
)
plt.savefig(f"results/original_data.png")

# ## Preparação de Dados Para Treinamento
# Dataframe final
opt_price["Maturity"] = maturity
opt_price["Strike"] = k
opt_price["Risk_Free"] = r
opt_price["Volatility"] = vol
model_data = opt_price.drop(["Sim_1_scaled"], axis=1)

# Visualiza
print(text_separator())
print(model_data.head())
print(text_separator())


# Dados de treino e teste
train_data = model_data.iloc[: round(len(model_data) * 0.8)]
test_data = model_data.iloc[len(train_data) :]

X_train = train_data.drop(["Sim_1_Call"], axis=1).values
y_train = train_data["Sim_1_Call"].values

X_test = test_data.drop(["Sim_1_Call"], axis=1).values
y_test = test_data["Sim_1_Call"].values

min_max = MinMaxScaler()

X_train = min_max.fit_transform(X_train)
X_test = min_max.transform(X_test)

print(
    "X_train shape:",
    X_train.shape,
    "\n",
    "y_train shape:",
    y_train.shape,
    "\n",
    "X_test shape:",
    X_test.shape,
    "\n",
    "y_test shape:",
    y_test.shape,
)

# ## Treinamento
# Modelo
model = Network(mse, mse_prime)
model.add(Dense(5, 200))
model.add(ActLayer(relu, relu_prime))
model.add(Dense(200, 200))
model.add(ActLayer(relu, relu_prime))
model.add(Dense(200, 200))
model.add(ActLayer(relu, relu_prime))
model.add(Dense(200, 200))
model.add(ActLayer(relu, relu_prime))
model.add(Dense(200, 1))


def model_experiments(X_train, y_train, X_test, epochs=10, verbose=False):

    # Treinamento
    model.fit(X_train, y_train, epochs=epochs, lr=0.001, verbose=verbose)

    # Previsões
    y_pred = np.ravel(model.predict(X_test))  # achata a matriz para uma única dimensão
    y_pred = [float(a) for a in y_pred]

    return y_pred


def plot_previson_epochs(epoch):
    plt.figure(figsize=(40, 18))
    ax = sns.lineplot(
        data=all_preds[["Valor_Real", f"{epoch} Epochs"]],
        palette="bright",
        linewidth=2.5,
    )
    ax.set(
        xlabel="Data",
        ylabel="Preço do Ativo",
        title=f'Após {epoch} Epochs,  MSE:{round(mean_squared_error(all_preds.Valor_Real, all_preds[f"{epoch} Epochs"]), 3)}',
    )

    plt.savefig(f"results/{epoch}_epochs.png")


y_pred_10 = model_experiments(X_train, y_train, X_test, epochs=10, verbose=True)
y_pred_100 = model_experiments(X_train, y_train, X_test, epochs=100)
y_pred_200 = model_experiments(X_train, y_train, X_test, epochs=200)
y_pred_1000 = model_experiments(X_train, y_train, X_test, epochs=1_000)
# y_pred_5000 = model_experiments(X_train, y_train, X_test, epochs=5_000)


# ## Testando e Comparando os Resultados

# Ajusta o shape das perevisões para cada treinamento
y_pred_10 = np.array(y_pred_10).reshape(-1,)
y_pred_100 = np.array(y_pred_100).reshape(-1,)
y_pred_200 = np.array(y_pred_200).reshape(-1,)
y_pred_1000 = np.array(y_pred_1000).reshape(-1,)
#y_pred_5000 = np.array(y_pred_5000).reshape(-1,)

# Dataframe das previsões
all_preds = pd.DataFrame(
    {
        "Valor_Real": y_test,
        "10 Epochs": y_pred_10,
        "100 Epochs": y_pred_100,
        "200 Epochs": y_pred_200,
        "1000 Epochs": y_pred_1000,
        # "5000 Epochs": y_pred_5000,
    },
    index=test_data.index,
)


plot_previson_epochs(10)
plot_previson_epochs(100)
plot_previson_epochs(200)
plot_previson_epochs(1_000)
# plot_previson_epochs(5_000)

# ## Conclusão
#
# Treinar o modelo por poucas epochs ou por epochs demais afeta negativamente a performance do modelo. A construção de um modelo equilibrado depende do ponto ideal de treinamento, o que requer experimentação.
