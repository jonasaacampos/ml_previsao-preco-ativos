def text_separator():
    print()
    print('=' * 60)


def print_predict(y_train, y_pred):
    print("Valor Real:", "\n",
      list(y_train.reshape(-1,)), "\n",
      "------------", "\n",
      "Valor Previsto:", "\n",
      [round(float(a)) for a in y_pred])
