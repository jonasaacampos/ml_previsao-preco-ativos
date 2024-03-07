def text_separator():
    return '=' * 60

def line_blank():
    return ''


def print_predict(y_train, y_pred):
    print("Valor Real:", "\n",
      list(y_train.reshape(-1,)), "\n",
      "------------", "\n",
      "Valor Previsto:", "\n",
      [round(float(a)) for a in y_pred])
