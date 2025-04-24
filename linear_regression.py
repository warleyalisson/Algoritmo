# linear_regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def treinar_modelo(X_ref: pd.DataFrame, y_ref: pd.DataFrame):
    """
    Treina um modelo de regressão linear múltipla usando dados de referência.

    Parâmetros:
        X_ref (DataFrame): Dados de entrada (umidade, proteína, cinzas, fibras)
        y_ref (DataFrame): Dados de saída padrão (valores de referência)

    Retorno:
        modelo treinado (LinearRegression)
    """
    modelo = LinearRegression()
    modelo.fit(X_ref, y_ref)

    # Diagnóstico no terminal (apenas para debug offline)
    print("Coeficientes por variável:", modelo.coef_)
    print("Intercepto:", modelo.intercept_)

    # Avaliação de treinamento (opcional)
    y_pred_train = modelo.predict(X_ref)
    for i, col in enumerate(y_ref.columns):
        r2 = r2_score(y_ref[col], y_pred_train[:, i])
        mae = mean_absolute_error(y_ref[col], y_pred_train[:, i])
        print(f"[{col}] - R²: {r2:.4f} | Erro médio absoluto: {mae:.4f}")

    return modelo
