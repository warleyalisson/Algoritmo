# mlp_regression.py

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def treinar_modelo(X_ref: pd.DataFrame, y_ref: pd.DataFrame):
    """
    Treina um modelo de rede neural (MLP) para prever os valores de referência
    com base nos dados nutricionais da araruta.

    Parâmetros:
        X_ref (DataFrame): Dados de entrada (umidade, proteína, cinzas, fibras)
        y_ref (DataFrame): Dados de saída (valores de referência padrão)

    Retorno:
        modelo (MLPRegressor)
        scaler_X (StandardScaler): Escalonador dos dados de entrada
        scaler_y (StandardScaler): Escalonador dos dados de saída
    """
    # Escalonar os dados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_ref)
    y_scaled = scaler_y.fit_transform(y_ref)

    # Definir e treinar o modelo
    modelo = MLPRegressor(hidden_layer_sizes=(100, 50),
                          activation='relu',
                          solver='adam',
                          max_iter=1000,
                          early_stopping=True,
                          random_state=42)

    modelo.fit(X_scaled, y_scaled)

    # Avaliação de desempenho (opcional)
    y_pred_train_scaled = modelo.predict(X_scaled)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled)

    for i, col in enumerate(y_ref.columns):
        r2 = r2_score(y_ref[col], y_pred_train[:, i])
        mae = mean_absolute_error(y_ref[col], y_pred_train[:, i])
        print(f"[{col}] - R²: {r2:.4f} | Erro médio absoluto: {mae:.4f}")

    return modelo, scaler_X, scaler_y
