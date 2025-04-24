# decision_tree.py

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def treinar_modelo(X_ref: pd.DataFrame, y_ref: pd.DataFrame):
    modelo = DecisionTreeRegressor(random_state=42)
    modelo.fit(X_ref, y_ref)

    y_pred_train = modelo.predict(X_ref)
    for i, col in enumerate(y_ref.columns):
        r2 = r2_score(y_ref[col], y_pred_train[:, i])
        mae = mean_absolute_error(y_ref[col], y_pred_train[:, i])
        print(f"[{col}] - R²: {r2:.4f} | MAE: {mae:.4f}")

    return modelo

