import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🔬 Sistema Inteligente para Validação de Dados Nutricionais – Araruta")

st.markdown("Envie os **dados a corrigir** e a **base de referência** contendo os valores padrão.")

alvo_file = st.file_uploader("📂 Arquivo com os dados a corrigir:", type="csv")
ref_file = st.file_uploader("📘 Base de referência com *_ref:", type="csv")

if alvo_file and ref_file:
    df_alvo = pd.read_csv(alvo_file)
    df_ref = pd.read_csv(ref_file)

    col_X = ["umidade", "proteina", "cinzas", "fibras"]
    col_y = ["umidade_ref", "proteina_ref", "cinzas_ref", "fibras_ref"]

    try:
        X_ref = df_ref[col_X]
        y_ref = df_ref[col_y]
        X_alvo = df_alvo[col_X]

        df_resultado = df_alvo.copy()

        # Modelos e resultados
        resultados = {}

        # Regressão Linear
        lr = LinearRegression().fit(X_ref, y_ref)
        y_lr = lr.predict(X_alvo)
        resultados["Regressão Linear"] = {
            "modelo": lr,
            "y_pred": y_lr,
            "R²": r2_score(y_ref, lr.predict(X_ref)),
            "MAE": mean_absolute_error(y_ref, lr.predict(X_ref))
        }

        # Árvore de Decisão
        dt = DecisionTreeRegressor(random_state=0).fit(X_ref, y_ref)
        y_dt = dt.predict(X_alvo)
        resultados["Árvore de Decisão"] = {
            "modelo": dt,
            "y_pred": y_dt,
            "R²": r2_score(y_ref, dt.predict(X_ref)),
            "MAE": mean_absolute_error(y_ref, dt.predict(X_ref))
        }

        # Rede Neural MLP
        scalerX = StandardScaler().fit(X_ref)
        scalerY = StandardScaler().fit(y_ref)
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0, early_stopping=True)
        mlp.fit(scalerX.transform(X_ref), scalerY.transform(y_ref))
        y_mlp = scalerY.inverse_transform(mlp.predict(scalerX.transform(X_alvo)))
        resultados["Rede Neural (MLP)"] = {
            "modelo": mlp,
            "y_pred": y_mlp,
            "R²": r2_score(y_ref, scalerY.inverse_transform(mlp.predict(scalerX.transform(X_ref)))),
            "MAE": mean_absolute_error(y_ref, scalerY.inverse_transform(mlp.predict(scalerX.transform(X_ref))))
        }

        # Escolher melhor modelo
        melhor_modelo = max(resultados, key=lambda m: resultados[m]["R²"])
        st.success(f"🏆 Melhor modelo com base no R²: **{melhor_modelo}**")

        # Aplicar e exibir
        y_corrigido = resultados[melhor_modelo]["y_pred"]
        y_corrigido_df = pd.DataFrame(y_corrigido, columns=["umidade_corr", "proteina_corr", "cinzas_corr", "fibras_corr"])
        df_resultado = pd.concat([df_resultado, y_corrigido_df], axis=1)

        # Variação percentual e Z-score
        for col in col_X:
            col_corr = col + "_corr"
            df_resultado[col + "_var_%"] = ((df_resultado[col_corr] - df_resultado[col]) / df_resultado[col]) * 100
            df_resultado[col + "_zscore"] = np.abs((df_resultado[col + "_var_%"] - df_resultado[col + "_var_%"].mean()) / df_resultado[col + "_var_%"].std())

        st.dataframe(df_resultado.style
                     .highlight_between(subset=[c+"_zscore" for c in col_X], left=2, right=100, color='tomato'))

        # Gráficos interativos
        st.markdown("### 📈 Comparativo Visual")
        for col in col_X:
            fig = px.scatter(df_resultado, x=col, y=col + "_corr", title=f"Correção de {col.capitalize()}",
                             labels={col: "Original", col + "_corr": "Corrigido"})
            st.plotly_chart(fig, use_container_width=True)

        # Métricas
        st.markdown("### 📊 Desempenho dos Modelos")
        st.table(pd.DataFrame({
            modelo: {
                "R²": f"{res['R²']:.3f}",
                "MAE": f"{res['MAE']:.3f}"
            } for modelo, res in resultados.items()
        }).T)

        st.download_button("⬇️ Baixar resultado corrigido", df_resultado.to_csv(index=False), file_name="resultado_analitico.csv")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
