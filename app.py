
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
st.title("🔬 Sistema Inteligente para Validação Nutricional – Araruta")

alvo_file = st.file_uploader("📂 Arquivo com os dados a corrigir (umidade, proteínas, cinzas, lipídeos):", type="csv")
ref_file = st.file_uploader("📘 Base de referência (com *_ref):", type="csv")

col_X = ["umidade", "proteina", "cinzas", "lipideos"]
col_y = ["umidade_ref", "proteina_ref", "cinzas_ref", "lipideos_ref"]

if alvo_file and ref_file:
    df_alvo = pd.read_csv(alvo_file)
    df_ref = pd.read_csv(ref_file)

    try:
        # Calcular carboidratos por diferença
        df_alvo["carboidratos"] = 100 - df_alvo[col_X].sum(axis=1)
        df_ref["carboidratos_ref"] = 100 - df_ref[[c for c in col_y]].sum(axis=1)

        X_ref = df_ref[col_X]
        y_ref = df_ref[["umidade_ref", "proteina_ref", "cinzas_ref", "lipideos_ref", "carboidratos_ref"]]
        X_alvo = df_alvo[col_X]

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

        # Árvore
        dt = DecisionTreeRegressor(random_state=0).fit(X_ref, y_ref)
        y_dt = dt.predict(X_alvo)
        resultados["Árvore de Decisão"] = {
            "modelo": dt,
            "y_pred": y_dt,
            "R²": r2_score(y_ref, dt.predict(X_ref)),
            "MAE": mean_absolute_error(y_ref, dt.predict(X_ref))
        }

        # MLP
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

        melhor_modelo = max(resultados, key=lambda m: resultados[m]["R²"])
        st.success(f"🏆 Melhor modelo com base no R²: **{melhor_modelo}**")

        y_corr = resultados[melhor_modelo]["y_pred"]
        y_corr_df = pd.DataFrame(y_corr, columns=["umidade_corr", "proteina_corr", "cinzas_corr", "lipideos_corr", "carboidratos_corr"])
        df_resultado = pd.concat([df_alvo, y_corr_df], axis=1)

        # Cálculo de variação e z-score
        for col in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]:
            df_resultado[col + "_var_%"] = ((df_resultado[col + "_corr"] - df_resultado[col]) / df_resultado[col]) * 100
            df_resultado[col + "_zscore"] = np.abs((df_resultado[col + "_var_%"] - df_resultado[col + "_var_%"].mean()) / df_resultado[col + "_var_%"].std())

        st.dataframe(df_resultado.style.highlight_between(
            subset=[c+"_zscore" for c in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]],
            left=2, right=100, color='tomato'))

        # Gráficos interativos
        st.markdown("### 📈 Gráficos Comparativos")
        for col in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]:
            fig = px.scatter(df_resultado, x=col, y=col + "_corr", title=f"Correção de {col.capitalize()}",
                             labels={col: "Original", col + "_corr": "Corrigido"})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Desempenho dos Modelos")
        st.table(pd.DataFrame({
            modelo: {
                "R²": f"{res['R²']:.3f}",
                "MAE": f"{res['MAE']:.3f}"
            } for modelo, res in resultados.items()
        }).T)

        st.download_button("⬇️ Baixar resultado corrigido", df_resultado.to_csv(index=False), file_name="resultado_final.csv")

    except Exception as e:
        st.error(f"Erro: {e}")
