
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
st.title("🔬 Validador IA – Composição Proximal (Umidade, Cinzas, Proteínas, Lipídeos, Carboidratos)")

st.markdown("Envie os **dados a corrigir** e a **base de referência** contendo os valores padrão.")

alvo_file = st.file_uploader("📂 Arquivo com os dados a corrigir:", type="csv")
ref_file = st.file_uploader("📘 Base de referência com *_ref:", type="csv")

if alvo_file and ref_file:
    df_alvo = pd.read_csv(alvo_file)
    df_ref = pd.read_csv(ref_file)

    col_base = ["umidade", "proteina", "cinzas", "lipideos"]
    col_corrigido = ["umidade_corr", "proteina_corr", "cinzas_corr", "lipideos_corr", "carboidratos_corr"]
    col_y = ["umidade_ref", "proteina_ref", "cinzas_ref", "lipideos_ref", "carboidratos_ref"]

    # Verifica se as colunas existem e calcula carboidratos
    for df in [df_alvo, df_ref]:
        required_cols = ["umidade", "proteina", "cinzas", "lipideos"]
        if all(col in df.columns for col in required_cols):
            if "carboidratos" not in df.columns:
                df["carboidratos"] = 100 - (df["umidade"] + df["proteina"] + df["cinzas"] + df["lipideos"])
        else:
            st.error(f"❌ Arquivo está incompleto. As colunas obrigatórias são: {', '.join(required_cols)}.")
            st.stop()

    # Gerar colunas *_ref se não existirem
    for base_col in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]:
        ref_col = base_col + "_ref"
        if ref_col not in df_ref.columns:
            df_ref[ref_col] = df_ref[base_col]

    try:
        X_ref = df_ref[["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]]
        y_ref = df_ref[["umidade_ref", "proteina_ref", "cinzas_ref", "lipideos_ref", "carboidratos_ref"]]
        X_alvo = df_alvo[["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]]

        df_resultado = df_alvo.copy()
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

        melhor_modelo = max(resultados, key=lambda m: resultados[m]["R²"])
        st.success(f"🏆 Melhor modelo com base no R²: **{melhor_modelo}**")

        y_corrigido = resultados[melhor_modelo]["y_pred"]
        y_corrigido_df = pd.DataFrame(y_corrigido, columns=col_corrigido)
        df_resultado = pd.concat([df_resultado, y_corrigido_df], axis=1)

        # Variação percentual e zscore
        for col in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]:
            col_corr = col + "_corr"
            df_resultado[col + "_var_%"] = ((df_resultado[col_corr] - df_resultado[col]) / df_resultado[col]) * 100
            df_resultado[col + "_zscore"] = np.abs((df_resultado[col + "_var_%"] - df_resultado[col + "_var_%"].mean()) / df_resultado[col + "_var_%"].std())

        st.dataframe(df_resultado.style.highlight_between(
            subset=[c+"_zscore" for c in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]],
            left=2, right=100, color='tomato'))

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

        st.download_button("⬇️ Baixar resultado corrigido", df_resultado.to_csv(index=False), file_name="resultado_corrigido_proximal.csv")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
