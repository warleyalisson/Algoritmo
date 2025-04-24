
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
st.title("🔬 Validador IA – Composição Proximal com Análise Estatística")

st.markdown("Envie os **dados a corrigir** e a **base de referência** contendo os valores padrão.")

alvo_file = st.file_uploader("📂 Arquivo com os dados a corrigir:", type="csv")
ref_file = st.file_uploader("📘 Base de referência com *_ref:", type="csv")

if alvo_file and ref_file:
    df_alvo = pd.read_csv(alvo_file)
    df_ref = pd.read_csv(ref_file)

    col_base = ["umidade", "proteina", "cinzas", "lipideos"]
    col_corrigido = ["umidade_corr", "proteina_corr", "cinzas_corr", "lipideos_corr", "carboidratos_corr"]

    for df in [df_alvo, df_ref]:
        required = col_base
        if all(col in df.columns for col in required):
            if "carboidratos" not in df.columns:
                df["carboidratos"] = 100 - (df["umidade"] + df["proteina"] + df["cinzas"] + df["lipideos"])
        else:
            st.error(f"❌ Faltam colunas obrigatórias: {', '.join(required)}.")
            st.stop()

    for col in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]:
        if col + "_ref" not in df_ref.columns:
            df_ref[col + "_ref"] = df_ref[col]

    try:
        X_ref = df_ref[["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]]
        y_ref = df_ref[["umidade_ref", "proteina_ref", "cinzas_ref", "lipideos_ref", "carboidratos_ref"]]
        X_alvo = df_alvo[["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]]

        df_resultado = df_alvo.copy()
        resultados = {}

        # Modelos
        lr = LinearRegression().fit(X_ref, y_ref)
        dt = DecisionTreeRegressor(random_state=0).fit(X_ref, y_ref)
        scalerX, scalerY = StandardScaler().fit(X_ref), StandardScaler().fit(y_ref)
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0, early_stopping=True)
        mlp.fit(scalerX.transform(X_ref), scalerY.transform(y_ref))

        y_preds = {
            "Regressão Linear": lr.predict(X_alvo),
            "Árvore de Decisão": dt.predict(X_alvo),
            "Rede Neural (MLP)": scalerY.inverse_transform(mlp.predict(scalerX.transform(X_alvo)))
        }

        scores = {
            nome: r2_score(y_ref, pred)
            for nome, pred in {
                "Regressão Linear": lr.predict(X_ref),
                "Árvore de Decisão": dt.predict(X_ref),
                "Rede Neural (MLP)": scalerY.inverse_transform(mlp.predict(scalerX.transform(X_ref)))
            }.items()
        }

        melhor = max(scores, key=scores.get)
        st.success(f"🏆 Melhor modelo: {melhor}")
        y_corrigido = y_preds[melhor]

        df_corrigido = pd.DataFrame(y_corrigido, columns=col_corrigido)
        df_final = pd.concat([df_alvo, df_corrigido], axis=1)

        for c in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]:
            df_final[c + "_var_%"] = ((df_final[c + "_corr"] - df_final[c]) / df_final[c]) * 100

        # Tabelas separadas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📂 Dados Originais")
            st.dataframe(df_alvo)
        with col2:
            st.subheader("📘 Base de Referência")
            st.dataframe(df_ref)

        st.subheader("✅ Dados Corrigidos")
        st.dataframe(df_corrigido)

        # Estatísticas
        st.subheader("📊 Comparação Estatística")
        estat = pd.DataFrame()
        for col in ["umidade", "proteina", "cinzas", "lipideos", "carboidratos"]:
            estat.loc[col, "Média Original"] = df_final[col].mean()
            estat.loc[col, "Média Corrigida"] = df_final[col + "_corr"].mean()
            estat.loc[col, "Desvio (%)"] = ((estat.loc[col, "Média Corrigida"] - estat.loc[col, "Média Original"]) / estat.loc[col, "Média Original"]) * 100

        def cor_valor(val):
            return "background-color: lightgreen" if -10 <= val <= 10 else "background-color: tomato"

        st.dataframe(estat.style.applymap(cor_valor, subset=["Desvio (%)"]).format("{:.2f}"))

        st.download_button("⬇️ Baixar Resultado Final", df_final.to_csv(index=False), file_name="resultado_completo.csv")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
