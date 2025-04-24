
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🔬 Validador IA – Composição Proximal com Fibras e Análise Científica")

st.markdown("""
Este sistema foi desenvolvido para validar automaticamente análises bromatológicas
relacionadas à **araruta (Maranta arundinacea)** e similares.  
Inclui análise de: `Umidade`, `Proteína`, `Cinzas`, `Lipídeos`, `Fibras`, `Carboidratos`.

📌 **Desvios padrão e precisão esperada** são calculados para indicar confiabilidade das análises,
com destaque para variações fora do intervalo considerado confiável (±10%).

As correções são realizadas por modelos de Regressão Linear, Árvore de Decisão e Rede Neural (MLP),
com escolha automática do modelo de melhor desempenho (maior R²).
""")

alvo_file = st.file_uploader("📂 Arquivo com os dados a corrigir:", type="csv")
ref_file = st.file_uploader("📘 Base de referência com *_ref:", type="csv")

if alvo_file and ref_file:
    df_alvo = pd.read_csv(alvo_file)
    df_ref = pd.read_csv(ref_file)

    col_base = ["umidade", "proteina", "cinzas", "lipideos", "fibras"]
    col_corrigido = [c + "_corr" for c in col_base + ["carboidratos"]]

    for df in [df_alvo, df_ref]:
        if all(col in df.columns for col in col_base):
            df["carboidratos"] = 100 - df[col_base].sum(axis=1)
        else:
            st.error(f"❌ Faltam colunas obrigatórias: {', '.join(col_base)}.")
            st.stop()

    for col in col_base + ["carboidratos"]:
        if col + "_ref" not in df_ref.columns:
            df_ref[col + "_ref"] = df_ref[col]

    try:
        X_ref = df_ref[col_base + ["carboidratos"]]
        y_ref = df_ref[[c + "_ref" for c in col_base + ["carboidratos"]]]
        X_alvo = df_alvo[col_base + ["carboidratos"]]

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

        for c in col_base + ["carboidratos"]:
            df_final[c + "_var_%"] = ((df_final[c + "_corr"] - df_final[c]) / df_final[c]) * 100

        # Tabelas
        st.subheader("📂 Dados Originais")
        st.dataframe(df_alvo)

        st.subheader("📘 Base de Referência")
        st.dataframe(df_ref)

        st.subheader("✅ Dados Corrigidos")
        st.dataframe(df_corrigido)

        st.subheader("📊 Estatísticas de Validação Científica")
        estat = pd.DataFrame()
        for col in col_base + ["carboidratos"]:
            estat.loc[col, "Média Original"] = df_final[col].mean()
            estat.loc[col, "Média Corrigida"] = df_final[col + "_corr"].mean()
            estat.loc[col, "Desvio (%)"] = ((estat.loc[col, "Média Corrigida"] - estat.loc[col, "Média Original"]) / estat.loc[col, "Média Original"]) * 100
            estat.loc[col, "Desvio Padrão Corrigido"] = df_final[col + "_corr"].std()
            estat.loc[col, "Erro Quadrático Médio (MSE)"] = mean_squared_error(df_final[col], df_final[col + "_corr"])
            estat.loc[col, "Erro Absoluto Médio (MAE)"] = mean_absolute_error(df_final[col], df_final[col + "_corr"])

        def cor_valor(val):
            return "background-color: lightgreen" if -10 <= val <= 10 else "background-color: tomato"

        st.dataframe(estat.style.applymap(cor_valor, subset=["Desvio (%)"]).format("{:.3f}"))

        st.download_button("⬇️ Baixar Resultado Final", df_final.to_csv(index=False), file_name="resultado_validacao_cientifica.csv")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
