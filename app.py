
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
st.title("🔬 Validador Inteligente de Composição Proximal com Estatística Científica")

st.markdown("""
Este sistema aplica modelos de IA para correção e validação de dados de composição proximal de alimentos,
considerando os parâmetros: **umidade, proteínas, cinzas, lipídeos, fibras e carboidratos** (calculado por diferença).

**Variações esperadas em estudos laboratoriais**:
- 🟢 **±2%**: Alta precisão (padrão ouro – controle de qualidade rigoroso)
- 🟡 **±5%**: Precisão aceitável para estudos acadêmicos
- 🔴 **±10% ou mais**: Dados inconsistentes ou com alto risco de erro

**Métricas estatísticas apresentadas**:
- Média original e corrigida
- Desvio padrão
- Variação percentual média
- Coloração automática (verde, amarelo, vermelho)

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
        st.success(f"🏆 Melhor modelo com base no R²: **{melhor}**")
        y_corrigido = y_preds[melhor]

        df_corrigido = pd.DataFrame(y_corrigido, columns=col_corrigido)
        df_final = pd.concat([df_alvo, df_corrigido], axis=1)

        for c in col_base + ["carboidratos"]:
            df_final[c + "_var_%"] = ((df_final[c + "_corr"] - df_final[c]) / df_final[c]) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📂 Dados Originais")
            st.dataframe(df_alvo)
        with col2:
            st.subheader("📘 Base de Referência")
            st.dataframe(df_ref)

        st.subheader("✅ Dados Corrigidos")
        st.dataframe(df_corrigido)

        # Estatísticas ampliadas
        st.subheader("📊 Estatísticas Analíticas Comparativas")
        estat = pd.DataFrame()
        for col in col_base + ["carboidratos"]:
            estat.loc[col, "Média Original"] = df_final[col].mean()
            estat.loc[col, "Desvio Original"] = df_final[col].std()
            estat.loc[col, "Média Corrigida"] = df_final[col + "_corr"].mean()
            estat.loc[col, "Desvio Corrigido"] = df_final[col + "_corr"].std()
            estat.loc[col, "Variação Média (%)"] = df_final[col + "_var_%"].mean()

        def cor_variacao(val):
            if abs(val) <= 2:
                return "background-color: lightgreen"
            elif abs(val) <= 5:
                return "background-color: khaki"
            else:
                return "background-color: tomato"

        st.dataframe(estat.style.applymap(cor_variacao, subset=["Variação Média (%)"]).format("{:.2f}"))

        st.download_button("⬇️ Baixar Resultado Final", df_final.to_csv(index=False), file_name="resultado_avaliado.csv")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
