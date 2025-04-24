
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
st.title("🔬 Validador Inteligente de Composição Proximal com Estatística Científica Avançada")

st.markdown("""
Este sistema aplica modelos de IA para correção e validação de dados de composição proximal de alimentos, considerando:
**umidade, proteínas, cinzas, lipídeos, fibras e carboidratos** (calculado por diferença).

### 📏 Variações esperadas:
- 🟢 ±2%: Alta precisão (controle de qualidade rigoroso)
- 🟡 ±5%: Precisão acadêmica aceitável
- 🔴 >10%: Dados inconsistentes

### 📊 Indicadores apresentados:
- Média, desvio padrão e coeficiente de variação (CV%)
- Erro médio absoluto
- Coloração condicional: verde (ok), amarelo (aceitável), vermelho (alerta)
""")

alvo_file = st.file_uploader("📂 Dados a Corrigir (CSV):", type="csv")
ref_file = st.file_uploader("📘 Base de Referência (CSV):", type="csv")

if alvo_file and ref_file:
    df_alvo = pd.read_csv(alvo_file)
    df_ref = pd.read_csv(ref_file)

    col_base = ["umidade", "proteina", "cinzas", "lipideos", "fibras"]
    col_corrigido = [c + "_corr" for c in col_base + ["carboidratos"]]

    for df in [df_alvo, df_ref]:
        if all(col in df.columns for col in col_base):
            df["carboidratos"] = 100 - df[col_base].sum(axis=1)
        else:
            st.error("❌ Faltam colunas obrigatórias.")
            st.stop()

    for col in col_base + ["carboidratos"]:
        if col + "_ref" not in df_ref.columns:
            df_ref[col + "_ref"] = df_ref[col]

    try:
        X_ref = df_ref[col_base + ["carboidratos"]]
        y_ref = df_ref[[c + "_ref" for c in col_base + ["carboidratos"]]]
        X_alvo = df_alvo[col_base + ["carboidratos"]]

        # Modelos
        scalerX, scalerY = StandardScaler().fit(X_ref), StandardScaler().fit(y_ref)
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=0)
        mlp.fit(scalerX.transform(X_ref), scalerY.transform(y_ref))
        y_pred_corr = scalerY.inverse_transform(mlp.predict(scalerX.transform(X_alvo)))

        df_resultado = df_alvo.copy()
        df_corrigido = pd.DataFrame(y_pred_corr, columns=col_corrigido)
        df_final = pd.concat([df_resultado, df_corrigido], axis=1)

        for c in col_base + ["carboidratos"]:
            df_final[c + "_var_%"] = ((df_final[c + "_corr"] - df_final[c]) / df_final[c]) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## 📂 Dados Originais")
            st.dataframe(df_alvo)
        with col2:
            st.markdown("## 📘 Base de Referência")
            st.dataframe(df_ref)

        st.markdown("## ✅ Dados Corrigidos")
        st.dataframe(df_corrigido)

        st.markdown("## 📊 Estatísticas Analíticas Comparativas")
        estat = pd.DataFrame()
        for col in col_base + ["carboidratos"]:
            estat.loc[col, "Média Original"] = df_final[col].mean()
            estat.loc[col, "Desvio Original"] = df_final[col].std()
            estat.loc[col, "Média Corrigida"] = df_final[col + "_corr"].mean()
            estat.loc[col, "Desvio Corrigido"] = df_final[col + "_corr"].std()
            estat.loc[col, "Erro Médio Absoluto"] = abs(df_final[col + "_corr"] - df_final[col]).mean()
            estat.loc[col, "CV Corrigido (%)"] = 100 * estat.loc[col, "Desvio Corrigido"] / estat.loc[col, "Média Corrigida"]
            estat.loc[col, "Variação Média (%)"] = df_final[col + "_var_%"].mean()

        def cor_variacao(val):
            if abs(val) <= 2:
                return "background-color: lightgreen"
            elif abs(val) <= 5:
                return "background-color: khaki"
            else:
                return "background-color: tomato"

        st.dataframe(estat.style
                     .applymap(cor_variacao, subset=["Variação Média (%)"])
                     .format("{:.2f}"))

        st.markdown("## 📈 Comparação Visual")
        fig = px.bar(estat.reset_index(), x='index', y=["Média Original", "Média Corrigida"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 📝 Resumo Técnico")
        media_erro = estat["Erro Médio Absoluto"].mean()
        media_var = estat["Variação Média (%)"].mean()
        if media_var < 2:
            qualidade = "excelente precisão (nível de controle laboratorial)"
        elif media_var < 5:
            qualidade = "boa precisão dentro do esperado em pesquisas"
        else:
            qualidade = "alta variação, recomenda-se revisão dos dados originais"

        st.info(f"O modelo corrigiu os dados com média de erro absoluto de {media_erro:.2f} e variação média de {media_var:.2f}%. "
                f"Essa correção representa {qualidade}.")

        # Downloads
        st.download_button("⬇️ Baixar Dados Corrigidos", df_final.to_csv(index=False), file_name="resultado_corrigido.csv")
        st.download_button("⬇️ Baixar Estatísticas", estat.to_csv(index=True), file_name="estatisticas.csv")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
