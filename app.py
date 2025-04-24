import streamlit as st
import pandas as pd
from linear_regression import treinar_modelo as treinar_linear
from mlp_regression import treinar_modelo as treinar_mlp
from decision_tree import treinar_modelo as treinar_tree

st.set_page_config(page_title="Validador IA de Dados da Araruta")

st.title("🧠 Validação Inteligente de Dados Nutricionais")
st.markdown("Carregue os **dados a corrigir** e uma ou mais **bases de referência** para treinar o modelo.")

# Upload dos dados a corrigir
alvo_file = st.file_uploader("📄 Envie o arquivo com os dados a serem corrigidos (dados_alvo.csv):", type="csv")

# Upload múltiplo das bases de referência
referencia_files = st.file_uploader("📁 Envie uma ou mais bases de referência:", type="csv", accept_multiple_files=True)

# Escolher o modelo
modelo_opcao = st.selectbox("🧠 Escolha o modelo de IA:", ["Regressão Linear", "Rede Neural (MLP)", "Árvore de Decisão"])

# Quando todos os arquivos estiverem carregados
if alvo_file and referencia_files:

    # Ler e unificar as bases de referência
    dfs_ref = [pd.read_csv(ref) for ref in referencia_files]
    df_referencia = pd.concat(dfs_ref, ignore_index=True)

    # Ler os dados alvo
    df_alvo = pd.read_csv(alvo_file)

    # Separar variáveis
    X_ref = df_referencia[["umidade", "proteina", "cinzas", "fibras"]]
    y_ref = df_referencia[["umidade_ref", "proteina_ref", "cinzas_ref", "fibras_ref"]]
    X_alvo = df_alvo[["umidade", "proteina", "cinzas", "fibras"]]

    # Treinar e aplicar modelo
    if modelo_opcao == "Regressão Linear":
        modelo = treinar_linear(X_ref, y_ref)
        y_pred = modelo.predict(X_alvo)

    elif modelo_opcao == "Rede Neural (MLP)":
        modelo, scalerX, scalerY = treinar_mlp(X_ref, y_ref)
        X_alvo_scaled = scalerX.transform(X_alvo)
        y_pred_scaled = modelo.predict(X_alvo_scaled)
        y_pred = scalerY.inverse_transform(y_pred_scaled)

    elif modelo_opcao == "Árvore de Decisão":
        modelo = treinar_tree(X_ref, y_ref)
        y_pred = modelo.predict(X_alvo)

    # Mostrar resultado
    df_corrigido = pd.DataFrame(y_pred, columns=["umidade_corr", "proteina_corr", "cinzas_corr", "fibras_corr"])
    resultado = pd.concat([df_alvo, df_corrigido], axis=1)

    st.subheader("📊 Resultado Corrigido")
    st.dataframe(resultado)

    # Botão de download
    st.download_button("⬇️ Baixar Resultado Corrigido", resultado.to_csv(index=False), file_name="resultado_corrigido.csv")

