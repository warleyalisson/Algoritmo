import streamlit as st
import pandas as pd
from linear_regression import treinar_modelo as treinar_linear
from mlp_regression import treinar_modelo as treinar_mlp
from decision_tree import treinar_modelo as treinar_tree

# Configuração da interface
st.set_page_config(page_title="Validador IA de Dados da Araruta", layout="centered")
st.title("🧠 Validação Inteligente de Dados Nutricionais da Araruta")

# Upload dos arquivos
st.markdown("### 📥 Envie seus arquivos:")
alvo_file = st.file_uploader("📄 Arquivo com os dados a serem corrigidos (dados_alvo.csv):", type="csv")
referencia_files = st.file_uploader("📁 Arquivo(s) com bases de referência (com valores padrão):", type="csv", accept_multiple_files=True)

# Seleção do modelo
modelo_opcao = st.selectbox("🧠 Escolha o modelo de IA:", [
    "Regressão Linear", 
    "Rede Neural (MLP)", 
    "Árvore de Decisão"
])

# Ação principal
if alvo_file and referencia_files:
    try:
        # Leitura dos arquivos
        df_alvo = pd.read_csv(alvo_file)
        dfs_referencia = [pd.read_csv(ref) for ref in referencia_files]
        df_referencia = pd.concat(dfs_referencia, ignore_index=True)

        # Verificação de colunas obrigatórias
        col_X = ["umidade", "proteina", "cinzas", "fibras"]
        col_y = ["umidade_ref", "proteina_ref", "cinzas_ref", "fibras_ref"]
        
        if not all(col in df_alvo.columns for col in col_X):
            st.error("⚠️ O arquivo alvo deve conter as colunas: umidade, proteina, cinzas, fibras.")
        elif not all(col in df_referencia.columns for col in col_X + col_y):
            st.error("⚠️ As bases de referência devem conter as colunas: umidade, proteina, cinzas, fibras + os respectivos *_ref.")
        else:
            # Separar entradas e saídas
            X_ref = df_referencia[col_X]
            y_ref = df_referencia[col_y]
            X_alvo = df_alvo[col_X]

            # Aplicar o modelo
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

            # Resultado final
            df_corrigido = pd.DataFrame(y_pred, columns=["umidade_corr", "proteina_corr", "cinzas_corr", "fibras_corr"])
            df_resultado = pd.concat([df_alvo, df_corrigido], axis=1)

            st.success("✅ Dados corrigidos com sucesso!")
            st.dataframe(df_resultado)
            st.download_button("⬇️ Baixar resultado corrigido", df_resultado.to_csv(index=False), file_name="resultado_corrigido.csv")

    except Exception as e:
        st.error(f"❌ Erro ao processar os arquivos: {e}")
else:
    st.info("🔁 Aguardando envio dos dois arquivos...")
