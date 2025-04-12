import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, confusion_matrix
import seaborn as sns

# Configurações iniciais
df_result = pd.read_parquet("./data/processed/resultados_producao.parquet")

st.set_page_config(page_title="Dashboard de Monitoramento", layout="wide")
st.title("Dashboard de Monitoramento do Modelo em Produção")

# Sidebar com filtros
st.sidebar.header("Filtros")
periodos = df_result["period"].unique()
periodo_sel = st.sidebar.selectbox("Filtrar por Período do Jogo", options=sorted(periodos))

# Filtrar dados
df_filtered = df_result[df_result["period"] == periodo_sel]

# Métricas globais
st.subheader("Métricas de Performance")
logloss = log_loss(df_filtered["real"], df_filtered["probabilidade"])
f1 = f1_score(df_filtered["real"], df_filtered["previsto"])
st.metric("Log Loss", f"{logloss:.4f}")
st.metric("F1-Score", f"{f1:.4f}")

# Matriz de confusão
st.subheader(":bar_chart: Matriz de Confusão")
cm = confusion_matrix(df_filtered["real"], df_filtered["previsto"])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Erro", "Acerto"], yticklabels=["Erro", "Acerto"], ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
st.pyplot(fig)

# Distribuição das probabilidades
st.subheader(":chart_with_upwards_trend: Distribuição das Probabilidades Previstas")
fig2, ax2 = plt.subplots()
sns.histplot(df_filtered["probabilidade"], bins=20, kde=True, ax=ax2)
ax2.set_title("Distribuição das Probabilidades de Acerto")
st.pyplot(fig2)

# Tabela com previsões detalhadas
st.subheader(":clipboard: Amostra de Previsões")
st.dataframe(df_filtered.sample(10).reset_index(drop=True))