import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dashboard Fase 1 - Análise de Currículos", layout="wide")

# inserir imagem e alterar cor side bar
with st.sidebar:
    st.image("agenteia.png", width=220)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #E6E0F8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Funções básicas mantidas ---

def carregar_arquivo_resultado(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()

def extrair_tabela_resumo(texto):
    tabela_match = re.search(r"=== TABELA RESUMO ===\n+```(.*?)```", texto, re.DOTALL)
    if tabela_match:
        tabela_bruta = tabela_match.group(1)
        linhas = [l.strip() for l in tabela_bruta.split("\n") if l.strip() and not l.startswith("|") == False]
        dados = []
        for linha in linhas[1:]:
            colunas = [c.strip() for c in linha.strip("|").split("|")]
            if len(colunas) == 2:
                nome, nota = colunas
                try:
                    dados.append({"Nome": nome, "Nota Final": float(nota)})
                except ValueError:
                    pass
        return pd.DataFrame(dados).sort_values("Nota Final", ascending=False)
    return pd.DataFrame()

@st.cache_data
def extrair_analises_individuais(texto):
    blocos = re.split(r"Candidato: (.+?\.pdf)\n", texto)[1:]
    candidatos = []
    for i in range(0, len(blocos), 2):
        nome_pdf = blocos[i]
        conteudo = blocos[i+1].strip()
        nome_real = re.search(r"### An\u00e1lise do Curr\u00edculo de (.+)", conteudo)
        nome_real = nome_real.group(1).strip() if nome_real else nome_pdf.replace(".pdf", "")
        tabela_match = re.search(r"#### Tabela de Pontua\u00e7\u00e3o\n\n(.*?)\n\| \*\*Pontua\u00e7\u00e3o Final\*\*", conteudo, re.DOTALL)
        criterios = []
        if tabela_match:
            linhas = [l for l in tabela_match.group(1).split("\n") if l.startswith("|")]
            for linha in linhas[1:]:
                partes = [p.strip() for p in linha.strip("|").split("|")]
                if len(partes) == 4:
                    try:
                        nota = float(partes[1])
                        peso = float(partes[2])
                        pontuacao = float(partes[3])
                        criterios.append({
                            "Critério": partes[0],
                            "Nota": nota,
                            "Peso (%)": peso,
                            "Pontuação": pontuacao
                        })
                    except ValueError:
                        continue
        candidatos.append({"arquivo": nome_pdf, "nome": nome_real, "conteudo": conteudo, "criterios": criterios})
    return candidatos

def criar_df_criterios_todos(candidatos):
    linhas = []
    for c in candidatos:
        nome = c["nome"]
        for crit in c["criterios"]:
            linhas.append({
                "Nome": nome,
                "Critério": crit["Critério"],
                "Nota": crit["Nota"],
                "Peso (%)": crit["Peso (%)"],
                "Pontuação": crit["Pontuação"]
            })
    return pd.DataFrame(linhas)

def plotar_grafico_criterios_interativo(criterios, nome):
    import plotly.graph_objects as go
    df = pd.DataFrame(criterios)
    fig = go.Figure(go.Bar(
        x=df["Pontuação"],
        y=df["Critério"],
        orientation='h',
        marker=dict(color='rgba(128, 0, 128, 0.7)', line=dict(color='rgba(128, 0, 128, 1.0)', width=1))
    ))
    fig.update_layout(title=f"Pontuação por Critério - {nome}",
                      xaxis_title="Pontuação Total (Nota x Peso)",
                      yaxis_title="Critério",
                      yaxis=dict(autorange="reversed"))
    fig.update_traces(text=df["Pontuação"].round(2), textposition='outside')
    return fig

def gerar_grafico_histograma_matplotlib(df_notas):
    fig, ax = plt.subplots()
    ax.hist(df_notas["Nota Final"], bins=20, color='#A020F0', edgecolor='black')
    ax.set_title("Distribuição das Notas Finais dos Candidatos")
    ax.set_xlabel("Nota Final")
    ax.set_ylabel("Frequência")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def top_n_por_criterio(df_criterios, top_n=3):
    st.subheader(f"🏅 Top {top_n} candidatos por critério")
    criterios_unicos = df_criterios["Critério"].unique()
    
    for crit in criterios_unicos:
        df_top = (
            df_criterios[df_criterios["Critério"] == crit]
            .sort_values("Pontuação", ascending=False)
            .head(top_n)
        )
        st.markdown(f"**{crit}**")
        st.dataframe(df_top[["Nome", "Pontuação"]].reset_index(drop=True), use_container_width=True)

def gerar_grafico_barra_matplotlib(df_media_criterio):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df_media_criterio["Critério"], df_media_criterio["Média_Pontuação"], color='#800080')
    ax.set_title("Média da Pontuação por Critério")
    ax.set_xlabel("Pontuação Média")
    ax.set_ylabel("Critério")
    ax.invert_yaxis()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def gerar_grafico_correlacao_matplotlib(corr):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap='Purples')
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlação entre Critérios", pad=20)
    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val < 0.5 else 'black')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def gerar_relatorio_pdf(df_notas, df_media_criterio, corr):
    st.info("Gerando relatório PDF, aguarde...")

    # Gera imagens matplotlib
    img_hist = gerar_grafico_histograma_matplotlib(df_notas)
    img_bar = gerar_grafico_barra_matplotlib(df_media_criterio)
    img_corr = gerar_grafico_correlacao_matplotlib(corr)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Relatório de Análise de Currículos - Analytics", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "1. Distribuição das Notas Finais", ln=True)

    # Salva imagem do buffer em arquivo temporário
    with open("temp_hist.png", "wb") as f:
        f.write(img_hist.read())
    pdf.image("temp_hist.png", w=170)
    img_hist.seek(0)  # Reset buffer para o caso de reutilização

    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, f"""
Estatísticas principais:
Média: {df_notas['Nota Final'].mean():.2f}
Mediana: {df_notas['Nota Final'].median():.2f}
Mínimo: {df_notas['Nota Final'].min():.2f}
Máximo: {df_notas['Nota Final'].max():.2f}
Desvio Padrão: {df_notas['Nota Final'].std():.2f}
    """)

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "2. Média de Pontuação por Critério", ln=True)

    with open("temp_bar.png", "wb") as f:
        f.write(img_bar.read())
    pdf.image("temp_bar.png", w=170)
    img_bar.seek(0)

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "3. Matriz de Correlação entre Critérios", ln=True)

    with open("temp_corr.png", "wb") as f:
        f.write(img_corr.read())
    pdf.image("temp_corr.png", w=170)
    img_corr.seek(0)

    # Remove arquivos temporários
    import os
    os.remove("temp_hist.png")
    os.remove("temp_bar.png")
    os.remove("temp_corr.png")

    return pdf.output(dest='S').encode('latin1')


def analytics_aba(candidatos):
    st.title("📊 Analytics - Insights para Tomada de Decisão")
    df_criterios = criar_df_criterios_todos(candidatos)
    if df_criterios.empty:
        st.warning("Sem dados suficientes para análise.")
        return

    st.subheader("1. Distribuição das Notas Finais")
    df_notas = pd.DataFrame({
        "Nome": [c["nome"] for c in candidatos],
        "Nota Final": [sum([x["Pontuação"] for x in c["criterios"]]) for c in candidatos]
    })

    import plotly.express as px
    bins = list(range(0, 110, 10))
    labels = [f"{i}–{i+10}" for i in bins[:-1]]
    df_notas["Faixa de Nota"] = pd.cut(df_notas["Nota Final"], bins=bins, labels=labels, include_lowest=True)

    df_faixas = df_notas["Faixa de Nota"].value_counts().sort_index().reset_index()
    df_faixas.columns = ["Faixa de Nota", "Quantidade de Candidatos"]

    fig = px.bar(
        df_faixas,
        x="Quantidade de Candidatos",
        y="Faixa de Nota",
        orientation="h",
        title="Distribuição dos Candidatos por Faixa de Nota",
        labels={"Quantidade de Candidatos": "Qtd. de Candidatos", "Faixa de Nota": "Faixa de Nota Final"},
        text="Quantidade de Candidatos",
        color_discrete_sequence=["#6a0dad"]  # púrpura escuro
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_categoryorder='category ascending',
                      plot_bgcolor='rgba(240, 230, 250, 0.3)',
                      paper_bgcolor='rgba(240, 230, 250, 0.3)',
                      font=dict(color="#4b0082"))  # roxo índigo

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
**Estatísticas:**
- Média: {df_notas['Nota Final'].mean():.2f}  
- Mediana: {df_notas['Nota Final'].median():.2f}  
- Mínimo: {df_notas['Nota Final'].min():.2f}  
- Máximo: {df_notas['Nota Final'].max():.2f}  
- Desvio Padrão: {df_notas['Nota Final'].std():.2f}  
""")

    st.subheader("2. Média de Pontuação por Critério")
    df_media_criterio = df_criterios.groupby("Critério").agg(
        Média_Pontuação=("Pontuação", "mean"),
        Count=("Pontuação", "count")
    ).reset_index().sort_values(by="Média_Pontuação", ascending=False)

    fig_bar = px.bar(
        df_media_criterio,
        x="Média_Pontuação",
        y="Critério",
        orientation="h",
        color="Média_Pontuação",
        color_continuous_scale=px.colors.sequential.Purples,  # escala púrpura
        title="Média da Pontuação por Critério"
    )
    fig_bar.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor='rgba(240, 230, 250, 0.3)',
        paper_bgcolor='rgba(240, 230, 250, 0.3)',
        font=dict(color="#4b0082")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("3. Matriz de Correlação entre Critérios")
    df_pivot = df_criterios.pivot(index="Nome", columns="Critério", values="Pontuação").fillna(0)
    corr = df_pivot.corr()

    import plotly.figure_factory as ff
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale=px.colors.sequential.Purples,
        showscale=True,
        reversescale=False,
        annotation_text=corr.round(2).values,
        hoverinfo="z"
    )
    fig_corr.update_layout(
        title="Correlação entre Critérios",
        plot_bgcolor='rgba(240, 230, 250, 0.3)',
        paper_bgcolor='rgba(240, 230, 250, 0.3)',
        font=dict(color="#4b0082")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📊 Top candidatos por critério", expanded=False):
        top_n = st.slider("Número de candidatos por critério", 1, 10, 3)
        top_n_por_criterio(df_criterios, top_n)

    if st.button("📄 Gerar relatório PDF"):
        pdf_bytes = gerar_relatorio_pdf(df_notas, df_media_criterio, corr)
        st.success("Relatório PDF gerado com sucesso!")
        st.download_button(
            label="⬇️ Baixar Relatório PDF",
            data=pdf_bytes,
            file_name="relatorio_analytics_curriculos.pdf",
            mime="application/pdf"
        )


def comparar_candidatos(candidatos):
    st.title("🔍 Comparativo de Candidatos")
    nomes_disponiveis = [c["nome"] for c in candidatos]
    if len(nomes_disponiveis) < 2:
        st.warning("É necessário pelo menos dois candidatos na lista filtrada para fazer a comparação.")
        return

    col1, col2 = st.columns(2)
    with col1:
        nome1 = st.selectbox("Candidato 1", nomes_disponiveis, key="cand1")
    with col2:
        nome2 = st.selectbox("Candidato 2", nomes_disponiveis, key="cand2")

    if nome1 == nome2:
        st.warning("Selecione dois candidatos diferentes para comparar.")
        return

    c1 = next(c for c in candidatos if c["nome"] == nome1)
    c2 = next(c for c in candidatos if c["nome"] == nome2)

    st.subheader("📊 Tabela Comparativa de Critérios")
    criterios1 = {c["Critério"]: c for c in c1["criterios"]}
    criterios2 = {c["Critério"]: c for c in c2["criterios"]}
    todos_criterios = sorted(set(criterios1.keys()).union(criterios2.keys()))

    dados_comparativos = []
    for crit in todos_criterios:
        linha = {
            "Critério": crit,
            f"Nota - {nome1}": criterios1.get(crit, {}).get("Nota", "—"),
            f"Nota - {nome2}": criterios2.get(crit, {}).get("Nota", "—"),
            f"Pontuação - {nome1}": criterios1.get(crit, {}).get("Pontuação", "—"),
            f"Pontuação - {nome2}": criterios2.get(crit, {}).get("Pontuação", "—"),
        }
        dados_comparativos.append(linha)

    df_comp = pd.DataFrame(dados_comparativos)
    st.dataframe(df_comp, use_container_width=True)

    st.subheader("📝 Análises Individuais")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {nome1}")
        st.markdown(c1["conteudo"])
    with col2:
        st.markdown(f"### {nome2}")
        st.markdown(c2["conteudo"])


def main():
    texto_resultado = carregar_arquivo_resultado("resultado_completo.txt")
    df_resumo = extrair_tabela_resumo(texto_resultado)
    candidatos = extrair_analises_individuais(texto_resultado)

    if not df_resumo.empty:
        min_nota = float(df_resumo["Nota Final"].min())
        max_nota = float(df_resumo["Nota Final"].max())
    else:
        min_nota, max_nota = 0.0, 10.0

    nota_min, nota_max = st.sidebar.slider(
        "Filtrar candidatos por faixa de Nota Final:",
        min_value=min_nota,
        max_value=max_nota,
        value=(min_nota, max_nota),
        step=0.1
    )

    df_filtrado = df_resumo[(df_resumo["Nota Final"] >= nota_min) & (df_resumo["Nota Final"] <= nota_max)]
    nomes_filtrados = df_filtrado["Nome"].tolist()
    candidatos_filtrados = [c for c in candidatos if c["nome"] in nomes_filtrados]

    aba = st.sidebar.radio("Navegação", ["Ranking", "Comparar Candidatos", "Analytics"])

    if aba == "Ranking":
        st.title("📋 Dashboard - Análise Individual de Currículos (Fase 1)")
        st.subheader("🏆 Ranking de Candidatos")
        st.dataframe(df_filtrado, use_container_width=True)

        if nomes_filtrados:
            selecao = st.selectbox("Selecione um candidato para ver os detalhes:", nomes_filtrados)
            candidato = next(c for c in candidatos_filtrados if c["nome"] == selecao)

            st.markdown(f"### 🧑‍💼 {candidato['nome']}")
            if candidato["criterios"]:
                fig = plotar_grafico_criterios_interativo(candidato["criterios"], candidato["nome"])
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(candidato['conteudo'])
        else:
            st.warning("Nenhum candidato encontrado para a faixa de notas selecionada.")

    elif aba == "Comparar Candidatos":
        comparar_candidatos(candidatos_filtrados)

    elif aba == "Analytics":
        analytics_aba(candidatos_filtrados)


if __name__ == "__main__":
    main()

