"""
AEGIS — Painel de previsão de dengue (Streamlit)
Global Solution 2026.2 · CitizenTech · FIAP · Data Science & Statistical Computing

Como rodar (dentro da pasta deste arquivo):
    pip install -r requirements.txt
    streamlit run app.py

O painel lê a base tratada gerada pelo notebook (dengue_base_tratada.csv), treina os mesmos
dois modelos com a mesma divisão treino/teste, refaz a validação cruzada no tempo (que escolhe o
limiar do alerta) e confere os resultados com metricas_notebook.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Visual dos gráficos (matplotlib): fundo branco, traços discretos e títulos em negrito
plt.rcParams.update({
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "font.size": 10,
    "axes.edgecolor": "#c9d1dc",
    "axes.labelcolor": "#3d4652",
    "axes.titlesize": 11,
    "xtick.color": "#5f6b7a",
    "ytick.color": "#5f6b7a",
    "legend.frameon": False,
})

# Visual da página (CSS): cabeçalho, cartões dos indicadores e abas; funciona no tema claro e no escuro
ESTILO_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}
.aegis-topo {border-left: 5px solid #ff4b4b; background: rgba(128, 128, 128, 0.08); border-radius: 10px;
             padding: 18px 24px; margin: 0 0 1.2rem 0;}
.aegis-selo {font-size: 0.75rem; letter-spacing: .07em; text-transform: uppercase; opacity: 0.7; font-weight: 600;}
.aegis-titulo {font-size: 2.05rem; font-weight: 700; margin: 2px 0 6px 0; line-height: 1.2;}
.aegis-subtitulo {font-size: 0.98rem; opacity: 0.85; max-width: 1000px; line-height: 1.5;}
.aegis-rodape {text-align: center; opacity: 0.6; font-size: 0.8rem; padding-top: 0.4rem;}
[data-testid="stMetric"] {background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.25);
                          border-radius: 10px; padding: 12px 16px;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 650;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {padding: 10px 14px; font-weight: 600;}
h4 {margin-top: 1.2rem;}
</style>
"""

# =============================================================================
# Configurações (as mesmas do notebook)
# =============================================================================
PASTA = Path(__file__).parent
ARQUIVO_BASE = PASTA / "dengue_base_tratada.csv"
ARQUIVO_METRICAS = PASTA / "metricas_notebook.json"

FEATURES_LINEAR = ["incidencia", "incidencia_1sem_antes", "incidencia_2sem_antes", "tempmed", "umidmed"]
FEATURES_LOGISTICA = FEATURES_LINEAR + ["acima_limiar_hoje"]
ALVO_LINEAR = "incidencia_em_4_sem"
ALVO_LOGISTICA = "epidemica_em_4_sem"
COLUNAS_OBRIGATORIAS = ["municipio_nome", "SE", "data_iniSE", "pop", "casos", "nivel_inc", "SE_alvo", "data_alvo",
                        "conjunto", "nivel_inc_em_4_sem"] + FEATURES_LOGISTICA + [ALVO_LINEAR, ALVO_LOGISTICA]
NOMES = {
    "incidencia": "incidência hoje",
    "incidencia_1sem_antes": "incidência 1 semana antes",
    "incidencia_2sem_antes": "incidência 2 semanas antes",
    "tempmed": "temperatura média (°C)",
    "umidmed": "umidade média (%)",
    "acima_limiar_hoje": "acima do limiar hoje (0/1)",
    ALVO_LINEAR: "ALVO: incidência em 4 semanas",
    ALVO_LOGISTICA: "ALVO: epidêmica em 4 semanas",
}
CORES = {
    "São Paulo": "#2a78d6",
    "Osasco": "#eb6834",
    "Santo André": "#1baf7a",
    "São Bernardo do Campo": "#eda100",
    "São Caetano do Sul": "#e87ba4",
}
MUNICIPIOS = list(CORES)
SITUACOES = ["1. Continua abaixo do limiar", "2. Continua acima do limiar",
             "3. Entra em período epidêmico", "4. Sai do período epidêmico"]
EQUIPE = [("Enzo Fernandes Ramos", "563705"), ("Felipe Henrique de Souza Cerazi", "562746"),
          ("Gustavo Peaguda de Castro", "562923"), ("Lorenzo Andolfatto Coque", "563385")]


# =============================================================================
# Funções de apoio
# =============================================================================
def br(numero, casas=2):
    """Formata um número no padrão brasileiro: 1.234,56"""
    if numero is None or pd.isna(numero):
        return "—"
    texto = f"{numero:,.{casas}f}"
    return texto.replace(",", "#").replace(".", ",").replace("#", ".")


def variacao(valor, casas):
    """Texto da variação para o st.metric (sempre começa com + ou -, para a seta sair certa)."""
    return ("+" if valor >= 0 else "") + br(valor, casas)


def semana(se):
    """202633 -> 'SE 33/2026'"""
    se = int(se)
    return f"SE {se % 100:02d}/{se // 100}"


def formatar(df, casas):
    """Copia a tabela trocando os números por texto no padrão brasileiro."""
    saida = df.copy()
    for coluna in saida.columns:
        if pd.api.types.is_numeric_dtype(saida[coluna]):
            saida[coluna] = saida[coluna].apply(lambda v: br(v, casas))
    return saida


def metricas_regressao(real, previsto):
    return {
        "MAE": float(mean_absolute_error(real, previsto)),
        "RMSE": float(np.sqrt(mean_squared_error(real, previsto))),
        "R²": float(r2_score(real, previsto)),
    }


def metricas_classificacao(real, previsto):
    return {
        "Acurácia": float(accuracy_score(real, previsto)),
        "Precisão": float(precision_score(real, previsto, zero_division=0)),
        "Recall": float(recall_score(real, previsto, zero_division=0)),
        "F1": float(f1_score(real, previsto, zero_division=0)),
    }


def mostrar_figura(fig):
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# Dados e modelos
# =============================================================================
@st.cache_data(show_spinner="Carregando a base tratada...")
def carregar_base():
    base = pd.read_csv(ARQUIVO_BASE)
    for coluna in ["data_iniSE", "data_alvo"]:
        if coluna in base.columns:
            base[coluna] = pd.to_datetime(base[coluna])
    return base


def problemas_da_base(base):
    """Confere se o CSV tem o que o painel precisa. Devolve a lista de problemas (vazia = tudo certo)."""
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in base.columns]
    if faltando:
        return ["Colunas ausentes: " + ", ".join(faltando)]
    problemas = []
    vazios = int(base[FEATURES_LOGISTICA + [ALVO_LINEAR, ALVO_LOGISTICA]].isna().sum().sum())
    if vazios:
        problemas.append(f"{vazios} valores ausentes nas entradas ou nos alvos")
    if not set(base["conjunto"]).issubset({"treino", "teste"}) or base["conjunto"].nunique() != 2:
        problemas.append("a coluna 'conjunto' deve ter apenas os valores 'treino' e 'teste'")
    if not set(MUNICIPIOS).issubset(set(base["municipio_nome"])):
        problemas.append("faltam municípios na base")
    if base.duplicated(["municipio_nome", "SE"]).any():
        problemas.append("há semanas repetidas para o mesmo município")
    return problemas


@st.cache_data(show_spinner="Treinando os modelos...")
def treinar_modelos():
    """Mesma divisão treino/teste e mesmos modelos do notebook (seções 2.6, 4 e 5)."""
    base = carregar_base()
    treino = base[base["conjunto"] == "treino"].copy()
    teste = base[base["conjunto"] == "teste"].copy()

    linear = LinearRegression().fit(treino[FEATURES_LINEAR], treino[ALVO_LINEAR])
    padronizador = StandardScaler().fit(treino[FEATURES_LOGISTICA])  # aprende só com o treino
    logistica = LogisticRegression(max_iter=1000).fit(padronizador.transform(treino[FEATURES_LOGISTICA]),
                                                      treino[ALVO_LOGISTICA])
    teste["previsto"] = linear.predict(teste[FEATURES_LINEAR])
    teste["probabilidade"] = logistica.predict_proba(padronizador.transform(teste[FEATURES_LOGISTICA]))[:, 1]
    return {"treino": treino, "teste": teste, "linear": linear, "padronizador": padronizador, "logistica": logistica}


def ler_metricas_notebook():
    if not ARQUIVO_METRICAS.exists():
        return None
    try:
        return json.loads(ARQUIVO_METRICAS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# =============================================================================
# Gráficos (matplotlib)
# =============================================================================
def figura_matriz(matriz, limiar):
    rotulos = [["Acertou\n(não era epidemia)", "Alarme falso\n(FP)"],
               ["Epidemia não antecipada\n(FN)", "Acertou\n(epidemia antecipada)"]]
    cores = [["#d6f0e1", "#fbd5d5"], ["#fbd5d5", "#d6f0e1"]]  # verde = acerto, vermelho = erro
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=cores[i][j], edgecolor="white", linewidth=4))
            ax.text(j + 0.5, 1 - i + 0.62, str(matriz[i, j]), ha="center", va="center", fontsize=24,
                    fontweight="bold", color="#1f2328")
            ax.text(j + 0.5, 1 - i + 0.28, rotulos[i][j], ha="center", va="center", fontsize=9.5, color="#1f2328")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["Previsto: não epidêmica", "Previsto: epidêmica"])
    ax.set_yticks([1.5, 0.5], ["Real: não\nepidêmica", "Real:\nepidêmica"])
    ax.tick_params(length=0)
    for borda in ax.spines.values():
        borda.set_visible(False)
    ax.set_title(f"Matriz de confusão — teste (limiar {br(limiar, 2)})", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figura_probabilidade(dados, limiar, municipio):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(dados["data_alvo"], 0, 1, where=(dados[ALVO_LOGISTICA] == 1).to_numpy(),
                    step="mid", color="#c3c2b7", alpha=0.6, linewidth=0, label="Semana realmente epidêmica")
    ax.plot(dados["data_alvo"], dados["probabilidade"], color="#2a78d6", linewidth=2,
            label="Probabilidade prevista (4 semanas antes)")
    ax.axhline(limiar, color="#3d3d3a", linestyle="--", linewidth=1, label=f"Limiar {br(limiar, 2)}")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda valor, _: br(valor, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_title(f"{municipio}: probabilidade prevista × semanas epidêmicas reais (teste)", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def figura_sigmoide(pontuacao, probabilidade, real, limiar, mostrar_reta):
    """Curva sigmoide: cada semana do teste fica no 0 (não epidêmica) ou no 1 (epidêmica), na posição da pontuação."""
    z = np.asarray(pontuacao, dtype=float)
    real = np.asarray(real, dtype=int)
    acerto = (np.asarray(probabilidade) >= limiar).astype(int) == real
    z_corte = float(np.log(limiar / (1 - limiar)))  # pontuação em que a chance é igual ao limiar
    deslocamento = np.random.default_rng(0).uniform(-0.035, 0.035, len(real))  # só para os pontos não se sobreporem
    y = real + deslocamento
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.scatter(z[acerto], y[acerto], s=16, color="#1baf7a", alpha=0.55, linewidths=0,
               label=f"semana que o modelo acertou ({int(acerto.sum())})")
    ax.scatter(z[~acerto], y[~acerto], s=24, color="#e5484d", alpha=0.9, linewidths=0,
               label=f"semana que o modelo errou ({int((~acerto).sum())})")
    grade = np.linspace(min(z.min(), -6.0) - 0.5, max(z.max(), 6.0) + 0.5, 400)
    ax.plot(grade, 1 / (1 + np.exp(-grade)), color="#2a78d6", linewidth=2.6,
            label="curva sigmoide = chance prevista")
    if mostrar_reta:
        inclinacao, intercepto = np.polyfit(z, real, 1)
        ax.plot(grade, inclinacao * grade + intercepto, color="#8a8f98", linestyle="--", linewidth=1.6,
                label="reta (como seria uma regressão linear)")
        ax.set_ylim(-0.6, 1.6)
    else:
        ax.set_ylim(-0.12, 1.12)
    ax.axhline(limiar, color="#3d3d3a", linestyle="--", linewidth=1, label=f"limiar {br(limiar, 2)}")
    ax.axvline(z_corte, color="#3d3d3a", linestyle=":", linewidth=1)
    ax.set_xlim(grade.min(), grade.max())
    ax.text(z_corte - 0.25, 0.62, "sem alerta  ←", ha="right", va="center", fontsize=9, color="#5f6b7a")
    ax.text(z_corte + 0.25, 0.62, "→  alerta", ha="left", va="center", fontsize=9, color="#5f6b7a")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(lambda valor, _: br(valor, 2))
    ax.set_xlabel("pontuação do modelo (z): soma dos pesos × pistas da semana")
    ax.set_ylabel("chance prevista  |  real: 0 ou 1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_title("Regressão logística: a curva sigmoide transforma a pontuação em chance (teste)", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def figura_roc(teste, limiar):
    real = teste[ALVO_LOGISTICA]
    taxa_fp, taxa_vp, _ = roc_curve(real, teste["probabilidade"])
    auc_modelo = roc_auc_score(real, teste["probabilidade"])
    auc_ref = roc_auc_score(real, teste["acima_limiar_hoje"])
    vn, fp, fn, vp = confusion_matrix(real, (teste["probabilidade"] >= limiar).astype(int), labels=[0, 1]).ravel()
    rvn, rfp, rfn, rvp = confusion_matrix(real, teste["acima_limiar_hoje"], labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(taxa_fp, taxa_vp, color="#2a78d6", linewidth=2.2, label=f"Regressão logística (AUC = {br(auc_modelo, 3)})")
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linestyle="--", linewidth=1, label="Sorteio (AUC = 0,5)")
    ax.scatter([fp / (fp + vn)], [vp / (vp + fn)], s=110, facecolors="none", edgecolors="#eb6834", linewidths=2,
               zorder=3, label=f"limiar {br(limiar, 2)}")
    ax.scatter([rfp / (rfp + rvn)], [rvp / (rvp + rfn)], s=60, color="#3d3d3a", zorder=3,
               label=f"Referência: repetir hoje (AUC = {br(auc_ref, 3)})")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.xaxis.set_major_formatter(lambda valor, _: br(valor, 1))
    ax.yaxis.set_major_formatter(lambda valor, _: br(valor, 1))
    ax.set_xlabel("taxa de falsos positivos")
    ax.set_ylabel("taxa de verdadeiros positivos (recall)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Curva ROC — teste", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig, auc_modelo, auc_ref


def figura_pesos(rob):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for (nome, f1s), cor in zip(rob["curvas_f1_pesos"].items(), ["#2a78d6", "#eb6834"]):
        ax.plot(f1s.index, f1s.values, marker="o", color=cor, label=nome)
    ax.xaxis.set_major_formatter(lambda valor, _: br(valor, 2))
    ax.yaxis.set_major_formatter(lambda valor, _: br(valor, 3))
    ax.set_xlabel("limiar do alerta")
    ax.set_ylabel("F1 na validação")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("F1 por limiar: sem pesos × class_weight='balanced'", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figura_real_previsto(teste, municipio):
    """Mesmo gráfico da seção 4.5 do notebook, com o município da barra lateral em destaque."""
    real, previsto = teste[ALVO_LINEAR], teste["previsto"]
    destaque = (teste["municipio_nome"] == municipio).to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(real[~destaque], previsto[~destaque], s=18, alpha=0.45, color="#9ec5f4", linewidths=0,
               label="outros municípios")
    ax.scatter(real[destaque], previsto[destaque], s=22, alpha=0.8, color=CORES[municipio], linewidths=0,
               label=municipio)
    limite = max(real.max(), previsto.max()) * 1.05
    ax.plot([0, limite], [0, limite], color="#3d3d3a", linestyle="--", linewidth=1, label="previsão perfeita")
    ax.text(0.03, 0.97, "acima da linha:\no modelo previu a mais", transform=ax.transAxes, va="top", fontsize=9,
            color="#5f6b7a")
    ax.text(0.97, 0.03, "abaixo da linha:\no modelo previu a menos", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#5f6b7a")
    ax.set_xlabel("Incidência real (casos/100 mil)")
    ax.set_ylabel("Incidência prevista (casos/100 mil)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=8.5)
    ax.set_title("Real × previsto — teste\n(linha tracejada = previsão perfeita)", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figura_residuos(residuos):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(residuos, bins=50, color="#eb6834", alpha=0.85)
    ax.axvline(0, color="#3d3d3a", linestyle="--", linewidth=1)
    ax.set_xlabel("Resíduo = real − previsto (casos/100 mil)")
    ax.set_ylabel("nº de semanas")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_title("Distribuição dos resíduos — teste", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figura_correlacao(dados):
    variaveis = FEATURES_LOGISTICA + [ALVO_LINEAR, ALVO_LOGISTICA]
    corr = dados[variaveis].corr()
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.imshow(corr.fillna(0).values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(variaveis)):
        for j in range(len(variaveis)):
            valor = corr.values[i, j]
            valor = 0.0 if abs(valor) < 0.005 else valor  # evita escrever -0,00
            ax.text(j, i, br(valor, 2), ha="center", va="center", fontsize=8.5,
                    color="white" if abs(valor) > 0.6 else "#1f2328")
    rotulos = [NOMES[v] for v in variaveis]
    ax.set_xticks(range(len(variaveis)), rotulos, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(variaveis)), rotulos, fontsize=9)
    ax.set_title("Matriz de correlação de Pearson", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


# =============================================================================
# Abas
# =============================================================================
def aba_visao_geral(base, modelos, municipio):
    teste = modelos["teste"]
    st.markdown("#### O projeto em números")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semanas-município na base tratada", br(len(base), 0),
              help="Cada linha é um município em uma semana, já com entradas e alvos completos.")
    c2.metric("Municípios", str(base["municipio_nome"].nunique()))
    c3.metric("Treino / teste", f"{br(len(modelos['treino']), 0)} / {br(len(teste), 0)}",
              help="Divisão pelo tempo: treino = semanas-alvo de 2018 a 2024; teste = 2025 e 2026.")
    c4.metric("Período (semanas epidemiológicas)",
              f"{semana(base['SE'].min()).replace('SE ', '')} a {semana(base['SE'].max()).replace('SE ', '')}")

    st.markdown("#### Como funciona")
    etapas = [
        ("1. Dados", "Casos de dengue e clima por semana, do InfoDengue (Fiocruz/FGV)."),
        ("2. Tratamento", "Limpeza, incidência por 100 mil habitantes e semanas anteriores como pistas."),
        ("3. Treino e teste", "Os modelos estudam com 2018–2024 e são avaliados em 2025–2026."),
        ("4. Modelos", "Regressão linear (quantos casos) e logística (chance de epidemia)."),
        ("5. Ação", "Recomendações para orientar a população e priorizar vistorias."),
    ]
    for coluna, (titulo, texto) in zip(st.columns(len(etapas)), etapas):
        with coluna:
            with st.container(border=True):
                st.markdown(f"**{titulo}**")
                st.caption(texto)

    st.divider()
    d = base[base["municipio_nome"] == municipio]
    pico = d.loc[d["incidencia"].idxmax()]
    st.markdown(f"#### {municipio}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("População (IBGE)", br(d["pop"].iloc[-1], 0))
    c2.metric("Casos notificados em 2024", br(d.loc[d["SE"] // 100 == 2024, "casos"].sum(), 0),
              help="2024 foi o ano com mais casos notificados da série nos 5 municípios.")
    c3.metric("Pico semanal de incidência", f"{br(pico['incidencia'], 0)} por 100 mil",
              help=f"Semana iniciada em {pico['data_iniSE']:%d/%m/%Y}.")
    c4.metric("Semanas acima do limiar epidêmico", f"{int((d['nivel_inc'] == 2).sum())} de {len(d)}",
              help="Classificação do InfoDengue (nivel_inc = 2) nas semanas da base tratada.")

    ultima = teste[teste["municipio_nome"] == municipio].sort_values("SE").iloc[-1]
    st.markdown(f"**Última previsão avaliada:** feita na {semana(ultima['SE'])} para a {semana(ultima['SE_alvo'])}.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Incidência prevista", f"{br(max(ultima['previsto'], 0), 1)} por 100 mil")
    c2.metric("Incidência real", f"{br(ultima[ALVO_LINEAR], 1)} por 100 mil")
    c3.metric("Probabilidade de semana epidêmica", f"{br(100 * ultima['probabilidade'], 0)}%",
              help="Na semana-alvo, a classificação real foi: "
                   + ("epidêmica." if ultima[ALVO_LOGISTICA] == 1 else "não epidêmica."))

    st.markdown("#### Incidência semanal (casos por 100 mil habitantes)")
    comparar = st.checkbox("Comparar com os outros municípios", value=False)
    if comparar:
        serie = base.pivot_table(index="data_iniSE", columns="municipio_nome", values="incidencia")[MUNICIPIOS]
        st.line_chart(serie, color=[CORES[m] for m in serie.columns])
    else:
        serie = d.set_index("data_iniSE")[["incidencia"]].rename(columns={"incidencia": municipio})
        st.line_chart(serie, color=CORES[municipio])
    st.caption("Quanto mais alta a linha, mais casos por 100 mil moradores naquela semana. "
               "Passe o mouse sobre o gráfico para ver os valores.")


def aba_explorar(base):
    st.markdown("Explore a **base tratada** (`dengue_base_tratada.csv`), gerada pelo notebook na seção 2.7. "
                "Os filtros desta aba não alteram os modelos.")
    f1, f2, f3 = st.columns([2, 1.4, 1])
    with f1:
        escolhidos = st.multiselect("Municípios", MUNICIPIOS, default=MUNICIPIOS)
    with f2:
        ano_min, ano_max = int(base["SE"].min() // 100), int(base["SE"].max() // 100)
        anos = st.slider("Anos (semana t)", min_value=ano_min, max_value=ano_max, value=(ano_min, ano_max))
    with f3:
        conjunto = st.radio("Conjunto", ["Todos", "Treino", "Teste"], horizontal=True)

    if not escolhidos:
        st.warning("Nenhum município selecionado: mostrando todos.")
        escolhidos = MUNICIPIOS
    filtro = base["municipio_nome"].isin(escolhidos) & (base["SE"] // 100).between(anos[0], anos[1])
    if conjunto != "Todos":
        filtro &= base["conjunto"] == conjunto.lower()
    dados = base[filtro]
    if dados.empty:
        st.warning("Nenhuma semana com esses filtros. Ajuste os anos ou o conjunto.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Semanas-município", br(len(dados), 0))
    c2.metric("Semanas epidêmicas 4 semanas depois", f"{br(100 * dados[ALVO_LOGISTICA].mean(), 1)}%")
    c3.metric("Incidência mediana em 4 semanas", f"{br(dados[ALVO_LINEAR].median(), 2)} por 100 mil")

    st.markdown("#### Incidência semanal")
    serie = dados.pivot_table(index="data_iniSE", columns="municipio_nome", values="incidencia")
    colunas = [m for m in MUNICIPIOS if m in serie.columns]
    cores = [CORES[m] for m in colunas]
    st.line_chart(serie[colunas], color=cores if len(cores) > 1 else cores[0])

    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        st.markdown("#### Matriz de correlação")
        if len(dados) >= 10:
            mostrar_figura(figura_correlacao(dados))
            st.caption("Calculada com os dados filtrados. No notebook (seção 3.5) ela usa só o treino. As três "
                       "incidências são muito parecidas entre si, então os pesos delas devem ser lidos em conjunto.")
        else:
            st.info("São necessárias pelo menos 10 semanas para calcular a correlação.")
    with col_b:
        st.markdown("#### Estatísticas descritivas")
        estatisticas = dados[FEATURES_LOGISTICA + [ALVO_LINEAR]].describe().T.rename(index=NOMES)
        estatisticas = estatisticas.rename(columns={"count": "linhas", "mean": "média", "std": "desvio", "min": "mín.",
                                                    "25%": "Q1", "50%": "mediana", "75%": "Q3", "max": "máx."})
        estatisticas["linhas"] = estatisticas["linhas"].astype(int)
        st.dataframe(formatar(estatisticas, 2))

    st.markdown("#### Amostra da base filtrada")
    st.dataframe(dados.head(200), hide_index=True)
    st.download_button("Baixar a base filtrada (CSV)", data=dados.to_csv(index=False).encode("utf-8"),
                       file_name="dengue_base_tratada_filtrada.csv", mime="text/csv")


def aba_linear(modelos, municipio):
    teste, linear = modelos["teste"], modelos["linear"]
    st.markdown("A regressão linear prevê **quantos casos por 100 mil habitantes** o município terá daqui a "
                "4 semanas. É uma **fórmula de soma**: cada pista é multiplicada por um peso, e tudo é somado.")
    c = dict(zip(FEATURES_LINEAR, linear.coef_))
    sinal = lambda v: "+" if v >= 0 else "−"  # noqa: E731
    st.code("incidência daqui a 4 semanas ≈ " + br(linear.intercept_, 3) + "\n"
            + "\n".join(f"    {sinal(c[v])} {br(abs(c[v]), 3)} × {NOMES[v]}" for v in FEATURES_LINEAR),
            language="text")

    real = teste[ALVO_LINEAR]
    m = metricas_regressao(real, teste["previsto"])
    r = metricas_regressao(real, teste["incidencia"])
    st.markdown("#### Resultado no teste (2025–2026), todos os municípios")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (erro médio)", f"{br(m['MAE'], 1)} por 100 mil", variacao(m["MAE"] - r["MAE"], 1) + " vs. repetir hoje",
              delta_color="inverse", help="Em média, quanto a previsão erra. Menor é melhor.")
    c2.metric("RMSE (pesa mais os erros grandes)", f"{br(m['RMSE'], 1)} por 100 mil",
              variacao(m["RMSE"] - r["RMSE"], 1) + " vs. repetir hoje", delta_color="inverse",
              help="Como o MAE, mas castiga mais os erros grandes. Menor é melhor.")
    c3.metric("R² (variação explicada)", br(m["R²"], 3), variacao(m["R²"] - r["R²"], 3) + " vs. repetir hoje",
              help="1 = acerta tudo; 0 = não é melhor que chutar sempre a média. Maior é melhor.")
    st.caption("A seta compara com a referência simples: supor que daqui a 4 semanas a incidência será igual à de "
               "hoje. Verde = o modelo foi melhor que a referência.")

    st.markdown("#### Erro por município (teste)")
    linhas = []
    for mun in MUNICIPIOS:
        dm = teste[teste["municipio_nome"] == mun]
        mm = metricas_regressao(dm[ALVO_LINEAR], dm["previsto"])
        linhas.append({"município": mun, "MAE": mm["MAE"], "RMSE": mm["RMSE"],
                       "MAE em casos por semana": mm["MAE"] * dm["pop"].mean() / 100_000,
                       "previsões negativas": int((dm["previsto"] < 0).sum())})
    por_municipio = pd.DataFrame(linhas)
    exibicao = formatar(por_municipio, 1)
    exibicao["previsões negativas"] = por_municipio["previsões negativas"].astype(str)
    st.dataframe(exibicao, hide_index=True)

    st.markdown(f"#### {municipio}: real × previsto (teste)")
    dm = teste[teste["municipio_nome"] == municipio]
    grafico = (dm.set_index("data_alvo")[[ALVO_LINEAR, "previsto"]]
               .rename(columns={ALVO_LINEAR: "Real", "previsto": "Previsto 4 semanas antes"}))
    grafico["Previsto 4 semanas antes"] = grafico["Previsto 4 semanas antes"].clip(lower=0)
    st.line_chart(grafico, color=["#2a78d6", "#eb6834"])
    st.caption("Quanto mais as duas linhas se sobrepõem, melhor a previsão. No gráfico, previsões negativas aparecem "
               "como zero; as métricas usam os valores originais do modelo.")

    st.markdown("#### Real × previsto e resíduos (teste, todos os municípios)")
    residuos = real - teste["previsto"]
    col_a, col_b = st.columns([1, 1])
    with col_a:
        mostrar_figura(figura_real_previsto(teste, municipio))
    with col_b:
        mostrar_figura(figura_residuos(residuos))
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Como ler o real × previsto**")
        st.markdown(f"- Cada ponto é uma semana do teste ({len(teste)} no total); os pontos de **{municipio}** "
                    "estão em destaque (troque o município na barra lateral).\n"
                    "- Pontos sobre a linha tracejada são previsões perfeitas. Quanto mais perto dela, melhor.\n"
                    f"- R² de **{br(m['R²'], 3)}**: o modelo explica cerca de {br(100 * m['R²'], 0)}% da variação da "
                    "incidência 4 semanas à frente.\n"
                    f"- Nas incidências altas, a maioria dos pontos fica abaixo da linha: das "
                    f"{int((real > 50).sum())} semanas com mais de 50 casos por 100 mil, o modelo previu menos em "
                    f"**{int((residuos[real > 50] > 0).sum())}**. Nas subidas fortes, ele reage com atraso.")
    with col_b:
        st.markdown("**Como ler os resíduos**")
        st.markdown(f"- O resíduo é o erro de cada previsão (real − previsto).\n"
                    f"- Mediana: **{br(residuos.median(), 2)}**. Nas semanas comuns, o modelo tende a prever um pouco "
                    f"acima do real.\n"
                    f"- **{br(100 * (residuos.abs() <= 10).mean(), 1)}%** das previsões erram até 10 casos por 100 mil.\n"
                    f"- Os erros grandes ficam à direita de 0: nas subidas, o modelo reage com atraso.")

    with st.expander("Ver os pesos (coeficientes) do modelo"):
        pesos = pd.DataFrame({"variável": [NOMES[v] for v in FEATURES_LINEAR], "coeficiente": linear.coef_})
        st.dataframe(formatar(pesos, 3), hide_index=True)
        st.caption(f"Soma dos três pesos de incidência: {br(linear.coef_[:3].sum(), 3)}. As três incidências são muito "
                   "correlacionadas, então devem ser lidas juntas: o modelo parte de cerca de 80% da incidência de hoje "
                   "e soma a tendência das últimas semanas.")


def aba_logistica(modelos, municipio, limiar, limiar_escolhido):
    teste, logistica = modelos["teste"], modelos["logistica"]
    st.markdown("A regressão logística devolve a **chance** (de 0% a 100%) de o município estar em epidemia daqui "
                f"a 4 semanas. Se a chance passar do limiar (**{br(limiar * 100, 0)}%**, ajustável na barra lateral), "
                f"o modelo dá o alerta. O limiar padrão, **{br(limiar_escolhido, 2)}**, foi escolhido na validação "
                "cruzada, sem olhar o teste (aba Robustez).")
    real = teste[ALVO_LOGISTICA]
    previsto = (teste["probabilidade"] >= limiar).astype(int)
    m = metricas_classificacao(real, previsto)
    r = metricas_classificacao(real, teste["acima_limiar_hoje"])
    matriz = confusion_matrix(real, previsto, labels=[0, 1])
    vn, fp, fn, vp = matriz.ravel()

    st.markdown(f"#### Resultado no teste (2025–2026), todos os municípios, limiar {br(limiar, 2)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precisão (alertas certos)", br(m["Precisão"], 3), variacao(m["Precisão"] - r["Precisão"], 3) + " vs. repetir hoje",
              help="Quando o modelo dá alerta, quantas vezes está certo.")
    c2.metric("Recall (epidemias avisadas)", br(m["Recall"], 3), variacao(m["Recall"] - r["Recall"], 3) + " vs. repetir hoje",
              help="Das semanas que foram epidêmicas, quantas o modelo avisou antes.")
    c3.metric("F1 (nota geral)", br(m["F1"], 3), variacao(m["F1"] - r["F1"], 3) + " vs. repetir hoje",
              help="Equilibra precisão e recall.")
    c4.metric("Acurácia", br(m["Acurácia"], 3), variacao(m["Acurácia"] - r["Acurácia"], 3) + " vs. repetir hoje",
              help="Quantas previsões acertou no geral. Engana quando as semanas epidêmicas são minoria.")
    st.success(f"Com limiar {br(limiar, 2)}, o modelo antecipou **{vp} de {vp + fn}** semanas epidêmicas e deu "
               f"**{fp}** alarmes falsos.")

    st.markdown("#### Como o modelo transforma as pistas em chance: a curva sigmoide")
    entradas = modelos["padronizador"].transform(teste[FEATURES_LOGISTICA])
    pontuacao = logistica.decision_function(entradas)
    pesos_z = dict(zip(FEATURES_LOGISTICA, logistica.coef_[0]))
    sinal = lambda v: "+" if v >= 0 else "−"  # noqa: E731
    st.code("z (pontuação) = " + br(logistica.intercept_[0], 3) + "\n"
            + "\n".join(f"    {sinal(pesos_z[v])} {br(abs(pesos_z[v]), 3)} × {NOMES[v]} (padronizada)"
                        for v in FEATURES_LOGISTICA)
            + "\n\nchance de semana epidêmica = 1 / (1 + e^(−z))", language="text")
    mostrar_reta = st.checkbox("Comparar com uma reta (como seria se usássemos regressão linear)", value=False)
    mostrar_figura(figura_sigmoide(pontuacao, teste["probabilidade"], real, limiar, mostrar_reta))
    z_corte = float(np.log(limiar / (1 - limiar)))
    st.markdown(f"- Cada ponto é uma semana do teste ({len(teste)} no total, todos os municípios). Os pontos ficam "
                "**em cima (1)** se a semana foi epidêmica e **embaixo (0)** se não foi; a posição na horizontal é a "
                "pontuação que o modelo deu 4 semanas antes.\n"
                "- A **curva sigmoide** (azul) transforma a pontuação em chance e nunca sai do intervalo de 0 a 1. "
                "Uma reta passaria de 0 e de 1, o que não faz sentido para uma probabilidade; por isso a logística "
                "é a escolha certa para um alvo de sim/não.\n"
                f"- Com limiar **{br(limiar, 2)}**, o alerta acontece quando z passa de **{br(z_corte, 2)}** (linha "
                "pontilhada). Pontos vermelhos são erros: **em cima e à esquerda** são epidemias não antecipadas (FN); "
                "**embaixo e à direita** são alarmes falsos (FP), os mesmos da matriz de confusão abaixo.")

    st.markdown("#### Curva ROC e AUC: o modelo em todos os limiares ao mesmo tempo")
    col_a, col_b = st.columns([1, 1.1])
    fig_roc, auc_modelo, auc_ref = figura_roc(teste, limiar)
    with col_a:
        mostrar_figura(fig_roc)
    with col_b:
        c1, c2 = st.columns(2)
        c1.metric("AUC (teste)", br(auc_modelo, 3), variacao(auc_modelo - auc_ref, 3) + " vs. repetir hoje",
                  help="Área sob a curva ROC. 1 = separa perfeitamente; 0,5 = igual a sortear.")
        c2.metric("AUC da referência", br(auc_ref, 3), help="Repetir a situação de hoje só responde 0 ou 1.")
        st.markdown("- Cada ponto da curva azul é um limiar diferente: quanto mais a curva sobe rápido para o canto "
                    "superior esquerdo, melhor o modelo separa as semanas epidêmicas das não epidêmicas.\n"
                    "- A **AUC** resume a curva num número e **não depende do limiar**.\n"
                    "- O círculo laranja é o limiar da barra lateral; o ponto escuro é a referência (repetir hoje), que "
                    "só responde 0 ou 1. Com o limiar 0,45, o modelo tem a mesma taxa de falsos positivos da referência, "
                    "mas antecipa mais semanas epidêmicas.")

    col_a, col_b = st.columns([1, 1.15])
    with col_a:
        mostrar_figura(figura_matriz(matriz, limiar))
    with col_b:
        st.markdown("**Como ler a matriz:** quadrados verdes são acertos e vermelhos são erros. Em cima ficam as "
                    "semanas que **não** foram epidêmicas; embaixo, as que **foram**.")
        situacoes = teste[["acima_limiar_hoje", ALVO_LOGISTICA]].copy()
        situacoes["acertou"] = previsto == real
        situacoes["situação"] = np.select(
            [(situacoes["acima_limiar_hoje"] == 0) & (situacoes[ALVO_LOGISTICA] == 0),
             (situacoes["acima_limiar_hoje"] == 1) & (situacoes[ALVO_LOGISTICA] == 1),
             (situacoes["acima_limiar_hoje"] == 0) & (situacoes[ALVO_LOGISTICA] == 1),
             (situacoes["acima_limiar_hoje"] == 1) & (situacoes[ALVO_LOGISTICA] == 0)],
            SITUACOES, default="?")
        por_situacao = (situacoes.groupby("situação").agg(semanas=("situação", "size"), acertos=("acertou", "sum"))
                        .reindex(SITUACOES, fill_value=0))
        st.markdown("**Onde o modelo erra:** quando a situação se mantém, ele quase sempre acerta; nas **viradas** "
                    "(entrar ou sair de um período epidêmico) ele acerta pouco.")
        st.dataframe(formatar(por_situacao, 0))

    st.markdown(f"#### {municipio}: probabilidade prevista no teste")
    mostrar_figura(figura_probabilidade(teste[teste["municipio_nome"] == municipio], limiar, municipio))
    st.caption("As faixas cinzas são as semanas que realmente foram epidêmicas. O ideal é a linha azul ficar acima "
               "da linha tracejada dentro das faixas cinzas e abaixo dela fora delas.")

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("**Comparar limiares**")
        linhas = []
        for valor in sorted({0.3, 0.4, 0.5, 0.6, 0.7, limiar_escolhido}):
            prev_l = (teste["probabilidade"] >= valor).astype(int)
            _, fp_l, fn_l, _ = confusion_matrix(real, prev_l, labels=[0, 1]).ravel()
            linhas.append({"limiar": br(valor, 2) + (" (escolhido)" if np.isclose(valor, limiar_escolhido) else ""),
                           "alarmes falsos (FP)": str(fp_l),
                           "epidemias não antecipadas (FN)": str(fn_l),
                           **{k: br(v, 3) for k, v in metricas_classificacao(real, prev_l).items()}})
        st.dataframe(pd.DataFrame(linhas), hide_index=True)
    with col_b:
        st.caption("Só para informação: o limiar foi escolhido na validação, não nesta tabela. Limiar mais baixo = mais "
                   "alertas: perdem-se menos epidemias, mas aparecem mais alarmes falsos.")

    with st.expander("Ver os pesos (coeficientes) do modelo"):
        pesos = pd.DataFrame({"variável": [NOMES[v] for v in FEATURES_LOGISTICA], "coeficiente": logistica.coef_[0],
                              "razão de chances (por 1 desvio-padrão)": np.exp(logistica.coef_[0])})
        st.dataframe(formatar(pesos, 3), hide_index=True)
        st.caption("Todos os pesos são positivos: mais casos, mais calor, mais umidade e já estar acima do limiar "
                   "aumentam a chance de epidemia. As entradas foram padronizadas com média e desvio do treino.")


def aba_simulador(base, modelos, municipio, limiar):
    st.markdown("Digite os valores de uma semana e veja as duas previsões para **4 semanas depois**. Os campos "
                f"começam com a semana mais recente da base tratada para **{municipio}**.")
    treino = modelos["treino"]
    ultima = base[base["municipio_nome"] == municipio].sort_values("SE").iloc[-1]
    st.caption(f"Valores iniciais: {semana(ultima['SE'])} de {municipio}.")

    c1, c2, c3 = st.columns(3)
    with c1:
        inc0 = st.number_input("Incidência hoje (casos/100 mil)", min_value=0.0, max_value=5000.0,
                               value=round(float(ultima["incidencia"]), 2), step=0.5)
        inc1 = st.number_input("Incidência 1 semana antes", min_value=0.0, max_value=5000.0,
                               value=round(float(ultima["incidencia_1sem_antes"]), 2), step=0.5)
    with c2:
        inc2 = st.number_input("Incidência 2 semanas antes", min_value=0.0, max_value=5000.0,
                               value=round(float(ultima["incidencia_2sem_antes"]), 2), step=0.5)
        temp = st.number_input("Temperatura média (°C)", min_value=0.0, max_value=45.0,
                               value=round(float(ultima["tempmed"]), 1), step=0.5)
    with c3:
        umid = st.number_input("Umidade média (%)", min_value=0.0, max_value=100.0,
                               value=round(float(ultima["umidmed"]), 1), step=1.0)
        acima = st.checkbox("A semana atual já está acima do limiar epidêmico",
                            value=bool(ultima["acima_limiar_hoje"] == 1))

    entrada = pd.DataFrame([{"incidencia": inc0, "incidencia_1sem_antes": inc1, "incidencia_2sem_antes": inc2,
                             "tempmed": temp, "umidmed": umid, "acima_limiar_hoje": int(acima)}])
    fora = [NOMES[v] for v in FEATURES_LINEAR
            if not treino[v].min() <= entrada[v].iloc[0] <= treino[v].max()]
    if fora:
        st.warning("Valores fora da faixa vista no treino: " + ", ".join(fora)
                   + ". A previsão é uma extrapolação e merece menos confiança.")

    previsto = float(modelos["linear"].predict(entrada[FEATURES_LINEAR])[0])
    probabilidade = float(modelos["logistica"].predict_proba(
        modelos["padronizador"].transform(entrada[FEATURES_LOGISTICA]))[0, 1])
    populacao = float(ultima["pop"])

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Incidência prevista em 4 semanas", f"{br(max(previsto, 0), 1)} por 100 mil",
              help="Regressão linear. Valores negativos são mostrados como zero.")
    r2.metric(f"Casos por semana em {municipio} (aprox.)", br(max(previsto, 0) * populacao / 100_000, 0),
              help="Incidência prevista convertida em casos com a população do município.")
    r3.metric("Chance de semana epidêmica", f"{br(100 * probabilidade, 1)}%", help="Regressão logística.")
    if probabilidade >= limiar:
        st.error(f"ALERTA: a chance ({br(100 * probabilidade, 1)}%) passou do limiar ({br(100 * limiar, 0)}%). "
                 "Recomendação: reforçar a orientação à população e priorizar vistorias neste município.")
    else:
        st.success(f"Sem alerta: a chance ({br(100 * probabilidade, 1)}%) está abaixo do limiar "
                   f"({br(100 * limiar, 0)}%). Manter o acompanhamento semanal.")
    if previsto < 0:
        st.info("A regressão linear deu um valor negativo, o que é impossível para uma incidência. Ele foi mostrado "
                "como zero; é uma limitação conhecida do modelo linear.")

    with st.expander("Como a incidência prevista foi calculada"):
        linear = modelos["linear"]
        partes = pd.DataFrame({"termo": ["intercepto"] + [NOMES[v] for v in FEATURES_LINEAR],
                               "valor digitado": [np.nan] + [entrada[v].iloc[0] for v in FEATURES_LINEAR],
                               "peso": [np.nan] + list(linear.coef_),
                               "contribuição": [linear.intercept_] + [linear.coef_[i] * entrada[v].iloc[0]
                                                                      for i, v in enumerate(FEATURES_LINEAR)]})
        st.dataframe(formatar(partes, 3), hide_index=True)
        st.caption(f"Soma das contribuições = {br(previsto, 3)} casos por 100 mil.")


def aba_recomendacoes(modelos, rob):
    teste = modelos["teste"]
    real = teste[ALVO_LOGISTICA]
    limiar = rob["limiar"]
    prev05 = (teste["probabilidade"] >= limiar).astype(int)
    m_log = metricas_classificacao(real, prev05)
    vn, fp, fn, vp = confusion_matrix(real, prev05, labels=[0, 1]).ravel()
    m_lin = metricas_regressao(teste[ALVO_LINEAR], teste["previsto"])
    m_ref = metricas_regressao(teste[ALVO_LINEAR], teste["incidencia"])
    entradas = teste[(teste["acima_limiar_hoje"] == 0) & (real == 1)]
    sp = teste[teste["municipio_nome"] == "São Paulo"]
    erro_sp = mean_absolute_error(sp[ALVO_LINEAR], sp["previsto"]) * sp["pop"].mean() / 100_000

    anos_lin, h_log = rob["anos_linear"], rob["hipotese_logistica"]
    st.markdown(f"Recomendações baseadas nos resultados do período de teste (2025–2026), com o limiar "
                f"{br(limiar, 2)} escolhido na validação cruzada. O texto completo está na seção 6 do notebook.")
    recomendacoes = pd.DataFrame([
        {"recomendação": f"Usar a probabilidade da regressão logística como alerta antecipado por município, "
                         f"recalculada toda semana, com limiar {br(limiar, 2)}. Ao passar do limiar: reforçar a "
                         "orientação à população e priorizar as vistorias daquele município.",
         "base nos resultados": f"Antecipou {vp} de {vp + fn} semanas epidêmicas (recall {br(m_log['Recall'], 3)}); "
                                f"{br(100 * m_log['Precisão'], 0)}% dos alertas estavam certos. O limiar foi escolhido "
                                "na validação cruzada, sem olhar o teste."},
        {"recomendação": f"Ajustar o limiar ao custo de errar: {br(limiar, 2)} é o padrão; se a prioridade for não "
                         "deixar passar semanas epidêmicas, os outros limiares empatados na validação dão mais alertas.",
         "base nos resultados": "Limiares empatados no F1 da validação: "
                                + ", ".join(br(l, 2) for l in rob["limiares_empatados"]) + " (aba Robustez)."},
        {"recomendação": "Usar a regressão linear para dimensionar o esforço (ordem de grandeza e tendência), não "
                         "como número exato.",
         "base nos resultados": f"Erro médio de {br(m_lin['MAE'], 1)} casos por 100 mil (cerca de {br(erro_sp, 0)} "
                                f"casos por semana em São Paulo); RMSE {br(m_lin['RMSE'], 1)} contra "
                                f"{br(m_ref['RMSE'], 1)} da referência; venceu a referência em "
                                f"{int((anos_lin['RMSE'] < anos_lin['RMSE da referência']).sum())} dos 5 anos de validação."},
        {"recomendação": "Não depender só do modelo para detectar o início de um surto: combinar com o "
                         "acompanhamento semanal e com os alertas oficiais do InfoDengue.",
         "base nos resultados": f"Antecipou só {int(prev05[entradas.index].sum())} das {len(entradas)} entradas em "
                                "período epidêmico."},
        {"recomendação": "Considerar o clima junto com os casos na decisão de alerta.",
         "base nos resultados": f"Sem o clima, o F1 da logística na validação cai de "
                                f"{br(h_log.loc['Completo (6 variáveis)', 'F1 (validação)'], 3)} para "
                                f"{br(h_log.loc['Sem o clima', 'F1 (validação)'], 3)}."},
        {"recomendação": "Retreinar o modelo e revalidar o limiar periodicamente (por exemplo, a cada temporada), "
                         "com a mesma validação cruzada no tempo.",
         "base nos resultados": f"O R² da linear variou de {br(anos_lin['R²'].min(), 3)} a {br(anos_lin['R²'].max(), 3)} "
                                "entre os anos de validação, e o limiar epidêmico do InfoDengue é recalculado."},
    ])
    recomendacoes.index = range(1, len(recomendacoes) + 1)
    st.dataframe(recomendacoes)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Limitações")
        st.markdown(
            "- Dados **por município e por semana**: o modelo não diz em qual bairro ou imóvel está o risco.\n"
            "- O modelo antecipa pouco o **começo de um surto**; acerta quase sempre quando a situação se mantém.\n"
            "- **2024** foi um ano extremo, e o teste cobre só **2025 e parte de 2026**. Na validação, o desempenho "
            f"variou bastante entre os anos (R² da linear de {br(anos_lin['R²'].min(), 3)} a "
            f"{br(anos_lin['R²'].max(), 3)}).\n"
            "- A regressão linear pode dar **previsões negativas**.\n"
            "- O limiar epidêmico do InfoDengue é **recalculado**; o modelo precisa ser retreinado.\n"
            "- Só entram incidência, temperatura e umidade. **Correlação não é causa**.")
    with col_b:
        st.markdown("#### Impactos")
        st.markdown(
            f"- **Cidadão:** orientação antes do pico, com até 4 semanas de antecedência ({vp} de {vp + fn} semanas "
            "epidêmicas antecipadas no teste).\n"
            "- **Gestão:** a probabilidade e a incidência prevista permitem comparar municípios e priorizar vistorias.\n"
            f"- **Riscos:** alarmes falsos ({fp} no teste) geram desgaste; epidemias não antecipadas ({fn}) dão falsa "
            "sensação de segurança. A decisão final deve ser de pessoas.")


def aba_metodologia(base, modelos, rob):
    st.markdown("#### Passo a passo")
    st.markdown(
        "1. **Dados:** InfoDengue (API `alertcity`), 5 municípios, SE 01/2018 a SE 43/2026, arquivo "
        "`dengue_infodengue_2018_2026_5municipios.csv`.\n"
        "2. **Tratamento (notebook, seção 2.3):** remoção de colunas vazias, constantes e sem uso; conversão de tipos; "
        "checagem de duplicidades; remoção das semanas ainda em aberto; cálculo da incidência; defasagens e alvos "
        "com `shift` dentro de cada município; remoção das linhas incompletas.\n"
        "3. **Treino e teste pelo tempo:** treino = semanas-alvo de 2018 a 2024; teste = 2025 e 2026. A padronização "
        "da logística aprende só com o treino.\n"
        "4. **Validação cruzada no tempo (só treino):** 5 rodadas, validando em 2020 a 2024. Escolhe o limiar do "
        "alerta, mede a estabilidade e testa a hipótese.\n"
        "5. **Modelos:** regressão linear (incidência em 4 semanas) e regressão logística (semana epidêmica em 4 "
        "semanas), comparadas com a referência \"repetir a situação de hoje\".")

    st.markdown("#### Verificação: notebook × painel")
    ref = ler_metricas_notebook()
    teste = modelos["teste"]
    agora = {
        "limiar escolhido": rob["limiar"],
        **metricas_regressao(teste[ALVO_LINEAR], teste["previsto"]),
        **metricas_classificacao(teste[ALVO_LOGISTICA], (teste["probabilidade"] >= rob["limiar"]).astype(int)),
        "RMSE da linear na validação": rob["rmse_validacao_linear"],
        "F1 da logística na validação": rob["f1_validacao"],
    }
    if ref is None or "logistica_teste" not in ref:
        st.info("Arquivo `metricas_notebook.json` não encontrado ou de uma versão antiga: rode o notebook para "
                "gerá-lo e poder comparar.")
    else:
        notebook = {"limiar escolhido": ref["limiar_escolhido"], **ref["linear_teste"], **ref["logistica_teste"],
                    "RMSE da linear na validação": ref["validacao_rmse_linear"],
                    "F1 da logística na validação": ref["validacao_f1_logistica"]}
        comparacao = pd.DataFrame({"notebook": notebook, "painel": {k: agora[k] for k in notebook}})
        comparacao["diferença"] = (comparacao["painel"] - comparacao["notebook"]).abs()
        st.dataframe(formatar(comparacao[["notebook", "painel"]], 4))
        if comparacao["diferença"].max() < 1e-6 and ref.get("linhas_teste") == len(teste):
            st.success(f"Os números do painel são os mesmos do notebook (teste e validação cruzada, limiar "
                       f"{br(rob['limiar'], 2)}).")
        else:
            st.warning("Os números do painel diferem do notebook. Confira se o CSV e o JSON vieram da mesma "
                       "execução do notebook.")

    st.markdown("#### Glossário")
    st.markdown(
        "| Termo | Em palavras simples |\n|---|---|\n"
        "| Incidência | Casos para cada 100 mil moradores. Permite comparar cidades grandes e pequenas. |\n"
        "| Limiar epidêmico | A \"linha vermelha\" do InfoDengue: acima dela, o município está em epidemia. |\n"
        "| Treino e teste | O modelo estuda com uma parte dos dados e faz a \"prova\" com outra, que nunca viu. |\n"
        "| MAE / RMSE | Quanto a previsão erra, em média (o RMSE pesa mais os erros grandes). |\n"
        "| R² | Quanto da variação o modelo explica (1 = perfeito). |\n"
        "| Precisão / Recall | Alertas certos / epidemias avisadas antes. |\n"
        "| F1 | Nota única que equilibra precisão e recall. |\n"
        "| Validação cruzada no tempo | \"Simulados\" dentro do treino: treina com os anos anteriores e testa no seguinte. |\n"
        "| Limiar do alerta | A partir de qual chance o modelo dá o alerta (escolhido na validação). |")

    st.markdown("#### Arquivos usados pelo painel")
    st.markdown(f"- `{ARQUIVO_BASE.name}`: base tratada ({br(len(base), 0)} linhas), gerada pelo notebook.\n"
                f"- `{ARQUIVO_METRICAS.name}`: métricas do notebook, usadas só na verificação acima.\n"
                "- Fonte dos dados: [InfoDengue](https://info.dengue.mat.br) (Fiocruz/FGV).")


# =============================================================================
# Robustez: validação cruzada no tempo e teste da hipótese (notebook, seções 2.6, 4.2, 4.3, 5.2 e 5.3)
# =============================================================================
ANOS_VALIDACAO = [2020, 2021, 2022, 2023, 2024]
LIMIARES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LIMIARES_AMPLOS = [round(float(l), 2) for l in np.arange(0.30, 0.951, 0.05)]
GRUPOS_LINEAR = {
    "Completo (5 variáveis)": FEATURES_LINEAR,
    "Sem o clima": ["incidencia", "incidencia_1sem_antes", "incidencia_2sem_antes"],
    "Sem as semanas anteriores": ["incidencia", "tempmed", "umidmed"],
    "Só a incidência de hoje": ["incidencia"],
    "Só o clima": ["tempmed", "umidmed"],
}
GRUPOS_LOGISTICA = {
    "Completo (6 variáveis)": FEATURES_LOGISTICA,
    "Sem o clima": ["incidencia", "incidencia_1sem_antes", "incidencia_2sem_antes", "acima_limiar_hoje"],
    "Sem as semanas anteriores": ["incidencia", "tempmed", "umidmed", "acima_limiar_hoje"],
    "Sem a situação de hoje (acima_limiar_hoje)": FEATURES_LINEAR,
    "Só o clima": ["tempmed", "umidmed"],
}


def rodadas(treino):
    ano_alvo = treino["SE_alvo"] // 100
    return [(ano, treino.index[ano_alvo < ano], treino.index[ano_alvo == ano]) for ano in ANOS_VALIDACAO]


def validar_linear(treino, colunas):
    partes = []
    for ano, idx_e, idx_v in rodadas(treino):
        estudo, validacao = treino.loc[idx_e], treino.loc[idx_v]
        modelo = LinearRegression().fit(estudo[colunas], estudo[ALVO_LINEAR])
        partes.append(pd.DataFrame({"ano": ano, "real": validacao[ALVO_LINEAR],
                                    "previsto": modelo.predict(validacao[colunas]),
                                    "referencia": validacao["incidencia"]}))
    return pd.concat(partes)


def por_ano_linear(previsoes):
    linhas = []
    for ano, g in previsoes.groupby("ano"):
        linhas.append({"ano de validação": str(ano), **metricas_regressao(g["real"], g["previsto"]),
                       "RMSE da referência": float(np.sqrt(mean_squared_error(g["real"], g["referencia"]))),
                       "R² da referência": float(r2_score(g["real"], g["referencia"]))})
    return pd.DataFrame(linhas).set_index("ano de validação")


def resumo_linear(previsoes):
    anos = por_ano_linear(previsoes)
    return {"RMSE (validação)": float(np.sqrt(mean_squared_error(previsoes["real"], previsoes["previsto"]))),
            "MAE (validação)": float(mean_absolute_error(previsoes["real"], previsoes["previsto"])),
            "R² médio por ano": float(anos["R²"].mean()),
            "anos em que vence a referência": int((anos["RMSE"] < anos["RMSE da referência"]).sum())}


def validar_logistica(treino, colunas, pesos=None):
    partes = []
    for ano, idx_e, idx_v in rodadas(treino):
        estudo, validacao = treino.loc[idx_e], treino.loc[idx_v]
        pad = StandardScaler().fit(estudo[colunas])
        modelo = LogisticRegression(max_iter=1000, class_weight=pesos).fit(pad.transform(estudo[colunas]),
                                                                            estudo[ALVO_LOGISTICA])
        partes.append(pd.DataFrame({"ano": ano, "real": validacao[ALVO_LOGISTICA],
                                    "probabilidade": modelo.predict_proba(pad.transform(validacao[colunas]))[:, 1],
                                    "referencia": validacao["acima_limiar_hoje"]}))
    return pd.concat(partes)


@st.cache_data(show_spinner="Calculando a validação cruzada no tempo...")
def analisar_robustez():
    """Mesmas análises do notebook (seções 2.6, 4.2, 4.3, 5.2 e 5.3), usando só o treino da base tratada."""
    treino = treinar_modelos()["treino"]
    r = {}
    r["rodadas"] = pd.DataFrame([{"rodada": i + 1, "treina com semanas-alvo de": f"2018 a {ano - 1}",
                                  "valida em": str(ano), "linhas de treino": len(idx_e),
                                  "linhas de validação": len(idx_v),
                                  "semanas epidêmicas na validação": int(treino.loc[idx_v, ALVO_LOGISTICA].sum())}
                                 for i, (ano, idx_e, idx_v) in enumerate(rodadas(treino))])

    # Regressão linear: estabilidade ano a ano e teste da hipótese
    base_lin = validar_linear(treino, FEATURES_LINEAR)
    r["anos_linear"] = por_ano_linear(base_lin)
    r["rmse_validacao_linear"] = float(np.sqrt(mean_squared_error(base_lin["real"], base_lin["previsto"])))
    r["rmse_validacao_referencia"] = float(np.sqrt(mean_squared_error(base_lin["real"], base_lin["referencia"])))
    r["hipotese_linear"] = pd.DataFrame({nome: resumo_linear(base_lin if nome.startswith("Completo")
                                                             else validar_linear(treino, colunas))
                                         for nome, colunas in GRUPOS_LINEAR.items()}).T

    # Regressão logística: escolha do limiar (maior F1; diferença < 1% = empate; fica o mais perto de 0,5)
    base_log = validar_logistica(treino, FEATURES_LOGISTICA)
    linhas = []
    for limiar in LIMIARES:
        alerta = (base_log["probabilidade"] >= limiar).astype(int)
        linhas.append({"limiar": limiar,
                       "precisão": float(precision_score(base_log["real"], alerta, zero_division=0)),
                       "recall": float(recall_score(base_log["real"], alerta, zero_division=0)),
                       "F1": float(f1_score(base_log["real"], alerta, zero_division=0))})
    f1_limiares = pd.DataFrame(linhas).set_index("limiar")
    melhor = float(f1_limiares["F1"].max())
    empatados = f1_limiares.index[f1_limiares["F1"] >= melhor * 0.99]
    r["limiar"] = float(min(empatados, key=lambda limiar: abs(limiar - 0.5)))
    r["f1_limiares"] = f1_limiares
    r["faixa_empate"] = (melhor * 0.99, melhor)
    r["limiares_empatados"] = list(empatados)
    r["f1_validacao"] = float(f1_limiares.loc[r["limiar"], "F1"])
    r["f1_referencia_validacao"] = float(f1_score(base_log["real"], base_log["referencia"]))
    r["auc_validacao"] = float(roc_auc_score(base_log["real"], base_log["probabilidade"]))

    # Desbalanceamento: modelo sem pesos x class_weight='balanced' (mesma regra; limiares de 0,30 a 0,95)
    curvas, linhas = {}, {}
    for nome, prev in {"Sem pesos (padrão)": base_log,
                       "Com class_weight='balanced'": validar_logistica(treino, FEATURES_LOGISTICA, "balanced")}.items():
        f1s = pd.Series({l: float(f1_score(prev["real"], (prev["probabilidade"] >= l).astype(int), zero_division=0))
                         for l in LIMIARES_AMPLOS})
        curvas[nome] = f1s
        linhas[nome] = {"AUC (validação)": float(roc_auc_score(prev["real"], prev["probabilidade"])),
                        "chance média dada às semanas não epidêmicas": float(prev.loc[prev["real"] == 0, "probabilidade"].mean()),
                        "maior F1 (limiares de 0,30 a 0,95)": float(f1s.max()),
                        "limiar desse F1": float(f1s.idxmax()),
                        f"F1 com o limiar {br(r['limiar'], 2)}": float(f1s[r["limiar"]])}
    r["curvas_f1_pesos"] = curvas
    r["comparacao_pesos"] = pd.DataFrame(linhas).T
    f1_sem, f1_com = r["comparacao_pesos"]["maior F1 (limiares de 0,30 a 0,95)"]
    r["pesos_vencem"] = bool((f1_com - f1_sem) / f1_sem >= 0.01)

    linhas = []
    for ano, g in base_log.groupby("ano"):
        alerta = (g["probabilidade"] >= r["limiar"]).astype(int)
        linhas.append({"ano de validação": str(ano), "semanas epidêmicas": int(g["real"].sum()),
                       "precisão": float(precision_score(g["real"], alerta, zero_division=0)),
                       "recall": float(recall_score(g["real"], alerta, zero_division=0)),
                       "F1": float(f1_score(g["real"], alerta, zero_division=0)),
                       "F1 da referência": float(f1_score(g["real"], g["referencia"], zero_division=0))})
    r["anos_logistica"] = pd.DataFrame(linhas).set_index("ano de validação")

    linhas = {}
    for nome, colunas in GRUPOS_LOGISTICA.items():
        prev = base_log if nome.startswith("Completo") else validar_logistica(treino, colunas)
        alerta = (prev["probabilidade"] >= r["limiar"]).astype(int)
        linhas[nome] = {"F1 (validação)": float(f1_score(prev["real"], alerta, zero_division=0)),
                        "precisão": float(precision_score(prev["real"], alerta, zero_division=0)),
                        "recall": float(recall_score(prev["real"], alerta, zero_division=0))}
    r["hipotese_logistica"] = pd.DataFrame(linhas).T
    return r


def figura_r2_anos(anos):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    posicoes = np.arange(len(anos))
    b1 = ax.bar(posicoes - 0.2, anos["R²"], width=0.38, color="#2a78d6", label="Regressão linear")
    b2 = ax.bar(posicoes + 0.2, anos["R² da referência"], width=0.38, color="#c3c2b7", label="Referência: repetir hoje")
    ax.bar_label(b1, labels=[br(v, 2) for v in anos["R²"]], padding=2, fontsize=9)
    ax.bar_label(b2, labels=[br(v, 2) for v in anos["R² da referência"]], padding=2, fontsize=9)
    ax.set_xticks(posicoes, anos.index)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(lambda valor, _: br(valor, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("R² em cada ano de validação", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figura_limiar_validacao(rob, limiar_atual):
    f1 = rob["f1_limiares"]["F1"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhspan(*rob["faixa_empate"], color="#2a78d6", alpha=0.10, label="faixa de empate (até 1% do maior F1)")
    ax.plot(f1.index, f1.values, marker="o", color="#2a78d6", label="Regressão logística")
    ax.axhline(rob["f1_referencia_validacao"], color="#3d3d3a", linestyle="--", linewidth=1, label="Referência: repetir hoje")
    ax.scatter([rob["limiar"]], [f1[rob["limiar"]]], s=140, facecolors="none", edgecolors="#eb6834", linewidths=2,
               zorder=3, label=f"escolha: {br(rob['limiar'], 2)}")
    if not np.isclose(limiar_atual, rob["limiar"]) and limiar_atual in f1.index:
        ax.scatter([limiar_atual], [f1[limiar_atual]], s=60, color="#3d3d3a", zorder=3,
                   label=f"limiar da barra lateral: {br(limiar_atual, 2)}")
    ax.xaxis.set_major_formatter(lambda valor, _: br(valor, 2))
    ax.yaxis.set_major_formatter(lambda valor, _: br(valor, 3))
    ax.set_xlabel("limiar do alerta")
    ax.set_ylabel("F1 na validação")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Escolha do limiar na validação cruzada", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figura_hipotese(rob):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, serie, titulo, limites in [
            (eixos[0], rob["hipotese_linear"]["R² médio por ano"], "Linear: R² médio por ano", (-0.5, 0.7)),
            (eixos[1], rob["hipotese_logistica"]["F1 (validação)"], f"Logística: F1 (limiar {br(rob['limiar'], 2)})",
             (0, 1.05))]:
        ax.barh(serie.index, serie.values, color=["#2a78d6"] + ["#9ec5f4"] * (len(serie) - 1), height=0.6)
        ax.bar_label(ax.containers[0], labels=[br(v, 3) for v in serie.values], padding=3)
        ax.set_xlim(*limites)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(lambda valor, _: br(valor, 1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(titulo + " (validação)", loc="left", fontweight="bold")
    eixos[0].axvline(0, color="#3d3d3a", linewidth=0.8)
    fig.tight_layout()
    return fig


def aba_robustez(rob, modelos, limiar):
    st.markdown("Esta aba mostra **o quanto dá para confiar** nos modelos: se o desempenho se mantém de um ano para "
                "outro, se o limiar foi escolhido sem olhar o teste e quanto cada grupo de variáveis ajuda. É o mesmo "
                "conteúdo das seções 2.6, 4 e 5 do notebook.")
    parte = st.radio("Mostrar", ["Validação cruzada no tempo", "Teste da hipótese", "Resultados por município"],
                     horizontal=True)

    if parte == "Validação cruzada no tempo":
        st.markdown("Em cada rodada, o modelo treina com os anos anteriores e é avaliado no ano seguinte, sempre "
                    "dentro do treino. O limiar do alerta é escolhido aqui; o teste (2025–2026) só é usado no fim.")
        st.dataframe(formatar(rob["rodadas"], 0), hide_index=True)

        st.markdown("#### Regressão linear: estabilidade ano a ano")
        col_a, col_b = st.columns([1.1, 1])
        anos = rob["anos_linear"]
        with col_a:
            mostrar_figura(figura_r2_anos(anos))
        with col_b:
            vence = int((anos["RMSE"] < anos["RMSE da referência"]).sum())
            st.markdown(f"- O modelo venceu a referência em **{vence} de {len(anos)}** anos de validação.\n"
                        f"- Com todas as previsões da validação juntas: RMSE de **{br(rob['rmse_validacao_linear'], 3)}**, "
                        f"contra **{br(rob['rmse_validacao_referencia'], 3)}** da referência.\n"
                        f"- O R² variou de **{br(anos['R²'].min(), 3)}** a **{br(anos['R²'].max(), 3)}**: funciona em "
                        "todos os anos, mas a qualidade muda bastante.")
        st.dataframe(formatar(anos, 3))

        st.markdown("#### Regressão logística: escolha do limiar")
        f1_tab = rob["f1_limiares"].copy()
        f1_tab.index = [br(l, 2) + (" (escolhido)" if np.isclose(l, rob["limiar"]) else "") for l in f1_tab.index]
        col_a, col_b = st.columns([1.1, 1])
        with col_a:
            mostrar_figura(figura_limiar_validacao(rob, limiar))
        with col_b:
            st.dataframe(formatar(f1_tab, 3))
            st.caption("Regra: maior F1 com todas as previsões da validação juntas; diferenças menores que 1% são "
                       "empate, e no empate fica o limiar mais próximo de 0,5. Empatados: "
                       + ", ".join(br(l, 2) for l in rob["limiares_empatados"])
                       + f". Escolha: **{br(rob['limiar'], 2)}**.")
        st.dataframe(formatar(rob["anos_logistica"], 3).assign(
            **{"semanas epidêmicas": rob["anos_logistica"]["semanas epidêmicas"].astype(str)}))
        st.caption(f"Resultado de cada ano de validação com o limiar {br(rob['limiar'], 2)}. O F1 varia muito de um "
                   "ano para outro (2021 teve só 2 semanas epidêmicas).")

        st.markdown("#### Regressão logística: vale usar `class_weight='balanced'`?")
        st.markdown("As semanas epidêmicas são minoria. O `class_weight='balanced'` dá mais peso a elas no treino. "
                    "Comparamos as duas versões na mesma validação cruzada e com a mesma regra (diferença menor que "
                    "1% no maior F1 = empate, e no empate fica o modelo mais simples, sem pesos).")
        col_a, col_b = st.columns([1.1, 1])
        with col_a:
            mostrar_figura(figura_pesos(rob))
        with col_b:
            comp = rob["comparacao_pesos"]
            st.dataframe(formatar(comp, 3))
            f1_sem, f1_com = comp["maior F1 (limiares de 0,30 a 0,95)"]
            st.markdown(f"- Maior F1: **{br(f1_sem, 4)}** sem pesos e **{br(f1_com, 4)}** com pesos; AUC de "
                        f"{br(comp['AUC (validação)'].iloc[0], 3)} e {br(comp['AUC (validação)'].iloc[1], 3)}.\n"
                        "- Os pesos não melhoram o modelo: só **empurram as chances para cima**, e por isso o mesmo F1 "
                        "só aparece com um limiar bem mais alto.\n"
                        + ("- Decisão: **usar os pesos**." if rob["pesos_vencem"] else
                           "- Decisão: fica o modelo **sem pesos** (mais simples, mesmo desempenho)."))

    elif parte == "Teste da hipótese":
        st.markdown("Repetimos a validação cruzada tirando um grupo de variáveis por vez. Se o resultado piora quando "
                    "um grupo sai, esse grupo ajuda.")
        mostrar_figura(figura_hipotese(rob))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Regressão linear**")
            tab = formatar(rob["hipotese_linear"], 3)
            tab["anos em que vence a referência"] = rob["hipotese_linear"]["anos em que vence a referência"].astype(int).astype(str)
            st.dataframe(tab)
        with col_b:
            st.markdown(f"**Regressão logística (limiar {br(rob['limiar'], 2)})**")
            st.dataframe(formatar(rob["hipotese_logistica"], 3))
        h_lin, h_log = rob["hipotese_linear"], rob["hipotese_logistica"]
        st.markdown(
            f"- **A incidência recente é a pista principal.** Só com o clima, a linear tem R² médio de "
            f"{br(h_lin.loc['Só o clima', 'R² médio por ano'], 3)}, pior do que chutar a média.\n"
            f"- **O clima quase não ajuda a estimar o número de casos** (R² médio de "
            f"{br(h_lin.loc['Completo (5 variáveis)', 'R² médio por ano'], 3)} com ele e "
            f"{br(h_lin.loc['Sem o clima', 'R² médio por ano'], 3)} sem ele), **mas é importante para prever epidemia**: "
            f"sem ele, o F1 da logística cai de {br(h_log.loc['Completo (6 variáveis)', 'F1 (validação)'], 3)} para "
            f"{br(h_log.loc['Sem o clima', 'F1 (validação)'], 3)}.")

    else:
        teste = modelos["teste"]
        st.markdown(f"Desempenho dos dois modelos em cada município no período de teste (logística com limiar "
                    f"{br(limiar, 2)}, o da barra lateral).")
        linhas = []
        for mun in MUNICIPIOS:
            g = teste[teste["municipio_nome"] == mun]
            alerta = (g["probabilidade"] >= limiar).astype(int)
            m_log = metricas_classificacao(g[ALVO_LOGISTICA], alerta)
            m_lin = metricas_regressao(g[ALVO_LINEAR], g["previsto"])
            linhas.append({"município": mun, "MAE (linear)": m_lin["MAE"],
                           "semanas epidêmicas": str(int(g[ALVO_LOGISTICA].sum())), "alertas": str(int(alerta.sum())),
                           "precisão": m_log["Precisão"], "recall": m_log["Recall"], "F1": m_log["F1"]})
        st.dataframe(formatar(pd.DataFrame(linhas), 3), hide_index=True)
        st.caption("Um único modelo atende os 5 municípios, então ele não se ajusta igualmente bem a todos.")


# =============================================================================
# Página
# =============================================================================
st.set_page_config(page_title="AEGIS · Previsão de dengue", layout="wide")
st.markdown(ESTILO_CSS, unsafe_allow_html=True)

if not ARQUIVO_BASE.exists():
    st.error(f"Arquivo `{ARQUIVO_BASE.name}` não encontrado na pasta do app. Ele é gerado pelo notebook "
             "(seção 2.7). Deixe o CSV na mesma pasta do `app.py` e abra o painel de novo.")
    st.stop()
BASE = carregar_base()
PROBLEMAS = problemas_da_base(BASE)
if PROBLEMAS:
    st.error("A base tratada não está no formato esperado: " + "; ".join(PROBLEMAS)
             + ". Rode o notebook de novo para gerar o arquivo.")
    st.stop()
MODELOS = treinar_modelos()
ROBUSTEZ = analisar_robustez()

with st.sidebar:
    st.markdown("## AEGIS")
    st.caption("Vigilância e prevenção da dengue")
    st.markdown("### Controles")
    MUNICIPIO = st.selectbox("Município", MUNICIPIOS)
    LIMIAR = st.slider("Limiar do alerta (probabilidade)", min_value=0.30, max_value=0.70,
                       value=ROBUSTEZ["limiar"], step=0.05,
                       help="A partir desta probabilidade o modelo dá o alerta de semana epidêmica. O padrão "
                            f"({br(ROBUSTEZ['limiar'], 2)}) foi escolhido na validação cruzada, sem olhar o teste.")
    LIMIAR = round(float(LIMIAR), 2)
    st.caption("O município e o limiar valem para as abas de visão geral, modelos e simulador.")
    st.divider()
    _t = MODELOS["teste"]
    _lin = metricas_regressao(_t[ALVO_LINEAR], _t["previsto"])
    _log = metricas_classificacao(_t[ALVO_LOGISTICA], (_t["probabilidade"] >= ROBUSTEZ["limiar"]).astype(int))
    st.markdown("### Desempenho no teste")
    st.markdown(f"**Regressão linear:** MAE {br(_lin['MAE'], 1)} · R² {br(_lin['R²'], 3)}  \n"
                f"**Regressão logística ({br(ROBUSTEZ['limiar'], 2)}):** F1 {br(_log['F1'], 3)} · "
                f"recall {br(_log['Recall'], 3)}")
    st.divider()
    with st.expander("Equipe AEGIS"):
        st.markdown("\n".join(f"- {nome} (RM {rm})" for nome, rm in EQUIPE))
    st.caption("Dados: [InfoDengue](https://info.dengue.mat.br) (Fiocruz/FGV)")

st.markdown(
    '<div class="aegis-topo">'
    '<div class="aegis-selo">Global Solution 2026.2 · CitizenTech · FIAP</div>'
    '<div class="aegis-titulo">AEGIS — Previsão de dengue</div>'
    '<div class="aegis-subtitulo">São Paulo, Osasco e ABC paulista, com <b>4 semanas de antecedência</b>. A partir dos '
    'casos das últimas semanas e do clima, o painel mostra <b>quantos casos</b> por 100 mil moradores (regressão '
    'linear) e a <b>chance de epidemia</b> (regressão logística).</div>'
    '</div>', unsafe_allow_html=True)

abas = st.tabs(["Visão geral", "Explorar os dados", "Regressão linear", "Regressão logística", "Robustez",
                "Simulador", "Recomendações", "Metodologia"])
with abas[0]:
    aba_visao_geral(BASE, MODELOS, MUNICIPIO)
with abas[1]:
    aba_explorar(BASE)
with abas[2]:
    aba_linear(MODELOS, MUNICIPIO)
with abas[3]:
    aba_logistica(MODELOS, MUNICIPIO, LIMIAR, ROBUSTEZ["limiar"])
with abas[4]:
    aba_robustez(ROBUSTEZ, MODELOS, LIMIAR)
with abas[5]:
    aba_simulador(BASE, MODELOS, MUNICIPIO, LIMIAR)
with abas[6]:
    aba_recomendacoes(MODELOS, ROBUSTEZ)
with abas[7]:
    aba_metodologia(BASE, MODELOS, ROBUSTEZ)

st.divider()
st.markdown('<div class="aegis-rodape">AEGIS · Global Solution 2026.2 · CitizenTech · FIAP · '
            'Dados: InfoDengue (Fiocruz/FGV)</div>', unsafe_allow_html=True)
