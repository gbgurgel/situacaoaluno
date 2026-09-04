"""
Previsor de Situação do Aluno (v2 - Streamlit)
-----------------------------------------------
Modelo de árvore de decisão + interface Streamlit interativa.

Convertido da versão original em Gradio (Blocks) mantendo:
- Dataset sintético de 400 alunos gerado por regra de negócio explícita
- Camada de segurança lógica (regra de bom senso) além da árvore
- Árvore de decisão colorida no tema escuro
- Cards de resultado com barras de probabilidade animadas
- Exemplos rápidos (equivalente ao gr.Examples)

Para rodar:
    streamlit run app.py
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

matplotlib.use("Agg")

st.set_page_config(page_title="Previsor de Situação do Aluno", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------------
# 1) Regra de negócio que define a situação "de verdade"
#    (usada tanto para gerar os dados de treino quanto como rede de segurança)
# ---------------------------------------------------------------------------
NOTA_APROVACAO = 7.0
NOTA_RECUPERACAO_MIN = 5.0
FALTAS_LIMITE = 20  # acima disso -> reprovado por falta, não importa a nota


def definir_situacao(nota_aluno: float, faltas_aluno: float) -> str:
    if faltas_aluno > FALTAS_LIMITE:
        return "Reprovado"
    if nota_aluno >= NOTA_APROVACAO:
        return "Aprovado"
    if nota_aluno >= NOTA_RECUPERACAO_MIN:
        return "Recuperação"
    return "Reprovado"


def _regra_bom_senso(nota_aluno: float, faltas_aluno: float):
    """Rede de segurança: só se manifesta em casos extremos e inequívocos.
    Retorna None quando o caso é de fronteira (aí confia-se no modelo)."""
    if faltas_aluno > FALTAS_LIMITE:
        return "Reprovado"
    if nota_aluno < 3:
        return "Reprovado"
    if nota_aluno >= 9 and faltas_aluno <= 10:
        return "Aprovado"
    return None


# Paleta única usada em todo lugar (cards, barras, árvore)
CORES = {
    "Aprovado": "#16a34a",
    "Recuperação": "#d97706",
    "Reprovado": "#dc2626",
}
EMOJI = {"Aprovado": "🎉", "Recuperação": "📘", "Reprovado": "📕"}
FUNDO_ARVORE = "#0b1020"


# ---------------------------------------------------------------------------
# 2) Dataset sintético + treino do modelo (cacheado - só roda uma vez)
# ---------------------------------------------------------------------------
@st.cache_resource
def treinar_modelo():
    rng = np.random.default_rng(7)
    n_amostras = 400

    horas = rng.uniform(0, 20, n_amostras)
    faltas = rng.uniform(0, 30, n_amostras)
    ruido = rng.normal(0, 0.9, n_amostras)
    nota = np.clip(3.6 + 0.30 * horas - 0.06 * faltas + ruido, 0, 10)

    situacao = [definir_situacao(n, f) for n, f in zip(nota, faltas)]

    df = pd.DataFrame(
        {
            "Horas_de_estudo": np.round(horas, 1),
            "Faltas": np.round(faltas).astype(int),
            "Nota": np.round(nota, 1),
            "Situacao": situacao,
        }
    )

    x = df[["Horas_de_estudo", "Faltas", "Nota"]]
    y = df["Situacao"]  # Series (1D), não DataFrame

    x_train, x_teste, y_train, y_teste = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = DecisionTreeClassifier(random_state=42, max_depth=6)
    modelo.fit(x_train, y_train)
    acuracia_teste = modelo.score(x_teste, y_teste)

    return modelo, list(x.columns), list(modelo.classes_), acuracia_teste, n_amostras


modelo, FEATURE_NAMES, CLASSES, ACURACIA_TESTE, N_AMOSTRAS = treinar_modelo()


@st.cache_resource
def gerar_imagem_arvore(_modelo, feature_names, classes):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(FUNDO_ARVORE)
    ax.set_facecolor(FUNDO_ARVORE)
    plot_tree(
        _modelo,
        feature_names=feature_names,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    # Recolore os nós e as linhas de conexão para o tema escuro
    for artist in ax.get_children():
        if isinstance(artist, matplotlib.text.Annotation):
            texto = artist.get_text()
            cor_no = None
            for classe, cor in CORES.items():
                if f"class = {classe}" in texto:
                    cor_no = cor
                    break
            bbox = artist.get_bbox_patch()
            if bbox is not None:
                bbox.set_facecolor(cor_no if cor_no else "#1e293b")
                bbox.set_alpha(0.92)
                bbox.set_edgecolor("#475569")
            artist.set_color("#f8fafc")
            if artist.arrow_patch is not None:
                artist.arrow_patch.set_color("#64748b")
    fig.tight_layout()
    return fig


FIG_ARVORE = gerar_imagem_arvore(modelo, FEATURE_NAMES, CLASSES)


# ---------------------------------------------------------------------------
# 3) Função de previsão (modelo + validação + rede de segurança lógica)
# ---------------------------------------------------------------------------
def prever_situacao(horas_in, faltas_in, nota_in):
    df_novo = pd.DataFrame(
        [[horas_in, faltas_in, nota_in]],
        columns=["Horas_de_estudo", "Faltas", "Nota"],
    )
    previsao_modelo = modelo.predict(df_novo)[0]
    probabilidades = dict(zip(CLASSES, modelo.predict_proba(df_novo)[0]))

    ajuste = _regra_bom_senso(nota_in, faltas_in)
    resultado_final = ajuste if (ajuste and ajuste != previsao_modelo) else previsao_modelo

    aviso_ajuste = ""
    if ajuste and ajuste != previsao_modelo:
        aviso_ajuste = (
            '<div class="aviso-ajuste">⚙️ O modelo sugeriu "'
            f'{previsao_modelo}", mas o resultado foi ajustado para "{ajuste}" '
            "por uma regra de consistência (caso extremo e inequívoco).</div>"
        )
        # refletir o ajuste também nas barras, para não parecer contraditório
        probabilidades = {c: (1.0 if c == ajuste else 0.0) for c in CLASSES}

    cor = CORES.get(resultado_final, "#334155")
    emoji = EMOJI.get(resultado_final, "🔎")

    linhas_prob = "".join(
        f"""
        <div class="prob-row">
          <span class="prob-label" style="color:{CORES.get(classe, '#475569')}">{classe}</span>
          <div class="prob-barra-fundo">
            <div class="prob-barra" style="width:{probabilidades[classe]*100:.1f}%; background:{CORES.get(classe, '#64748b')};"></div>
          </div>
          <span class="prob-valor">{probabilidades[classe]*100:.1f}%</span>
        </div>
        """
        for classe in CLASSES
    )

    html = f"""
    <div class="resultado-card" style="border-color:{cor}">
      <div class="resultado-emoji">{emoji}</div>
      <div class="resultado-texto" style="color:{cor}">{resultado_final}</div>
      <div class="prob-container">{linhas_prob}</div>
      {aviso_ajuste}
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# 4) CSS — tema escuro + animações + paleta consistente
# ---------------------------------------------------------------------------
CSS = """
<style>
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes popIn {
  0%   { transform: scale(0.85); opacity: 0; }
  70%  { transform: scale(1.04); opacity: 1; }
  100% { transform: scale(1); }
}
@keyframes barGrow { from { width: 0%; } }
@keyframes glow {
  0%, 100% { box-shadow: 0 0 0 rgba(129, 140, 248, 0); }
  50% { box-shadow: 0 0 22px rgba(129, 140, 248, 0.25); }
}

.stApp {
  background: radial-gradient(circle at 15% 0%, #1e1b4b 0%, #0b1020 45%, #030712 100%) !important;
  color: #e2e8f0 !important;
}
#titulo { animation: fadeInUp 0.6s ease-out; text-align: center; }
h1 { color: #c7d2fe !important; text-shadow: 0 0 24px rgba(129, 140, 248, 0.35); }

.card-form, .card-tree {
  animation: fadeInUp 0.7s ease-out;
  border-radius: 16px !important;
  background: #131a2c !important;
  border: 1px solid #262f47 !important;
  color: #e2e8f0 !important;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
  transition: box-shadow 0.25s ease, transform 0.25s ease;
  padding: 18px 20px;
}
.card-form:hover, .card-tree:hover {
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6);
  transform: translateY(-2px);
}
.card-form h3, .card-tree h3 { color: #a5b4fc !important; margin-top: 0; }

div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #4338ca, #7e22ce) !important;
  border: none !important;
  color: #f8fafc !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  font-weight: 600 !important;
  width: 100%;
}
div[data-testid="stButton"] > button:hover { transform: scale(1.03); box-shadow: 0 6px 20px rgba(126, 34, 206, 0.5); }
div[data-testid="stButton"] > button:active { transform: scale(0.97); }

.resultado-card {
  animation: popIn 0.45s ease-out;
  border: 2px solid #334155;
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  background: #0d1424;
}
.resultado-emoji { font-size: 42px; line-height: 1; margin-bottom: 6px; }
.resultado-texto { font-size: 26px; font-weight: 800; margin-bottom: 14px; }
.prob-container { display: flex; flex-direction: column; gap: 8px; text-align: left; }
.prob-row { display: grid; grid-template-columns: 110px 1fr 52px; align-items: center; gap: 8px; font-size: 13px; }
.prob-label { font-weight: 700; }
.prob-barra-fundo { background: #1e293b; border-radius: 999px; height: 10px; overflow: hidden; }
.prob-barra { height: 100%; border-radius: 999px; animation: barGrow 0.7s ease-out; }
.prob-valor { text-align: right; color: #94a3b8; font-variant-numeric: tabular-nums; }

.aviso-ajuste {
  margin-top: 14px;
  padding: 10px 12px;
  border-radius: 10px;
  background: #1e1b4b;
  color: #c7d2fe;
  font-size: 12.5px;
  text-align: left;
  border-left: 3px solid #818cf8;
}

#aviso-dados {
  border-left: 4px solid #b45309;
  background: #2a1a05 !important;
  color: #fcd34d !important;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 13px;
}
#aviso-dados * { color: #fcd34d !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 5) Interface
# ---------------------------------------------------------------------------
EXEMPLOS = [
    {"horas": 0, "faltas": 0, "nota": 0.0},
    {"horas": 10, "faltas": 1, "nota": 9.0},
    {"horas": 3, "faltas": 12, "nota": 4.0},
    {"horas": 6, "faltas": 5, "nota": 6.0},
    {"horas": 15, "faltas": 25, "nota": 8.5},
]

# Valores padrão dos sliders, controláveis via session_state (para os exemplos)
if "horas" not in st.session_state:
    st.session_state["horas"] = 6
if "faltas" not in st.session_state:
    st.session_state["faltas"] = 2
if "nota" not in st.session_state:
    st.session_state["nota"] = 7.0

st.markdown('<h1 id="titulo">🎓 Previsor de Situação do Aluno</h1>', unsafe_allow_html=True)
st.markdown(
    f"""<div id="aviso-dados">Modelo treinado com {N_AMOSTRAS} alunos simulados a partir de uma
    regra realista (nota e faltas definem a situação). Acurácia no conjunto de teste:
    <b>{ACURACIA_TESTE * 100:.1f}%</b>.</div>""",
    unsafe_allow_html=True,
)
st.write("")

col_form, col_tree = st.columns(2)

with col_form:
    st.markdown('<div class="card-form">', unsafe_allow_html=True)
    st.markdown("### 📋 Dados do aluno")
    horas_in = st.slider("Horas de estudo por semana", 0, 20, step=1, key="horas")
    faltas_in = st.slider("Número de faltas", 0, 30, step=1, key="faltas")
    nota_in = st.slider("Nota", 0.0, 10.0, step=0.1, key="nota")

    prever_clicado = st.button("🔮 Prever situação")

    resultado_placeholder = st.empty()

    if prever_clicado:
        html_resultado = prever_situacao(horas_in, faltas_in, nota_in)
        resultado_placeholder.markdown(html_resultado, unsafe_allow_html=True)

    st.markdown("**Exemplos rápidos:**")
    cols_exemplos = st.columns(len(EXEMPLOS))
    for i, ex in enumerate(EXEMPLOS):
        rotulo = f"{ex['horas']}h · {ex['faltas']}f · {ex['nota']}"
        if cols_exemplos[i].button(rotulo, key=f"exemplo_{i}"):
            st.session_state["horas"] = ex["horas"]
            st.session_state["faltas"] = ex["faltas"]
            st.session_state["nota"] = ex["nota"]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with col_tree:
    st.markdown('<div class="card-tree">', unsafe_allow_html=True)
    st.markdown("### 🌳 Árvore de decisão treinada")
    st.pyplot(FIG_ARVORE, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
