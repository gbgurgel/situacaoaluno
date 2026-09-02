"""
Previsor de Situação do Aluno (v2)
-----------------------------------
Modelo de árvore de decisão + interface Gradio (Blocks) interativa e animada.

O QUE MUDOU NESTA VERSÃO (correção do bug relatado: tudo em 0 previa "Aprovado")
---------------------------------------------------------------------------
Causa raiz: o dataset original tinha só 5 linhas. Com tão poucos exemplos, a
árvore "aprendeu" uma regra tosca (ex.: "se Faltas <= 4 então Aprovado") sem
nunca considerar a Nota nesse ramo — por isso 0 faltas, 0 horas e 0 de nota
caía em "Aprovado". Não era mais um erro de sintaxe, era falta de dado
representativo por trás da lógica.

Correções:
1. Dataset sintético com 400 alunos, gerado a partir de uma REGRA DE NEGÓCIO
   explícita (ver `definir_situacao`): Nota e Faltas determinam a situação;
   Horas de estudo influencia a Nota (correlação realista), mas não decide
   sozinha o resultado.
2. Camada de segurança lógica (`_regra_bom_senso`): além da árvore, casos
   extremos e óbvios (nota muito baixa, excesso de faltas, nota altíssima
   com poucas faltas) são verificados por uma regra determinística. Se a
   árvore discordar da regra nesses casos claros, a regra prevalece e a
   interface avisa que o resultado foi ajustado por consistência.
3. `y` como Series (não DataFrame), split estratificado, árvore com
   max_depth para não overfitar no ruído.
4. Cores revisadas: mesma paleta (verde/âmbar/vermelho) usada nos cards, nas
   barras de probabilidade e nos nós da árvore (recoloridos manualmente).
   Tema forçado para claro, já que o Gradio estava herdando o modo escuro
   do sistema operacional do usuário.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

matplotlib.use("Agg")

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


# ---------------------------------------------------------------------------
# 2) Dataset sintético (realista): horas influencia nota, nota+faltas
#    determinam a situação pela regra acima.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
N_AMOSTRAS = 400

horas = rng.uniform(0, 20, N_AMOSTRAS)
faltas = rng.uniform(0, 30, N_AMOSTRAS)
ruido = rng.normal(0, 0.9, N_AMOSTRAS)
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

# ---------------------------------------------------------------------------
# 3) Treino do modelo
# ---------------------------------------------------------------------------
x = df[["Horas_de_estudo", "Faltas", "Nota"]]
y = df["Situacao"]  # Series (1D), não DataFrame

x_train, x_teste, y_train, y_teste = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

modelo = DecisionTreeClassifier(random_state=42, max_depth=6)
modelo.fit(x_train, y_train)
ACURACIA_TESTE = modelo.score(x_teste, y_teste)

CLASSES = list(modelo.classes_)

# Paleta única usada em todo lugar (cards, barras, árvore)
CORES = {
    "Aprovado": "#16a34a",
    "Recuperação": "#d97706",
    "Reprovado": "#dc2626",
}
EMOJI = {"Aprovado": "🎉", "Recuperação": "📘", "Reprovado": "📕"}


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


FUNDO_ARVORE = "#0b1020"


def gerar_imagem_arvore() -> str:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(FUNDO_ARVORE)
    ax.set_facecolor(FUNDO_ARVORE)
    plot_tree(
        modelo,
        feature_names=list(x.columns),
        class_names=CLASSES,
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
    caminho = "arvore_decisao.png"
    fig.savefig(caminho, dpi=150, facecolor=FUNDO_ARVORE)
    plt.close(fig)
    return caminho


IMG_ARVORE = gerar_imagem_arvore()


# ---------------------------------------------------------------------------
# 4) Função de previsão (modelo + validação + rede de segurança lógica)
# ---------------------------------------------------------------------------
def prever_situacao(horas_in, faltas_in, nota_in):
    if horas_in is None or faltas_in is None or nota_in is None:
        raise gr.Error("Preencha os três campos antes de prever.")
    if horas_in < 0 or faltas_in < 0:
        raise gr.Error("Horas de estudo e faltas não podem ser negativas.")
    if not (0 <= nota_in <= 10):
        raise gr.Error("A nota deve estar entre 0 e 10.")

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
# 5) CSS — tema claro forçado + animações + paleta consistente
# ---------------------------------------------------------------------------
CSS = """
:root, .dark { color-scheme: light; }

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

body, .gradio-container, .dark .gradio-container {
  background: radial-gradient(circle at 15% 0%, #1e1b4b 0%, #0b1020 45%, #030712 100%) !important;
  color: #e2e8f0 !important;
}
#titulo { animation: fadeInUp 0.6s ease-out; text-align: center; }
#titulo h1 { color: #c7d2fe !important; text-shadow: 0 0 24px rgba(129, 140, 248, 0.35); }

#card-form, #card-tree, .dark #card-form, .dark #card-tree {
  animation: fadeInUp 0.7s ease-out;
  border-radius: 16px !important;
  background: #131a2c !important;
  border: 1px solid #262f47 !important;
  color: #e2e8f0 !important;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
  transition: box-shadow 0.25s ease, transform 0.25s ease;
}
#card-form:hover, #card-tree:hover {
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6);
  transform: translateY(-2px);
  animation: glow 2.4s ease-in-out infinite;
}
#card-form h3, #card-tree h3, .dark #card-form h3, .dark #card-tree h3 { color: #a5b4fc !important; }
#card-form label span, .dark #card-form label span { color: #cbd5e1 !important; }

#botao-prever {
  background: linear-gradient(135deg, #4338ca, #7e22ce) !important;
  border: none !important;
  color: #f8fafc !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  font-weight: 600 !important;
}
#botao-prever:hover { transform: scale(1.03); box-shadow: 0 6px 20px rgba(126, 34, 206, 0.5); }
#botao-prever:active { transform: scale(0.97); }

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

input[type="range"] { accent-color: #818cf8; }
"""

# Garante o tema escuro mesmo se o navegador do usuário estiver em modo claro
FORCE_DARK_JS = """
() => {
    document.documentElement.classList.add('dark');
}
"""

# ---------------------------------------------------------------------------
# 6) Interface
# ---------------------------------------------------------------------------
with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="indigo"), js=FORCE_DARK_JS) as interface:
    gr.Markdown("# 🎓 Previsor de Situação do Aluno", elem_id="titulo")
    gr.Markdown(
        f"Modelo treinado com {N_AMOSTRAS} alunos simulados a partir de uma regra "
        f"realista (nota e faltas definem a situação). Acurácia no conjunto de "
        f"teste: **{ACURACIA_TESTE*100:.1f}%**.",
        elem_id="aviso-dados",
    )

    with gr.Row():
        with gr.Column(elem_id="card-form", variant="panel"):
            gr.Markdown("### 📋 Dados do aluno")
            horas_slider = gr.Slider(0, 20, value=6, step=1, label="Horas de estudo por semana")
            faltas_slider = gr.Slider(0, 30, value=2, step=1, label="Número de faltas")
            nota_slider = gr.Slider(0, 10, value=7.0, step=0.1, label="Nota")
            botao = gr.Button("🔮 Prever situação", elem_id="botao-prever", variant="primary")
            saida = gr.HTML()

        with gr.Column(elem_id="card-tree", variant="panel"):
            gr.Markdown("### 🌳 Árvore de decisão treinada")
            gr.Image(value=IMG_ARVORE, show_label=False, container=False)

    botao.click(fn=prever_situacao, inputs=[horas_slider, faltas_slider, nota_slider], outputs=saida)

    gr.Examples(
        examples=[[0, 0, 0], [10, 1, 9.0], [3, 12, 4.0], [6, 5, 6.0], [15, 25, 8.5]],
        inputs=[horas_slider, faltas_slider, nota_slider],
    )

if __name__ == "__main__":
    interface.launch()
