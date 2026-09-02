# 🎓 Previsor de Situação do Aluno

Aplicação didática que treina uma árvore de decisão (`scikit-learn`) para
prever se um aluno será **Aprovado**, ficará em **Recuperação** ou será
**Reprovado**, com base em horas de estudo, número de faltas e nota. A
interface é feita em [Gradio](https://www.gradio.app/), com tema escuro,
animações e visualização interativa da árvore treinada.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple)

---

## 📸 Visão geral

- Formulário com sliders para **Horas de estudo**, **Faltas** e **Nota**.
- Resultado exibido em um card animado, com emoji, cor por situação e
  barras de probabilidade de cada classe.
- Árvore de decisão treinada, renderizada como imagem dentro da própria
  interface.
- Acurácia do modelo no conjunto de teste exibida no topo da tela.

---

## 🚀 Como rodar

### 1. Pré-requisitos
- Python 3.9 ou superior

### 2. Instalar as dependências
```bash
pip install pandas numpy scikit-learn matplotlib gradio
```

### 3. Executar
```bash
python previsor_situacao_aluno_v3.py
```

O terminal vai mostrar um link local (algo como `http://127.0.0.1:7860`).
Abra no navegador para usar a interface.

---

## 🧠 Como o modelo funciona

### Dados de treino
O dataset original do protótipo tinha apenas 5 linhas — poucos exemplos
para uma árvore aprender uma regra confiável (ela acabava "decorando" os
poucos casos em vez de generalizar). Por isso, o projeto gera um **dataset
sintético com 400 alunos simulados**, seguindo uma regra de negócio
explícita:

```python
def definir_situacao(nota, faltas):
    if faltas > 20:
        return "Reprovado"      # reprovado por excesso de faltas
    if nota >= 7:
        return "Aprovado"
    if nota >= 5:
        return "Recuperação"
    return "Reprovado"
```

As **horas de estudo** influenciam a nota simulada (correlação positiva
com ruído aleatório), mas quem decide a situação final são **nota** e
**faltas** — assim como funcionaria em um cenário real.

### Rede de segurança lógica
Além da árvore de decisão, existe uma checagem extra (`_regra_bom_senso`)
para casos extremos e óbvios (ex.: nota muito baixa, excesso de faltas,
nota altíssima com poucas faltas). Se o modelo estatístico discordar da
regra nesses casos inequívocos, a regra prevalece e a interface avisa que
o resultado foi ajustado por consistência. Isso evita situações absurdas
(como prever "Aprovado" para um aluno com nota 0 e nenhum registro de
estudo) mesmo quando o modelo, por natureza estatística, erra perto das
bordas da árvore.

### Modelo
- Algoritmo: `DecisionTreeClassifier` (scikit-learn)
- `max_depth=6` para reduzir overfitting
- Split treino/teste estratificado (`test_size=0.2`)

---

## 📁 Estrutura do arquivo

| Seção no código | O que faz |
|---|---|
| Regra de negócio | Define a situação "de verdade" a partir de nota/faltas |
| Geração do dataset | Cria os 400 alunos sintéticos |
| Treino do modelo | Split + treino da árvore de decisão |
| `_regra_bom_senso` | Rede de segurança para casos extremos |
| `gerar_imagem_arvore` | Renderiza a árvore treinada com cores customizadas |
| `prever_situacao` | Função chamada pela interface a cada previsão |
| CSS / tema | Estilo escuro, animações e paleta de cores |
| `gr.Blocks(...)` | Montagem da interface |

---

## ⚠️ Limitações conhecidas

- O dataset é **sintético**, gerado a partir de uma regra simplificada —
  não representa dados reais de uma escola.
- Por ser uma árvore de decisão, previsões muito próximas dos limiares da
  regra (ex.: nota exatamente igual ao valor de corte) podem divergir da
  regra "ideal" — é um comportamento esperado de modelos estatísticos, não
  um bug.
- Este projeto tem fins didáticos (ensino de Machine Learning e Gradio) e
  **não deve ser usado para decisões acadêmicas reais**.

---

## 🛠️ Possíveis melhorias futuras

- Trocar a árvore por um `RandomForestClassifier` para maior robustez.
- Permitir upload de um dataset real (CSV) pela própria interface.
- Adicionar métricas adicionais (matriz de confusão, importância das
  features) na UI.

---

## 📄 Licença

Livre para uso educacional. Adapte como quiser para seus estudos.
