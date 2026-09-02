"""
Gera gráficos de convergência comparando GA, COA, DE e PSO.

Saídas individuais:
  - convergencia_ieee14.png
  - convergencia_wscc9.png
  - convergencia_case3.png

Saída combinada (3 | 9 | 14):
  - convergencia_todos.png

Eixo X: número de avaliações da função objetivo (comparação justa).
Eixo Y: perdas ativas da incumbente [MW].
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DIR = Path(__file__).resolve().parent

CASOS = {
    "case3": {
        "titulo": "3 barras",
        "saida": "convergencia_case3.png",
        "arquivos": {
            "PSO": "convergencia_case3_pso.csv",
            "GA": "convergencia_case3_ga.csv",
            "DE": "convergencia_case3_de.csv",
            "COA": "convergencia_case3_coa.csv",
        },
    },
    "wscc9": {
        "titulo": "WSCC 9 barras",
        "saida": "convergencia_wscc9.png",
        "arquivos": {
            "PSO": "convergencia_wscc9_pso.csv",
            "GA": "convergencia_wscc9_ga.csv",
            "DE": "convergencia_wscc9_de.csv",
            "COA": "convergencia_wscc9_coa.csv",
        },
    },
    "ieee14": {
        "titulo": "IEEE 14 barras",
        "saida": "convergencia_ieee14.png",
        "arquivos": {
            "PSO": "convergencia_ieee14_pso.csv",
            "GA": "convergencia_ieee14_ga.csv",
            "DE": "convergencia_ieee14_de.csv",
            "COA": "convergencia_ieee14_coa.csv",
        },
    },
}

# Ordem pedida: 3, 9 e 14 barras lado a lado.
ORDEM_PAINEIS = ("case3", "wscc9", "ieee14")
SAIDA_COMBINADA = "convergencia_todos.png"

ESTILO = {
    "PSO": {"color": "#0984e3", "linestyle": "-", "linewidth": 2.0},
    "GA": {"color": "#e17055", "linestyle": "--", "linewidth": 2.0},
    "DE": {"color": "#6c5ce7", "linestyle": ":", "linewidth": 2.2},
    "COA": {"color": "#00b894", "linestyle": "-.", "linewidth": 2.0},
}


def carregar_serie(caminho: Path) -> pd.DataFrame:
    """Lê um CSV e devolve colunas padronizadas: avaliacoes, perdas_mw."""
    df = pd.read_csv(caminho)

    if "avaliacoes" in df.columns:
        x = df["avaliacoes"]
    elif "avaliacao" in df.columns:
        x = df["avaliacao"]
    elif "iteracao" in df.columns:
        x = df["iteracao"]
    elif "geracao" in df.columns:
        x = df["geracao"]
    else:
        raise ValueError(f"Sem coluna de eixo X em {caminho.name}")

    if "perdas_mw" not in df.columns:
        raise ValueError(f"Sem coluna perdas_mw em {caminho.name}")

    out = pd.DataFrame(
        {
            "avaliacoes": pd.to_numeric(x, errors="coerce"),
            "perdas_mw": pd.to_numeric(df["perdas_mw"], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("avaliacoes").reset_index(drop=True)


def desenhar_caso(ax, cfg: dict, mostrar_legenda: bool = True) -> None:
    """Desenha as quatro curvas de um sistema em um eixo."""
    for algoritmo, arquivo in cfg["arquivos"].items():
        caminho = DIR / arquivo
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        serie = carregar_serie(caminho)
        estilo = ESTILO[algoritmo]
        ax.plot(
            serie["avaliacoes"],
            serie["perdas_mw"],
            label=algoritmo,
            color=estilo["color"],
            linestyle=estilo["linestyle"],
            linewidth=estilo["linewidth"],
        )
        ax.scatter(
            serie["avaliacoes"].iloc[-1],
            serie["perdas_mw"].iloc[-1],
            color=estilo["color"],
            s=28,
            zorder=5,
            edgecolor="black",
            linewidths=0.35,
        )

    ax.set_title(cfg["titulo"], fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Avaliações da função objetivo", fontweight="bold")
    ax.set_ylabel("Perdas ativas [MW]", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    if mostrar_legenda:
        ax.legend(framealpha=0.95, loc="best", fontsize=8)


def plotar_caso(nome: str, cfg: dict) -> Path:
    """Plota as quatro curvas de um sistema e salva o PNG individual."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    desenhar_caso(ax, cfg, mostrar_legenda=True)
    fig.tight_layout()

    saida = DIR / cfg["saida"]
    fig.savefig(saida, bbox_inches="tight")
    plt.close(fig)
    return saida


def plotar_todos_lado_a_lado() -> Path:
    """Gera um único PNG com case3 | WSCC9 | IEEE14."""
    fig, eixos = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    for ax, nome in zip(eixos, ORDEM_PAINEIS):
        desenhar_caso(ax, CASOS[nome], mostrar_legenda=False)

    handles, labels = eixos[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        framealpha=0.95,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Convergência ORPD — 3, 9 e 14 barras",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    fig.tight_layout()

    saida = DIR / SAIDA_COMBINADA
    fig.savefig(saida, bbox_inches="tight")
    plt.close(fig)
    return saida


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.1

    for nome, cfg in CASOS.items():
        caminho = plotar_caso(nome, cfg)
        print(f"Gerado: {caminho}")

    combinado = plotar_todos_lado_a_lado()
    print(f"Gerado: {combinado}")


if __name__ == "__main__":
    main()
