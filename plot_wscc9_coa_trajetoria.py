"""
Trajetória 3D do COA no espaço de busca do WSCC 9 barras.

Reconstrói a sequência exata dos vetores avaliados a partir do mapa logístico,
da semente e dos parâmetros da execução já gravada em
``minimos_locais_wscc9_coa.csv``. Não chama a função objetivo nem reexecuta
a otimização.

Eixos:
  X = Vslack [p.u.], Y = Vg1 [p.u.], Z = Vg2 [p.u.]

Saídas:
  - trajetoria_wscc9_coa.png
  - trajetoria_wscc9_coa.pdf
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from rpd_wscc9_coa import (
    ARQUIVO_MINIMOS,
    ETA_INICIAL,
    ETA_MAXIMO,
    S1,
    S2,
    SEED,
    T3,
    estado_caotico_inicial,
    mapa_logistico,
)
from rpd_wscc9_pso import N_CONTROLE, VM_LB, VM_UB

DIR = Path(__file__).resolve().parent
SAIDA_PNG = DIR / "trajetoria_wscc9_coa.png"
SAIDA_PDF = DIR / "trajetoria_wscc9_coa.pdf"
TOLERANCIA_INCUMBENTE = 5.0e-10

COR_ONDA1 = "#0072B2"
COR_ONDA2 = "#D55E00"
COR_PRIMEIRO = "#111111"
COR_MELHOR = "#009E73"
COR_CUBO = "#9AA0A6"


def carregar_incumbentes(caminho: Path) -> list[dict]:
    """Lê os mínimos/incumbentes gravados na execução original."""
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        registros = []
        for linha in csv.DictReader(arquivo):
            registros.append(
                {
                    "fase": linha["fase"],
                    "avaliacao": int(linha["avaliacao"]),
                    "custo": float(linha["custo"]),
                    "x": np.asarray(
                        [
                            float(linha["vm_slack"]),
                            float(linha["vm_gen_0"]),
                            float(linha["vm_gen_1"]),
                        ],
                        dtype=float,
                    ),
                }
            )
    if not registros:
        raise RuntimeError(f"Nenhum registro em {caminho}.")
    return registros


def reconstruir_trajetoria(
    incumbentes: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduz os candidatos na ordem das avaliações, sem otimizar.

    A primeira onda só depende do mapa caótico. A segunda usa a incumbente
    vigente, atualizada nos instantes já registrados no CSV.
    """
    por_avaliacao = {reg["avaliacao"]: reg for reg in incumbentes}
    lb = np.full(N_CONTROLE, VM_LB, dtype=float)
    ub = np.full(N_CONTROLE, VM_UB, dtype=float)
    amplitude = ub - lb

    pontos: list[np.ndarray] = []
    fases: list[str] = []
    divergencias: list[str] = []

    def conferir(avaliacao: int, candidato: np.ndarray) -> None:
        registro = por_avaliacao.get(avaliacao)
        if registro is None:
            return
        erro = float(np.max(np.abs(candidato - registro["x"])))
        if erro > TOLERANCIA_INCUMBENTE:
            divergencias.append(
                f"aval. {avaliacao}: |Δx|_∞ = {erro:.3e} "
                f"(fase {registro['fase']})"
            )

    x_base = por_avaliacao[1]["x"].copy()
    pontos.append(x_base)
    fases.append("caso_base")
    melhor_x = x_base.copy()

    rng = np.random.default_rng(SEED)
    z = estado_caotico_inicial(N_CONTROLE, rng)

    for i in range(S1):
        avaliacao = 2 + i
        candidato = lb + z * amplitude
        conferir(avaliacao, candidato)
        if avaliacao in por_avaliacao:
            melhor_x = candidato.copy()
        pontos.append(candidato.copy())
        fases.append("primeira_onda")
        z = mapa_logistico(z)

    eta_fracao = ETA_INICIAL
    avaliacao = 1 + S1
    while eta_fracao <= ETA_MAXIMO + 1.0e-12:
        eta = eta_fracao * amplitude
        for _ in range(S2):
            z = mapa_logistico(z)
            avaliacao += 1
            candidato = np.clip(melhor_x + eta * (z - 0.5), lb, ub)
            conferir(avaliacao, candidato)
            if avaliacao in por_avaliacao:
                melhor_x = candidato.copy()
            pontos.append(candidato.copy())
            fases.append("segunda_onda")
        eta_fracao *= T3

    if divergencias:
        raise RuntimeError(
            "A trajetória reconstruída não reproduz os incumbentes gravados:\n"
            + "\n".join(divergencias)
        )

    esperado = 1 + S1
    eta_fracao = ETA_INICIAL
    while eta_fracao <= ETA_MAXIMO + 1.0e-12:
        esperado += S2
        eta_fracao *= T3
    if len(pontos) != esperado:
        raise RuntimeError(
            f"Número de pontos reconstruídos ({len(pontos)}) "
            f"diferente do esperado ({esperado})."
        )

    return np.vstack(pontos), np.asarray(fases)


def _arestas_cubo(vmin: float, vmax: float) -> np.ndarray:
    """Segmentos das 12 arestas do paralelepípedo [vmin, vmax]^3."""
    r = (vmin, vmax)
    vertices = np.array([(x, y, z) for x in r for y in r for z in r], dtype=float)
    segmentos = []
    for i, a in enumerate(vertices):
        for b in vertices[i + 1 :]:
            if np.count_nonzero(np.abs(a - b) > 1.0e-15) == 1:
                segmentos.append((a, b))
    return np.asarray(segmentos)


def _face_sombra(vmin: float, vmax: float, plano: str) -> np.ndarray:
    """Polígono da face do cubo usada como sombra de projeção."""
    if plano == "xy":
        return np.array(
            [
                [vmin, vmin, vmin],
                [vmax, vmin, vmin],
                [vmax, vmax, vmin],
                [vmin, vmax, vmin],
            ],
            dtype=float,
        )
    if plano == "xz":
        return np.array(
            [
                [vmin, vmax, vmin],
                [vmax, vmax, vmin],
                [vmax, vmax, vmax],
                [vmin, vmax, vmax],
            ],
            dtype=float,
        )
    return np.array(
        [
            [vmin, vmin, vmin],
            [vmin, vmax, vmin],
            [vmin, vmax, vmax],
            [vmin, vmin, vmax],
        ],
        dtype=float,
    )


def plot_trajetoria(
    pontos: np.ndarray,
    fases: np.ndarray,
    incumbentes: list[dict],
    saida_png: Path | str = SAIDA_PNG,
    saida_pdf: Path | str = SAIDA_PDF,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Desenha as duas ondas no cubo [VM_LB, VM_UB]^3 e grava PNG/PDF."""
    mascara1 = fases == "primeira_onda"
    mascara2 = fases == "segunda_onda"
    onda1 = pontos[mascara1]
    onda2 = pontos[mascara2]
    primeiro = pontos[0]
    melhor = incumbentes[-1]
    x_melhor = melhor["x"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.linewidth": 1.1,
        }
    )

    fig = plt.figure(figsize=(10.4, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    # Sem isto o mplot3d ordena pelo z médio da coleção e a nuvem
    # laranja cobre a estrela do ótimo, mesmo com zorder maior.
    ax.computed_zorder = False

    cubo = _arestas_cubo(VM_LB, VM_UB)
    ax.add_collection3d(
        Line3DCollection(cubo, colors=COR_CUBO, linewidths=1.0, alpha=0.9)
    )
    for plano in ("xy", "xz", "yz"):
        ax.add_collection3d(
            Poly3DCollection(
                [_face_sombra(VM_LB, VM_UB, plano)],
                facecolors="#f7f8fa",
                edgecolors="none",
                alpha=0.45,
                zorder=0,
            )
        )

    def projetar(xyz: np.ndarray, cor: str, s: float, alpha: float) -> None:
        ax.scatter(
            xyz[:, 0], xyz[:, 1], np.full(len(xyz), VM_LB),
            s=s, c=cor, alpha=alpha, linewidths=0, depthshade=False, zorder=1,
        )
        ax.scatter(
            xyz[:, 0], np.full(len(xyz), VM_UB), xyz[:, 2],
            s=s, c=cor, alpha=alpha, linewidths=0, depthshade=False, zorder=1,
        )
        ax.scatter(
            np.full(len(xyz), VM_LB), xyz[:, 1], xyz[:, 2],
            s=s, c=cor, alpha=alpha, linewidths=0, depthshade=False, zorder=1,
        )

    projetar(onda1, COR_ONDA1, s=3.0, alpha=0.08)
    projetar(onda2, COR_ONDA2, s=6.0, alpha=0.18)

    caixa2 = _arestas_cubo(0.0, 1.0)
    mins2, maxs2 = onda2.min(axis=0), onda2.max(axis=0)
    caixa2 = mins2 + caixa2 * (maxs2 - mins2)
    ax.add_collection3d(
        Line3DCollection(
            caixa2,
            colors=COR_ONDA2,
            linewidths=1.15,
            linestyles="--",
            alpha=0.85,
        )
    )

    # Nuvem da primeira onda (exploração): pontos na ordem avaliada, sem
    # polilinha — o mapa caótico saltaria pelo cubo inteiro e obscureceria
    # o refinamento.
    ax.scatter(
        onda1[:, 0],
        onda1[:, 1],
        onda1[:, 2],
        s=8,
        c=COR_ONDA1,
        alpha=0.22,
        linewidths=0,
        depthshade=False,
        zorder=3,
    )

    # Trajetória da segunda onda (refinamento): ordem das avaliações.
    ax.plot(
        onda2[:, 0],
        onda2[:, 1],
        onda2[:, 2],
        color=COR_ONDA2,
        linewidth=0.7,
        alpha=0.45,
        zorder=4,
    )
    ax.scatter(
        onda2[:, 0],
        onda2[:, 1],
        onda2[:, 2],
        s=18,
        c=COR_ONDA2,
        alpha=0.62,
        linewidths=0,
        depthshade=False,
        zorder=5,
    )

    ax.scatter(
        [primeiro[0]],
        [primeiro[1]],
        [primeiro[2]],
        s=110,
        marker="s",
        facecolors="white",
        edgecolors=COR_PRIMEIRO,
        linewidths=1.8,
        depthshade=False,
        zorder=8,
    )
    ax.text(
        primeiro[0] - 0.004,
        primeiro[1],
        primeiro[2] - 0.018,
        "início",
        fontsize=9,
        color=COR_PRIMEIRO,
        ha="right",
        va="top",
        zorder=10,
    )
    ax.plot(
        [x_melhor[0]],
        [x_melhor[1]],
        [x_melhor[2]],
        marker="o",
        markersize=16,
        markerfacecolor="white",
        markeredgecolor="white",
        markeredgewidth=0,
        linestyle="none",
        zorder=9,
        clip_on=False,
    )
    ax.plot(
        [x_melhor[0]],
        [x_melhor[1]],
        [x_melhor[2]],
        marker="*",
        markersize=22,
        markerfacecolor=COR_MELHOR,
        markeredgecolor="black",
        markeredgewidth=1.1,
        linestyle="none",
        zorder=10,
        clip_on=False,
    )
    ax.text(
        x_melhor[0],
        x_melhor[1],
        min(VM_UB - 0.004, x_melhor[2] + 0.022),
        rf"$F^* = {melhor['custo']:.4f}$",
        fontsize=10.5,
        color="black",
        ha="center",
        va="bottom",
        zorder=11,
    )

    ax.set_xlabel(r"$V_{\mathrm{slack}}$ [p.u.]", labelpad=8)
    ax.set_ylabel(r"$V_{\mathrm{g1}}$ [p.u.]", labelpad=8)
    ax.set_zlabel(r"$V_{\mathrm{g2}}$ [p.u.]", labelpad=8)
    ax.set_xlim(VM_LB, VM_UB)
    ax.set_ylim(VM_LB, VM_UB)
    ax.set_zlim(VM_LB, VM_UB)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ticks = np.linspace(VM_LB, VM_UB, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.tick_params(labelsize=9, pad=2)
    ax.view_init(elev=20, azim=-52)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(COR_CUBO)
    ax.yaxis.pane.set_edgecolor(COR_CUBO)
    ax.zaxis.pane.set_edgecolor(COR_CUBO)
    ax.grid(False)

    ax.set_title(
        "Trajetória do COA no espaço de busca — WSCC 9 barras",
        fontsize=13,
        pad=14,
    )
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=COR_ONDA1,
            markeredgecolor="none",
            markersize=8,
            label="Primeira onda — exploração global",
        ),
        Line2D(
            [0], [0],
            color=COR_ONDA2,
            linewidth=1.6,
            marker="o",
            markerfacecolor=COR_ONDA2,
            markeredgecolor="none",
            markersize=7,
            label="Segunda onda — refinamento local",
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COR_PRIMEIRO,
            markeredgewidth=1.6,
            markersize=8,
            label="Primeiro ponto avaliado",
        ),
        Line2D(
            [0], [0],
            marker="*",
            color="none",
            markerfacecolor=COR_MELHOR,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=14,
            label=rf"Melhor ponto  $F^* = {melhor['custo']:.4f}$",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.04),
        frameon=True,
        framealpha=0.95,
        fontsize=9.5,
        borderpad=0.55,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.88)
    caminho_png = Path(saida_png).resolve()
    caminho_pdf = Path(saida_pdf).resolve()
    fig.savefig(caminho_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(caminho_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return caminho_png, caminho_pdf


def main() -> None:
    csv_minimos = DIR / ARQUIVO_MINIMOS
    print("=" * 68)
    print(" Trajetória 3D do COA — WSCC 9 barras")
    print("=" * 68)
    print(f" Incumbentes       : {csv_minimos.name}")
    print(f" Limites           : [{VM_LB}, {VM_UB}] p.u. (Vslack, Vg1, Vg2)")
    print(f" Primeira onda     : S1 = {S1}")
    print(f" Segunda onda      : S2 = {S2} por amplitude")
    print(f" Semente           : {SEED}")
    print("-" * 68)

    incumbentes = carregar_incumbentes(csv_minimos)
    pontos, fases = reconstruir_trajetoria(incumbentes)
    n1 = int(np.sum(fases == "primeira_onda"))
    n2 = int(np.sum(fases == "segunda_onda"))
    melhor = incumbentes[-1]
    print(f" Avaliações        : {len(pontos)}")
    print(f"   caso base       : 1")
    print(f"   primeira onda   : {n1}")
    print(f"   segunda onda    : {n2}")
    print(f" Incumbentes CSV   : {len(incumbentes)}")
    print(
        " Melhor ponto      : "
        f"Vslack={melhor['x'][0]:.6f}, "
        f"Vg1={melhor['x'][1]:.6f}, "
        f"Vg2={melhor['x'][2]:.6f}"
    )
    print(f" F*                : {melhor['custo']:.10f}")
    print(" Reconstrução      : incumbentes reproduzidos dentro da tolerância")

    png, pdf = plot_trajetoria(pontos, fases, incumbentes)
    print(f" PNG salvo em      : {png}")
    print(f" PDF salvo em      : {pdf}")
    print("=" * 68)


if __name__ == "__main__":
    main()
