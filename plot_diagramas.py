"""
Gera diagramas unifilares dos três sistemas usados no ORPD:

  - diagrama_case3.png
  - diagrama_wscc9.png
  - diagrama_ieee14.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
from matplotlib.lines import Line2D

from rede_case3 import case3_orpd

DIR = Path(__file__).resolve().parent

CASOS = {
    "case3": {
        "titulo": "Diagrama unifilar — 3 barras (ORPD sintético)",
        "saida": "diagrama_case3.png",
        "carregar": case3_orpd,
        # Coordenadas manuais alinhadas à topologia do case3.
        "coords": {0: (0.0, 1.0), 1: (2.0, 1.0), 2: (1.0, 0.0)},
    },
    "wscc9": {
        "titulo": "Diagrama unifilar — WSCC 9 barras",
        "saida": "diagrama_wscc9.png",
        "carregar": nw.case9,
        "coords": None,
    },
    "ieee14": {
        "titulo": "Diagrama unifilar — IEEE 14 barras",
        "saida": "diagrama_ieee14.png",
        "carregar": nw.case14,
        "coords": None,
    },
}


def definir_coordenadas(net: pp.pandapowerNet, coords: dict[int, tuple[float, float]]) -> None:
    """Grava coordenadas no formato GeoJSON usado pelo pandapower atual."""
    geos = []
    for barra in net.bus.index:
        x, y = coords[int(barra)]
        geos.append(json.dumps({"type": "Point", "coordinates": [float(x), float(y)]}))
    net.bus["geo"] = geos


def coordenada_barra(net: pp.pandapowerNet, barra: int) -> tuple[float, float]:
    """Lê a coordenada (x, y) de uma barra a partir de bus.geo."""
    geo = net.bus.at[barra, "geo"]
    if isinstance(geo, str):
        geo = json.loads(geo)
    x, y = geo["coordinates"]
    return float(x), float(y)


def garantir_coordenadas(net: pp.pandapowerNet, coords: dict | None) -> None:
    """Define coordenadas manuais ou gera automaticamente."""
    if coords is not None:
        definir_coordenadas(net, coords)
        return
    plot.create_generic_coordinates(net, overwrite=True)


def desenhar_rede(net: pp.pandapowerNet, ax, titulo: str) -> None:
    """Desenha a topologia com rótulos de barras e equipamentos."""
    plot.simple_plot(
        net,
        ax=ax,
        show_plot=False,
        plot_loads=True,
        plot_gens=True,
        bus_size=1.4,
        gen_size=1.2,
        load_size=1.2,
        trafo_size=1.3,
        line_width=2.2,
        scale_size=True,
        bus_color="#1c3f52",
        line_color="#7f8c8d",
        trafo_color="#2c3e50",
        ext_grid_color="#c0392b",
        enable_hover=False,
        plot_colorbars=False,
    )

    xs = []
    ys = []
    for barra in net.bus.index:
        x, y = coordenada_barra(net, barra)
        xs.append(x)
        ys.append(y)
        ax.text(
            x,
            y + 0.06,
            f"B{barra}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
            zorder=10,
        )

    dx = max(xs) - min(xs) if xs else 1.0
    dy = max(ys) - min(ys) if ys else 1.0
    offset = 0.04 * max(dx, dy, 1.0)

    for _, row in net.ext_grid.iterrows():
        x, y = coordenada_barra(net, int(row.bus))
        ax.text(
            x - offset,
            y - offset,
            "Slack",
            ha="right",
            va="top",
            fontsize=8,
            color="#c0392b",
            zorder=10,
        )

    for i, row in net.gen.iterrows():
        x, y = coordenada_barra(net, int(row.bus))
        ax.text(
            x + offset,
            y - offset,
            f"G{i}",
            ha="left",
            va="top",
            fontsize=8,
            color="#0984e3",
            zorder=10,
        )

    for i, row in net.shunt.iterrows():
        x, y = coordenada_barra(net, int(row.bus))
        ax.text(
            x,
            y - 2.2 * offset,
            f"Sh{i}",
            ha="center",
            va="top",
            fontsize=8,
            color="#00b894",
            zorder=10,
        )

    for i, row in net.trafo.iterrows():
        x1, y1 = coordenada_barra(net, int(row.hv_bus))
        x2, y2 = coordenada_barra(net, int(row.lv_bus))
        ax.text(
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
            f"T{i}",
            ha="center",
            va="center",
            fontsize=8,
            color="#2c3e50",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            zorder=10,
        )

    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=10)
    ax.set_axis_off()

    legenda = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1c3f52",
               markersize=9, label="Barra"),
        Line2D([0], [0], color="#7f8c8d", linewidth=2.2, label="Linha"),
        Line2D([0], [0], color="#2c3e50", linewidth=2.2, label="Transformador"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#c0392b",
               markersize=8, label="Slack"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#0984e3",
               markersize=9, label="Gerador"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e17055",
               markersize=9, label="Carga"),
    ]
    if len(net.shunt):
        legenda.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#00b894",
                   markersize=8, label="Shunt")
        )
    ax.legend(handles=legenda, loc="best", framealpha=0.95, fontsize=8)


def gerar_diagrama(nome: str, cfg: dict) -> Path:
    """Gera e salva o diagrama de um caso."""
    net = cfg["carregar"]()
    garantir_coordenadas(net, cfg["coords"])

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    desenhar_rede(net, ax, cfg["titulo"])

    saida = DIR / cfg["saida"]
    fig.savefig(saida, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return saida


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    for nome, cfg in CASOS.items():
        caminho = gerar_diagrama(nome, cfg)
        print(f"Gerado: {caminho}")


if __name__ == "__main__":
    main()
