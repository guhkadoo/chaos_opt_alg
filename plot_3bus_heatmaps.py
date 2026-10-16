"""
Heatmaps da função objetivo do ORPD no sistema de 3 barras.

Para cada combinação discreta (tap, shunt), fixa essas variáveis e varre a
superfície contínua (Vslack × Vpv), colorindo pelo valor da função objetivo
já definida em ``rpd_case3_pso.criar_funcao_objetivo`` (perdas ativas +
penalidades de tensão).

Escala de cores (compartilhada):
  - F ∈ [0, 1]: mesmo tom (azul), mais intenso à medida que F → 0
  - F ∈ [1, 10]: gradiente distinto; valores > 10 saturam no extremo quente

Saídas:
  - heatmap_case3.png
  - heatmap_case3.pdf
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from rpd_case3_pso import (
    CUSTO_DIVERGENCIA,
    TAP_MAX,
    TAP_MIN,
    VM_LB,
    VM_UB,
    carregar_rede,
    criar_funcao_objetivo,
)

DIR = Path(__file__).resolve().parent
SAIDA_PNG = DIR / "heatmap_case3.png"
SAIDA_PDF = DIR / "heatmap_case3.pdf"
CACHE_SUPERFICIES = DIR / "heatmap_case3_superficies.npz"

# Resolução da malha contínua (não altera limites nem a função objetivo).
N_GRID = 41

TAPS = list(range(TAP_MIN[0], TAP_MAX[0] + 1))  # {-3,...,3}
SHUNTS = [0, 1]

# Cortes da escala visual (MW). Acima de F_VMAX a cor satura.
F_BREAK = 1.0
F_VMAX = 10.0
# Fração do colormap reservada à faixa [0, F_BREAK] (mais resolução nos ótimos).
FRAC_BAIXA = 0.65


def _avaliar_superficie_worker(
    tap: int,
    shunt: int,
    vslack: np.ndarray,
    vpv: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """Worker de processo: avalia F(Vslack, Vpv) com tap e shunt fixos."""
    funcao_objetivo = criar_funcao_objetivo(carregar_rede())
    z = np.empty((len(vpv), len(vslack)), dtype=float)
    for i, y in enumerate(vpv):
        for j, x in enumerate(vslack):
            z[i, j] = float(
                funcao_objetivo(np.asarray([x, y, tap, shunt], dtype=float))
            )
    return tap, shunt, z


def _localizar_minimo(
    z: np.ndarray,
    vslack: np.ndarray,
    vpv: np.ndarray,
) -> tuple[float, float, float]:
    """Retorna (Vslack*, Vpv*, F*) ignorando divergências quando possível."""
    mascarado = np.ma.masked_greater_equal(z, CUSTO_DIVERGENCIA * 0.5)
    if mascarado.count() == 0:
        i_min, j_min = np.unravel_index(np.argmin(z), z.shape)
    else:
        i_min, j_min = np.unravel_index(np.argmin(mascarado), z.shape)
    return float(vslack[j_min]), float(vpv[i_min]), float(z[i_min, j_min])


class FaixaDuplaNorm(Normalize):
    """
    Normalização em duas faixas para expandir o contraste em [0, F_BREAK].

    [0, F_BREAK]  → [0, FRAC_BAIXA] do colormap (com gama > 1 para
                    dar mais contraste perto de F_BREAK, onde estão os mínimos)
    [F_BREAK, F_VMAX] → [FRAC_BAIXA, 1]; valores > F_VMAX saturam.
    """

    def __init__(
        self,
        v_break: float = F_BREAK,
        vmax: float = F_VMAX,
        frac_baixa: float = FRAC_BAIXA,
        gamma_baixa: float = 2.0,
    ):
        super().__init__(vmin=0.0, vmax=vmax, clip=True)
        self.v_break = float(v_break)
        self.frac_baixa = float(frac_baixa)
        self.gamma_baixa = float(gamma_baixa)

    def __call__(self, value, clip=None):
        x = np.ma.asarray(value, dtype=float)
        resultado = np.ma.empty(x.shape, dtype=float)
        resultado.mask = np.ma.getmaskarray(x)

        baixo = x <= self.v_break
        alto = ~baixo

        # Gama > 1: comprime perto de 0 e abre o contraste em ~0,5–1.
        u = np.clip(x[baixo] / self.v_break, 0.0, 1.0)
        resultado[baixo] = self.frac_baixa * (u**self.gamma_baixa)
        resultado[alto] = self.frac_baixa + (1.0 - self.frac_baixa) * (
            (np.clip(x[alto], self.v_break, self.vmax) - self.v_break)
            / (self.vmax - self.v_break)
        )
        return np.ma.clip(resultado, 0.0, 1.0)

    def inverse(self, value):
        y = np.asarray(value, dtype=float)
        out = np.empty_like(y)
        baixo = y <= self.frac_baixa
        out[baixo] = self.v_break * (
            (y[baixo] / self.frac_baixa) ** (1.0 / self.gamma_baixa)
        )
        out[~baixo] = self.v_break + (self.vmax - self.v_break) * (
            (y[~baixo] - self.frac_baixa) / (1.0 - self.frac_baixa)
        )
        return out


def _criar_cmap_objetivo() -> LinearSegmentedColormap:
    """
    Colormap em duas famílias:
      - [0, F_BREAK]: azul único, mais forte (escuro/saturado) perto de 0
      - [F_BREAK, F_VMAX]: transição ciano → verde → amarelo → laranja → vermelho
    """
    # Posições no espaço do colormap [0, 1] alinhadas a FaixaDuplaNorm.
    # Contraste ampliado em [0,1]: os mínimos reais (~0,6–1,0) precisam
    # variar de azul médio a azul bem claro, não ficar numa mancha única.
    nos = [
        (0.00, "#000814"),  # F ≈ 0: azul quase preto (mais forte)
        (0.15, "#001d3d"),
        (0.30, "#003566"),
        (0.42, "#0077b6"),
        (0.52, "#00b4d8"),
        (0.60, "#48cae4"),
        (FRAC_BAIXA, "#ade8f4"),  # F = 1: azul bem claro
        (0.72, "#80ed99"),  # ~2–3
        (0.80, "#caffbf"),  # ~4
        (0.87, "#ffd60a"),  # ~5–6
        (0.93, "#ff9f1c"),  # ~7–8
        (1.00, "#e63946"),  # F ≥ 10
    ]
    return LinearSegmentedColormap.from_list("objetivo_orpd", nos, N=512)


def _chave_cache(tap: int, shunt: int) -> str:
    return f"z_tap{tap:+d}_shunt{shunt}"


def _salvar_cache(
    superficies: dict[tuple[int, int], np.ndarray],
    vslack: np.ndarray,
    vpv: np.ndarray,
    caminho: Path = CACHE_SUPERFICIES,
) -> None:
    payload = {"vslack": vslack, "vpv": vpv, "n_grid": np.asarray(len(vslack))}
    for (tap, shunt), z in superficies.items():
        payload[_chave_cache(tap, shunt)] = z
    np.savez_compressed(caminho, **payload)


def _carregar_cache(
    caminho: Path = CACHE_SUPERFICIES,
    n_grid: int = N_GRID,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]] | None:
    if not caminho.is_file():
        return None
    dados = np.load(caminho)
    if int(dados["n_grid"]) != n_grid:
        return None
    vslack = dados["vslack"]
    vpv = dados["vpv"]
    superficies = {}
    for shunt in SHUNTS:
        for tap in TAPS:
            chave = _chave_cache(tap, shunt)
            if chave not in dados:
                return None
            superficies[(tap, shunt)] = dados[chave]
    return vslack, vpv, superficies


def _avaliar_todas_superficies(
    vslack: np.ndarray,
    vpv: np.ndarray,
    n_workers: int | None,
) -> dict[tuple[int, int], np.ndarray]:
    pares = [(tap, shunt) for shunt in SHUNTS for tap in TAPS]
    superficies: dict[tuple[int, int], np.ndarray] = {}
    print(f"Avaliando {len(pares)} superfícies em malha {len(vslack)}×{len(vpv)}...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futuros = {
            executor.submit(_avaliar_superficie_worker, tap, shunt, vslack, vpv): (tap, shunt)
            for tap, shunt in pares
        }
        concluidos = 0
        for futuro in as_completed(futuros):
            tap, shunt, z = futuro.result()
            superficies[(tap, shunt)] = z
            concluidos += 1
            f_min = _localizar_minimo(z, vslack, vpv)[2]
            print(
                f"  [{concluidos:2d}/{len(pares)}] tap={tap:+d}, shunt={shunt} "
                f"-> min F = {f_min:.6f}"
            )
    return superficies


def plot_3bus_heatmaps(
    n_grid: int = N_GRID,
    saida_png: Path | str = SAIDA_PNG,
    saida_pdf: Path | str = SAIDA_PDF,
    dpi: int = 300,
    n_workers: int | None = None,
    usar_cache: bool = True,
    forcar_recalculo: bool = False,
) -> tuple[Path, Path]:
    """
    Gera 14 heatmaps (7 taps × 2 shunts) em uma única figura com escala comum.

    Retorna os caminhos absolutos do PNG e do PDF salvos.
    """
    cache = None if forcar_recalculo or not usar_cache else _carregar_cache(n_grid=n_grid)
    if cache is not None:
        vslack, vpv, superficies = cache
        print(f"Cache carregado: {CACHE_SUPERFICIES.name} ({n_grid}×{n_grid})")
    else:
        vslack = np.linspace(VM_LB, VM_UB, n_grid)
        vpv = np.linspace(VM_LB, VM_UB, n_grid)
        superficies = _avaliar_todas_superficies(vslack, vpv, n_workers)
        _salvar_cache(superficies, vslack, vpv)
        print(f"Cache salvo em: {CACHE_SUPERFICIES}")

    minimos = {
        chave: _localizar_minimo(z, vslack, vpv) for chave, z in superficies.items()
    }

    norma = FaixaDuplaNorm(v_break=F_BREAK, vmax=F_VMAX, frac_baixa=FRAC_BAIXA)
    cmap = _criar_cmap_objetivo()
    # Valores acima de F_VMAX usam a cor do extremo; divergências ficam mascaradas.
    cmap.set_over(cmap(1.0))
    cmap.set_bad("#d0d0d0")

    n_cols = len(TAPS)
    n_rows = len(SHUNTS)
    fig, eixos = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.5 * n_rows + 0.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    imagem_ref = None
    for i_row, shunt in enumerate(SHUNTS):
        for i_col, tap in enumerate(TAPS):
            ax = eixos[i_row, i_col]
            z = superficies[(tap, shunt)]
            z_plot = np.ma.masked_greater_equal(z, CUSTO_DIVERGENCIA * 0.5)

            imagem_ref = ax.pcolormesh(
                vslack,
                vpv,
                z_plot,
                shading="auto",
                cmap=cmap,
                norm=norma,
            )

            v_s, v_p, f_min = minimos[(tap, shunt)]
            ax.plot(
                v_s,
                v_p,
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="crimson",
                markeredgewidth=1.2,
                linestyle="none",
                zorder=5,
            )
            ax.set_title(f"tap = {tap:+d}, shunt = {shunt}", fontsize=9)
            if i_row == n_rows - 1:
                ax.set_xlabel(r"$V_{\mathrm{slack}}$ [p.u.]")
            if i_col == 0:
                ax.set_ylabel(r"$V_{\mathrm{pv}}$ [p.u.]")
            ax.set_xlim(VM_LB, VM_UB)
            ax.set_ylim(VM_LB, VM_UB)
            ax.set_aspect("equal")
            ax.text(
                0.97,
                0.03,
                f"min = {f_min:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "black",
                    "alpha": 0.45,
                    "edgecolor": "none",
                },
            )

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    cbar = fig.colorbar(
        imagem_ref,
        ax=eixos.ravel().tolist(),
        shrink=0.85,
        extend="max",
        ticks=ticks,
        label="Função objetivo $F(x)$ [MW]",
    )
    cbar.ax.set_yticklabels([f"{t:g}" for t in ticks])
    fig.suptitle(
        "Superfícies da função objetivo — sistema de 3 barras "
        rf"($V_{{\mathrm{{slack}}}}, V_{{\mathrm{{pv}}}} \in [{VM_LB:.2f},\ {VM_UB:.2f}]$ p.u.)",
        fontsize=12,
    )

    caminho_png = Path(saida_png).resolve()
    caminho_pdf = Path(saida_pdf).resolve()
    fig.savefig(caminho_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(caminho_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return caminho_png, caminho_pdf


def main() -> None:
    print("=" * 66)
    print(" Heatmaps ORPD — 3 barras (14 combinações tap × shunt)")
    print("=" * 66)
    print(f" Limites Vslack/Vpv : [{VM_LB}, {VM_UB}] p.u.")
    print(f" Taps               : {TAPS}")
    print(f" Shunts             : {SHUNTS}")
    print(f" Malha              : {N_GRID} × {N_GRID}")
    print(f" Escala de cores    : [0, {F_BREAK}] azul intenso→claro; "
          f"[{F_BREAK}, {F_VMAX}] multicor (>{F_VMAX} satura)")
    print("-" * 66)

    png, pdf = plot_3bus_heatmaps()
    print(f" PNG salvo em : {png}")
    print(f" PDF salvo em : {pdf}")
    print("=" * 66)


if __name__ == "__main__":
    main()
