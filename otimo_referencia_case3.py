"""
Ótimo de referência do sistema de 3 barras por enumeração exaustiva.

O espaço discreto do case3 tem apenas 14 combinações (7 taps x 2 shunts).
Para cada uma delas, fixa (tap, shunt) e resolve o subproblema contínuo em
(Vslack, Vpv) por refinamento local a partir do mínimo da malha 41x41 já
tabulada em ``heatmap_case3_superficies.npz`` (mais quatro partidas de
segurança), obtendo o mínimo global do problema com precisão muito superior
à dos quatro algoritmos comparados.

Serve de referência independente para avaliar as soluções de COA/PSO/GA/DE.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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
CACHE_SUPERFICIES = DIR / "heatmap_case3_superficies.npz"

TAPS = list(range(TAP_MIN[0], TAP_MAX[0] + 1))
SHUNTS = [0, 1]

# Solucoes finais dos quatro algoritmos (extraidas dos CSVs de convergencia).
SOLUCOES = {
    "COA": 0.6318060464,
    "PSO": 0.6317223805,
    "GA": 0.6422808190,
    "DE": 0.6319926191,
}


def _partidas(tap: int, shunt: int) -> list[tuple[float, float]]:
    """Pontos iniciais: minimo da malha tabulada + quatro partidas fixas."""
    pontos = [(1.02, 1.02), (1.06, 1.06), (1.10, 1.10), (1.10, 0.98)]
    if CACHE_SUPERFICIES.is_file():
        dados = np.load(CACHE_SUPERFICIES)
        chave = f"z_tap{tap:+d}_shunt{shunt}"
        if chave in dados:
            z = dados[chave]
            mascarado = np.ma.masked_greater_equal(z, CUSTO_DIVERGENCIA * 0.5)
            if mascarado.count():
                i, j = np.unravel_index(np.argmin(mascarado), z.shape)
                pontos.insert(0, (float(dados["vslack"][j]), float(dados["vpv"][i])))
    return pontos


def otimo_da_configuracao(tap: int, shunt: int) -> tuple[int, int, float, float, float]:
    """Retorna (tap, shunt, F*, Vslack*, Vpv*) do subproblema continuo."""
    funcao_objetivo = criar_funcao_objetivo(carregar_rede())

    def f(v: np.ndarray) -> float:
        return float(funcao_objetivo(np.asarray([v[0], v[1], tap, shunt], dtype=float)))

    melhor = (np.inf, np.nan, np.nan)
    for v0, v1 in _partidas(tap, shunt):
        r = minimize(
            f,
            x0=np.array([v0, v1]),
            method="Nelder-Mead",
            bounds=[(VM_LB, VM_UB), (VM_LB, VM_UB)],
            options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 600},
        )
        if r.fun < melhor[0]:
            melhor = (float(r.fun), float(r.x[0]), float(r.x[1]))
    return tap, shunt, melhor[0], melhor[1], melhor[2]


def main() -> None:
    pares = [(tap, shunt) for shunt in SHUNTS for tap in TAPS]
    with ProcessPoolExecutor() as executor:
        saidas = list(executor.map(otimo_da_configuracao, *zip(*pares)))

    resultados = sorted((f, tap, shunt, vs, vp) for tap, shunt, f, vs, vp in saidas)

    print("--- otimo continuo de cada configuracao discreta (ranking) ---")
    for f, tap, shunt, vs, vp in resultados:
        print(
            f"F*={f:.8f}  tap={tap:+d} shunt={shunt}  "
            f"Vslack={vs:.6f} Vpv={vp:.6f}"
        )

    f_ref, tap_ref, shunt_ref, vs_ref, vp_ref = resultados[0]
    print(f"\notimo global de referencia: F* = {f_ref:.8f} MW")
    print(f"  tap={tap_ref:+d}, shunt={shunt_ref}, "
          f"Vslack={vs_ref:.6f}, Vpv={vp_ref:.6f}")

    base = 0.7235973760
    print(f"  reducao sobre o caso base = {(base - f_ref) / base * 100:.2f}%")

    print("\n--- desvio das solucoes dos algoritmos em relacao ao otimo ---")
    for nome, custo in SOLUCOES.items():
        print(f"  {nome:>3}: {custo:.7f}  desvio = {(custo - f_ref) / f_ref * 100:.3f}%")


if __name__ == "__main__":
    main()
