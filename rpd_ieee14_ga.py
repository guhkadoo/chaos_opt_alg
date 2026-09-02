"""
Despacho Ótimo de Potência Reativa — IEEE 14 barras com Algoritmo Genético.

Usa ``sko.GA.GA`` e a mesma caixa-preta pandapower, controles e penalidades
adotados nos scripts PSO/COA. Taps e shunt são codificados com precisão inteira.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sko.GA import GA

from rpd_ieee14_pso import (
    N_CONTROLE,
    N_SHUNT,
    N_VM,
    TAP_INDICES,
    TAP_MAX,
    TAP_MIN,
    VM_CARGA_MAX,
    VM_CARGA_MIN,
    VM_GEN_MAX,
    VM_GEN_MIN,
    VM_LB,
    VM_UB,
    avaliar_solucao,
    carregar_rede,
    criar_funcao_objetivo,
    obter_variaveis_controle,
)

POP_SIZE = 40
MAX_GERACOES = 80
PROB_MUTACAO = 0.01
PRECISAO_VM = 1.0e-4
SEED = 42
ARQUIVO_CONVERGENCIA = "convergencia_ieee14_ga.csv"


def limites_e_precisao(net) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Limites e resolução de tensões, taps e shunt."""
    lb = np.asarray([VM_LB] * N_VM + list(TAP_MIN) + [0] * N_SHUNT, dtype=float)
    ub = np.asarray(
        [VM_UB] * N_VM
        + list(TAP_MAX)
        + [int(net.shunt.at[i, "max_step"]) for i in net.shunt.index],
        dtype=float,
    )
    precisao = np.asarray(
        [PRECISAO_VM] * N_VM + [1.0] * (len(TAP_INDICES) + N_SHUNT),
        dtype=float,
    )
    return lb, ub, precisao


def canonizar_controles(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Aplica limites e torna taps/shunt explicitamente inteiros."""
    resultado = np.clip(np.asarray(x, dtype=float).ravel(), lb, ub)
    resultado[N_VM:] = np.rint(resultado[N_VM:])
    return resultado


def registro_convergencia(
    geracao: int,
    avaliacoes: int,
    x: np.ndarray,
    detalhe: dict,
) -> dict:
    """Monta uma linha com a perda real da incumbente na geração."""
    return {
        "geracao": geracao,
        "avaliacoes": avaliacoes,
        "custo": f"{detalhe['custo']:.10f}",
        "perdas_mw": f"{detalhe['perdas_mw']:.10f}",
        "penalidade": f"{detalhe['penalidade']:.10f}",
        "violacoes_tensao": detalhe["n_violacoes_tensao"],
        "vm_min": f"{detalhe['vm_min']:.10f}",
        "vm_max": f"{detalhe['vm_max']:.10f}",
        "vm_slack": f"{x[0]:.10f}",
        "vm_gen_0": f"{x[1]:.10f}",
        "vm_gen_1": f"{x[2]:.10f}",
        "vm_gen_2": f"{x[3]:.10f}",
        "vm_gen_3": f"{x[4]:.10f}",
        "tap_0": int(x[5]),
        "tap_1": int(x[6]),
        "tap_2": int(x[7]),
        "shunt_step": int(x[8]),
    }


def salvar_convergencia_csv(registros: list[dict], caminho: str | Path) -> Path:
    """Salva a curva geração versus perdas ativas."""
    destino = Path(caminho)
    campos = [
        "geracao", "avaliacoes", "custo", "perdas_mw", "penalidade",
        "violacoes_tensao", "vm_min", "vm_max",
        "vm_slack", "vm_gen_0", "vm_gen_1", "vm_gen_2", "vm_gen_3",
        "tap_0", "tap_1", "tap_2", "shunt_step",
    ]
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(registros)
    return destino.resolve()


def otimizar_rpd_ga(
    pop: int = POP_SIZE,
    max_geracoes: int = MAX_GERACOES,
    prob_mutacao: float = PROB_MUTACAO,
    seed: int | None = SEED,
    arquivo_csv: str | Path | None = ARQUIVO_CONVERGENCIA,
) -> tuple[np.ndarray, float, dict]:
    """Executa o GA e preserva o caso base como incumbente."""
    if pop < 2 or pop % 2:
        raise ValueError("pop deve ser par e maior ou igual a 2.")
    if max_geracoes < 1:
        raise ValueError("max_geracoes deve ser positivo.")
    if seed is not None:
        np.random.seed(seed)

    net_base = carregar_rede()
    lb, ub, precisao = limites_e_precisao(net_base)
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)

    ga = GA(
        func=funcao_objetivo,
        n_dim=N_CONTROLE,
        size_pop=pop,
        max_iter=max_geracoes,
        prob_mut=prob_mutacao,
        lb=lb,
        ub=ub,
        precision=precisao,
    )

    melhor_x = x_base.copy()
    melhor_custo = float(detalhe_base["custo"])
    registros = [registro_convergencia(0, 1, melhor_x, detalhe_base)]
    historico_custo = [melhor_custo]

    # Laço equivalente ao GA.run(), exposto para registrar a incumbente real.
    for geracao in range(1, max_geracoes + 1):
        ga.X = ga.chrom2x(ga.Chrom)
        ga.Y = ga.x2y()
        ga.ranking()

        indice = int(np.argmin(ga.Y))
        custo_geracao = float(np.asarray(ga.Y[indice]).ravel()[0])
        if custo_geracao < melhor_custo:
            melhor_x = canonizar_controles(ga.X[indice], lb, ub)
            melhor_custo = custo_geracao

        detalhe_geracao = avaliar_solucao(net_base, melhor_x)
        melhor_custo = float(detalhe_geracao["custo"])
        registros.append(
            registro_convergencia(
                geracao,
                1 + geracao * pop,
                melhor_x,
                detalhe_geracao,
            )
        )
        historico_custo.append(melhor_custo)

        ga.selection()
        ga.crossover()
        ga.mutation()

    detalhe = avaliar_solucao(net_base, melhor_x)
    if detalhe["custo"] > detalhe_base["custo"] + 1.0e-9:
        melhor_x = x_base.copy()
        detalhe = avaliar_solucao(net_base, melhor_x)

    melhor_custo = float(detalhe["custo"])
    detalhe["caso_base"] = detalhe_base
    detalhe["historico_custo"] = historico_custo
    detalhe["historico_convergencia"] = registros
    detalhe["n_avaliacoes"] = 1 + pop * max_geracoes
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(registros, arquivo_csv)
        if arquivo_csv is not None
        else None
    )
    return melhor_x, melhor_custo, detalhe


def main() -> None:
    print("=" * 68)
    print(" RPD — IEEE 14 barras | GA (scikit-opt) + pandapower")
    print("=" * 68)
    print(f" Controles       : {N_CONTROLE} (5 tensões, 3 taps, 1 shunt)")
    print(f" GA              : pop = {POP_SIZE}, gerações = {MAX_GERACOES}")
    print(f" Mutação         : {PROB_MUTACAO}")
    print(f" Limite barras PV: [{VM_GEN_MIN}, {VM_GEN_MAX}] pu")
    print(f" Limite barras PQ: [{VM_CARGA_MIN}, {VM_CARGA_MAX}] pu")
    print("-" * 68)

    melhor_x, melhor_custo, detalhe = otimizar_rpd_ga()
    base = detalhe["caso_base"]
    print(
        f"Caso base -> perdas = {base['perdas_mw']:.6f} MW | "
        f"penalidade = {base['penalidade']:.6f} | objetivo = {base['custo']:.6f}"
    )

    rotulos = [
        "ext_grid (slack, barra 0)", "gen[0] (barra 1)",
        "gen[1] (barra 2)", "gen[2] (barra 5)", "gen[3] (barra 7)",
    ]
    print("\nConfiguração ótima de tensões [pu]:")
    for rotulo, vm in zip(rotulos, detalhe["tensoes_pu"]):
        print(f"  {rotulo:<28s}  vm_pu = {vm:.6f}")

    print("\nPosições ótimas dos taps:")
    for indice, tap in zip(TAP_INDICES, detalhe["tap_pos"]):
        trafo = detalhe["net"].trafo.loc[indice]
        relacao = 1.0 + (
            (tap - trafo.tap_neutral) * trafo.tap_step_percent / 100.0
        )
        print(
            f"  trafo[{indice}] (barras {trafo.hv_bus}-{trafo.lv_bus}): "
            f"tap_pos = {tap:+d}, relação = {relacao:.4f}"
        )

    print("\nConfiguração ótima do shunt:")
    for indice, (step, q_mvar) in enumerate(
        zip(detalhe["shunt_step"], detalhe["shunt_q_mvar"])
    ):
        barra = int(detalhe["net"].shunt.at[indice, "bus"])
        print(f"  shunt[{indice}] (barra {barra}): step = {step}, Q = {q_mvar:.4f} MVAr")

    reducao = base["perdas_mw"] - detalhe["perdas_mw"]
    print("\nResultados:")
    print(f"  Função objetivo                      : {melhor_custo:.6f}")
    print(f"  Perdas ativas mínimas                : {detalhe['perdas_mw']:.6f} MW")
    print(f"  Perdas do caso base                  : {base['perdas_mw']:.6f} MW")
    print(
        f"  Redução de perdas                    : {reducao:+.6f} MW "
        f"({100.0 * reducao / base['perdas_mw']:+.2f} %)"
    )
    print(f"  Penalidade                           : {detalhe['penalidade']:.6f}")
    print(f"  Violações de tensão                  : {detalhe['n_violacoes_tensao']}")
    print(f"  Avaliações da função objetivo        : {detalhe['n_avaliacoes']}")
    if detalhe["arquivo_convergencia_csv"] is not None:
        print(f"  CSV de convergência                  : "
              f"{detalhe['arquivo_convergencia_csv']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
