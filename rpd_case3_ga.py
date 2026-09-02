"""
Despacho Ótimo de Potência Reativa — 3 barras com Algoritmo Genético.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sko.GA import GA

from rpd_case3_pso import (
    N_CONTROLE,
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
ARQUIVO_CONVERGENCIA = "convergencia_case3_ga.csv"


def limites_e_precisao(net):
    lb = np.asarray([VM_LB] * N_VM + list(TAP_MIN) + [0], dtype=float)
    ub = np.asarray(
        [VM_UB] * N_VM + list(TAP_MAX) + [int(net.shunt.at[0, "max_step"])],
        dtype=float,
    )
    precisao = np.asarray([PRECISAO_VM] * N_VM + [1.0] * (len(TAP_INDICES) + 1))
    return lb, ub, precisao


def canonizar_controles(x, lb, ub):
    resultado = np.clip(np.asarray(x, dtype=float).ravel(), lb, ub)
    resultado[N_VM:] = np.rint(resultado[N_VM:])
    return resultado


def registro_convergencia(geracao, avaliacoes, x, detalhe):
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
        "tap_0": int(x[2]),
        "shunt_step": int(x[3]),
    }


def salvar_convergencia_csv(registros, caminho):
    destino = Path(caminho)
    campos = [
        "geracao", "avaliacoes", "custo", "perdas_mw", "penalidade",
        "violacoes_tensao", "vm_min", "vm_max",
        "vm_slack", "vm_gen_0", "tap_0", "shunt_step",
    ]
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(registros)
    return destino.resolve()


def otimizar_rpd_ga(
    pop=POP_SIZE,
    max_geracoes=MAX_GERACOES,
    prob_mutacao=PROB_MUTACAO,
    seed=SEED,
    arquivo_csv=ARQUIVO_CONVERGENCIA,
):
    if pop < 2 or pop % 2:
        raise ValueError("pop deve ser par e >= 2.")
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
            registro_convergencia(geracao, 1 + geracao * pop, melhor_x, detalhe_geracao)
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
    detalhe["n_avaliacoes"] = 1 + pop * max_geracoes
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(registros, arquivo_csv)
        if arquivo_csv is not None else None
    )
    return melhor_x, melhor_custo, detalhe


def main() -> None:
    print("=" * 68)
    print(" RPD — 3 barras | GA (scikit-opt) + pandapower")
    print("=" * 68)
    print(f" Controles : {N_CONTROLE} | pop = {POP_SIZE} | gerações = {MAX_GERACOES}")
    print("-" * 68)

    melhor_x, melhor_custo, detalhe = otimizar_rpd_ga()
    base = detalhe["caso_base"]
    print(f"Caso base -> perdas = {base['perdas_mw']:.6f} MW")
    print("\nConfiguração ótima:")
    print(f"  slack vm_pu = {melhor_x[0]:.6f}")
    print(f"  gen   vm_pu = {melhor_x[1]:.6f}")
    print(f"  tap_pos     = {int(melhor_x[2]):+d}")
    print(f"  shunt step  = {int(melhor_x[3])}")

    reducao = base["perdas_mw"] - detalhe["perdas_mw"]
    print("\nResultados:")
    print(f"  Perdas ativas : {detalhe['perdas_mw']:.6f} MW")
    print(f"  Redução       : {reducao:+.6f} MW "
          f"({100.0 * reducao / base['perdas_mw']:+.2f} %)")
    print(f"  Violações     : {detalhe['n_violacoes_tensao']}")
    if detalhe["arquivo_convergencia_csv"] is not None:
        print(f"  CSV           : {detalhe['arquivo_convergencia_csv']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
