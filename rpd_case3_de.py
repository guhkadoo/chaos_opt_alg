"""
Despacho Ótimo de Potência Reativa — 3 barras com Differential Evolution.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sko.DE import DE

from rpd_case3_pso import (
    N_CONTROLE,
    N_VM,
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
MAX_GERACOES = 40  # DE avalia 2*pop por geração → ~3200 avaliações
F = 0.5
PROB_CRUZAMENTO = 0.3
SEED = 42
ARQUIVO_CONVERGENCIA = "convergencia_case3_de.csv"


def limites_controle(net):
    lb = np.asarray([VM_LB] * N_VM + list(TAP_MIN) + [0], dtype=float)
    ub = np.asarray(
        [VM_UB] * N_VM + list(TAP_MAX) + [int(net.shunt.at[0, "max_step"])],
        dtype=float,
    )
    return lb, ub


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


def selecionar_populacao(de: DE) -> None:
    alvo = de.X.copy()
    de.X = alvo
    custo_alvo = de.x2y().copy()
    teste = de.U.copy()
    de.X = teste
    custo_teste = de.x2y().copy()
    usar_alvo = custo_alvo <= custo_teste
    de.X = np.where(usar_alvo.reshape(-1, 1), alvo, teste)
    de.Y = np.where(usar_alvo, custo_alvo, custo_teste)


def otimizar_rpd_de(
    pop=POP_SIZE,
    max_geracoes=MAX_GERACOES,
    fator_diferencial=F,
    prob_cruzamento=PROB_CRUZAMENTO,
    seed=SEED,
    arquivo_csv=ARQUIVO_CONVERGENCIA,
):
    if pop < 4:
        raise ValueError("pop deve ser >= 4.")
    if seed is not None:
        np.random.seed(seed)

    net_base = carregar_rede()
    lb, ub = limites_controle(net_base)
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)

    de = DE(
        func=funcao_objetivo,
        n_dim=N_CONTROLE,
        F=fator_diferencial,
        size_pop=pop,
        max_iter=max_geracoes,
        prob_mut=prob_cruzamento,
        lb=lb,
        ub=ub,
    )
    de.X[0] = x_base

    melhor_x = x_base.copy()
    melhor_custo = float(detalhe_base["custo"])
    registros = [registro_convergencia(0, 1, melhor_x, detalhe_base)]
    historico_custo = [melhor_custo]

    for geracao in range(1, max_geracoes + 1):
        de.mutation()
        de.crossover()
        selecionar_populacao(de)

        indice = int(np.argmin(de.Y))
        custo_geracao = float(np.asarray(de.Y[indice]).ravel()[0])
        if custo_geracao < melhor_custo:
            melhor_x = canonizar_controles(de.X[indice], lb, ub)
            melhor_custo = custo_geracao

        detalhe_geracao = avaliar_solucao(net_base, melhor_x)
        melhor_custo = float(detalhe_geracao["custo"])
        registros.append(
            registro_convergencia(
                geracao, 1 + 2 * pop * geracao, melhor_x, detalhe_geracao
            )
        )
        historico_custo.append(melhor_custo)

    detalhe = avaliar_solucao(net_base, melhor_x)
    if detalhe["custo"] > detalhe_base["custo"] + 1.0e-9:
        melhor_x = x_base.copy()
        detalhe = avaliar_solucao(net_base, melhor_x)

    melhor_custo = float(detalhe["custo"])
    detalhe["caso_base"] = detalhe_base
    detalhe["historico_custo"] = historico_custo
    detalhe["n_avaliacoes"] = 1 + 2 * pop * max_geracoes
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(registros, arquivo_csv)
        if arquivo_csv is not None else None
    )
    return melhor_x, melhor_custo, detalhe


def main() -> None:
    print("=" * 68)
    print(" RPD — 3 barras | DE (scikit-opt) + pandapower")
    print("=" * 68)
    print(f" Controles : {N_CONTROLE} | pop = {POP_SIZE} | gerações = {MAX_GERACOES}")
    print(f" F / CR    : {F} / {PROB_CRUZAMENTO}")
    print("-" * 68)

    melhor_x, melhor_custo, detalhe = otimizar_rpd_de()
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
