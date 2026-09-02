"""
Despacho Ótimo de Potência Reativa — WSCC 9 barras com Algoritmo Genético.

Usa ``sko.GA.GA`` com a mesma caixa-preta pandapower e as mesmas penalidades
dos scripts PSO/COA. O case9 nativo possui somente três controles de tensão.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sko.GA import GA

from rpd_wscc9_pso import (
    N_CONTROLE,
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
ARQUIVO_CONVERGENCIA = "convergencia_wscc9_ga.csv"


def registro_convergencia(
    geracao: int,
    avaliacoes: int,
    x: np.ndarray,
    detalhe: dict,
) -> dict:
    """Monta uma linha com a perda real da incumbente."""
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
    }


def salvar_convergencia_csv(registros: list[dict], caminho: str | Path) -> Path:
    """Salva a curva geração versus perdas ativas."""
    destino = Path(caminho)
    campos = [
        "geracao", "avaliacoes", "custo", "perdas_mw", "penalidade",
        "violacoes_tensao", "vm_min", "vm_max",
        "vm_slack", "vm_gen_0", "vm_gen_1",
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
    """Executa o GA e mantém o caso base como incumbente inicial."""
    if pop < 2 or pop % 2:
        raise ValueError("pop deve ser par e maior ou igual a 2.")
    if max_geracoes < 1:
        raise ValueError("max_geracoes deve ser positivo.")
    if seed is not None:
        np.random.seed(seed)

    net_base = carregar_rede()
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)
    lb = np.full(N_CONTROLE, VM_LB)
    ub = np.full(N_CONTROLE, VM_UB)
    precisao = np.full(N_CONTROLE, PRECISAO_VM)

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
            melhor_x = np.clip(ga.X[indice].copy(), lb, ub)
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
    print(" RPD — WSCC 9 barras | GA (scikit-opt) + pandapower")
    print("=" * 68)
    print(f" Controles       : {N_CONTROLE} tensões de geração")
    print(" Taps / shunts   : inexistentes no case9 nativo")
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

    print("\nConfiguração ótima de tensões [pu]:")
    rotulos = [
        "ext_grid (slack, barra 0)",
        "gen[0] (barra 1)",
        "gen[1] (barra 2)",
    ]
    for rotulo, vm in zip(rotulos, melhor_x):
        print(f"  {rotulo:<28s}  vm_pu = {vm:.6f}")

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
