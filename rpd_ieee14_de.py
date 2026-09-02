"""
Despacho Ótimo de Potência Reativa — IEEE 14 barras com Differential Evolution.

Usa ``sko.DE.DE`` com a mesma caixa-preta pandapower, controles e penalidades
dos demais algoritmos. Taps e shunt são arredondados antes do fluxo de carga.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sko.DE import DE

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
# O DE do scikit-opt faz 2*pop avaliações/geração; 40 gerações ≈ 3200 avaliações.
MAX_GERACOES = 40
F = 0.5
PROB_CRUZAMENTO = 0.3
SEED = 42
ARQUIVO_CONVERGENCIA = "convergencia_ieee14_de.csv"


def limites_controle(net) -> tuple[np.ndarray, np.ndarray]:
    """Limites das cinco tensões, três taps e um shunt."""
    lb = np.asarray([VM_LB] * N_VM + list(TAP_MIN) + [0] * N_SHUNT, dtype=float)
    ub = np.asarray(
        [VM_UB] * N_VM
        + list(TAP_MAX)
        + [int(net.shunt.at[i, "max_step"]) for i in net.shunt.index],
        dtype=float,
    )
    return lb, ub


def canonizar_controles(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Aplica limites e converte taps/shunt às posições discretas."""
    resultado = np.clip(np.asarray(x, dtype=float).ravel(), lb, ub)
    resultado[N_VM:] = np.rint(resultado[N_VM:])
    return resultado


def registro_convergencia(
    geracao: int,
    avaliacoes: int,
    x: np.ndarray,
    detalhe: dict,
) -> dict:
    """Monta uma linha com as perdas reais da incumbente."""
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
    """Salva a curva geração/avaliações versus perdas."""
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


def selecionar_populacao(de: DE) -> None:
    """
    Seleção gulosa do DE mantendo X e Y consistentes.

    A implementação da biblioteca deixa Y associado à população teste depois
    de selecionar X. Aqui ambos são selecionados pela mesma máscara.
    """
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
    pop: int = POP_SIZE,
    max_geracoes: int = MAX_GERACOES,
    fator_diferencial: float = F,
    prob_cruzamento: float = PROB_CRUZAMENTO,
    seed: int | None = SEED,
    arquivo_csv: str | Path | None = ARQUIVO_CONVERGENCIA,
) -> tuple[np.ndarray, float, dict]:
    """Executa o DE, incluindo o caso base na população e como incumbente."""
    if pop < 4:
        raise ValueError("pop deve ser maior ou igual a 4.")
    if max_geracoes < 1:
        raise ValueError("max_geracoes deve ser positivo.")
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
                geracao,
                1 + 2 * pop * geracao,
                melhor_x,
                detalhe_geracao,
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
    detalhe["historico_convergencia"] = registros
    detalhe["n_avaliacoes"] = 1 + 2 * pop * max_geracoes
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(registros, arquivo_csv)
        if arquivo_csv is not None
        else None
    )
    return melhor_x, melhor_custo, detalhe


def main() -> None:
    print("=" * 68)
    print(" RPD — IEEE 14 barras | DE (scikit-opt) + pandapower")
    print("=" * 68)
    print(f" Controles       : {N_CONTROLE} (5 tensões, 3 taps, 1 shunt)")
    print(f" DE              : pop = {POP_SIZE}, gerações = {MAX_GERACOES}")
    print(f" F / cruzamento  : {F} / {PROB_CRUZAMENTO}")
    print(f" Limite barras PV: [{VM_GEN_MIN}, {VM_GEN_MAX}] pu")
    print(f" Limite barras PQ: [{VM_CARGA_MIN}, {VM_CARGA_MAX}] pu")
    print("-" * 68)

    melhor_x, melhor_custo, detalhe = otimizar_rpd_de()
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
