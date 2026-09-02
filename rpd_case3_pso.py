"""
Despacho Ótimo de Potência Reativa (RPD) — 3 barras sintéticas com PSO.

Usa a rede ``case3_orpd`` (slack + gerador PV + carga, com tap e shunt).
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pandapower as pp
from sko.PSO import PSO

from rede_case3 import case3_orpd

N_VM = 2
TAP_INDICES = (0,)
TAP_MIN = (-3,)
TAP_MAX = (3,)
N_SHUNT = 1
N_CONTROLE = N_VM + len(TAP_INDICES) + N_SHUNT

VM_LB, VM_UB = 0.95, 1.10
VM_GEN_MIN, VM_GEN_MAX = 0.95, 1.10
VM_CARGA_MIN, VM_CARGA_MAX = 0.95, 1.06

PESO_PENALIDADE_LINEAR = 1.0e3
PESO_PENALIDADE_QUADRATICO = 1.0e4
CUSTO_DIVERGENCIA = 1.0e6

POP_SIZE = 40
MAX_ITER = 80
W = 0.8
C1 = 0.5
C2 = 0.5
SEED = 42
ARQUIVO_CONVERGENCIA = "convergencia_case3_pso.csv"


def carregar_rede() -> pp.pandapowerNet:
    """Carrega a rede sintética de 3 barras."""
    net = case3_orpd()
    assert len(net.ext_grid) == 1 and len(net.gen) == 1
    assert len(net.trafo) == 1 and len(net.shunt) == N_SHUNT
    return net


def aplicar_variaveis_controle(net: pp.pandapowerNet, variaveis: np.ndarray) -> None:
    """Injeta tensões, tap e estágio do shunt."""
    x = np.asarray(variaveis, dtype=float).ravel()
    if len(x) != N_CONTROLE:
        raise ValueError(f"Esperado vetor de dimensão {N_CONTROLE}, recebido {len(x)}.")

    net.ext_grid.at[0, "vm_pu"] = float(x[0])
    net.gen.at[0, "vm_pu"] = float(x[1])
    net.trafo.at[0, "tap_pos"] = int(np.rint(x[2]))
    net.shunt.at[0, "step"] = int(np.rint(x[3]))


def obter_variaveis_controle(net: pp.pandapowerNet) -> np.ndarray:
    """Extrai o vetor de controle atual da rede."""
    return np.asarray(
        [
            float(net.ext_grid.at[0, "vm_pu"]),
            float(net.gen.at[0, "vm_pu"]),
            float(net.trafo.at[0, "tap_pos"]),
            float(net.shunt.at[0, "step"]),
        ],
        dtype=float,
    )


def perdas_ativas_mw(net: pp.pandapowerNet) -> float:
    """Soma das perdas ativas em linhas e transformadores [MW]."""
    perdas_linha = float(net.res_line["pl_mw"].sum()) if len(net.res_line) else 0.0
    perdas_trafo = float(net.res_trafo["pl_mw"].sum()) if len(net.res_trafo) else 0.0
    return perdas_linha + perdas_trafo


def limites_tensao_por_barra(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """Limites de tensão por tipo de barra (PV/slack vs PQ)."""
    barras_pv = set(net.gen["bus"].tolist()) | set(net.ext_grid["bus"].tolist())
    e_pv = np.asarray([b in barras_pv for b in net.bus.index])
    inferior = np.where(e_pv, VM_GEN_MIN, VM_CARGA_MIN)
    superior = np.where(e_pv, VM_GEN_MAX, VM_CARGA_MAX)
    return inferior, superior


def violacoes_tensao(
    net: pp.pandapowerNet,
    inferior: np.ndarray,
    superior: np.ndarray,
) -> np.ndarray:
    """Magnitude da violação de tensão por barra [pu]."""
    vm = net.res_bus["vm_pu"].to_numpy()
    return np.maximum(0.0, vm - superior) + np.maximum(0.0, inferior - vm)


def penalidade_violacoes(violacao: np.ndarray) -> float:
    """Penalidade linear-quadrática sobre violações de tensão."""
    return float(
        PESO_PENALIDADE_LINEAR * np.sum(violacao)
        + PESO_PENALIDADE_QUADRATICO * np.sum(violacao**2)
    )


def criar_funcao_objetivo(net_base: pp.pandapowerNet):
    """Função caixa-preta: perdas + penalidades."""
    net = copy.deepcopy(net_base)
    inferior, superior = limites_tensao_por_barra(net)

    def funcao_objetivo(variaveis_controle: np.ndarray) -> float:
        aplicar_variaveis_controle(net, variaveis_controle)
        try:
            pp.runpp(net, numba=False)
        except Exception:
            return CUSTO_DIVERGENCIA
        perdas = perdas_ativas_mw(net)
        return perdas + penalidade_violacoes(violacoes_tensao(net, inferior, superior))

    return funcao_objetivo


def avaliar_solucao(net_base: pp.pandapowerNet, variaveis: np.ndarray) -> dict:
    """Reavalia uma solução e devolve métricas elétricas."""
    net = copy.deepcopy(net_base)
    x = np.asarray(variaveis, dtype=float).ravel().copy()
    x[N_VM:] = np.rint(x[N_VM:])
    aplicar_variaveis_controle(net, x)
    pp.runpp(net, numba=False)

    inferior, superior = limites_tensao_por_barra(net)
    violacao = violacoes_tensao(net, inferior, superior)
    penalidade = penalidade_violacoes(violacao)
    perdas = perdas_ativas_mw(net)
    vm = net.res_bus["vm_pu"]

    return {
        "tensoes_pu": x[:N_VM],
        "tap_pos": np.asarray([int(x[2])]),
        "shunt_step": np.asarray([int(x[3])]),
        "shunt_q_mvar": net.res_shunt["q_mvar"].to_numpy(),
        "perdas_mw": perdas,
        "custo": perdas + penalidade,
        "penalidade": penalidade,
        "n_violacoes_tensao": int((violacao > 1.0e-9).sum()),
        "violacao_max_pu": float(violacao.max()),
        "vm_min": float(vm.min()),
        "vm_max": float(vm.max()),
        "net": net,
    }


def salvar_convergencia_csv(registros: list[dict], caminho: str | Path) -> Path:
    """Exporta perdas da incumbente por iteração."""
    destino = Path(caminho)
    campos = [
        "iteracao", "avaliacoes", "custo", "perdas_mw", "penalidade",
        "violacoes_tensao", "vm_min", "vm_max",
        "vm_slack", "vm_gen_0", "tap_0", "shunt_step",
    ]
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(registros)
    return destino.resolve()


def registro_convergencia(
    iteracao: int,
    avaliacoes: int,
    x: np.ndarray,
    detalhe: dict,
) -> dict:
    """Monta uma linha do CSV com perdas reais."""
    x = np.asarray(x, dtype=float).ravel().copy()
    x[N_VM:] = np.rint(x[N_VM:])
    return {
        "iteracao": iteracao,
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


def atualizar_gbest_consistente(pso: PSO) -> None:
    """Sincroniza gbest_x com o melhor pessoal."""
    indice = int(np.argmin(pso.pbest_y))
    custo = float(np.asarray(pso.pbest_y[indice]).ravel()[0])
    custo_global = float(np.asarray(pso.gbest_y).ravel()[0])
    if custo < custo_global:
        pso.gbest_x = pso.pbest_x[indice].copy()
        pso.gbest_y = np.asarray([custo])


def otimizar_rpd_pso(
    pop: int = POP_SIZE,
    max_iter: int = MAX_ITER,
    seed: int | None = SEED,
    arquivo_csv: str | Path | None = ARQUIVO_CONVERGENCIA,
) -> tuple[np.ndarray, float, dict]:
    """Executa o PSO com o caso base como incumbente."""
    if seed is not None:
        np.random.seed(seed)

    net_base = carregar_rede()
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)

    lb = [VM_LB] * N_VM + list(TAP_MIN) + [0]
    ub = [VM_UB] * N_VM + list(TAP_MAX) + [int(net_base.shunt.at[0, "max_step"])]

    pso = PSO(
        func=funcao_objetivo,
        n_dim=N_CONTROLE,
        pop=pop,
        max_iter=max_iter,
        lb=lb,
        ub=ub,
        w=W,
        c1=C1,
        c2=C2,
    )
    pso.X[0] = x_base
    pso.V[0] = 0.0
    pso.cal_y()
    pso.pbest_x = pso.X.copy()
    pso.pbest_y = pso.Y.copy()
    pso.gbest_y = np.inf
    atualizar_gbest_consistente(pso)
    pso.best_x, pso.best_y = pso.gbest_x.copy(), pso.gbest_y

    registros = [registro_convergencia(0, pop, x_base, detalhe_base)]
    historico_custo = [float(detalhe_base["custo"])]

    for iteracao in range(1, max_iter + 1):
        pso.update_V()
        pso.update_X()
        pso.cal_y()
        pso.update_pbest()
        atualizar_gbest_consistente(pso)

        x_iteracao = np.asarray(pso.gbest_x, dtype=float).ravel()
        detalhe_iteracao = avaliar_solucao(net_base, x_iteracao)
        registros.append(
            registro_convergencia(
                iteracao, pop * (iteracao + 1), x_iteracao, detalhe_iteracao
            )
        )
        historico_custo.append(float(detalhe_iteracao["custo"]))

    melhor_x = np.asarray(pso.gbest_x, dtype=float).ravel()
    melhor_x[N_VM:] = np.rint(melhor_x[N_VM:])
    detalhe = avaliar_solucao(net_base, melhor_x)

    if detalhe["custo"] > detalhe_base["custo"] + 1.0e-9:
        melhor_x = x_base.copy()
        detalhe = avaliar_solucao(net_base, melhor_x)

    melhor_y = float(detalhe["custo"])
    detalhe["caso_base"] = detalhe_base
    detalhe["historico_custo"] = historico_custo
    detalhe["historico_convergencia"] = registros
    detalhe["n_avaliacoes"] = pop * (max_iter + 1)
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(registros, arquivo_csv)
        if arquivo_csv is not None
        else None
    )
    return melhor_x, melhor_y, detalhe


def main() -> None:
    print("=" * 66)
    print(" RPD — 3 barras | PSO (scikit-opt) + pandapower")
    print("=" * 66)
    print(f" Controles       : {N_CONTROLE} (2 tensões, 1 tap, 1 shunt)")
    print(f" PSO             : pop = {POP_SIZE}, max_iter = {MAX_ITER}")
    print(f" Limite barras PV: [{VM_GEN_MIN}, {VM_GEN_MAX}] pu")
    print(f" Limite barras PQ: [{VM_CARGA_MIN}, {VM_CARGA_MAX}] pu")
    print("-" * 66)

    melhor_x, melhor_y, detalhe = otimizar_rpd_pso()
    base = detalhe["caso_base"]
    print(
        f"Caso base -> perdas = {base['perdas_mw']:.6f} MW | "
        f"penalidade = {base['penalidade']:.6f} | objetivo = {base['custo']:.6f}"
    )

    print("\nConfiguração ótima:")
    print(f"  slack (barra 0)  vm_pu = {melhor_x[0]:.6f}")
    print(f"  gen[0] (barra 1) vm_pu = {melhor_x[1]:.6f}")
    print(f"  trafo[0] tap_pos = {int(melhor_x[2]):+d}")
    print(f"  shunt[0] step    = {int(melhor_x[3])}")

    reducao = base["perdas_mw"] - detalhe["perdas_mw"]
    print("\nResultados:")
    print(f"  Função objetivo                      : {melhor_y:.6f}")
    print(f"  Perdas ativas mínimas                : {detalhe['perdas_mw']:.6f} MW")
    print(f"  Perdas do caso base                  : {base['perdas_mw']:.6f} MW")
    print(
        f"  Redução de perdas                    : {reducao:+.6f} MW "
        f"({100.0 * reducao / base['perdas_mw']:+.2f} %)"
    )
    print(f"  Penalidade                           : {detalhe['penalidade']:.6f}")
    print(f"  Violações de tensão                  : {detalhe['n_violacoes_tensao']}")
    if detalhe["arquivo_convergencia_csv"] is not None:
        print(f"  CSV de convergência                  : "
              f"{detalhe['arquivo_convergencia_csv']}")
    print("=" * 66)


if __name__ == "__main__":
    main()
