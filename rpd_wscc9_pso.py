"""
Despacho Ótimo de Potência Reativa (RPD) — WSCC 9 barras com PSO.

O ``pandapower.networks.case9()`` possui três fontes controladoras de tensão
(uma barra slack e dois geradores), sem transformadores com tap e sem shunts.
Assim, o vetor de controle contém somente os três setpoints de tensão.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as nw
from sko.PSO import PSO

N_VM = 3
N_CONTROLE = N_VM
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
ARQUIVO_CONVERGENCIA = "convergencia_wscc9_pso.csv"


def carregar_rede() -> pp.pandapowerNet:
    """Carrega a rede WSCC 9 barras nativa do pandapower."""
    net = nw.case9()
    if len(net.ext_grid) != 1 or len(net.gen) != 2:
        raise RuntimeError("case9 inesperado: esperado 1 ext_grid e 2 geradores.")
    if len(net.trafo) or len(net.shunt):
        raise RuntimeError("case9 inesperado: a rede nativa não deve ter trafos ou shunts.")
    return net


def aplicar_variaveis_controle(net: pp.pandapowerNet, variaveis: np.ndarray) -> None:
    """Aplica os três setpoints de tensão à slack e aos dois geradores."""
    x = np.asarray(variaveis, dtype=float).ravel()
    if len(x) != N_CONTROLE:
        raise ValueError(f"Esperado vetor de dimensão {N_CONTROLE}, recebido {len(x)}.")

    net.ext_grid.at[0, "vm_pu"] = float(x[0])
    for i in range(len(net.gen)):
        net.gen.at[i, "vm_pu"] = float(x[i + 1])


def obter_variaveis_controle(net: pp.pandapowerNet) -> np.ndarray:
    """Extrai os setpoints atuais da slack e dos geradores."""
    return np.asarray(
        [float(net.ext_grid.at[0, "vm_pu"]), *net.gen["vm_pu"].astype(float)],
        dtype=float,
    )


def perdas_ativas_mw(net: pp.pandapowerNet) -> float:
    """Retorna as perdas ativas totais em linhas e transformadores [MW]."""
    perdas_linhas = float(net.res_line["pl_mw"].sum()) if len(net.res_line) else 0.0
    perdas_trafos = float(net.res_trafo["pl_mw"].sum()) if len(net.res_trafo) else 0.0
    return perdas_linhas + perdas_trafos


def limites_tensao_por_barra(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """Constrói limites de tensão distintos para barras PV/slack e PQ."""
    barras_pv = set(net.gen["bus"].tolist()) | set(net.ext_grid["bus"].tolist())
    e_pv = np.asarray([barra in barras_pv for barra in net.bus.index])
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
    """Penalidade linear-quadrática para violações de tensão."""
    return float(
        PESO_PENALIDADE_LINEAR * np.sum(violacao)
        + PESO_PENALIDADE_QUADRATICO * np.sum(violacao**2)
    )


def criar_funcao_objetivo(net_base: pp.pandapowerNet):
    """Cria a função caixa-preta de perdas mais penalidades."""
    net = copy.deepcopy(net_base)
    inferior, superior = limites_tensao_por_barra(net)

    def funcao_objetivo(variaveis_controle: np.ndarray) -> float:
        aplicar_variaveis_controle(net, variaveis_controle)
        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except Exception:
            return CUSTO_DIVERGENCIA

        perdas = perdas_ativas_mw(net)
        violacao = violacoes_tensao(net, inferior, superior)
        return perdas + penalidade_violacoes(violacao)

    return funcao_objetivo


def avaliar_solucao(net_base: pp.pandapowerNet, variaveis: np.ndarray) -> dict:
    """Reavalia uma solução e retorna suas métricas elétricas."""
    net = copy.deepcopy(net_base)
    aplicar_variaveis_controle(net, variaveis)
    pp.runpp(net, numba=False)

    inferior, superior = limites_tensao_por_barra(net)
    violacao = violacoes_tensao(net, inferior, superior)
    penalidade = penalidade_violacoes(violacao)
    perdas = perdas_ativas_mw(net)
    vm = net.res_bus["vm_pu"]

    return {
        "tensoes_pu": np.asarray(variaveis, dtype=float).ravel(),
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
    """Exporta a perda ativa da incumbente em cada iteração."""
    destino = Path(caminho)
    campos = [
        "iteracao", "avaliacoes", "custo", "perdas_mw", "penalidade",
        "violacoes_tensao", "vm_min", "vm_max",
        "vm_slack", "vm_gen_0", "vm_gen_1",
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
    """Monta uma linha do CSV com as perdas reais da solução incumbente."""
    x = np.asarray(x, dtype=float).ravel()
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
        "vm_gen_1": f"{x[2]:.10f}",
    }


def atualizar_gbest_consistente(pso: PSO) -> None:
    """Atualiza posição e custo globais a partir do mesmo melhor pessoal."""
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
    """Executa o PSO, preservando o caso base como incumbente inicial."""
    if pop < 1 or max_iter < 1:
        raise ValueError("pop e max_iter devem ser inteiros positivos.")
    if seed is not None:
        np.random.seed(seed)

    net_base = carregar_rede()
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)
    lb = np.full(N_CONTROLE, VM_LB)
    ub = np.full(N_CONTROLE, VM_UB)

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

    # Inclui o caso base e corrige os estados internos inicializados pelo sko.
    pso.X[0] = x_base
    pso.V[0] = 0.0
    pso.cal_y()
    pso.pbest_x = pso.X.copy()
    pso.pbest_y = pso.Y.copy()
    pso.gbest_y = np.inf
    atualizar_gbest_consistente(pso)
    pso.best_x, pso.best_y = pso.gbest_x.copy(), pso.gbest_y

    # O histórico padrão do sko contém fitness penalizado, não perdas em MW.
    # Executamos o laço explicitamente e reavaliamos a incumbente por iteração.
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
                iteracao,
                pop * (iteracao + 1),
                x_iteracao,
                detalhe_iteracao,
            )
        )
        historico_custo.append(float(detalhe_iteracao["custo"]))

    melhor_x = np.asarray(pso.gbest_x, dtype=float).ravel()
    melhor_y = float(np.asarray(pso.gbest_y, dtype=float).ravel()[0])
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
    print(" RPD — WSCC 9 barras | PSO (scikit-opt) + pandapower")
    print("=" * 66)
    print(f" Controles       : {N_CONTROLE} tensões de geração")
    print(" Taps / shunts   : inexistentes no case9 nativo")
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
    print(f"  Função objetivo                      : {melhor_y:.6f}")
    print(f"  Perdas ativas mínimas                : {detalhe['perdas_mw']:.6f} MW")
    print(f"  Perdas do caso base                  : {base['perdas_mw']:.6f} MW")
    print(
        f"  Redução de perdas                    : {reducao:+.6f} MW "
        f"({100.0 * reducao / base['perdas_mw']:+.2f} %)"
    )
    print(f"  Penalidade                           : {detalhe['penalidade']:.6f}")
    print(
        f"  Violações de tensão                  : "
        f"{detalhe['n_violacoes_tensao']} "
        f"(máx. {detalhe['violacao_max_pu']:.5f} pu)"
    )
    print(
        f"  vm_pu rede min / max                 : "
        f"{detalhe['vm_min']:.4f} / {detalhe['vm_max']:.4f} pu"
    )
    if detalhe["arquivo_convergencia_csv"] is not None:
        print(
            f"  CSV de convergência                  : "
            f"{detalhe['arquivo_convergencia_csv']}"
        )
    print("=" * 66)


if __name__ == "__main__":
    main()
