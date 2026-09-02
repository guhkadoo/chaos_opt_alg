"""
Despacho Ótimo de Potência Reativa (RPD) — IEEE 14 barras.

Minimiza perdas ativas de transmissão usando PSO (scikit-opt), com o fluxo de
carga AC tratado como caixa-preta via pandapower.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as nw
from sko.PSO import PSO

# ---------------------------------------------------------------------------
# Parâmetros do problema
# ---------------------------------------------------------------------------
N_VM = 5  # 1 slack (ext_grid) + 4 geradores (net.gen)
TAP_INDICES = (0, 1, 2)  # transformadores com comutador no case14
# Limites discretos que mantêm a relação de transformação próxima de 0.9–1.1.
TAP_MIN = (-4, -3, -1)
TAP_MAX = (4, 3, 1)
N_SHUNT = 1  # banco de 19 MVAr na barra 8 (índice pandapower)
N_CONTROLE = N_VM + len(TAP_INDICES) + N_SHUNT
VM_LB = 0.95
VM_UB = 1.10

# Barras PV/slack seguem a faixa das próprias variáveis de controle; barras PQ
# têm a faixa operativa mais estreita usual em estudos de RPD do IEEE 14.
VM_GEN_MIN, VM_GEN_MAX = 0.95, 1.10
VM_CARGA_MIN, VM_CARGA_MAX = 0.95, 1.06

# Penalidade contínua no lugar de uma contagem fixa por barra: dá ao PSO uma
# direção de melhora. O termo quadrático pune violações grandes; o linear
# mantém gradiente perto de zero e evita aceitar violações residuais.
PESO_PENALIDADE_LINEAR = 1.0e3
PESO_PENALIDADE_QUADRATICO = 1.0e4
CUSTO_DIVERGENCIA = 1.0e6

# Parâmetros do PSO
POP_SIZE = 40
MAX_ITER = 80
W = 0.8
C1 = 0.5
C2 = 0.5
ARQUIVO_CONVERGENCIA = "convergencia_ieee14_pso.csv"


def carregar_rede() -> pp.pandapowerNet:
    """Carrega a rede IEEE 14 barras nativa do pandapower."""
    net = nw.case14()
    for indice, tap_min, tap_max in zip(TAP_INDICES, TAP_MIN, TAP_MAX):
        net.trafo.at[indice, "tap_min"] = tap_min
        net.trafo.at[indice, "tap_max"] = tap_max
    return net


def aplicar_variaveis_controle(net: pp.pandapowerNet, variaveis: np.ndarray) -> None:
    """
    Injeta tensões, posições dos taps e estágios dos shunts na rede.

    O PSO trabalha em espaço contínuo; taps e shunts são arredondados para
    respeitar a natureza discreta desses equipamentos.
    """
    if len(variaveis) != N_CONTROLE:
        raise ValueError(f"Esperado vetor de dimensão {N_CONTROLE}, recebido {len(variaveis)}.")

    net.ext_grid.at[0, "vm_pu"] = float(variaveis[0])
    for i in range(len(net.gen)):
        net.gen.at[i, "vm_pu"] = float(variaveis[i + 1])

    inicio_taps = N_VM
    for offset, indice in enumerate(TAP_INDICES):
        tap = int(np.rint(variaveis[inicio_taps + offset]))
        net.trafo.at[indice, "tap_pos"] = tap

    inicio_shunts = N_VM + len(TAP_INDICES)
    for i in range(len(net.shunt)):
        step = int(np.rint(variaveis[inicio_shunts + i]))
        net.shunt.at[i, "step"] = step


def obter_variaveis_controle(net: pp.pandapowerNet) -> np.ndarray:
    """Extrai da rede o vetor de controle usado pelo PSO."""
    tensoes = [float(net.ext_grid.at[0, "vm_pu"]), *net.gen["vm_pu"].astype(float)]
    taps = net.trafo.loc[list(TAP_INDICES), "tap_pos"].astype(float).tolist()
    shunts = net.shunt["step"].astype(float).tolist()
    return np.asarray([*tensoes, *taps, *shunts], dtype=float)


def perdas_ativas_mw(net: pp.pandapowerNet) -> float:
    """Soma das perdas ativas em linhas e transformadores [MW]."""
    perdas_linha = float(net.res_line["pl_mw"].sum()) if len(net.res_line) else 0.0
    perdas_trafo = float(net.res_trafo["pl_mw"].sum()) if len(net.res_trafo) else 0.0
    return perdas_linha + perdas_trafo


def limites_tensao_por_barra(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """Vetores de limite inferior/superior de tensão alinhados com net.bus."""
    barras_pv = set(net.gen["bus"].tolist()) | set(net.ext_grid["bus"].tolist())
    e_pv = np.array([b in barras_pv for b in net.bus.index])

    inferior = np.where(e_pv, VM_GEN_MIN, VM_CARGA_MIN)
    superior = np.where(e_pv, VM_GEN_MAX, VM_CARGA_MAX)
    return inferior, superior


def violacoes_tensao(
    net: pp.pandapowerNet,
    inferior: np.ndarray,
    superior: np.ndarray,
) -> np.ndarray:
    """Magnitude da violação de tensão de cada barra [pu]; zero se dentro da faixa."""
    vm = net.res_bus["vm_pu"].to_numpy()
    return np.maximum(0.0, vm - superior) + np.maximum(0.0, inferior - vm)


def penalidade_violacoes(violacao: np.ndarray) -> float:
    """Penalidade linear-quadrática a partir das violações de tensão [pu]."""
    return float(
        PESO_PENALIDADE_LINEAR * np.sum(violacao)
        + PESO_PENALIDADE_QUADRATICO * np.sum(violacao**2)
    )


def criar_funcao_objetivo(net_base: pp.pandapowerNet):
    """
    Fabrica a função objetivo caixa-preta vinculada a uma cópia da rede.

    Retorna (custo, perdas_mw) internamente; a interface pública é só o escalar
    de custo exigido pelo otimizador.
    """
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
    """Reavalia uma solução e devolve métricas detalhadas."""
    net = copy.deepcopy(net_base)
    aplicar_variaveis_controle(net, variaveis)
    pp.runpp(net, numba=False)

    inferior, superior = limites_tensao_por_barra(net)
    violacao = violacoes_tensao(net, inferior, superior)
    penalidade = penalidade_violacoes(violacao)
    perdas = perdas_ativas_mw(net)
    vm = net.res_bus["vm_pu"]

    return {
        "tensoes_pu": np.asarray(variaveis[:N_VM], dtype=float),
        "tap_pos": net.trafo.loc[list(TAP_INDICES), "tap_pos"].astype(int).to_numpy(),
        "shunt_step": net.shunt["step"].astype(int).to_numpy(),
        "shunt_q_mvar": net.res_shunt["q_mvar"].to_numpy(),
        "perdas_mw": perdas,
        "custo": perdas + penalidade,
        "penalidade": penalidade,
        "n_violacoes_tensao": int((violacao > 1e-9).sum()),
        "violacao_max_pu": float(violacao.max()),
        "vm_min": float(vm.min()),
        "vm_max": float(vm.max()),
        "net": net,
    }


def salvar_convergencia_csv(registros: list[dict], caminho: str | Path) -> Path:
    """Exporta perdas e métricas da incumbente em cada iteração do PSO."""
    destino = Path(caminho)
    campos = [
        "iteracao", "avaliacoes", "custo", "perdas_mw", "penalidade",
        "violacoes_tensao", "vm_min", "vm_max",
        "vm_slack", "vm_gen_0", "vm_gen_1", "vm_gen_2", "vm_gen_3",
        "tap_0", "tap_1", "tap_2", "shunt_step",
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
    """Monta uma linha do histórico usando perdas reais, não apenas o fitness."""
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
        "vm_gen_1": f"{x[2]:.10f}",
        "vm_gen_2": f"{x[3]:.10f}",
        "vm_gen_3": f"{x[4]:.10f}",
        "tap_0": int(x[5]),
        "tap_1": int(x[6]),
        "tap_2": int(x[7]),
        "shunt_step": int(x[8]),
    }


def atualizar_gbest_consistente(pso: PSO) -> None:
    """Atualiza gbest_x a partir de pbest_x, evitando inconsistência do sko."""
    indice = int(np.argmin(pso.pbest_y))
    custo = float(np.asarray(pso.pbest_y[indice]).ravel()[0])
    custo_global = float(np.asarray(pso.gbest_y).ravel()[0])
    if custo < custo_global:
        pso.gbest_x = pso.pbest_x[indice].copy()
        pso.gbest_y = np.asarray([custo])


def otimizar_rpd(
    pop: int = POP_SIZE,
    max_iter: int = MAX_ITER,
    seed: int | None = 42,
    arquivo_csv: str | Path | None = ARQUIVO_CONVERGENCIA,
) -> tuple[np.ndarray, float, dict]:
    """
    Executa o PSO para o RPD.

    Returns
    -------
    melhor_x : ndarray
        Variáveis ótimas: tensões, taps e estágios dos shunts.
    melhor_y : float
        Valor mínimo da função objetivo (perdas + penalidades).
    detalhe : dict
        Métricas da reavaliação da melhor solução.
    """
    if seed is not None:
        np.random.seed(seed)

    net_base = carregar_rede()
    assert (
        len(net_base.gen) == 4
        and len(net_base.ext_grid) == 1
        and len(net_base.shunt) == N_SHUNT
    ), (
        "case14 inesperado: esperado 4 gens, 1 ext_grid e 1 shunt."
    )
    # Fluxo de carga e vetor de controle do caso base nativo.
    # Esse vetor será incluído explicitamente na população inicial.
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    print(f"Caso base -> perdas = {detalhe_base['perdas_mw']:.6f} MW | "
          f"penalidade = {detalhe_base['penalidade']:.6f} | "
          f"objetivo = {detalhe_base['custo']:.6f}")

    funcao_objetivo = criar_funcao_objetivo(net_base)

    lb = [VM_LB] * N_VM + list(TAP_MIN) + [0] * N_SHUNT
    ub = [VM_UB] * N_VM + list(TAP_MAX) + [
        int(net_base.shunt.at[i, "max_step"]) for i in net_base.shunt.index
    ]

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

    # O scikit-opt cria uma população aleatória. Substituímos a primeira
    # partícula pelo caso base e reinicializamos os melhores pessoais/global.
    # Assim, o PSO mantém o caso base como incumbente e nunca retorna uma
    # função objetivo pior que a dele.
    pso.X[0] = x_base
    pso.V[0] = 0.0
    pso.cal_y()
    pso.pbest_x = pso.X.copy()
    pso.pbest_y = pso.Y.copy()
    pso.gbest_y = np.inf
    atualizar_gbest_consistente(pso)
    pso.best_x, pso.best_y = pso.gbest_x.copy(), pso.gbest_y

    # Executa explicitamente as iterações para registrar a perda elétrica real
    # da incumbente, pois gbest_y_hist contém apenas a função penalizada.
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
    melhor_x = np.asarray(melhor_x, dtype=float).ravel()
    # Normaliza as variáveis discretas para que o resultado exibido coincida
    # exatamente com o equipamento aplicado no fluxo de carga.
    melhor_x[N_VM:] = np.rint(melhor_x[N_VM:])
    melhor_y = float(np.asarray(melhor_y, dtype=float).ravel()[0])

    detalhe = avaliar_solucao(net_base, melhor_x)

    # Salvaguarda numérica: em caso de comportamento inesperado da biblioteca,
    # devolve o caso base em vez de aceitar uma solução de custo superior.
    if detalhe["custo"] > detalhe_base["custo"] + 1e-9:
        melhor_x = x_base.copy()
        melhor_y = float(detalhe_base["custo"])
        detalhe = avaliar_solucao(net_base, melhor_x)

    detalhe["caso_base"] = detalhe_base
    melhor_y = float(detalhe["custo"])
    detalhe["historico_custo"] = historico_custo
    detalhe["historico_convergencia"] = registros
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(registros, arquivo_csv)
        if arquivo_csv is not None
        else None
    )

    return melhor_x, melhor_y, detalhe


def main() -> None:
    print("=" * 64)
    print(" RPD — IEEE 14 barras | PSO (scikit-opt) + pandapower")
    print("=" * 64)
    print(f" Tensões         : [{VM_LB}, {VM_UB}] pu ({N_VM} controles)")
    print(f" Taps discretos  : {list(zip(TAP_MIN, TAP_MAX))}")
    print(f" Shunt discreto  : 0 a 1 estágio ({N_SHUNT} controle)")
    print(f" Dimensão total  : {N_CONTROLE}")
    print(f" PSO             : pop = {POP_SIZE}, max_iter = {MAX_ITER}")
    print(f" Limite barras PV: [{VM_GEN_MIN}, {VM_GEN_MAX}] pu")
    print(f" Limite barras PQ: [{VM_CARGA_MIN}, {VM_CARGA_MAX}] pu")
    print("-" * 64)

    melhor_x, melhor_y, detalhe = otimizar_rpd()

    rotulos = [
        "ext_grid (slack, barra 0)",
        "gen[0] (barra 1)",
        "gen[1] (barra 2)",
        "gen[2] (barra 5)",
        "gen[3] (barra 7)",
    ]

    print("\nConfiguração ótima de tensões dos geradores [pu]:")
    for rotulo, vm in zip(rotulos, detalhe["tensoes_pu"]):
        print(f"  {rotulo:<28s}  vm_pu = {vm:.6f}")

    print("\nPosições ótimas dos taps:")
    for indice, tap in zip(TAP_INDICES, detalhe["tap_pos"]):
        trafo = detalhe["net"].trafo.loc[indice]
        relacao = 1.0 + (tap - trafo.tap_neutral) * trafo.tap_step_percent / 100.0
        print(f"  trafo[{indice}] (barras {trafo.hv_bus}-{trafo.lv_bus}): "
              f"tap_pos = {tap:+d}, relação = {relacao:.4f}")

    print("\nConfiguração ótima do shunt:")
    for indice, (step, q_mvar) in enumerate(
        zip(detalhe["shunt_step"], detalhe["shunt_q_mvar"])
    ):
        barra = int(detalhe["net"].shunt.at[indice, "bus"])
        print(f"  shunt[{indice}] (barra {barra}): step = {step}, "
              f"Q = {q_mvar:.4f} MVAr")

    base = detalhe["caso_base"]
    reducao = base["perdas_mw"] - detalhe["perdas_mw"]

    print("\nResultados:")
    print(f"  Função objetivo (perdas + penalidades): {melhor_y:.6f}")
    print(f"  Perdas ativas mínimas                 : {detalhe['perdas_mw']:.6f} MW")
    print(f"  Perdas do caso base                   : {base['perdas_mw']:.6f} MW")
    print(f"  Redução de perdas                     : {reducao:+.6f} MW "
          f"({100.0 * reducao / base['perdas_mw']:+.2f} %)")
    print(f"  Penalidade                            : {detalhe['penalidade']:.6f}")
    print(f"  Violações de tensão                   : {detalhe['n_violacoes_tensao']} "
          f"(máx. {detalhe['violacao_max_pu']:.5f} pu)")
    print(f"  vm_pu rede  min / max                 : "
          f"{detalhe['vm_min']:.4f} / {detalhe['vm_max']:.4f} pu")
    if detalhe["arquivo_convergencia_csv"] is not None:
        print(f"  CSV de convergência                   : "
              f"{detalhe['arquivo_convergencia_csv']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
