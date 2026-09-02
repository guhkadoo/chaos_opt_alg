"""
Despacho Ótimo de Potência Reativa (RPD) — WSCC 9 barras com COA.

Implementa as duas ondas portadoras do Chaos Optimization Algorithm:

    z(k + 1) = 4 z(k) [1 - z(k)]

A primeira onda percorre todo o espaço de controle; a segunda explora regiões
ao redor da melhor solução. O case9 nativo possui apenas três controles de
tensão e não contém transformadores com tap nem shunts.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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

S1 = 2000
S2 = 150
ETA_INICIAL = 0.01
ETA_MAXIMO = 0.10
T3 = 1.35
SEED = 42
ARQUIVO_MINIMOS = "minimos_locais_wscc9_coa.csv"
ARQUIVO_CONVERGENCIA = "convergencia_wscc9_coa.csv"


def mapa_logistico(z: np.ndarray) -> np.ndarray:
    """Avança uma iteração do mapa logístico caótico com a = 4."""
    proximo = 4.0 * z * (1.0 - z)
    eps = np.finfo(float).eps
    return np.clip(proximo, eps, 1.0 - eps)


def estado_caotico_inicial(
    n_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gera estados iniciais próximos, distintos e não periódicos."""
    centro = rng.uniform(0.15, 0.85)
    deslocamentos = rng.normal(0.0, 1.0e-4, size=n_dim)
    return np.clip(centro + deslocamentos, 1.0e-6, 1.0 - 1.0e-6)


def registrar_minimo(
    registros: list[dict],
    fase: str,
    avaliacao: int,
    x: np.ndarray,
    custo: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> None:
    """Registra uma solução que melhora estritamente a incumbente."""
    registros.append(
        {
            "fase": fase,
            "avaliacao": avaliacao,
            "custo": float(custo),
            "x": np.clip(np.asarray(x, dtype=float), lb, ub).copy(),
        }
    )


def busca_primeira_onda(
    funcao_objetivo,
    z: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    melhor_x: np.ndarray,
    melhor_custo: float,
    historico: list[float],
    registros: list[dict],
    n_buscas: int,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Executa a busca caótica global no espaço completo."""
    amplitude = ub - lb
    avaliacoes = 0

    for _ in range(n_buscas):
        candidato = lb + z * amplitude
        custo = float(funcao_objetivo(candidato))
        avaliacoes += 1

        if custo < melhor_custo:
            melhor_x = candidato.copy()
            melhor_custo = custo
            registrar_minimo(
                registros,
                "primeira_onda",
                len(historico) + 1,
                melhor_x,
                melhor_custo,
                lb,
                ub,
            )

        historico.append(melhor_custo)
        z = mapa_logistico(z)

    return melhor_x, melhor_custo, z, avaliacoes


def busca_segunda_onda(
    funcao_objetivo,
    z: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    melhor_x: np.ndarray,
    melhor_custo: float,
    historico: list[float],
    registros: list[dict],
    n_buscas: int,
    eta_inicial: float,
    eta_maximo: float,
    t3: float,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Explora regiões crescentes ao redor da incumbente."""
    amplitude_total = ub - lb
    eta_fracao = eta_inicial
    avaliacoes = 0

    while eta_fracao <= eta_maximo + 1.0e-12:
        eta = eta_fracao * amplitude_total

        for _ in range(n_buscas):
            z = mapa_logistico(z)
            candidato = np.clip(melhor_x + eta * (z - 0.5), lb, ub)
            custo = float(funcao_objetivo(candidato))
            avaliacoes += 1

            if custo < melhor_custo:
                melhor_x = candidato.copy()
                melhor_custo = custo
                registrar_minimo(
                    registros,
                    f"segunda_onda_eta_{eta_fracao:.6f}",
                    len(historico) + 1,
                    melhor_x,
                    melhor_custo,
                    lb,
                    ub,
                )

            historico.append(melhor_custo)

        eta_fracao *= t3

    return melhor_x, melhor_custo, z, avaliacoes


def salvar_minimos_csv(
    registros: list[dict],
    net_base,
    caminho: str | Path = ARQUIVO_MINIMOS,
) -> Path:
    """Exporta as incumbentes e suas métricas elétricas para CSV."""
    destino = Path(caminho)
    campos = [
        "ordem",
        "fase",
        "avaliacao",
        "custo",
        "perdas_mw",
        "penalidade",
        "violacoes_tensao",
        "violacao_max_pu",
        "vm_min",
        "vm_max",
        "vm_slack",
        "vm_gen_0",
        "vm_gen_1",
    ]

    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()

        for ordem, registro in enumerate(registros):
            x = registro["x"]
            detalhe = avaliar_solucao(net_base, x)
            escritor.writerow(
                {
                    "ordem": ordem,
                    "fase": registro["fase"],
                    "avaliacao": registro["avaliacao"],
                    "custo": f"{detalhe['custo']:.10f}",
                    "perdas_mw": f"{detalhe['perdas_mw']:.10f}",
                    "penalidade": f"{detalhe['penalidade']:.10f}",
                    "violacoes_tensao": detalhe["n_violacoes_tensao"],
                    "violacao_max_pu": f"{detalhe['violacao_max_pu']:.10f}",
                    "vm_min": f"{detalhe['vm_min']:.10f}",
                    "vm_max": f"{detalhe['vm_max']:.10f}",
                    "vm_slack": f"{x[0]:.10f}",
                    "vm_gen_0": f"{x[1]:.10f}",
                    "vm_gen_1": f"{x[2]:.10f}",
                }
            )

    return destino.resolve()


def salvar_convergencia_csv(
    registros: list[dict],
    net_base,
    total_avaliacoes: int,
    caminho: str | Path = ARQUIVO_CONVERGENCIA,
) -> Path:
    """Gera uma série densa da perda incumbente para cada avaliação."""
    destino = Path(caminho)
    pontos = []
    for registro in registros:
        detalhe = avaliar_solucao(net_base, registro["x"])
        pontos.append((int(registro["avaliacao"]), registro["fase"], detalhe))

    indice = 0
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        campos = ["avaliacao", "fase", "custo", "perdas_mw", "penalidade"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for avaliacao in range(1, total_avaliacoes + 1):
            while indice + 1 < len(pontos) and pontos[indice + 1][0] <= avaliacao:
                indice += 1
            _, fase, detalhe = pontos[indice]
            escritor.writerow(
                {
                    "avaliacao": avaliacao,
                    "fase": fase,
                    "custo": f"{detalhe['custo']:.10f}",
                    "perdas_mw": f"{detalhe['perdas_mw']:.10f}",
                    "penalidade": f"{detalhe['penalidade']:.10f}",
                }
            )
    return destino.resolve()


def otimizar_rpd_coa(
    s1: int = S1,
    s2: int = S2,
    seed: int = SEED,
    arquivo_csv: str | Path | None = ARQUIVO_MINIMOS,
    arquivo_convergencia: str | Path | None = ARQUIVO_CONVERGENCIA,
) -> tuple[np.ndarray, float, dict]:
    """Executa o COA para o RPD do WSCC 9 barras."""
    if s1 < 1 or s2 < 1:
        raise ValueError("s1 e s2 devem ser inteiros positivos.")

    net_base = carregar_rede()
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)
    lb = np.full(N_CONTROLE, VM_LB)
    ub = np.full(N_CONTROLE, VM_UB)

    # O caso base é a incumbente inicial, garantindo custo final não superior.
    melhor_x = x_base.copy()
    melhor_custo = float(detalhe_base["custo"])
    historico = [melhor_custo]
    avaliacoes = 1
    registros: list[dict] = []
    registrar_minimo(
        registros,
        "caso_base",
        avaliacoes,
        x_base,
        melhor_custo,
        lb,
        ub,
    )

    rng = np.random.default_rng(seed)
    z = estado_caotico_inicial(N_CONTROLE, rng)

    melhor_x, melhor_custo, z, n = busca_primeira_onda(
        funcao_objetivo,
        z,
        lb,
        ub,
        melhor_x,
        melhor_custo,
        historico,
        registros,
        s1,
    )
    avaliacoes += n

    melhor_x, melhor_custo, z, n = busca_segunda_onda(
        funcao_objetivo,
        z,
        lb,
        ub,
        melhor_x,
        melhor_custo,
        historico,
        registros,
        s2,
        ETA_INICIAL,
        ETA_MAXIMO,
        T3,
    )
    avaliacoes += n

    melhor_x = np.clip(melhor_x, lb, ub)
    detalhe = avaliar_solucao(net_base, melhor_x)
    if detalhe["custo"] > detalhe_base["custo"] + 1.0e-9:
        melhor_x = x_base.copy()
        detalhe = avaliar_solucao(net_base, melhor_x)

    melhor_custo = float(detalhe["custo"])
    detalhe["caso_base"] = detalhe_base
    detalhe["historico_custo"] = historico
    detalhe["n_avaliacoes"] = avaliacoes
    detalhe["estado_caotico_final"] = z
    detalhe["minimos_locais"] = registros
    detalhe["arquivo_minimos_csv"] = (
        salvar_minimos_csv(registros, net_base, arquivo_csv)
        if arquivo_csv is not None
        else None
    )
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(
            registros, net_base, avaliacoes, arquivo_convergencia
        )
        if arquivo_convergencia is not None
        else None
    )
    return melhor_x, melhor_custo, detalhe


def main() -> None:
    print("=" * 68)
    print(" RPD — WSCC 9 barras | COA (mapa logístico) + pandapower")
    print("=" * 68)
    print(f" Controles       : {N_CONTROLE} tensões de geração")
    print(" Taps / shunts   : inexistentes no case9 nativo")
    print(f" Primeira onda   : S1 = {S1}")
    print(f" Segunda onda    : S2 = {S2} por amplitude")
    print(f" Amplitude local : {ETA_INICIAL:.2%} até {ETA_MAXIMO:.2%}, t3 = {T3}")
    print(f" Limite barras PV: [{VM_GEN_MIN}, {VM_GEN_MAX}] pu")
    print(f" Limite barras PQ: [{VM_CARGA_MIN}, {VM_CARGA_MAX}] pu")
    print("-" * 68)

    melhor_x, melhor_custo, detalhe = otimizar_rpd_coa()
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
    print(
        f"  Violações de tensão                  : "
        f"{detalhe['n_violacoes_tensao']} "
        f"(máx. {detalhe['violacao_max_pu']:.5f} pu)"
    )
    print(
        f"  vm_pu rede min / max                 : "
        f"{detalhe['vm_min']:.4f} / {detalhe['vm_max']:.4f} pu"
    )
    print(f"  Avaliações da função objetivo        : {detalhe['n_avaliacoes']}")
    print(f"  Mínimos/incumbentes registrados      : {len(detalhe['minimos_locais'])}")
    if detalhe["arquivo_minimos_csv"] is not None:
        print(f"  CSV dos mínimos                      : {detalhe['arquivo_minimos_csv']}")
    if detalhe["arquivo_convergencia_csv"] is not None:
        print(
            f"  CSV de convergência                  : "
            f"{detalhe['arquivo_convergencia_csv']}"
        )
    print("=" * 68)


if __name__ == "__main__":
    main()
