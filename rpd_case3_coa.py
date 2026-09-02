"""
Despacho Ótimo de Potência Reativa — 3 barras com COA (mapa logístico).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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

S1 = 2000
S2 = 150
ETA_INICIAL = 0.01
ETA_MAXIMO = 0.10
T3 = 1.35
SEED = 42
ARQUIVO_MINIMOS = "minimos_locais_case3_coa.csv"
ARQUIVO_CONVERGENCIA = "convergencia_case3_coa.csv"


def mapa_logistico(z: np.ndarray) -> np.ndarray:
    proximo = 4.0 * z * (1.0 - z)
    eps = np.finfo(float).eps
    return np.clip(proximo, eps, 1.0 - eps)


def limites_controle(net) -> tuple[np.ndarray, np.ndarray]:
    lb = np.asarray([VM_LB] * N_VM + list(TAP_MIN) + [0], dtype=float)
    ub = np.asarray(
        [VM_UB] * N_VM + list(TAP_MAX) + [int(net.shunt.at[0, "max_step"])],
        dtype=float,
    )
    return lb, ub


def canonizar_discretas(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    resultado = np.clip(np.asarray(x, dtype=float), lb, ub)
    resultado[N_VM:] = np.rint(resultado[N_VM:])
    return resultado


def estado_caotico_inicial(n_dim: int, rng: np.random.Generator) -> np.ndarray:
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
    registros.append(
        {
            "fase": fase,
            "avaliacao": avaliacao,
            "custo": float(custo),
            "x": canonizar_discretas(x, lb, ub).copy(),
        }
    )


def busca_primeira_onda(
    funcao_objetivo, z, lb, ub, melhor_x, melhor_custo, historico, registros, n_buscas
):
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
                registros, "primeira_onda", len(historico) + 1,
                melhor_x, melhor_custo, lb, ub,
            )
        historico.append(melhor_custo)
        z = mapa_logistico(z)
    return melhor_x, melhor_custo, z, avaliacoes


def busca_segunda_onda(
    funcao_objetivo, z, lb, ub, melhor_x, melhor_custo, historico, registros,
    n_buscas, eta_inicial, eta_maximo, t3,
):
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
                    registros, f"segunda_onda_eta_{eta_fracao:.6f}",
                    len(historico) + 1, melhor_x, melhor_custo, lb, ub,
                )
            historico.append(melhor_custo)
        eta_fracao *= t3
    return melhor_x, melhor_custo, z, avaliacoes


def polimento_discreto(
    funcao_objetivo, x, custo, lb, ub, historico, registros
):
    melhor_x = canonizar_discretas(x, lb, ub)
    melhor_custo = float(funcao_objetivo(melhor_x))
    avaliacoes = 1
    historico.append(melhor_custo)

    houve_melhora = True
    while houve_melhora:
        houve_melhora = False
        for j in range(N_VM, N_VM + len(TAP_INDICES)):
            for delta in (-1.0, 1.0):
                candidato = melhor_x.copy()
                candidato[j] = np.clip(candidato[j] + delta, lb[j], ub[j])
                if candidato[j] == melhor_x[j]:
                    continue
                custo_candidato = float(funcao_objetivo(candidato))
                avaliacoes += 1
                if custo_candidato < melhor_custo:
                    melhor_x = candidato
                    melhor_custo = custo_candidato
                    houve_melhora = True
                    registrar_minimo(
                        registros, "polimento_tap", len(historico) + 1,
                        melhor_x, melhor_custo, lb, ub,
                    )
                historico.append(melhor_custo)

        for j in range(N_VM + len(TAP_INDICES), N_CONTROLE):
            candidato = melhor_x.copy()
            candidato[j] = ub[j] if melhor_x[j] == lb[j] else lb[j]
            custo_candidato = float(funcao_objetivo(candidato))
            avaliacoes += 1
            if custo_candidato < melhor_custo:
                melhor_x = candidato
                melhor_custo = custo_candidato
                houve_melhora = True
                registrar_minimo(
                    registros, "polimento_shunt", len(historico) + 1,
                    melhor_x, melhor_custo, lb, ub,
                )
            historico.append(melhor_custo)

    return melhor_x, melhor_custo, avaliacoes


def salvar_minimos_csv(registros, net_base, caminho=ARQUIVO_MINIMOS) -> Path:
    destino = Path(caminho)
    campos = [
        "ordem", "fase", "avaliacao", "custo", "perdas_mw", "penalidade",
        "violacoes_tensao", "vm_min", "vm_max",
        "vm_slack", "vm_gen_0", "tap_0", "shunt_step",
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
                    "vm_min": f"{detalhe['vm_min']:.10f}",
                    "vm_max": f"{detalhe['vm_max']:.10f}",
                    "vm_slack": f"{x[0]:.10f}",
                    "vm_gen_0": f"{x[1]:.10f}",
                    "tap_0": int(x[2]),
                    "shunt_step": int(x[3]),
                }
            )
    return destino.resolve()


def salvar_convergencia_csv(
    registros, net_base, total_avaliacoes, caminho=ARQUIVO_CONVERGENCIA
) -> Path:
    destino = Path(caminho)
    pontos = [
        (int(r["avaliacao"]), r["fase"], avaliar_solucao(net_base, r["x"]))
        for r in registros
    ]
    indice = 0
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=["avaliacao", "fase", "custo", "perdas_mw", "penalidade"],
        )
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
    net_base = carregar_rede()
    lb, ub = limites_controle(net_base)
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)

    melhor_x = x_base.copy()
    melhor_custo = float(detalhe_base["custo"])
    historico = [melhor_custo]
    avaliacoes = 1
    registros: list[dict] = []
    registrar_minimo(registros, "caso_base", avaliacoes, x_base, melhor_custo, lb, ub)

    rng = np.random.default_rng(seed)
    z = estado_caotico_inicial(N_CONTROLE, rng)

    melhor_x, melhor_custo, z, n = busca_primeira_onda(
        funcao_objetivo, z, lb, ub, melhor_x, melhor_custo, historico, registros, s1
    )
    avaliacoes += n

    melhor_x, melhor_custo, z, n = busca_segunda_onda(
        funcao_objetivo, z, lb, ub, melhor_x, melhor_custo, historico, registros,
        s2, ETA_INICIAL, ETA_MAXIMO, T3,
    )
    avaliacoes += n

    melhor_x, melhor_custo, n = polimento_discreto(
        funcao_objetivo, melhor_x, melhor_custo, lb, ub, historico, registros
    )
    avaliacoes += n

    detalhe = avaliar_solucao(net_base, melhor_x)
    if detalhe["custo"] > detalhe_base["custo"] + 1.0e-9:
        melhor_x = x_base.copy()
        detalhe = avaliar_solucao(net_base, melhor_x)

    melhor_custo = float(detalhe["custo"])
    detalhe["caso_base"] = detalhe_base
    detalhe["historico_custo"] = historico
    detalhe["n_avaliacoes"] = avaliacoes
    detalhe["minimos_locais"] = registros
    detalhe["arquivo_minimos_csv"] = (
        salvar_minimos_csv(registros, net_base, arquivo_csv)
        if arquivo_csv is not None else None
    )
    detalhe["arquivo_convergencia_csv"] = (
        salvar_convergencia_csv(registros, net_base, avaliacoes, arquivo_convergencia)
        if arquivo_convergencia is not None else None
    )
    return melhor_x, melhor_custo, detalhe


def main() -> None:
    print("=" * 68)
    print(" RPD — 3 barras | COA (mapa logístico) + pandapower")
    print("=" * 68)
    print(f" Controles       : {N_CONTROLE} (2 tensões, 1 tap, 1 shunt)")
    print(f" Primeira onda   : S1 = {S1}")
    print(f" Segunda onda    : S2 = {S2} por amplitude")
    print("-" * 68)

    melhor_x, melhor_custo, detalhe = otimizar_rpd_coa()
    base = detalhe["caso_base"]
    print(
        f"Caso base -> perdas = {base['perdas_mw']:.6f} MW | "
        f"objetivo = {base['custo']:.6f}"
    )
    print("\nConfiguração ótima:")
    print(f"  slack vm_pu = {melhor_x[0]:.6f}")
    print(f"  gen   vm_pu = {melhor_x[1]:.6f}")
    print(f"  tap_pos     = {int(melhor_x[2]):+d}")
    print(f"  shunt step  = {int(melhor_x[3])}")

    reducao = base["perdas_mw"] - detalhe["perdas_mw"]
    print("\nResultados:")
    print(f"  Perdas ativas mínimas : {detalhe['perdas_mw']:.6f} MW")
    print(f"  Redução               : {reducao:+.6f} MW "
          f"({100.0 * reducao / base['perdas_mw']:+.2f} %)")
    print(f"  Violações             : {detalhe['n_violacoes_tensao']}")
    print(f"  Avaliações            : {detalhe['n_avaliacoes']}")
    if detalhe["arquivo_convergencia_csv"] is not None:
        print(f"  CSV                   : {detalhe['arquivo_convergencia_csv']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
