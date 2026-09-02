"""
Despacho Ótimo de Potência Reativa (RPD) — IEEE 14 barras com COA.

Implementa as duas ondas portadoras do Chaos Optimization Algorithm descrito
por Tavazoei e Haeri (2007). O mapa logístico gera a sequência caótica:

    z(k + 1) = 4 z(k) [1 - z(k)]

A primeira onda realiza a busca global em todo o espaço; a segunda explora
regiões crescentes ao redor da melhor solução. A física e as penalidades são
as mesmas de ``rpd_ieee14_pso.py``, permitindo comparação direta.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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

# Parâmetros do COA. O total típico é próximo das 3200 avaliações usadas pelo
# PSO (40 partículas x 80 iterações), tornando a comparação mais justa.
S1 = 2000
S2 = 150
ETA_INICIAL = 0.01
ETA_MAXIMO = 0.10
T3 = 1.35
SEED = 42
ARQUIVO_MINIMOS = "minimos_locais_coa.csv"
ARQUIVO_CONVERGENCIA = "convergencia_ieee14_coa.csv"


def mapa_logistico(z: np.ndarray) -> np.ndarray:
    """Avança uma iteração do mapa logístico caótico com a = 4."""
    proximo = 4.0 * z * (1.0 - z)
    # Evita estados absorventes decorrentes de arredondamento em 0 ou 1.
    eps = np.finfo(float).eps
    return np.clip(proximo, eps, 1.0 - eps)


def limites_controle(net) -> tuple[np.ndarray, np.ndarray]:
    """Limites inferior e superior das nove variáveis de controle."""
    lb = np.asarray(
        [VM_LB] * N_VM + list(TAP_MIN) + [0] * N_SHUNT,
        dtype=float,
    )
    ub = np.asarray(
        [VM_UB] * N_VM
        + list(TAP_MAX)
        + [int(net.shunt.at[i, "max_step"]) for i in net.shunt.index],
        dtype=float,
    )
    return lb, ub


def canonizar_discretas(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Limita o vetor e converte taps e shunts às posições discretas aplicadas."""
    resultado = np.clip(np.asarray(x, dtype=float), lb, ub)
    resultado[N_VM:] = np.rint(resultado[N_VM:])
    return resultado


def estado_caotico_inicial(n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Cria estados próximos, mas distintos, conforme o passo 1 do artigo.

    Pontos exatamente iguais ou periódicos simples do mapa logístico devem ser
    evitados, pois não apresentam a ergodicidade desejada.
    """
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
    """
    Registra uma nova incumbente do COA.

    Neste script, “mínimo local encontrado” designa cada solução que melhora
    estritamente a incumbente ao longo das ondas caóticas ou do polimento.
    """
    registros.append(
        {
            "fase": fase,
            "avaliacao": avaliacao,
            "custo": float(custo),
            "x": canonizar_discretas(x, lb, ub).copy(),
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
    """Busca global: mapeia a órbita caótica em todo o espaço de controle."""
    avaliacoes = 0
    amplitude = ub - lb

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
    """
    Busca local pela segunda onda do artigo.

    A amplitude começa em 1% da faixa e cresce por t3 até 10%. Cada candidato
    obedece x = x* + eta * (z - 0.5).
    """
    avaliacoes = 0
    amplitude_total = ub - lb
    eta_fracao = eta_inicial

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


def polimento_discreto(
    funcao_objetivo,
    x: np.ndarray,
    custo: float,
    lb: np.ndarray,
    ub: np.ndarray,
    historico: list[float],
    registros: list[dict],
) -> tuple[np.ndarray, float, int]:
    """
    Testa vizinhos dos taps e os dois estados do shunt.

    A segunda onda pode não cruzar um limiar de arredondamento. Este pequeno
    refinamento garante que os equipamentos discretos vizinhos sejam avaliados.
    """
    melhor_x = canonizar_discretas(x, lb, ub)
    melhor_custo = float(funcao_objetivo(melhor_x))
    avaliacoes = 1
    historico.append(melhor_custo)

    houve_melhora = True
    while houve_melhora:
        houve_melhora = False

        # Vizinhança de uma posição para cada tap.
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
                        registros,
                        "polimento_tap",
                        len(historico) + 1,
                        melhor_x,
                        melhor_custo,
                        lb,
                        ub,
                    )
                historico.append(melhor_custo)

        # O shunt do case14 é binário; avalia explicitamente o outro estado.
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
                    registros,
                    "polimento_shunt",
                    len(historico) + 1,
                    melhor_x,
                    melhor_custo,
                    lb,
                    ub,
                )
            historico.append(melhor_custo)

    return melhor_x, melhor_custo, avaliacoes


def salvar_minimos_csv(
    registros: list[dict],
    net_base,
    caminho: str | Path = ARQUIVO_MINIMOS,
) -> Path:
    """Salva todas as incumbentes encontradas e suas métricas elétricas."""
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
        "vm_gen_2",
        "vm_gen_3",
        "tap_0",
        "tap_1",
        "tap_2",
        "shunt_step",
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
                    "vm_gen_2": f"{x[3]:.10f}",
                    "vm_gen_3": f"{x[4]:.10f}",
                    "tap_0": int(x[5]),
                    "tap_1": int(x[6]),
                    "tap_2": int(x[7]),
                    "shunt_step": int(x[8]),
                }
            )

    return destino.resolve()


def salvar_convergencia_csv(
    registros: list[dict],
    net_base,
    total_avaliacoes: int,
    caminho: str | Path = ARQUIVO_CONVERGENCIA,
) -> Path:
    """Expande as melhorias em uma série por avaliação para plot de convergência."""
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
    """Executa o COA e retorna variáveis ótimas, custo e métricas elétricas."""
    if s1 < 1 or s2 < 1:
        raise ValueError("s1 e s2 devem ser inteiros positivos.")

    net_base = carregar_rede()
    lb, ub = limites_controle(net_base)
    x_base = obter_variaveis_controle(net_base)
    detalhe_base = avaliar_solucao(net_base, x_base)
    funcao_objetivo = criar_funcao_objetivo(net_base)

    # O caso base é a incumbente inicial: o COA nunca devolve custo pior.
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

    melhor_x, melhor_custo, n = polimento_discreto(
        funcao_objetivo,
        melhor_x,
        melhor_custo,
        lb,
        ub,
        historico,
        registros,
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


def exibir_resultados(melhor_x: np.ndarray, melhor_custo: float, detalhe: dict) -> None:
    """Imprime controles e desempenho da solução encontrada."""
    rotulos = [
        "ext_grid (slack, barra 0)",
        "gen[0] (barra 1)",
        "gen[1] (barra 2)",
        "gen[2] (barra 5)",
        "gen[3] (barra 7)",
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
        print(
            f"  shunt[{indice}] (barra {barra}): step = {step}, "
            f"Q = {q_mvar:.4f} MVAr"
        )

    base = detalhe["caso_base"]
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
        print(f"  CSV de convergência                  : "
              f"{detalhe['arquivo_convergencia_csv']}")


def main() -> None:
    print("=" * 68)
    print(" RPD — IEEE 14 barras | COA (mapa logístico) + pandapower")
    print("=" * 68)
    print(f" Controles       : {N_CONTROLE} (5 tensões, 3 taps, 1 shunt)")
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
    exibir_resultados(melhor_x, melhor_custo, detalhe)
    print("=" * 68)


if __name__ == "__main__":
    main()
