"""Extrai as estatisticas usadas na secao de Resultados do relatorio."""

from __future__ import annotations

import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SISTEMAS = {"case3": "3 barras", "wscc9": "WSCC 9 barras", "ieee14": "IEEE 14 barras"}
ALGOS = ["coa", "pso", "ga", "de"]


def ler(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def trajetoria(sistema: str, algo: str):
    """Retorna [(avaliacoes, custo, perdas)] da incumbente."""
    linhas = ler(RAIZ / f"convergencia_{sistema}_{algo}.csv")
    pontos = []
    for lin in linhas:
        aval = int(lin.get("avaliacoes") or lin["avaliacao"])
        pontos.append((aval, float(lin["custo"]), float(lin["perdas_mw"])))
    # torna monotona (incumbente)
    melhor = float("inf")
    inc = []
    for aval, custo, perdas in pontos:
        if custo < melhor:
            melhor = custo
            inc.append((aval, custo, perdas))
    return pontos, inc


def avals_para_atingir(inc, alvo):
    for aval, custo, _ in inc:
        if custo <= alvo:
            return aval
    return None


print("=" * 78)
print("1. RESUMO POR SISTEMA / ALGORITMO")
print("=" * 78)

resumo = {}
for sis, nome in SISTEMAS.items():
    print(f"\n--- {nome} ---")
    finais = {}
    for algo in ALGOS:
        pontos, inc = trajetoria(sis, algo)
        aval_fim, custo_fim, perdas_fim = inc[-1]
        base = pontos[0]
        finais[algo] = (custo_fim, perdas_fim, aval_fim, len(inc), pontos[-1][0])
        print(
            f"{algo.upper():>4}: perdas={perdas_fim:9.5f} MW  custo={custo_fim:9.5f}  "
            f"ultima melhoria em aval={aval_fim:5d}  n_melhorias={len(inc):3d}  "
            f"total_avals={pontos[-1][0]}"
        )
        print(f"      caso base: custo={base[1]:.5f} perdas={base[2]:.5f}")
    resumo[sis] = finais

    melhor_geral = min(v[0] for v in finais.values())
    print(f"  melhor custo global do sistema: {melhor_geral:.6f}")
    for algo in ALGOS:
        gap = (finais[algo][0] - melhor_geral) / melhor_geral * 100
        print(f"    gap {algo.upper():>4} = {gap:8.4f} %")

print()
print("=" * 78)
print("2. VELOCIDADE DE CONVERGENCIA (avaliacoes para chegar a X% do melhor global)")
print("=" * 78)
for sis, nome in SISTEMAS.items():
    melhor_geral = min(v[0] for v in resumo[sis].values())
    print(f"\n--- {nome} (melhor global = {melhor_geral:.5f}) ---")
    for tol in (0.05, 0.01, 0.005, 0.001):
        alvo = melhor_geral * (1 + tol)
        linha = f"  dentro de {tol*100:5.1f}% : "
        for algo in ALGOS:
            _, inc = trajetoria(sis, algo)
            n = avals_para_atingir(inc, alvo)
            linha += f"{algo.upper()}={str(n) if n else '---':>6}  "
        print(linha)
    # 0.5% do PROPRIO valor final
    linha = "  0,5% do proprio: "
    for algo in ALGOS:
        _, inc = trajetoria(sis, algo)
        alvo = inc[-1][1] * 1.005
        linha += f"{algo.upper()}={str(avals_para_atingir(inc, alvo)):>6}  "
    print(linha)

print()
print("=" * 78)
print("3. FASES DO COA: quanto cada onda contribuiu")
print("=" * 78)
for sis, nome in SISTEMAS.items():
    linhas = ler(RAIZ / f"convergencia_{sis}_coa.csv")
    fases = {}
    ordem_fases = []
    melhor = float("inf")
    for lin in linhas:
        fase = lin["fase"]
        custo = float(lin["custo"])
        if fase not in fases:
            fases[fase] = {"n": 0, "inicio": melhor, "melhor": melhor, "aval_ini": int(lin["avaliacao"])}
            ordem_fases.append(fase)
        fases[fase]["n"] += 1
        melhor = min(melhor, custo)
        fases[fase]["melhor"] = melhor
    print(f"\n--- {nome} ---")
    # agrupa: caso_base / primeira_onda / segunda_onda_* / polimento
    grupos = {"caso_base": [], "primeira_onda": [], "segunda_onda": [], "polimento": []}
    for f in ordem_fases:
        if f.startswith("segunda_onda"):
            grupos["segunda_onda"].append(f)
        elif "polimento" in f:
            grupos["polimento"].append(f)
        elif f in grupos:
            grupos[f].append(f)
        else:
            grupos.setdefault(f, []).append(f)
    for g, fs in grupos.items():
        if not fs:
            continue
        n = sum(fases[f]["n"] for f in fs)
        ini = fases[fs[0]]["inicio"]
        fim = fases[fs[-1]]["melhor"]
        ini_txt = "inf" if ini == float("inf") else f"{ini:.5f}"
        print(f"  {g:16s} avals={n:5d}  custo {ini_txt:>9s} -> {fim:.5f}")
    total = sum(fases[f]["n"] for f in fases)
    print(f"  TOTAL avaliacoes = {total}")

print()
print("=" * 78)
print("4. MINIMOS LOCAIS REGISTRADOS PELO COA")
print("=" * 78)
arquivos = {
    "case3": "minimos_locais_case3_coa.csv",
    "wscc9": "minimos_locais_wscc9_coa.csv",
    "ieee14": "minimos_locais_coa.csv",
}
for sis, arq in arquivos.items():
    linhas = ler(RAIZ / arq)
    por_fase = {}
    for lin in linhas:
        f = lin["fase"]
        g = ("segunda_onda" if f.startswith("segunda_onda")
             else "polimento" if "polimento" in f else f)
        por_fase[g] = por_fase.get(g, 0) + 1
    print(f"\n--- {SISTEMAS[sis]} ({arq}) ---")
    print(f"  total de incumbentes registradas: {len(linhas)}")
    for g, n in por_fase.items():
        print(f"    {g:16s}: {n}")
    prim = [l for l in linhas if l["fase"] == "primeira_onda"]
    if prim:
        custos = [float(l["custo"]) for l in prim]
        print(f"  primeira onda: melhor={min(custos):.5f}  pior(1a incumbente)={max(custos):.5f}")
        print(f"  aval da ultima melhoria na 1a onda: {prim[-1]['avaliacao']}")
    ult = linhas[-1]
    print("  solucao final (variaveis de controle):")
    for k, v in ult.items():
        if k.startswith(("vm_", "tap_", "shunt")):
            print(f"    {k:12s} = {v}")

print()
print("=" * 78)
print("5. SOLUCOES FINAIS DOS METAHEURISTICOS (variaveis de controle)")
print("=" * 78)
for sis, nome in SISTEMAS.items():
    print(f"\n--- {nome} ---")
    for algo in ["pso", "ga", "de"]:
        linhas = ler(RAIZ / f"convergencia_{sis}_{algo}.csv")
        melhor_lin = min(linhas, key=lambda l: float(l["custo"]))
        campos = {k: v for k, v in melhor_lin.items()
                  if k.startswith(("vm_", "tap_", "shunt", "violac"))}
        print(f"  {algo.upper():>4}: perdas={float(melhor_lin['perdas_mw']):.5f} "
              f"viol={melhor_lin.get('violacoes_tensao')}")
        print(f"        {campos}")
