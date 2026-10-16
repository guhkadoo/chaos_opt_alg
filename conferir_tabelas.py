"""Reproduz exatamente as linhas das tabelas do relatorio, para conferencia."""

from __future__ import annotations

import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SIS = {"case3": "3 barras", "wscc9": "WSCC 9 barras", "ieee14": "IEEE 14 barras"}
MIN_ARQ = {"case3": "minimos_locais_case3_coa.csv",
           "wscc9": "minimos_locais_wscc9_coa.csv",
           "ieee14": "minimos_locais_coa.csv"}


def ler(p):
    with (RAIZ / p).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def br(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


def final(sis, algo):
    if algo == "coa":
        r = ler(MIN_ARQ[sis])[-1]
    else:
        r = min(ler(f"convergencia_{sis}_{algo}.csv"), key=lambda l: float(l["custo"]))
    return r


print("### TABELA 5 - caso base")
for sis, nome in SIS.items():
    r = ler(MIN_ARQ[sis])[0]
    print(f"{nome:14s} & {br(float(r['perdas_mw']),4)} & {br(float(r['penalidade']),4)} & "
          f"{br(float(r['custo']),4)} & {r['violacoes_tensao']} & "
          f"{br(float(r['vm_min']),4)} -- {br(float(r['vm_max']),4)} \\\\")

print("\n### TABELA 6 - perdas finais, reducao, desvio")
for sis, nome in SIS.items():
    base = float(ler(MIN_ARQ[sis])[0]["perdas_mw"])
    custos = {a: float(final(sis, a)["custo"]) for a in ("coa", "pso", "ga", "de")}
    melhor = min(custos.values())
    print(f"-- {nome} (base perdas={br(base,5)}) --")
    for a in ("coa", "pso", "ga", "de"):
        r = final(sis, a)
        p = float(r["perdas_mw"])
        red = (base - p) / base * 100
        dev = (custos[a] - melhor) / melhor * 100
        dev_s = "---" if dev < 1e-9 else br(dev, 3)
        print(f"  {a.upper():>4} & {br(p,5)} & {br(red,2)} & {dev_s} & {r['violacoes_tensao']} \\\\")
    v = [float(final(sis, a)["perdas_mw"]) for a in ("coa", "pso", "ga", "de")]
    print(f"  amplitude = {br(max(v)-min(v),4)} MW = {br((max(v)-min(v))/min(v)*100,2)}%")

print("\n### TABELA 8 - fases do COA")
for sis, nome in SIS.items():
    regs = ler(MIN_ARQ[sis])
    base = float(regs[0]["custo"])
    fim1 = min(float(r["custo"]) for r in regs if int(r["avaliacao"]) <= 2001)
    fim = float(regs[-1]["custo"])
    tot = base - fim
    print(f"{nome:14s} & {br(base,5)} & {br(fim1,5)} & {br(fim,5)} & "
          f"{br((base-fim1)/tot*100,1)} & {br((fim1-fim)/tot*100,1)} \\\\")

print("\n### TABELA 9 - controles IEEE 14")
campos = ["vm_slack", "vm_gen_0", "vm_gen_1", "vm_gen_2", "vm_gen_3",
          "tap_0", "tap_1", "tap_2", "shunt_step"]
for a in ("coa", "pso", "ga", "de"):
    r = final("ieee14", a)
    vs = " & ".join(br(float(r[c]), 4) for c in campos[:5])
    ds = " & ".join(f"{int(float(r[c])):+d}" if c.startswith("tap") else str(int(float(r[c])))
                    for c in campos[5:])
    print(f"{a.upper():>4} & {vs} & {ds} \\\\")

print("\n### AFIRMACOES DO TEXTO")
# max distancia COA-PSO
maxd = 0.0
for sis in SIS:
    d = float(final(sis, "coa")["perdas_mw"]) - float(final(sis, "pso")["perdas_mw"])
    print(f"  COA-PSO em {SIS[sis]:14s}: {d:.3e} MW")
    maxd = max(maxd, d)
print(f"  -> maior distancia COA-PSO = {maxd:.3e} MW")
# desperdicio 1a onda ieee14
regs = ler(MIN_ARQ["ieee14"])
ult1 = max(int(r["avaliacao"]) for r in regs if r["fase"] == "primeira_onda")
tot_av = len(ler("convergencia_ieee14_coa.csv"))
print(f"  ieee14: ultima melhoria 1a onda = aval {ult1}; 1a onda termina em 2001")
print(f"  -> {2001-ult1} avaliacoes sem progresso = {(2001-ult1)/tot_av*100:.1f}% do total ({tot_av})")
# numero de melhorias
for sis in SIS:
    n = {}
    for a in ("coa", "pso", "ga", "de"):
        rows = ler(f"convergencia_{sis}_{a}.csv")
        m, c = float("inf"), 0
        for r in rows:
            if float(r["custo"]) < m:
                m = float(r["custo"]); c += 1
        n[a.upper()] = c
    print(f"  melhorias da incumbente em {SIS[sis]:14s}: {n}")
