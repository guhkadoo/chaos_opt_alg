"""Segunda rodada: contribuicao correta de cada fase do COA e multimodalidade."""

from __future__ import annotations

import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SISTEMAS = {"case3": "3 barras", "wscc9": "WSCC 9 barras", "ieee14": "IEEE 14 barras"}
MIN_ARQ = {
    "case3": "minimos_locais_case3_coa.csv",
    "wscc9": "minimos_locais_wscc9_coa.csv",
    "ieee14": "minimos_locais_coa.csv",
}
S1, S2, T3 = 2000, 150, 1.35


def ler(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


n_eta = 0
eta = 0.01
while eta <= 0.10 + 1e-12:
    n_eta += 1
    eta *= T3
print(f"rodadas de eta na segunda onda = {n_eta}  -> {n_eta*S2} avaliacoes")
print(f"fronteiras: base=1 | 1a onda=2..{1+S1} | 2a onda={2+S1}..{1+S1+n_eta*S2} | polimento depois")
print()

print("=" * 78)
print("CONTRIBUICAO DE CADA FASE DO COA (em custo)")
print("=" * 78)
for sis, nome in SISTEMAS.items():
    regs = ler(RAIZ / MIN_ARQ[sis])
    total_avals = len(ler(RAIZ / f"convergencia_{sis}_coa.csv"))
    base = float(regs[0]["custo"])
    fim_1a = min(float(r["custo"]) for r in regs
                 if int(r["avaliacao"]) <= 1 + S1)
    final = float(regs[-1]["custo"])
    ganho_total = base - final
    g1 = base - fim_1a
    g2 = fim_1a - final
    n_pol = total_avals - (1 + S1 + n_eta * S2)
    tem_pol = any("polimento" in r["fase"] for r in regs)
    print(f"\n--- {nome} ---")
    print(f"  base        = {base:10.5f}")
    print(f"  fim 1a onda = {fim_1a:10.5f}   ganho = {g1:8.5f} ({g1/ganho_total*100:5.1f}% do total)")
    print(f"  final       = {final:10.5f}   ganho 2a onda = {g2:8.5f} ({g2/ganho_total*100:5.1f}%)")
    print(f"  ganho total = {ganho_total:.5f}  ({ganho_total/base*100:.2f}% do caso base)")
    print(f"  avaliacoes de polimento = {n_pol}   houve melhoria no polimento? {tem_pol}")
    n1 = sum(1 for r in regs if r["fase"] == "primeira_onda")
    n2 = sum(1 for r in regs if r["fase"].startswith("segunda_onda"))
    print(f"  incumbentes: 1a onda = {n1}   2a onda = {n2}")
    prim = [r for r in regs if r["fase"] == "primeira_onda"]
    print(f"  ultima melhoria da 1a onda na avaliacao {prim[-1]['avaliacao']} de {S1}")

print()
print("=" * 78)
print("MULTIMODALIDADE: solucoes quase-otimas com controles discretos distintos")
print("=" * 78)
for sis in ("case3", "ieee14"):
    print(f"\n--- {SISTEMAS[sis]} ---")
    finais = {}
    regs = ler(RAIZ / MIN_ARQ[sis])
    ult = regs[-1]
    disc = [k for k in ult if k.startswith(("tap_", "shunt"))]
    finais["COA"] = (float(ult["perdas_mw"]), tuple(ult[k] for k in disc))
    for algo in ("pso", "ga", "de"):
        lin = min(ler(RAIZ / f"convergencia_{sis}_{algo}.csv"),
                  key=lambda l: float(l["custo"]))
        finais[algo.upper()] = (float(lin["perdas_mw"]), tuple(lin[k] for k in disc))
    print(f"  variaveis discretas: {disc}")
    for a, (p, d) in finais.items():
        print(f"    {a:>4}: perdas={p:9.5f} MW   discretas={d}")
    combos = {d for _, d in finais.values()}
    print(f"  -> {len(combos)} combinacoes discretas distintas entre {len(finais)} algoritmos")

print()
print("=" * 78)
print("DISPERSAO DAS PERDAS FINAIS E PERFIL DE TENSAO")
print("=" * 78)
for sis, nome in SISTEMAS.items():
    regs = ler(RAIZ / MIN_ARQ[sis])
    ult = regs[-1]
    vals = {"COA": float(ult["perdas_mw"])}
    tens = {"COA": (float(ult["vm_min"]), float(ult["vm_max"]))}
    for algo in ("pso", "ga", "de"):
        lin = min(ler(RAIZ / f"convergencia_{sis}_{algo}.csv"),
                  key=lambda l: float(l["custo"]))
        vals[algo.upper()] = float(lin["perdas_mw"])
        tens[algo.upper()] = (float(lin["vm_min"]), float(lin["vm_max"]))
    v = list(vals.values())
    print(f"\n--- {nome} ---")
    print(f"  perdas: min={min(v):.5f} max={max(v):.5f} "
          f"amplitude={max(v)-min(v):.5f} MW ({(max(v)-min(v))/min(v)*100:.3f}%)")
    for a in ("COA", "PSO", "GA", "DE"):
        print(f"    {a:>4}: Vmin={tens[a][0]:.4f}  Vmax={tens[a][1]:.4f} p.u.")

print()
print("=" * 78)
print("CASO BASE: detalhes")
print("=" * 78)
for sis, nome in SISTEMAS.items():
    r = ler(RAIZ / MIN_ARQ[sis])[0]
    print(f"  {nome:16s} perdas={float(r['perdas_mw']):9.5f} custo={float(r['custo']):9.5f} "
          f"penal={float(r['penalidade']):.5f} viol={r['violacoes_tensao']} "
          f"Vmin={float(r['vm_min']):.4f} Vmax={float(r['vm_max']):.4f} "
          f"viol_max={r.get('violacao_max_pu', '-')}")
