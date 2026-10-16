#!/usr/bin/env bash
# Re-executa os quatro algoritmos do case3 para (a) medir tempos com a rede
# corrigida e (b) conferir se os CSVs atuais sao reproduzidos exatamente.
set -u
cd "$(dirname "$0")"

BK=_bk_case3
mkdir -p "$BK" _log_case3
cp convergencia_case3_*.csv minimos_locais_case3_coa.csv "$BK"/

for a in coa pso ga de; do
  echo "=== $a ==="
  s=$(date +%s.%N)
  .venv/bin/python "rpd_case3_${a}.py" > "_log_case3/${a}.txt" 2>&1
  rc=$?
  e=$(date +%s.%N)
  echo "tempo_s=$(echo "$e - $s" | bc)  retorno=$rc"
done

echo
echo "=== diff dos CSVs (vazio = reproduzido exatamente) ==="
for f in convergencia_case3_coa.csv convergencia_case3_pso.csv \
         convergencia_case3_ga.csv convergencia_case3_de.csv \
         minimos_locais_case3_coa.csv; do
  if diff -q "$BK/$f" "$f" > /dev/null; then
    echo "IDENTICO  $f"
  else
    echo "DIFERENTE $f"
  fi
done
