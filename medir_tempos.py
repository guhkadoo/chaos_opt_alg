"""
Mede o tempo de execução de cada algoritmo em cada sistema-teste.

Executa as 12 combinações (3 sistemas x 4 algoritmos) sequencialmente, no mesmo
processo-pai mas cada uma em um subprocesso limpo, e grava os tempos em
tempos_execucao.csv.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PYTHON = str(RAIZ / ".venv" / "bin" / "python")

SISTEMAS = ["case3", "wscc9", "ieee14"]
ALGORITMOS = ["coa", "pso", "ga", "de"]

SAIDA = RAIZ / "tempos_execucao.csv"


def main() -> None:
    with SAIDA.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["sistema", "algoritmo", "tempo_s", "retorno"])

        for sistema in SISTEMAS:
            for algoritmo in ALGORITMOS:
                script = RAIZ / f"rpd_{sistema}_{algoritmo}.py"
                print(f"[{time.strftime('%H:%M:%S')}] executando {script.name} ...",
                      flush=True)

                inicio = time.perf_counter()
                processo = subprocess.run(
                    [PYTHON, str(script)],
                    cwd=str(RAIZ),
                    capture_output=True,
                    text=True,
                )
                decorrido = time.perf_counter() - inicio

                print(f"    -> {decorrido:.1f} s (retorno {processo.returncode})",
                      flush=True)
                if processo.returncode != 0:
                    print(processo.stderr[-2000:], file=sys.stderr, flush=True)

                escritor.writerow(
                    [sistema, algoritmo, f"{decorrido:.3f}", processo.returncode]
                )
                arquivo.flush()

    print(f"\nTempos gravados em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
