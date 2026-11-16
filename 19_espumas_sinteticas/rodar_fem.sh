#!/usr/bin/env bash
# FEM das 8 espumas sintáticas de Liang & Li (2006). Rodar da raiz, em primeiro plano:
#     bash 19_espumas_sinteticas/rodar_fem.sh
# Razões da espuma com k_m = 0,2 do ocas.py: k_p = 0,14917 (vidro), k_c = 0,019 (gás).
# t/R = 1 − η: TK35 0,115; TK70 0,033 (parede fina, malha maior). Retoma de onde parou.
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ/09_particulas_ocas"
FEM="$RAIZ/19_espumas_sinteticas/fem"
mkdir -p "$FEM"
{
  echo "=== $(date '+%F %T') início"
  for p in 0.05 0.10 0.15 0.20; do
    python3 estudos/ocas/ocas.py --modo mesmo_phi --phi $p --k-particula 0.14917 --k-nucleo 0.019 \
      --rk 1e-10 --espessuras 0.115 0.033 --saida "$FEM/espumas.csv"
  done
  echo "=== $(date '+%F %T') fim"
} 2>&1 | tee -a "$RAIZ/19_espumas_sinteticas/rodar_fem.log"
cd "$RAIZ" && python3 19_espumas_sinteticas/validacao.py
