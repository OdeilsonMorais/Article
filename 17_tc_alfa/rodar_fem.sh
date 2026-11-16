#!/usr/bin/env bash
# Verificação por FEM de t_c(alfa) (reforço 1 do artigo). R = 50 nm fixo no
# ocas.py, então alfa = R_K k_m / R varia pelo R_K: 5e-8 → 0,2; 7,5e-8 → 0,3;
# 1e-7 → 0,4; 1,5e-7 → 0,6. Para cada alfa: t/R = 1,25 t_c, t_c e 0,75 t_c
# (phi = 5 %, núcleo de ar). Rodar da raiz do repositório, em primeiro plano:
#     bash 17_tc_alfa/rodar_fem.sh            # tudo (ZIF-8 primeiro, é barato)
#     bash 17_tc_alfa/rodar_fem.sh zif8
#     bash 17_tc_alfa/rodar_fem.sh sio2
# Retoma de onde parou. Depois: python3 17_tc_alfa/tc_alfa.py
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ/09_particulas_ocas"
FEM="$RAIZ/17_tc_alfa/fem"
mkdir -p "$FEM"
etapa="${1:-tudo}"
log="$RAIZ/17_tc_alfa/rodar_fem.log"

caso() {  # k_p R_K saida t1 t2 t3
  python3 estudos/ocas/ocas.py --modo mesmo_phi --phi 0.05 --k-particula "$1" --rk "$2" \
    --espessuras "$4" "$5" "$6" --saida "$FEM/$3"
}

zif8() {
  # Esperado k/k_m: 1,0051 / 1,0000 / 0,9922 (alfa 0,2); 1,0032 / 1,0000 / 0,9941 (alfa 0,3)
  caso 0.33 5e-8   fem_zif8.csv 0.5210 0.4168 0.3126
  caso 0.33 7.5e-8 fem_zif8.csv 0.6590 0.5272 0.3954
}

sio2() {
  # Esperado: 1,0080 / 1,0000 / 0,9901 (0,2); 1,0059 / 1,0000 / 0,9921 (0,4); 1,0038 / 1,0000 / 0,9945 (0,6)
  caso 1.36 5e-8   fem_sio2.csv 0.1046 0.0837 0.0628
  caso 1.36 1e-7   fem_sio2.csv 0.1443 0.1154 0.0866
  caso 1.36 1.5e-7 fem_sio2.csv 0.2260 0.1808 0.1356
}

{
  echo "=== $(date '+%F %T') etapa: $etapa"
  case "$etapa" in
    zif8) zif8 ;;
    sio2) sio2 ;;
    tudo) zif8; sio2 ;;
    *) echo "Etapa desconhecida: $etapa (use zif8, sio2 ou tudo)"; exit 1 ;;
  esac
  echo "=== $(date '+%F %T') fim"
} 2>&1 | tee -a "$log"
