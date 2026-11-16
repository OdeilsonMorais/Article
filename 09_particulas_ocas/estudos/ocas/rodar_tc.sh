#!/usr/bin/env bash
# Espessura crítica t_c: verificação por FEM (B1) e figura de campo.
# Rodar de 09_particulas_ocas, em primeiro plano:
#     bash estudos/ocas/rodar_tc.sh            # tudo: ZIF-8, sílica e figura
#     bash estudos/ocas/rodar_tc.sh zif8       # só ZIF-8 (barato)
#     bash estudos/ocas/rodar_tc.sh sio2       # só sílica (parede fina, malha maior)
#     bash estudos/ocas/rodar_tc.sh figura     # só a figura de campo
# Cada caso grava num CSV próprio e o ocas.py retoma de onde parou, então dá
# para interromper (Ctrl+C) e rodar de novo. O log fica em estudos/ocas/rodar_tc.log.
set -euo pipefail
cd "$(dirname "$0")/../.."          # 09_particulas_ocas
etapa="${1:-tudo}"
log=estudos/ocas/rodar_tc.log

zif8() {
  # t_c/R = 0,3187. Esperado: t/R 0,40 → 1,0077 / 1,0154 / 1,0309; 0,3187 → 1,0000; 0,25 → 0,9916 / 0,9833 / 0,9668
  for p in 0.05 0.10 0.20; do
    python3 estudos/ocas/ocas.py --modo mesmo_phi --phi $p --k-particula 0.33 --rk 1e-8 \
      --espessuras 0.40 0.3187 0.25 --saida estudos/ocas/tc_zif8_phi$p.csv
  done
}

sio2() {
  # t_c/R = 0,068. Esperado: t/R 0,10 → 1,0171 / 1,0344 / 1,0696; 0,068 → 1,0000; 0,05 → 0,9880 / 0,9761 / 0,9525
  # φ = 0,20 por último: parede fina com φ alto é a malha mais pesada.
  for p in 0.05 0.10 0.20; do
    python3 estudos/ocas/ocas.py --modo mesmo_phi --phi $p --k-particula 1.36 --rk 1e-8 \
      --espessuras 0.10 0.068 0.05 --saida estudos/ocas/tc_sio2_phi$p.csv
  done
}

figura() {
  # ZIF-8, φ 10 %, t/R 0,5 / 0,3187 / 0,15. Esperado 1,0289 / 1,0000 / 0,9516.
  python3 estudos/ocas/cortes_tc.py --calcular
}

{
  echo "=== $(date '+%F %T') etapa: $etapa"
  case "$etapa" in
    zif8) zif8 ;;
    sio2) sio2 ;;
    figura) figura ;;
    tudo) zif8; figura; sio2 ;;
    *) echo "Etapa desconhecida: $etapa (use zif8, sio2, figura ou tudo)"; exit 1 ;;
  esac
  echo "=== $(date '+%F %T') fim"
} 2>&1 | tee -a "$log"
