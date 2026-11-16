#!/bin/bash
# Convergência com refino global (h_bulk, h_int, h_lac juntos), malha reprodutível (1 thread).
# Rodar de 04_duas_particulas: nohup bash estudos/convergencia/conv2.sh > estudos/convergencia/conv2_log.txt 2>&1 &
SAIDA=estudos/convergencia/conv_global.txt
conv() {
  echo "== h_bulk=$1 h_int=$2 h_lac=$3"
  /usr/bin/time -f "tempo %e s, memória %M KB" \
    mpirun -np 2 python3 main.py --solver-kapitza --rk 1e-8 --malha-adaptativa --threads 1 \
      --h-relativo $1 --h-interface-relativo $2 --h-lacuna-relativo $3 2>&1 \
    | grep -E "nodes|k_eff|Área|Salto|tempo|Error|error"
}
{
  conv 0.5 0.10 0.0125
  conv 0.5 0.10 0.0125
  conv 0.4 0.08 0.01
  conv 0.32 0.064 0.008
  conv 0.256 0.0512 0.0064
} | tee "$SAIDA"
