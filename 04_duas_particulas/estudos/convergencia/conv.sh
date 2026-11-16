#!/bin/bash
# Convergência de malha, caso s/R = 0,05, R_K = 1e-8 (valores em parametros.py).
# Rodar dentro de 04_duas_particulas:  nohup bash conv.sh > conv_log.txt 2>&1 &

conv() {  # $1 = h_int/R, $2 = h_lacuna/R
  echo "== h_int=$1 h_lac=$2"
  /usr/bin/time -f "tempo %e s, memória %M KB" \
    mpirun -np 2 python3 main.py --solver-kapitza --rk 1e-8 --malha-adaptativa --threads 2 \
      --h-interface-relativo $1 --h-lacuna-relativo $2 2>&1 \
    | grep -E "nodes|k_eff|Área|Salto|tempo|Error|error"
}

# Série 1: lacuna, com a interface fixa em R/10
for l in 0.025 0.0125 0.00625; do conv 0.10 $l; done | tee conv_lacuna.txt
# Série 2: interface, com a lacuna fixa em R/80
for h in 0.125 0.10 0.075; do conv $h 0.0125; done | tee conv_interface.txt
