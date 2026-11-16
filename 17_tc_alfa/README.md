# 17_tc_alfa: espessura crítica da parede contra alfa

Reforço 1 do artigo. Mostra onde a resistência de Kapitza pesa na parede crítica
t_c das partículas ocas: a correção sobre Felske (2004) é 4–5 % em R = 50 nm com
R_K = 1e-8 (alfa = 0,04), mas cresce como ~1/(1 − alfa) e diverge em alfa_c.

- `tc_alfa.py`: teoria (CSV `teoria_tc_alfa.csv`) e figura `figuras/fig_tc_alfa.pdf`;
  marca os pontos de FEM se `fem/*.csv` existir. Não roda FEM.
- `rodar_fem.sh`: 15 casos FEM (sílica alfa 0,2/0,4/0,6 e ZIF-8 0,2/0,3) com
  `09_particulas_ocas/estudos/ocas/ocas.py`. Rodar da raiz, em primeiro plano.
