# Bolsões de ar (poros), sem partículas

Um poro é uma esfera de ar (k_ar = 0,026 W/mK) na epóxi, sem Kapitza. No FEM,
é a partícula maciça da pasta 09 com k_p = k_ar e R_K = 1e-12.

```bash
# curva k × porosidade e mapa poros × partículas (só Matplotlib, segundos)
python3 estudos/poros/teoria_poros.py

# cortes de T e |q| para 5, 10 e 20 % (FEniCSx, em série)
python3 estudos/poros/cortes_poros.py --calcular
```

`teoria_poros.py` lê os pontos do FEM de `estudos/ocas/poros.csv`, gerado com
`ocas.py --espessuras 1 --rk 1e-12 --k-particula 0.026 --saida estudos/ocas/poros.csv`.

## Resultado (05/11/2026)

- FEM e Maxwell batem dentro de 0,08 % (2, 5, 10 e 20 %).
- Sem partículas, chegar em k/k_m = 0,7 exige 26–27 % de ar, e 0,5 exige 46–49 %.
- Com 2 % de partículas maciças (5 % em massa), seriam 31 % e 53 %.
