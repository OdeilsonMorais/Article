# 11_muitas_particulas

Dezenas de partículas (padrão 80) em posições aleatórias, maciças ou ocas, com
a concentração dada em **% em massa**. Os módulos são uma cópia dos da pasta 09
(07/11/2026); a 09 fica como estava.

## O que mudou nos módulos

- `malha.py`: etiquetas em blocos de 1000. Partícula j = 1001 + j, interface =
  2001 + j, núcleo = 3001 + j, parede interna = 4001 + j. Assim cabem até 999
  partículas; na 09 cabiam 9. Também ganhou `gerar_malha(..., so_contar_nos=True)`,
  que gera a malha só com o Gmsh e devolve o número de nós.
- `solver.py`: as medidas usam a tupla de etiquetas (`dx((1001, 1002, ...))`),
  não uma soma de 80 termos.

## Tamanho da célula

A massa fixa a fração de óxido, φ_sólido = (w/ρ_p) / (w/ρ_p + (1 − w)/ρ_tinta).
A oca ocupa φ_externo = φ_sólido/(1 − (1 − t/R)³). Com N esferas de raio R,
L = R·(4πN/(3φ_externo))^(1/3). Para 80 esferas de 100 nm a 5 % em massa
(tinta a 1,5 g/cm³, valor ainda a confirmar):

| material | t/R | φ externo | L |
|---|---|---|---|
| γ-Al₂O₃ (3,65 g/cm³) | 1 | 2,1 % | 25,1R = 1,26 µm |
| SiO₂ (2,2 g/cm³) | 1 | 3,5 % | 21,3R = 1,07 µm |
| SiO₂ | 0,2 | 7,1 % | 16,8R = 0,84 µm |
| SiO₂ | 0,1 | 12,8 % | 13,8R = 0,69 µm |

## Uso

```bash
python3 estudos/muitas.py --material gama --so-malha     # só Gmsh: nós por nível
python3 estudos/muitas.py --material gama                # resolve (3 níveis + Aitken)
python3 estudos/muitas.py --material sio2 --espessura 1 0.2 --sementes 0 1
```

Malha da γ-Al₂O₃ maciça com 80 esferas (Gmsh, testado): h_bulk = 0,35R dá 400 mil
nós, 0,25R dá 850 mil e 0,18R dá 1,9 milhão. Por isso o padrão aqui é 0,5/0,35/0,25.
