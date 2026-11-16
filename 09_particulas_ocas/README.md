# 09_particulas_ocas

Partículas ocas: uma casca de óxido (k_p) com núcleo de ar (k_n), comparadas
com a partícula maciça. Os módulos são uma cópia dos da pasta 06 (05/11/2026)
com o núcleo acrescentado; a 06 fica como estava.

## O que mudou nos módulos

- `geometria.py`: `geometria_de_centros(..., raio_interno)`, com as
  propriedades `oca`, `espessura_parede` e `fracao_solida`.
- `parametros.py`: `k_nucleo` (0,026 W/mK, ar a ~300 K).
- `malha.py`: `construir_volumes` faz casca e núcleo. Etiquetas novas: núcleo
  da partícula j = 30 + j; parede interna = 40 + j. A interface de Kapitza
  continua sendo 11 + j, só a superfície externa. O refino inclui a parede
  interna, e h_interface = min(R/10, t/3), com no mínimo três elementos na
  espessura da casca.
- `solver.py` e `pos_processamento.py`: o núcleo usa o mesmo campo θ_p da
  casca (temperatura contínua na parede interna) com k_nucleo, e entra no
  fluxo médio. Com raio_interno = 0 tudo volta a ser idêntico à pasta 06.

A malha foi testada só com o Gmsh: os volumes de casca e núcleo batem com os
analíticos, e com t/R = 0,05 o nível mais fino tem cerca de 135 mil nós.

## Estudos (`estudos/ocas/`)

```bash
# teoria (só NumPy/Matplotlib, segundos)
python3 estudos/ocas/teoria.py

# FEM, mesmo φ externo (5 %): t/R = 1 (maciça), 0,5, 0,3, 0,2, 0,1; R_K = 1e-10 e 1e-8
python3 estudos/ocas/ocas.py

# FEM, mesma massa de óxido (φ sólido = 2 %, perto dos 5 % em massa da tinta)
python3 estudos/ocas/ocas.py --modo mesma_massa --phi 0.02
```

`ocas.py` grava `ocas.csv` com k (Aitken em três níveis), a teoria e o desvio.
Parede interna sem Kapitza: o gás quase não conduz, então isso não muda nada.

## Teoria (teoria.py)

Esfera oca equivalente (esfera revestida, exata para gradiente uniforme) e HJ
com k_p → k_eq. Primeiros números, R = 50 nm, k_p = 5:

- Mesmo φ: a oca sempre conduz menos que a maciça. Ela só deixa de aumentar k
  com parede abaixo de t/R ≈ 0,018 (menos de 1 nm). Com parede nanocristalina
  (k_p ≈ 1), isso já acontece em t/R ≈ 0,09 (4,7 nm).
- Mesma massa: paredes moderadas (t/R ≈ 0,05–0,1) aumentam k mais que a
  maciça, porque ocupam mais volume. Só paredes muito finas reduzem k, e aí
  φ externo passa de 20 %, onde o HJ diluído já não é confiável. É para isso
  que serve o FEM.
