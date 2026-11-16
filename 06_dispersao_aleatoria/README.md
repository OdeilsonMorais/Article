# 06_dispersao_aleatoria

Dispersões aleatórias de N esferas iguais na RVE periódica, comparadas com
Hasselman–Johnson. Os módulos (`geometria.py`, `malha.py`, `solver.py` etc.)
são uma cópia dos da pasta 05 em 03/11/2026, com o solver iterativo como
padrão. A pasta 05 fica congelada com os resultados dos aglomerados.

## Estudo

`estudos/aleatorio/aleatorio.py`: para cada φ e cada semente sorteia N centros
(padrão 8) sem sobreposição, resolve o tensor completo (x, y e z) em três
níveis de malha com Aitken e grava `k_media` e o desvio de HJ em
`estudos/aleatorio/aleatorio.csv`, junto com os centros para reproduzir cada
sorteio. A esfera isolada no mesmo φ (semente = −1) é rodada junto, como
referência.

```bash
# ver os sorteios sem resolver
python3 estudos/aleatorio/aleatorio.py --so-sortear --phi 0.05 --sementes 0 1 2

# estudo padrão: φ = 1, 2,5, 5 e 10 %, 5 sementes, R_K = 1e-10 e 1e-8
python3 estudos/aleatorio/aleatorio.py
```

Limitações do sorteio (vêm da malha): no máximo 9 esferas por célula, e
nenhuma esfera cruza as faces do cubo (centros a pelo menos 1,2R das faces).

## Mapas e animação

`estudos/aleatorio/cortes_aleatorios.py`: vista 3D de um sorteio e cortes de T e
|q| num plano xy ou xz que passa pelo par mais próximo (G sempre em x).
`estudos/aleatorio/animacao_aleatoria.py`: o plano xy (ou xz) varre a célula e
mostra T, |q| no plano e o fluxo normal ao plano; grava mp4 (ou GIF sem ffmpeg).
Os dois rodam em série e guardam os campos em `dados/` (fora do git).

```bash
python3 estudos/aleatorio/cortes_aleatorios.py --calcular --phi 0.1 --sementes 0 6 8
python3 estudos/aleatorio/animacao_aleatoria.py --calcular --phi 0.1 --semente 6
```
