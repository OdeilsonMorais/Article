# 07_resistencia_interface

Parâmetros físicos da interface epóxi/óxido, começando pela resistência de
Kapitza R_K. Os valores daqui alimentam as simulações da RVE (pastas 03–06),
que até agora usaram R_K = 10⁻¹⁰ e 10⁻⁸ m²K/W como chutes.

## Arquivos

- `parametros_materiais.py`: **o único lugar com números**. Densidade,
  velocidades do som, densidade atômica e k de cada material, com a fonte e
  um grau de confiança. Também tem os raios, os cristalitos, a fração mássica
  e as temperaturas da bancada, tirados da dissertação. Para trocar um valor,
  mude só este arquivo.
- `amm_dmm.py`: R_K pelos modelos de descasamento acústico (AMM) e difuso
  (DMM) com fônons de Debye. Calcula R_K(T) de 200 a 400 K, α = R_K k_m/R
  para o raio de cada partícula e o raio crítico de HJ.
- `resultados/`: `rk_amm_dmm.csv`, `resumo.csv` e `rk_amm_dmm.png`.

```bash
python3 amm_dmm.py      # só NumPy e Matplotlib, roda em segundos
```

## Primeiro resultado (04/11/2026)

Na faixa da bancada (25–80 °C), R_K quase não depende de T:

| Interface | R_K AMM | R_K DMM | α (DMM) | R_c (DMM) |
|---|---|---|---|---|
| epóxi/anatase | 1,2·10⁻⁸ | 6,5·10⁻⁹ | 0,026 (R = 50 nm) | 1,3 nm |
| epóxi/γ-Al₂O₃ | 2,0·10⁻⁸ | 1,1·10⁻⁸ | 0,064 (R = 35 nm) | 2,3 nm |
| epóxi/rutilo | 2,2·10⁻⁸ | 1,1·10⁻⁸ | 0,043 (R = 50 nm) | 2,2 nm |
| epóxi/α-Al₂O₃ | 3,5·10⁻⁸ | 1,6·10⁻⁸ | 0,093 (R = 35 nm) | 3,3 nm |

(R_K em m²K/W.) O valor 10⁻⁸ usado nas pastas 03–06 está na faixa certa.
Com partículas de 35–50 nm, α fica entre 0,03 e 0,2, muito abaixo de
α_c ≈ 0,96: pela interface, as partículas ainda devem aumentar k.

## Limitações

- Debye isotrópico: o epóxi é amorfo e tem muitos modos ópticos que o modelo
  ignora. O AMM e o DMM não veem a ligação química na interface.
- O epóxi é o lado mole: o ramo transversal dele (Θ ≈ 170 K) limita o fluxo,
  então R_K é sensível ao v_T do epóxi. Uma medida de ultrassom do
  revestimento curado melhora muito o resultado.
- A densidade é a do DGEBA puro; trocar pela da tinta N2680 (boletim técnico).
- γ-Al₂O₃ tem confiança baixa (módulos elásticos estimados, partícula porosa).

## Próximos passos

DFT (exciting) para a dispersão real de fônons e o kₚ de cada fase, depois
dinâmica molecular (LAMMPS, NEMD) na interface.
