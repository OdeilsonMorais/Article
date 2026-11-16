# 12_nanocristalino

Partículas nanocristalinas, com cristalitos de 1 a 25 nm, uma esfera ou 80
esferas a 5 % em massa.

## Por que em duas escalas

Num grão de 3 nm dentro de uma partícula de 100 nm há ~20 mil grãos por
partícula, e cada contorno precisaria de um salto de temperatura na malha.
Isso não cabe em nenhum computador para 80 partículas. Então:

1. **Escala do grão** (`nanocristal.py`): k da partícula pelo tamanho do
   cristalito d, com espalhamento de contorno dentro do grão (ℓ) e
   resistência de Kapitza R_gb em cada contorno de grão:
   k_nc = k_grão/(1 + R_gb k_grão/d), com k_grão = k0/(1 + ℓ/d).
2. **Escala da partícula** (FEM da pasta 11): esse k_nc vai para a partícula,
   e o Kapitza epóxi/óxido continua na superfície dela.

## Uso (de 12_nanocristalino; a pasta 11 precisa estar ao lado)

```bash
python3 estudos/teoria.py                      # tabela + figura, segundos
python3 estudos/rodar.py --caso uma            # 1 esfera, d = 1, 2, 3, 5, 10, 15, 25 nm
python3 estudos/rodar.py --caso muitas         # 80 esferas (~12 min por d)
python3 estudos/teoria.py                      # de novo, agora com os pontos do FEM
```

Opções: `--material gama|alfa|anatase|sio2|quartzo`, `--k0 5`, `--rgb 1e-10 1e-9 1e-8`,
`--diametro 200` (para raio de 100 nm). Cada material (e cada `--k0`) grava
seus próprios CSV: `estudos/nc_{uma,muitas,mapa}_<material>[_k0_X].csv`.

Rodar tudo de uma vez (noite, ~6 h):

```bash
for a in "--material gama --k0 5" "--material sio2" "--material gama" "--material quartzo"; do
  python3 estudos/rodar.py --caso uma $a && python3 estudos/rodar.py --caso muitas $a || break
done
```

Cuidado com d = 1 e 2 nm: é a escala da célula unitária (espinélio ≈ 0,8 nm),
então o modelo contínuo (ℓ, R_gb) está no limite; leia como tendência.

## Parâmetros (estimativas)

| | k0 (W/mK) | ℓ (nm) | confiança |
|---|---|---|---|
| γ-Al₂O₃ | 10 | 2 | baixa |
| α-Al₂O₃ | 35 | 5 | média |
| TiO₂ anatase | 8,5 | 2 | baixa |

R_gb de óxidos: 1e-10 a 1e-8 m²K/W (1e-9 é o valor central).

## Teoria (10/11/2026), γ-Al₂O₃, R_gb = 1e-9, R_K = 1e-8, φ = 2,1 %

| d (nm) | k_partícula | k_eff/k_m |
|---|---|---|
| 1 | 0,8 | 1,028 |
| 2 | 1,4 | 1,039 |
| 3 | 2,0 | 1,043 |
| 5 | 2,9 | 1,047 |
| 10 | 4,5 | 1,051 |
| 15 | 5,6 | 1,052 |
| 25 | 6,8 | 1,053 |
| monocristal | 10 | 1,054 |

Os contornos derrubam o k da partícula em até 5×, mas o compósito quase não
muda, porque k_p continua bem acima de k_m. Só com R_gb = 1e-8 e d ≈ 2 nm a
partícula chega perto da epóxi. O piso absoluto, com partículas isolantes, é
0,969.

## Esferas ocas nanocristalinas (11/11/2026)

Parede nanocristalina (k_nc(d)), núcleo de ar, mesma massa (5 %). Só entram
cristalitos menores que a parede (t = (t/R)·50 nm). Teoria: esfera revestida
equivalente + HJ (`estudos/teoria_ocas.py`). FEM: `rodar.py --espessura t/R`,
CSV `nc_*_<material>[_k0_X]_oca_tR_<t/R>.csv`.

Teoria, R_gb = 1e-9, k/k_m:

| | t/R 0,2 (10 nm) d=1 / 10 | t/R 0,1 (5 nm) d=1 / 5 | t/R 0,05 (2,5 nm) d=1 / 2 |
|---|---|---|---|
| SiO₂ amorfa | 0,996 / 1,059 | 0,935 / 1,005 | 0,810 / 0,847 |
| γ-Al₂O₃ (k0 = 5) | 1,015 / 1,076 | 0,983 / 1,067 | 0,907 / 0,956 |
