# 15_bidispersa

Células com esferas maciças de dois raios (padrão 25 e 75 nm, metade do volume de sólido em cada),
5 % em massa, para conferir a regra de mistura de Hasselman–Johnson
(14_reforcos_teoria/b5_distribuicao.py). `malha_bi.py` é a malha da pasta 11 com raio por esfera;
solver, periodicidade e pós-processamento vêm da pasta 11 sem mudança.

```bash
python3 estudos/bidispersa.py --so-sortear
python3 estudos/bidispersa.py --so-malha
python3 estudos/bidispersa.py                                   # SiO2, 25 + 75, R_K 1e-10 e 1e-8
python3 estudos/bidispersa.py --raios 25 --n-mono 54            # referência monodispersa
python3 estudos/bidispersa.py --material gama
```
