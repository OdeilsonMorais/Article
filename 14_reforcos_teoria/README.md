# 14_reforcos_teoria

Só teoria (NumPy/SciPy/Matplotlib), roda em segundos em qualquer máquina.

| script | reforço | saída |
|---|---|---|
| `b1_espessura_critica.py` | t_c da parede (k_eq = k_m/(1 − α)) e valores esperados do FEM | `b1_t_critico.csv` |
| `b3_tamanho_parede.py` | efeito de tamanho na parede (Matthiessen e Fuchs–Sondheimer) | `b3_tamanho_parede.csv`, figura |
| `b4_monte_carlo.py` | incertezas por Monte Carlo (1e5), P(k < k_m), Spearman | `b4_monte_carlo.csv`, `b4_sensibilidade.csv` |
| `b5_distribuicao.py` | HJ generalizado com log-normal e células bidispersas | `b5_distribuicao.csv`, figura |

`base.py` tem as fórmulas comuns (as mesmas das pastas 09, 11 e 12).
