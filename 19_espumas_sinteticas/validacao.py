"""Validação do modelo de partícula oca com medidas em espumas sintáticas.

Dados: polipropileno + microesferas de vidro TK35 (η = r_i/R = 0,885, 35 µm) e
TK70 (η = 0,967, 70 µm), 5–20 vol% (Liang & Li, Polym. Test. 25 (2006) 527),
com os parâmetros compilados por Gupta & Pinisetty (JOM 65 (2013) 234, Tabelas
II–III): k_m = 0,24, vidro 0,179, gás 0,0228 W/(m K). Os valores medidos têm
dois algarismos (±0,005 W/(m K)). Em 35–70 µm, α ≈ 1e-4: o teste é do modelo
oco com contato perfeito.

Teoria: esfera revestida + Hasselman–Johnson (as mesmas equações do artigo).
FEM: o ocas.py tem k_m = 0,2 fixo, então roda com as razões k/k_m da espuma
(k_p = 0,2·0,179/0,24, k_c = 0,2·0,0228/0,24) e R_K = 1e-10 (α = 4e-4);
k_eff = 0,24·(k/k_m). Rodar: bash 19_espumas_sinteticas/rodar_fem.sh
Uso: python3 19_espumas_sinteticas/validacao.py  → validacao.csv
"""
import csv
from pathlib import Path

PASTA = Path(__file__).resolve().parent
KM, KG, KA = 0.24, 0.179, 0.0228
MEDIDO = {(0.885, 5): 0.23, (0.885, 10): 0.23, (0.885, 15): 0.21, (0.885, 20): 0.18,
          (0.967, 5): 0.22, (0.967, 10): 0.21, (0.967, 15): 0.21, (0.967, 20): 0.19}
# Felske, Porfiri e Park, como em Gupta & Pinisetty (2013), Tabela III
FELSKE = {(0.885, 5): 0.227, (0.885, 10): 0.214, (0.885, 15): 0.201, (0.885, 20): 0.189,
          (0.967, 5): 0.224, (0.967, 10): 0.208, (0.967, 15): 0.193, (0.967, 20): 0.179}
PORFIRI = {(0.885, 5): 0.229, (0.885, 10): 0.217, (0.885, 15): 0.207, (0.885, 20): 0.196,
           (0.967, 5): 0.226, (0.967, 10): 0.212, (0.967, 15): 0.199, (0.967, 20): 0.187}


def k_eq(kp, kc, delta):
    return kp * ((kc + 2 * kp) + 2 * delta * (kc - kp)) / ((kc + 2 * kp) - delta * (kc - kp))


def hj(phi, kp, km=KM, alfa=0.0):
    a = kp * (1 + 2 * alfa) + 2 * km
    b = kp * (1 - alfa) - km
    return km * (a + 2 * phi * b) / (a - phi * b)


def ler_fem():
    fem = {}
    for arq in (PASTA / "fem").glob("*.csv"):
        with arq.open(encoding="utf-8") as f:
            for l in csv.DictReader(f):
                eta = round(1 - float(l["t_R"]), 3)
                fem[(eta, round(100 * float(l["phi_externo"])))] = KM * float(l["k_sobre_km"])
    return fem


def main():
    fem = ler_fem()
    linhas, erros, erros_fem = [], [], []
    print(f"{'η':>6} {'φ%':>4} {'medido':>7} {'modelo':>7} {'FEM':>7} {'Felske':>7} {'desvio %':>8}")
    for (eta, p), med in MEDIDO.items():
        k = hj(p / 100, k_eq(KG, KA, eta**3))
        kf = fem.get((eta, p))
        desvio = 100 * (k - med) / med
        erros.append(abs(desvio))
        if kf is not None:
            erros_fem.append(abs(100 * (kf - med) / med))
        print(f"{eta:6.3f} {p:4d} {med:7.3f} {k:7.4f} {kf if kf else float('nan'):7.4f} "
              f"{FELSKE[(eta, p)]:7.3f} {desvio:8.1f}")
        linhas.append([eta, p, med, f"{k:.5f}", "" if kf is None else f"{kf:.5f}",
                       FELSKE[(eta, p)], PORFIRI[(eta, p)], f"{desvio:.2f}"])
    print(f"desvio absoluto médio {sum(erros) / len(erros):.1f} %, máximo {max(erros):.1f} %")
    if erros_fem:
        print(f"FEM: médio {sum(erros_fem) / len(erros_fem):.1f} %, máximo {max(erros_fem):.1f} % "
              f"({len(erros_fem)} casos)")
    with (PASTA / "validacao.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eta", "phi_percent", "k_medido", "k_modelo", "k_fem", "k_felske", "k_porfiri",
                    "desvio_modelo_percent"])
        w.writerows(linhas)


if __name__ == "__main__":
    main()
