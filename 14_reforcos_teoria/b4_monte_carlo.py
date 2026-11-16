"""B4: propagação de incertezas por Monte Carlo (HJ + esfera revestida + nanocristal).

1e5 amostras por arquitetura. Log-uniforme para R_K e R_gb, uniforme para o resto:
  R_K 3e-9–1e-7 m²K/W; k_p SiO2 1,1–1,5, γ-Al2O3 1–10, ZIF-8 0,2–0,45; k_m 0,17–0,25;
  ρ_tinta 1,2–1,6 g/cm³; R_gb 5e-10–1e-8; k do gás 0,007–0,026; R 25–100 nm.
5 % em massa fixo; t/R fixo em cada arquitetura. A casca granular usa d = 3 nm
e o ℓ de nanocristal.py, com k0 = k_p sorteado (ℓ escala com k0).

Saídas: b4_monte_carlo.csv (resumo) e b4_sensibilidade.csv (Spearman).
Rodar:  python3 b4_monte_carlo.py [--n 100000] [--semente 1]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from base import MATERIAIS, beta, k_eq, k_nc

PASTA = Path(__file__).resolve().parent
FAIXAS_KP = {"sio2": (1.1, 1.5), "gama": (1.0, 10.0), "zif8": (0.2, 0.45)}
ARQ = [  # nome, material, t/R, d (nm) ou None
    ("SiO2 maciça", "sio2", 1.0, None),
    ("γ-Al2O3 maciça", "gama", 1.0, None),
    ("SiO2 oca t/R 0,2", "sio2", 0.2, None),
    ("SiO2 oca t/R 0,1", "sio2", 0.1, None),
    ("SiO2 oca t/R 0,05", "sio2", 0.05, None),
    ("SiO2 oca granular t/R 0,1, d 3 nm", "sio2", 0.1, 3.0),
    ("γ-Al2O3 oca granular t/R 0,1, d 3 nm", "gama", 0.1, 3.0),
    ("ZIF-8 oca t/R 0,2", "zif8", 0.2, None),
    ("ZIF-8 oca t/R 0,1", "zif8", 0.1, None),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--semente", type=int, default=1)
    args = p.parse_args()
    g = np.random.default_rng(args.semente)
    n = args.n
    logu = lambda a, b: np.exp(g.uniform(np.log(a), np.log(b), n))
    comuns = {"R_K": logu(3e-9, 1e-7), "k_m": g.uniform(0.17, 0.25, n), "rho_tinta": g.uniform(1.2, 1.6, n),
              "R_gb": logu(5e-10, 1e-8), "k_gas": g.uniform(0.007, 0.026, n), "R_nm": g.uniform(25, 100, n)}
    kps = {m: g.uniform(*f, n) for m, f in FAIXAS_KP.items()}
    resumo, sens = [], []
    print(f"{'arquitetura':40s} {'mediana':>8s} {'p5':>7s} {'p95':>7s} {'P(k<k_m)':>9s}")
    for nome, mat, tR, d in ARQ:
        _, k0_ref, ell_ref, rho_p = MATERIAIS[mat]
        kp = kps[mat]
        par = dict(comuns, k_p=kp)
        if d:
            ell = ell_ref * kp / k0_ref
            kp = np.array([k_nc(d, k0, l, r) for k0, l, r in zip(kp, ell, comuns["R_gb"])])
        km, rk, R = comuns["k_m"], comuns["R_K"], comuns["R_nm"] * 1e-9
        phi_sol = (0.05 / rho_p) / (0.05 / rho_p + 0.95 / comuns["rho_tinta"])
        if tR < 1:
            c = (1 - tR) ** 3
            keq = k_eq(kp, comuns["k_gas"], c)
            phi = phi_sol / (1 - c)
        else:
            keq, phi = kp, phi_sol
        a = rk * km / R
        b = beta(keq, a, km)
        k = (1 + 2 * phi * b) / (1 - phi * b)
        p5, med, p95 = np.percentile(k, [5, 50, 95])
        prob = np.mean(k < 1)
        print(f"{nome:40s} {med:8.4f} {p5:7.4f} {p95:7.4f} {prob:9.1%}")
        resumo.append([nome, f"{med:.5f}", f"{p5:.5f}", f"{p95:.5f}", f"{prob:.4f}"])
        usados = ["R_K", "k_p", "k_m", "rho_tinta", "R_nm"] + (["k_gas"] if tR < 1 else []) + (["R_gb"] if d else [])
        linha = {u: spearmanr(par[u], k)[0] for u in usados}
        sens.append([nome] + [f"{linha.get(u, float('nan')):.3f}" for u in
                              ("R_K", "k_p", "k_m", "rho_tinta", "R_nm", "k_gas", "R_gb")])
    with (PASTA / "b4_monte_carlo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquitetura", "mediana_k_sobre_km", "p5", "p95", "P_k_menor_km"])
        w.writerows(resumo)
    with (PASTA / "b4_sensibilidade.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquitetura", "R_K", "k_p", "k_m", "rho_tinta", "R_nm", "k_gas", "R_gb"])
        w.writerows(sens)
    print("\nSpearman (correlação de postos com k/k_m):")
    print(f"{'':40s} " + " ".join(f"{c:>8s}" for c in ("R_K", "k_p", "k_m", "ρ_tinta", "R", "k_gás", "R_gb")))
    for s in sens:
        print(f"{s[0]:40s} " + " ".join(f"{x:>8s}" for x in s[1:]))


if __name__ == "__main__":
    main()
