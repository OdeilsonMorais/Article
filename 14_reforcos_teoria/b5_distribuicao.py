"""B5: distribuição de tamanhos (teoria).

HJ generalizado para uma mistura de raios no mesmo φ total:
    (k − k_m)/(k + 2k_m) = Σ φ_i β_i,  β_i = [k_p(1−α_i) − k_m]/[k_p(1+2α_i) + 2k_m],  α_i = R_K k_m/R_i
Distribuição log-normal em NÚMERO, com mediana R_g e desvio geométrico σ_g; o
volume de cada classe pesa φ_i (maciça: ∝ n R³; oca na mesma massa: a massa de
casca de cada classe vai para φ_sólido, e φ_ext,i = φ_sol,i/(1 − c_i)).
Dois casos para as ocas: t/R fixo (todas igualmente ocas) e t fixo em nm (as
pequenas ficam mais "cheias").
Também calcula as células bidispersas R = 25 + 75 nm (meio a meio em volume
de sólido) que a rodada de FEM vai conferir.

Saídas: b5_distribuicao.csv e figuras/b5_distribuicao.png.
Rodar:  python3 b5_distribuicao.py
"""
import csv
from pathlib import Path

import numpy as np

from base import K_AR, K_M, MATERIAIS, beta, k_eq, phi_solido

PASTA = Path(__file__).resolve().parent
RK = 1e-8


def mistura(raios_nm, massa_rel, kp, phi_sol_total, t_R=None, t_nm=None, kn=K_AR, rk=RK):
    """k/k_m para classes de raio com massa relativa massa_rel (soma 1)."""
    soma = 0.0
    phi_ext_total = 0.0
    for R, m in zip(raios_nm, massa_rel):
        if t_R is None and t_nm is None:
            tR = 1.0
        elif t_R is not None:
            tR = t_R
        else:
            tR = min(1.0, t_nm / R)
        c = (1 - tR) ** 3
        keq = k_eq(kp, kn, c) if tR < 1 else kp
        phi_i = phi_sol_total * m / (1 - c)
        a = rk * K_M / (R * 1e-9)
        soma += phi_i * beta(keq, a)
        phi_ext_total += phi_i
    return (1 + 2 * soma) / (1 - soma), phi_ext_total


def lognormal(R_g, s_g, n=400):
    """Classes de raio e fração de VOLUME de sólido maciço (∝ n R³)."""
    ls = np.log(s_g)
    x = np.linspace(np.log(R_g) - 4 * ls, np.log(R_g) + 4 * ls, n) if s_g > 1 else np.array([np.log(R_g)])
    R = np.exp(x)
    pn = np.exp(-0.5 * ((x - np.log(R_g)) / ls) ** 2) if s_g > 1 else np.array([1.0])
    return R, pn


def massa_rel(R, pn, t_R=None, t_nm=None):
    """Massa de casca de cada classe (∝ n·(R³ − r_i³))."""
    if t_R is None and t_nm is None:
        v = R**3
    elif t_R is not None:
        v = R**3 * (1 - (1 - t_R) ** 3)
    else:
        v = R**3 - np.clip(R - t_nm, 0, None) ** 3
    w = pn * v
    return w / w.sum()


linhas = []
print("k/k_m, 5 % em massa, R_K = 1e-8, mediana em número R_g = 50 nm")
casos = [("SiO2 maciça", "sio2", None, None), ("γ-Al2O3 maciça (k=5)", "gama5", None, None),
         ("SiO2 oca t/R 0,1", "sio2", 0.1, None), ("SiO2 oca t = 5 nm", "sio2", None, 5.0),
         ("SiO2 oca t/R 0,05", "sio2", 0.05, None), ("SiO2 oca t = 2,5 nm", "sio2", None, 2.5),
         ("ZIF-8 oca t/R 0,2", "zif8", 0.2, None), ("ZIF-8 oca t = 10 nm", "zif8", None, 10.0)]
sigmas = (1.0, 1.2, 1.5, 2.0)
print(f"{'caso':24s} " + " ".join(f"σg={s:<4g}" for s in sigmas))
for nome, mat, tR, tnm in casos:
    _, kp, _, rho = MATERIAIS[mat]
    ps = phi_solido(rho)
    vals = []
    for s in sigmas:
        R, pn = lognormal(50.0, s)
        k, phi = mistura(R, massa_rel(R, pn, tR, tnm), kp, ps, tR, tnm)
        vals.append(f"{k:.4f}" + ("*" if phi > 0.30 else " "))
        linhas.append([nome, s, f"{k:.5f}", f"{phi:.5f}"])
    print(f"{nome:24s} " + " ".join(f"{v:>9s}" for v in vals))

print("* φ externo acima de 30 %: fora da validade de HJ. Com t fixo em nm, as partículas grandes têm\n"
      "  t/R pequeno, ocupam muito volume e dominam a mistura; o ganho vem de φ_ext alto, não da física nova.")
print("\nCélulas bidispersas para o FEM (R = 25 e 75 nm, metade da massa em cada; R_K = 1e-8):")
bid = []
for nome, mat, tR in (("SiO2 maciça", "sio2", None), ("γ-Al2O3 maciça (k=5)", "gama5", None),
                      ("SiO2 oca t/R 0,2", "sio2", 0.2)):
    _, kp, _, rho = MATERIAIS[mat]
    ps = phi_solido(rho)
    kb, phib = mistura([25.0, 75.0], [0.5, 0.5], kp, ps, tR)
    k25, _ = mistura([25.0], [1.0], kp, ps, tR)
    k75, _ = mistura([75.0], [1.0], kp, ps, tR)
    k50, _ = mistura([50.0], [1.0], kp, ps, tR)
    print(f"  {nome:22s} bidispersa {kb:.5f} (φ_ext {phib:.4f}) | só 25 nm {k25:.5f} | só 50 {k50:.5f} | só 75 {k75:.5f}")
    bid.append([nome, f"{kb:.5f}", f"{phib:.5f}", f"{k25:.5f}", f"{k50:.5f}", f"{k75:.5f}"])
with (PASTA / "b5_distribuicao.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["caso", "sigma_g", "k_sobre_km", "phi_externo"])
    w.writerows(linhas)
    w.writerow([])
    w.writerow(["bidispersa_25_75", "k_mistura", "phi_ext", "k_so_25", "k_so_50", "k_so_75"])
    w.writerows(bid)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
sg = np.linspace(1.0, 2.2, 40)
for nome, mat, tR, tnm in casos:
    _, kp, _, rho = MATERIAIS[mat]
    ps = phi_solido(rho)
    ys = []
    for s in sg:
        R, pn = lognormal(50.0, s)
        ys.append(mistura(R, massa_rel(R, pn, tR, tnm), kp, ps, tR, tnm)[0])
    ax.plot(sg, ys, "--" if tnm else "-", lw=2, label=nome)
ax.axhline(1, color="0.4", lw=1)
ax.set_xlabel("desvio geométrico σ_g (log-normal em número, mediana 50 nm)")
ax.set_ylabel("k_eff / k_m (5 % em massa)")
ax.legend(frameon=False, fontsize=8.5, ncol=2)
(PASTA / "figuras").mkdir(exist_ok=True)
fig.savefig(PASTA / "figuras" / "b5_distribuicao.png")
print("Figura: figuras/b5_distribuicao.png")
