"""B3: efeito de tamanho dos fônons na parede fina (só teoria).

Hoje a casca densa usa o k de volume. Numa parede de espessura t o
espalhamento difuso nas duas superfícies reduz o k no plano da casca.
Dois modelos, ambos cinzentos (um só ℓ):
  Matthiessen:      k = k0 / (1 + ℓ/t)
  Fuchs–Sondheimer: k/k0 = 1 − (3ℓ/2t) ∫₁^∞ (1/x³ − 1/x⁵)(1 − e^(−x t/ℓ)) dx  (p = 0, difuso)
Com grãos de tamanho d (pasta 12): ℓ_g = 1/(1/ℓ + 1/d) dentro do grão,
o filme reduz com ℓ_g, e R_gb entra em série a cada d.

Saídas: b3_tamanho_parede.csv e figuras/b3_tamanho_parede.png.
Rodar:  python3 b3_tamanho_parede.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from base import K_AR, K_M, MATERIAIS, R_NM, alfa, k_eq, k_oca, phi_solido

PASTA = Path(__file__).resolve().parent


def fs(t_nm, ell_nm):
    """Fuchs–Sondheimer, superfícies difusas: k_filme/k_volume."""
    kappa = t_nm / ell_nm
    integ, _ = quad(lambda x: (1 / x**3 - 1 / x**5) * (1 - np.exp(-kappa * x)), 1, np.inf)
    return 1 - 3 / (2 * kappa) * integ


def k_parede(t_nm, k0, ell, modelo="fs", d_nm=None, rgb=0.0):
    """k da casca com espessura t (nm), opcionalmente nanocristalina (d, R_gb)."""
    if ell is None:  # ZIF-8: sem ℓ, sem efeito
        return k0
    ell_g = ell if d_nm is None else 1 / (1 / ell + 1 / d_nm)
    kg = k0 / (1 + ell / d_nm) if d_nm else k0
    if modelo == "fs":
        k = kg * fs(t_nm, ell_g)
    elif modelo == "matthiessen":
        k = kg / (1 + ell_g / t_nm)
    else:
        k = kg
    if d_nm and rgb:
        k = k / (1 + rgb * k / (d_nm * 1e-9))
    return k


def t_c(k0, ell, modelo, d_nm=None, rgb=0.0, a=alfa()):
    kstar = K_M / (1 - a)

    def f(t):
        kp = k_parede(t, k0, ell, modelo, d_nm, rgb)
        return k_eq(kp, K_AR, (1 - t / R_NM) ** 3) - kstar
    try:
        return brentq(f, 0.05, R_NM * 0.999)
    except ValueError:
        return float("nan")


casos = [("sio2", None, 0.0), ("gama5", None, 0.0), ("gama", None, 0.0), ("quartzo", None, 0.0),
         ("anatase", None, 0.0), ("sio2", 3.0, 1e-9), ("gama5", 3.0, 1e-9)]
linhas = []
print("t_c (nm): volume / Matthiessen / Fuchs–Sondheimer   (R = 50 nm, R_K = 1e-8, núcleo de ar)")
for chave, d, rgb in casos:
    nome, k0, ell, rho = MATERIAIS[chave]
    vals = [t_c(k0, ell, m, d, rgb) for m in ("volume", "matthiessen", "fs")]
    rot = nome + (f", d = {d:g} nm, R_gb = {rgb:g}" if d else "")
    print(f"  {rot:42s} {vals[0]:6.2f}  {vals[1]:6.2f}  {vals[2]:6.2f}")
    phis = phi_solido(rho)
    for tR in (0.2, 0.1, 0.05):
        t = tR * R_NM
        if d and d > t:
            continue
        ks = [k_parede(t, k0, ell, m, d, rgb) for m in ("volume", "matthiessen", "fs")]
        kk = [k_oca(phis, tR, kp, K_AR, alfa())[0] for kp in ks]
        linhas.append([chave, d or "", rgb, tR, t, *(f"{x:.4f}" for x in ks), *(f"{x:.5f}" for x in kk),
                       *(f"{x:.3f}" for x in vals)])
with (PASTA / "b3_tamanho_parede.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["material", "d_nm", "R_gb", "t_R", "t_nm", "k_p_volume", "k_p_matthiessen", "k_p_fs",
                "k_sobre_km_volume", "k_sobre_km_matthiessen", "k_sobre_km_fs",
                "t_c_volume_nm", "t_c_matthiessen_nm", "t_c_fs_nm"])
    w.writerows(linhas)
print("\nk/k_m na mesma massa (5 %): volume / Matthiessen / FS")
for l in linhas:
    print(f"  {l[0]:8s} d={str(l[1]):>3s} t/R={l[3]:<5g} k_p {l[5]}/{l[6]}/{l[7]}  k/k_m {l[8]}/{l[9]}/{l[10]}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
tR = np.linspace(0.02, 0.5, 120)
cores = {"sio2": "#2f5d9b", "gama": "#c2410c", "quartzo": "#15803d"}
for chave, cor in cores.items():
    nome, k0, ell, rho = MATERIAIS[chave]
    phis = phi_solido(rho)
    for m, ls in (("volume", "--"), ("fs", "-")):
        kp = np.array([k_parede(x * R_NM, k0, ell, m) for x in tR])
        ax1.plot(tR * R_NM, kp / k0, ls, color=cor, lw=2,
                 label=f"{nome}, {'volume' if m == 'volume' else 'Fuchs–Sondheimer'}")
        ax2.plot(tR, [k_oca(phis, x, k, K_AR, alfa())[0] for x, k in zip(tR, kp)], ls, color=cor, lw=2)
ax1.set_xlabel("espessura da parede t (nm)")
ax1.set_ylabel("k_parede / k_volume")
ax1.set_xscale("log")
ax1.legend(frameon=False, fontsize=8.5)
ax2.axhline(1, color="0.4", lw=1)
ax2.set_xlabel("t/R")
ax2.set_ylabel("k_eff / k_m (5 % em massa)")
ax2.set_ylim(0.8, 1.35)
(PASTA / "figuras").mkdir(exist_ok=True)
fig.savefig(PASTA / "figuras" / "b3_tamanho_parede.png")
print("Figura: figuras/b3_tamanho_parede.png")
