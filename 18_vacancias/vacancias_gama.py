"""Estimativa de Callaway: quanto as vacâncias catiônicas da γ-Al2O3 baixam k_p,
sozinhas e junto com o espalhamento nos contornos de grão (d = 3,7 nm).

Modelo de Debye com uma velocidade média e a regra de Matthiessen:
    1/τ = 1/τ_U + 1/τ_V + 1/τ_B
    1/τ_U = B ω² T exp(−Θ/3T)            (Umklapp, B ajustado para k0)
    1/τ_V = V0 Γ ω⁴ / (4π v³)            (Klemens, defeitos pontuais)
    1/τ_B = v / d                        (contorno de grão)
Referência sem defeitos: rede de alumina densa com k0 = 35 W/(m K) (safira).
Vacâncias: Γ = x (ΔM/M̄ + 2)² com ΔM = M_Al (Ratsifaritana & Klemens 1987),
x = 2,67/56 sítios vagos por célula de espinélio → Γ ≈ 0,5 (limite superior;
Γ 0,05–0,5 varrido). Piso: condutividade mínima de Cahill–Pohl.
Uso: python3 18_vacancias/vacancias_gama.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

hbar, kB = 1.054571817e-34, 1.380649e-23
T = 300.0
rho, M_mol, nat = 3.65e3, 0.10196, 5          # γ-Al2O3
n = rho / M_mol * nat * 6.02214076e23          # átomos/m³
vL, vT = 10.0e3, 5.9e3                         # m/s (γ, ~8–10% abaixo da safira)
v = (3 / (1 / vL**3 + 2 / vT**3)) ** (1 / 3)
wD = v * (6 * np.pi**2 * n) ** (1 / 3)
Theta = hbar * wD / kB
V0 = 1 / n


def k_callaway(B, Gamma=0.0, d=np.inf):
    def f(w):
        x = hbar * w / (kB * T)
        C = kB * x**2 * np.exp(x) / np.expm1(x) ** 2       # por modo
        dos = 3 * w**2 / (2 * np.pi**2 * v**3)             # 3 ramos
        inv = B * w**2 * T * np.exp(-Theta / (3 * T)) + V0 * Gamma * w**4 / (4 * np.pi * v**3)
        if np.isfinite(d):
            inv += v / d
        return C * dos * v**2 / inv / 3
    return quad(f, 1e9, wD, limit=200)[0]


def k_min():
    """Cahill–Pohl (1992), três ramos."""
    tot = 0.0
    for vi in (vL, vT, vT):
        Ti = vi * hbar / kB * (6 * np.pi**2 * n) ** (1 / 3)
        I = quad(lambda x: x**3 * np.exp(x) / np.expm1(x) ** 2, 1e-6, Ti / T)[0]
        tot += vi * (T / Ti) ** 2 * I
    return (np.pi / 6) ** (1 / 3) * kB * n ** (2 / 3) * tot


k0 = 35.0
B = 10 ** brentq(lambda lb: k_callaway(10 ** lb) - k0, -22, -15)
x_vac = 2.67 / 56
Gmax = x_vac * (26.98 / (M_mol * 1e3 / nat) + 2) ** 2
print(f"v = {v:.0f} m/s, Θ = {Theta:.0f} K, k_min = {k_min():.2f} W/mK, Γ(vacâncias) ≈ {Gmax:.2f}")
print(f"{'Γ':>6} {'k (grão ∞)':>11} {'k (d=3,7 nm)':>13} {'k (d=3,7, Γ=0)':>15}")
kb0 = k_callaway(B, 0, 3.7e-9)
for G in (0, 0.05, 0.1, 0.2, Gmax):
    print(f"{G:6.2f} {k_callaway(B, G):11.2f} {k_callaway(B, G, 3.7e-9):13.2f} {kb0:15.2f}")
