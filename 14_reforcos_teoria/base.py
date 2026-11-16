"""Fórmulas comuns aos reforços do artigo (mesmas das pastas 09, 11 e 12).

Esfera revestida (oca) equivalente + Hasselman–Johnson, modelo nanocristalino
de duas escalas e gás de Knudsen. Unidades SI, exceto onde indicado (nm).
"""
import numpy as np

K_M, RHO_M, R_NM, RK, W = 0.2, 1.5, 50.0, 1e-8, 0.05
K_AR, LIVRE_NM, BETA = 0.026, 68.0, 1.64

# k0 (W/mK), ℓ cinzento (nm), densidade (g/cm³): os mesmos de 12_nanocristalino/nanocristal.py
MATERIAIS = {
    "sio2": ("SiO₂ amorfa", 1.36, 0.6, 2.20),
    "gama": ("γ-Al₂O₃ (k0 = 10)", 10.0, 2.0, 3.65),
    "gama5": ("γ-Al₂O₃ (k0 = 5)", 5.0, 1.0, 3.65),
    "quartzo": ("quartzo", 8.0, 3.0, 2.65),
    "anatase": ("TiO₂ anatásio", 8.5, 2.0, 3.89),
    "zif8": ("ZIF-8", 0.33, None, 0.95),
}


def alfa(rk=RK, km=K_M, r_nm=R_NM):
    return rk * km / (r_nm * 1e-9)


def beta(kp, a, km=K_M):
    """Polarizabilidade de HJ: (k − k_m)/(k + 2k_m) = φ·β."""
    return (kp * (1 - a) - km) / (kp * (1 + 2 * a) + 2 * km)


def hj(phi, a, kp, km=K_M):
    """k_eff/k_m de Hasselman–Johnson."""
    b = beta(kp, a, km)
    return (1 + 2 * phi * b) / (1 - phi * b)


def k_eq(kp, kn, c):
    """Esfera revestida: casca kp, núcleo kn, c = (r_i/R)³."""
    return kp * ((kn + 2 * kp) + 2 * c * (kn - kp)) / ((kn + 2 * kp) - c * (kn - kp))


def phi_solido(rho_p, w=W, rho_m=RHO_M):
    return (w / rho_p) / (w / rho_p + (1 - w) / rho_m)


def k_knudsen(d_cav_nm, k0=K_AR, livre=LIVRE_NM, b=BETA):
    """Kaganer: k = k0/(1 + 2β Kn), Kn = livre caminho/diâmetro da cavidade."""
    return k0 / (1 + 2 * b * livre / d_cav_nm)


def k_nc(d_nm, k0, ell_nm, rgb):
    """Duas escalas (pasta 12): Matthiessen no grão + R_gb em série a cada d."""
    kg = k0 / (1 + ell_nm / d_nm)
    return kg / (1 + rgb * kg / (d_nm * 1e-9))


def k_oca(phi_sol, tR, kp, kn, a, km=K_M):
    """k_eff/k_m da oca na mesma massa (φ_externo = φ_sólido/(1 − c))."""
    c = (1 - tR) ** 3
    keq = k_eq(kp, kn, c) if tR < 1 else kp
    return hj(phi_sol / (1 - c), a, keq, km), phi_sol / (1 - c)


def delta_c(kp, kc, kstar):
    """Fração de vazio (r_i/R)³ em que k_eq = k*."""
    return (2 * kp + kc) * (kp - kstar) / ((kp - kc) * (2 * kp + kstar))


def t_critico(kp, kc, a, r_nm=R_NM, km=K_M):
    """Espessura crítica (nm): o compósito cruza k_m quando k_eq = k_m/(1 − α)."""
    kstar = km / (1 - a)
    if kp <= kstar:
        return float("inf")  # a casca maciça já não aumenta k
    d = delta_c(kp, kc, kstar)
    return r_nm * (1 - d ** (1 / 3))
