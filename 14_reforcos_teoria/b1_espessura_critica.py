"""B1: espessura crítica de parede t_c, onde o compósito com esferas ocas cruza k_m.

k_eq(t_c) = k* = k_m/(1 − α)  ⇒  β = 0  ⇒  k_eff = k_m para QUALQUER φ (HJ).
Imprime t_c por material e os valores esperados das rodadas de FEM
(09_particulas_ocas/estudos/ocas/ocas.py --modo mesmo_phi) em torno de t_c.

Rodar:  python3 b1_espessura_critica.py
"""
import csv
from pathlib import Path

from base import K_AR, K_M, MATERIAIS, R_NM, alfa, hj, k_eq, k_knudsen, t_critico

PASTA = Path(__file__).resolve().parent

linhas = []
print("t_c (nm) e t_c/R, R = 50 nm, k_m = 0,2")
print(f"{'material':22s} {'k_p':>5s} | {'R_K 1e-8 ar':>12s} {'Knudsen':>10s} {'R_K 1e-10':>10s} | aprox. R k_m/(2 k_p)")
for chave in ("sio2", "zif8", "gama5", "gama", "anatase", "quartzo"):
    nome, kp, *_ = MATERIAIS[chave]
    a8, a10 = alfa(1e-8), alfa(1e-10)
    tc_ar = t_critico(kp, K_AR, a8)
    # Knudsen: o k do gás depende da própria cavidade (2R − 2t); duas iterações bastam
    tc_kn = tc_ar
    for _ in range(5):
        tc_kn = t_critico(kp, k_knudsen(2 * (R_NM - tc_kn)), a8)
    tc_10 = t_critico(kp, K_AR, a10)
    aprox = R_NM * K_M / (2 * kp)
    print(f"{nome:22s} {kp:5.2f} | {tc_ar:6.2f} ({tc_ar / R_NM:.3f}) {tc_kn:6.2f} {tc_10:9.2f}  | {aprox:6.2f}")
    linhas.append([chave, kp, f"{tc_ar:.3f}", f"{tc_ar / R_NM:.4f}", f"{tc_kn:.3f}", f"{tc_10:.3f}", f"{aprox:.3f}"])
with (PASTA / "b1_t_critico.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["material", "k_p", "t_c_nm", "t_c_R", "t_c_knudsen_nm", "t_c_rk1e-10_nm", "aprox_nm"])
    w.writerows(linhas)

print("\nValores esperados das rodadas de FEM (mesmo φ externo, R_K = 1e-8, núcleo de ar 0,026):")
for chave, esp in (("zif8", (0.40, 0.3187, 0.25)), ("sio2", (0.10, 0.068, 0.05))):
    nome, kp, *_ = MATERIAIS[chave]
    print(f"  {nome}")
    for tR in esp:
        keq = k_eq(kp, K_AR, (1 - tR) ** 3)
        vals = "  ".join(f"φ={p:.2f}: {hj(p, alfa(), keq):.5f}" for p in (0.05, 0.10, 0.20))
        print(f"    t/R = {tR:<6g} k_eq = {keq:.4f}   {vals}")
