"""Teoria das partículas ocas: Hasselman–Johnson com a esfera oca equivalente.

Uma esfera oca (casca de óxido k_p entre r_i e R, núcleo de gás k_n) responde
a um gradiente uniforme como uma esfera maciça de condutividade equivalente
(esfera revestida, solução exata do problema de dipolo):

    k_eq = k_p [ (k_n + 2k_p) + 2c (k_n − k_p) ] / [ (k_n + 2k_p) − c (k_n − k_p) ],
    c = (r_i/R)³.

A resistência de Kapitza na superfície externa fica em série com isso, e HJ
com k_p → k_eq é exato no limite diluído (é o mesmo que Maxwell com
k_eq/(1 + α k_eq/k_m), α = R_K k_m/R). A temperatura é contínua na parede
interna; uma resistência casca/gás ali não muda nada, porque o gás quase não
conduz.

Duas comparações com a maciça:
  * mesmo φ externo (mesmo número e tamanho de partículas);
  * mesma massa de óxido (φ_sólido fixo, como nos 5 % em massa da tinta):
    a oca ocupa φ_externo = φ_sólido/(1 − c), mais volume.

Rodar de 09_particulas_ocas:  python3 estudos/ocas/teoria.py
Saídas em estudos/ocas/: teoria_ocas.csv e teoria_ocas.png.
"""

import csv
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
K_M = 0.2
RAIO = 50e-9


def k_equivalente(k_p, k_n, c):
    return k_p * ((k_n + 2 * k_p) + 2 * c * (k_n - k_p)) / ((k_n + 2 * k_p) - c * (k_n - k_p))


def hasselman_johnson(phi, alfa, k_m, k_p):
    a = k_p * (1 + 2 * alfa) + 2 * k_m
    b = k_p * (1 - alfa) - k_m
    return k_m * (a + 2 * phi * b) / (a - phi * b)


def k_oca(phi, rk, t_R, k_p=5.0, k_n=0.026, k_m=K_M, raio=RAIO):
    """k_eff de HJ para partículas ocas com parede t/R (t/R = 1 é a maciça)."""
    c = (1 - t_R) ** 3
    return hasselman_johnson(phi, rk * k_m / raio, k_m, k_equivalente(k_p, k_n, c))


def espessura_neutra(rk, k_p, k_n, phi=0.05):
    """t/R em que a partícula oca não muda k (k_eff = k_m), ou None."""
    t = np.logspace(-4, 0, 4001)
    k = np.array([k_oca(phi, rk, x, k_p, k_n) for x in t])
    troca = np.where(np.diff(np.sign(k - K_M)) != 0)[0]
    return float(t[troca[0]]) if len(troca) else None


def main() -> None:
    t = np.logspace(-2, 0, 200)
    casos = [("Rₖ = 10⁻¹⁰, kₚ = 5", 1e-10, 5.0, 0.026, "#c2410c", "-"),
             ("Rₖ = 10⁻⁸, kₚ = 5", 1e-8, 5.0, 0.026, "#2f5d9b", "-"),
             ("Rₖ = 10⁻⁸, kₚ = 1 (parede nanocristalina)", 1e-8, 1.0, 0.026, "#2f5d9b", "--"),
             ("Rₖ = 10⁻⁸, kₚ = 5, gás de Knudsen (kₙ = 0,005)", 1e-8, 5.0, 0.005, "#8fa6c8", ":")]
    phi_externo, phi_solido = 0.05, 0.02

    linhas = []
    print(f"Limite isolante (núcleo de vácuo, parede → 0) em φ = {phi_externo:.0%}: "
          f"k/k_m = {(1 - phi_externo) / (1 + phi_externo / 2):.4f}\n")
    for rotulo, rk, k_p, k_n, *_ in casos:
        macica = k_oca(phi_externo, rk, 1.0, k_p, k_n)
        neutra = espessura_neutra(rk, k_p, k_n)
        texto = f"t/R = {neutra:.4f} ({neutra * 50:.2f} nm em R = 50 nm)" if neutra else "nunca"
        print(f"{rotulo}: maciça k/k_m = {macica / K_M:.4f}; neutra em {texto}")
        for x in t:
            c = (1 - x) ** 3
            k_mesmo_phi = k_oca(phi_externo, rk, x, k_p, k_n)
            phi_ext = phi_solido / (1 - c)
            k_mesma_massa = k_oca(phi_ext, rk, x, k_p, k_n) if phi_ext < 0.5 else float("nan")
            linhas.append([rotulo, rk, k_p, k_n, f"{x:.5f}", f"{k_mesmo_phi / K_M:.6f}",
                           f"{phi_ext:.5f}", f"{k_mesma_massa / K_M:.6f}"])
    with (PASTA / "teoria_ocas.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["caso", "R_K", "k_p", "k_nucleo", "t_R", "k_sobre_km_phi5",
                           "phi_externo_massa2", "k_sobre_km_massa2"])
        escritor.writerows(linhas)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200, sharey=False)
    for rotulo, rk, k_p, k_n, cor, estilo in casos:
        eixos[0].plot(t, [k_oca(phi_externo, rk, x, k_p, k_n) / K_M for x in t], estilo, color=cor,
                      lw=2, label=rotulo)
        y = []
        for x in t:
            phi_ext = phi_solido / (1 - (1 - x) ** 3)
            y.append(k_oca(phi_ext, rk, x, k_p, k_n) / K_M if phi_ext < 0.5 else np.nan)
        eixos[1].plot(t, y, estilo, color=cor, lw=2, label=rotulo)
    for eixo, titulo in ((eixos[0], f"Mesmo φ externo ({phi_externo:.0%})"),
                         (eixos[1], f"Mesma massa de óxido (φ sólido = {phi_solido:.0%})")):
        eixo.axhline(1, color="#6b7385", lw=1)
        eixo.set_xscale("log")
        eixo.set_xlabel("espessura da parede t/R  (t/R = 1: maciça)")
        eixo.set_ylabel("k_eff / k_m")
        eixo.set_title(titulo)
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.text(1.0, 1.0, "maciça", transform=eixo.get_xaxis_transform(), ha="right", va="top",
                  fontsize=8, color="#6b7385")
    t_20 = 1 - (1 - phi_solido / 0.2) ** (1 / 3)  # parede em que φ externo chega a 20 %
    eixos[1].axvspan(0.01, t_20, color="#6b7385", alpha=0.12, lw=0)
    eixos[1].text(0.0105, 0.02, "φ externo > 20 %:\nHJ diluído\npouco confiável", fontsize=7.5,
                  color="#6b7385", transform=eixos[1].get_xaxis_transform(), va="bottom")
    eixos[0].text(0.0105, 1.0005, "sem partículas", fontsize=8, color="#6b7385", va="bottom")
    eixos[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Partículas ocas (núcleo de gás) contra maciças: HJ com a esfera oca equivalente, R = 50 nm",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PASTA / "teoria_ocas.png")
    print(f"\nFigura: {PASTA / 'teoria_ocas.png'}")


if __name__ == "__main__":
    main()
