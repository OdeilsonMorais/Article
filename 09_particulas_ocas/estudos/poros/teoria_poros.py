"""Bolsões de ar na matriz, sem partículas: curva k × porosidade e mapa poros × partículas.

Um poro é uma esfera de ar (k_ar = 0,026 W/mK) sem resistência de interface.
Duas figuras, só NumPy/Matplotlib:

1. ``poros_curva.png``: k_eff/k_m contra a porosidade, por Maxwell (que, para
   poros mais isolantes que a matriz, é também o limite superior de
   Hashin–Shtrikman), pelo esquema diferencial (Bruggeman assimétrico, melhor
   em porosidade alta) e com poros de vácuo. Os pontos do FEM vêm de
   ``../ocas/poros.csv`` (uma esfera por célula), quando o arquivo existe. As
   faixas horizontais são as medidas da bancada (Al₂O₃ 0,7 ± 0,1; híbrido 0,5 ± 0,1).

2. ``poros_mapa.png``: k_eff/k_m em função da porosidade e da fração de
   partículas, por Maxwell com três fases (matriz, poros e partículas maciças com
   Kapitza). A linha tracejada marca φ_partículas = 2 % (5 % em massa da tinta).
   Mostra quanto vazio seria preciso, com as partículas presentes, para chegar
   nos valores medidos.

Rodar de 09_particulas_ocas:  python3 estudos/poros/teoria_poros.py
Opções: --k-particula 5, --rk 1e-8, --raio 50e-9, --k-ar 0.026.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
K_M = 0.2
TINTA = "#1f2a44"
TINTA_SECUNDARIA = "#5b6474"
GRADE = "#d9dde5"
AZUL, LARANJA, AZUL_CLARO = "#2f5d9b", "#c2410c", "#8fa6c8"
MEDIDAS = [("Al₂O₃ medido", 0.7, 0.1), ("híbrido medido", 0.5, 0.1)]


def beta(k_i, k_m=K_M):
    return (k_i - k_m) / (k_i + 2 * k_m)


def maxwell(fases, k_m=K_M):
    """Maxwell com várias fases: fases = [(φ_i, k_i efetivo), ...]."""
    s = sum(phi * beta(k, k_m) for phi, k in fases)
    return k_m * (1 + 2 * s) / (1 - s)


def diferencial(phi, k_d, k_m=K_M):
    """Esquema diferencial para esferas: (k − k_d)/(k_m − k_d)·(k_m/k)^(1/3) = 1 − φ."""
    if phi <= 0:
        return k_m
    a, b = min(k_d, k_m) + 1e-12, max(k_d, k_m)
    f = lambda k: (k - k_d) / (k_m - k_d) * (k_m / k) ** (1 / 3) - (1 - phi)
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def porosidade_para(alvo, modelo):
    """Porosidade em que k/k_m = alvo (bisseção em [0, 0,9])."""
    a, b = 0.0, 0.9
    for _ in range(100):
        m = 0.5 * (a + b)
        if modelo(m) / K_M > alvo:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def ler_fem(arquivo: Path):
    if not arquivo.exists():
        return []
    with arquivo.open(encoding="utf-8") as f:
        return sorted((float(l["phi_externo"]), float(l["k_xx"]) / K_M) for l in csv.DictReader(f))


def estilo(plt):
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.edgecolor": TINTA_SECUNDARIA, "axes.labelcolor": TINTA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
        "text.color": TINTA, "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200,
    })


def curva(plt, k_ar, fem, destino):
    p = np.linspace(0, 0.5, 251)
    k_max = np.array([maxwell([(x, k_ar)]) for x in p]) / K_M
    k_dif = np.array([diferencial(x, k_ar) for x in p]) / K_M
    k_vac = np.array([maxwell([(x, 0.0)]) for x in p]) / K_M
    fig, eixo = plt.subplots(figsize=(8.5, 5.6), layout="constrained")
    for rotulo, centro, meia in MEDIDAS:
        eixo.axhline(centro, color=LARANJA, lw=1, ls=(0, (4, 3)))
        eixo.text(3, centro + 0.008, f"{rotulo} ({centro:g} ± {meia:g})".replace(".", ","),
                  ha="left", va="bottom", fontsize=9, color=TINTA_SECUNDARIA)
    eixo.plot(p * 100, k_max, color=AZUL, lw=2, label="Maxwell (= limite superior HS)")
    eixo.plot(p * 100, k_dif, color=AZUL, lw=2, ls="--", label="diferencial (Bruggeman)")
    eixo.plot(p * 100, k_vac, color=AZUL_CLARO, lw=2, ls=":", label="poros de vácuo (Maxwell)")
    if fem:
        x, y = zip(*fem)
        eixo.plot(np.array(x) * 100, y, "o", ms=8, color=TINTA, mfc="white", mew=1.8,
                  label="FEM, uma esfera por célula")
    for _, centro, _ in MEDIDAS:
        px = porosidade_para(centro, lambda x: diferencial(x, k_ar))
        eixo.plot([px * 100], [centro], "o", ms=6, color=LARANJA)
        eixo.annotate(f"{px:.0%}".replace(".", ","), (px * 100, centro), xytext=(0, -16),
                      textcoords="offset points", ha="center", fontsize=9, color=LARANJA)
    eixo.axvline(2, color=GRADE, lw=1)
    eixo.text(2.4, 1.005, "φ = 2 %", fontsize=8.5, color=TINTA_SECUNDARIA)
    eixo.set_xlim(0, 50), eixo.set_ylim(0.3, 1.02)
    eixo.set_xlabel("porosidade (fração de volume de ar, %)")
    eixo.set_ylabel("k_eff / k_m")
    eixo.set_title(f"Bolsões de ar na epóxi, sem partículas (k_ar = {k_ar:g}, k_m = {K_M:g} W/mK)".replace(".", ","))
    eixo.grid(axis="y", color=GRADE, lw=0.8)
    eixo.spines[["top", "right"]].set_visible(False)
    eixo.legend(frameon=False, loc="lower left", fontsize=9.5, bbox_to_anchor=(0.01, 0.0))
    fig.savefig(destino)
    plt.close(fig)
    print(f"Figura: {destino}")


def mapa(plt, k_ar, k_p, rk, raio, destino):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    alfa = rk * K_M / raio
    k_p_ef = k_p / (1 + alfa * k_p / K_M)  # Kapitza em série (equivalente a HJ)
    poros = np.linspace(0, 0.6, 241)
    particulas = np.linspace(0, 0.10, 161)
    P, F = np.meshgrid(poros, particulas)
    K = np.vectorize(lambda p, f: maxwell([(p, k_ar), (f, k_p_ef)]))(P, F) / K_M
    cmap = LinearSegmentedColormap.from_list("div", [LARANJA, "#f1f2f4", AZUL])
    norma = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.25)
    fig, eixo = plt.subplots(figsize=(8.8, 5.8), layout="constrained")
    imagem = eixo.pcolormesh(P * 100, F * 100, K, cmap=cmap, norm=norma, shading="auto")
    niveis = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    linhas = eixo.contour(P * 100, F * 100, K, levels=niveis, colors=TINTA, linewidths=0.8)
    eixo.clabel(linhas, fmt=lambda v: f"{v:.1f}".replace(".", ","), fontsize=8.5)
    eixo.axhline(2, color=TINTA, lw=1.2, ls="--")
    eixo.text(0.8, 1.85, "5 % em massa (φ ≈ 2 %)", ha="left", va="top", fontsize=9)
    for (_, centro, _), dy in zip(MEDIDAS, (10, -18)):
        px = porosidade_para(centro, lambda x: maxwell([(x, k_ar), (0.02, k_p_ef)]))
        eixo.plot([px * 100], [2], "o", ms=7, color=LARANJA, mec="white", mew=1.2)
        eixo.annotate(f"k/k_m = {centro:g}: {px:.0%} de ar".replace(".", ","), (px * 100, 2),
                      xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8.5,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    eixo.set_xlabel("porosidade (%)")
    eixo.set_ylabel("partículas maciças (% em volume)")
    eixo.set_xlim(0, 60)
    eixo.set_title(f"k_eff/k_m com poros e partículas (Maxwell, 3 fases) · kₚ = {k_p:g}, "
                   f"Rₖ = {rk:g}, R = {raio * 1e9:g} nm".replace(".", ","))
    barra = fig.colorbar(imagem, ax=eixo, label="k_eff / k_m")
    barra.ax.axhline(1, color=TINTA, lw=1)
    fig.savefig(destino)
    plt.close(fig)
    print(f"Figura: {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k-ar", type=float, default=0.026)
    parser.add_argument("--k-particula", type=float, default=5.0)
    parser.add_argument("--rk", type=float, default=1e-8)
    parser.add_argument("--raio", type=float, default=50e-9)
    parser.add_argument("--fem", default=str(PASTA.parent / "ocas" / "poros.csv"))
    args = parser.parse_args()

    fem = ler_fem(Path(args.fem))
    print("Porosidade necessária (sem partículas):")
    for rotulo, centro, _ in MEDIDAS:
        pm = porosidade_para(centro, lambda x: maxwell([(x, args.k_ar)]))
        pd = porosidade_para(centro, lambda x: diferencial(x, args.k_ar))
        print(f"  {rotulo} (k/k_m = {centro}): Maxwell {pm:.1%}, diferencial {pd:.1%}")
    if fem:
        print("FEM contra Maxwell:")
        for phi, k in fem:
            print(f"  φ = {phi:.0%}: FEM {k:.4f}, Maxwell {maxwell([(phi, args.k_ar)]) / K_M:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    estilo(plt)
    (PASTA / "figuras").mkdir(exist_ok=True)
    curva(plt, args.k_ar, fem, PASTA / "figuras" / "poros_curva.png")
    mapa(plt, args.k_ar, args.k_particula, args.rk, args.raio, PASTA / "figuras" / "poros_mapa.png")


if __name__ == "__main__":
    main()
