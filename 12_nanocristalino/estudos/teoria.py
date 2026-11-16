"""Teoria das partículas nanocristalinas, com os pontos do FEM quando existirem.

Figura com dois painéis (estudos/figuras/nanocristal.png):
  esquerda: k da partícula contra o cristalito d, para R_gb = 0, 1e-10, 1e-9 e
            1e-8 m²K/W (com a linha do monocristal e a da matriz);
  direita:  k_eff/k_m do compósito a 5 % em massa (HJ, R_K = 1e-8 na
            superfície da partícula), com os círculos do FEM de rodar.py
            (uma esfera e 80 esferas).

Rodar de 12_nanocristalino:  python3 estudos/teoria.py [--material gama]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent))
from nanocristal import MATERIAIS, com_k0, fracao_volumetrica, hasselman_johnson, k_grao, k_nanocristalino  # noqa: E402

K_M = 0.2
TINTA, TINTA2, GRADE = "#1f2a44", "#5b6474", "#d9dde5"
RGBS = [(0.0, "sem contornos (só ℓ)", "#9aa3b2", ":"), (1e-10, "R_gb = 1e-10", "#8fa6c8", "-"),
        (1e-9, "R_gb = 1e-9", "#2f5d9b", "-"), (1e-8, "R_gb = 1e-8", "#c2410c", "-")]
CRISTALITOS = [1, 2, 3, 5, 10, 15, 25]


def ler_fem(arquivo, mapa, material):
    if not arquivo.exists() or not mapa.exists():
        return []
    with mapa.open(encoding="utf-8") as f:
        d_de = {(float(l["R_gb"]), round(float(l["k_p"]), 4)): float(l["d_nm"])
                for l in csv.DictReader(f) if l["material"] == material}
    pontos = []
    with arquivo.open(encoding="utf-8") as f:
        for l in csv.DictReader(f):
            k_p = round(float(l["k_p"]), 4)
            for (rgb, k), d in d_de.items():
                if k == k_p:
                    pontos.append((rgb, d, float(l["k_sobre_km"]), float(l["desvio_HJ_percent"])))
    return pontos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", choices=sorted(MATERIAIS), default="gama")
    parser.add_argument("--diametro", type=float, default=100.0)
    parser.add_argument("--rk", type=float, default=1e-8)
    parser.add_argument("--k0", type=float, default=None,
                        help="k do monocristal; ex.: 5 (o valor fixo usado até a pasta 11). ℓ escala junto.")
    args = parser.parse_args()
    ox = com_k0(MATERIAIS[args.material], args.k0)
    raio = args.diametro * 1e-9 / 2
    phi = fracao_volumetrica(0.05, ox.densidade)
    alfa = args.rk * K_M / raio

    print(f"{ox.nome}: k0 = {ox.k0:g}, ℓ = {ox.ell_nm:g} nm; 5 % em massa → φ = {phi:.2%}; α = {alfa:.3g}")
    print("d (nm)  k_grão   " + "  ".join(f"k_nc(R_gb={r:g})" for r, *_ in RGBS[1:]) + "   k/k_m (R_gb=1e-9)")
    for d in CRISTALITOS:
        ks = [k_nanocristalino(d, ox, r) for r, *_ in RGBS[1:]]
        print(f"{d:5g}   {k_grao(d, ox):6.2f}   " + "   ".join(f"{k:13.2f}" for k in ks)
              + f"   {hasselman_johnson(phi, alfa, K_M, ks[1]) / K_M:.4f}")
    print(f"denso, sem contornos: k/k_m = {hasselman_johnson(phi, alfa, K_M, ox.k0) / K_M:.4f}; "
          f"partícula isolante (k_p → 0): {hasselman_johnson(phi, alfa, K_M, 1e-9) / K_M:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    d = np.logspace(0, 2, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for rgb, rotulo, cor, ls in RGBS:
        kp = np.array([k_nanocristalino(x, ox, rgb) for x in d])
        ax1.plot(d, kp, ls, color=cor, lw=2, label=rotulo)
        ax2.plot(d, [hasselman_johnson(phi, alfa, K_M, k) / K_M for k in kp], ls, color=cor, lw=2, label=rotulo)
    ax1.axhline(ox.k0, color=TINTA2, lw=1, ls="--")
    ax1.text(1.05, ox.k0 * 1.05, f"denso, sem contornos ({ox.k0:g} W/mK)".replace(".", ","), fontsize=9, color=TINTA2)
    ax1.axhline(K_M, color=TINTA2, lw=1)
    ax1.text(1.05, K_M * 1.1, "matriz (epóxi)", fontsize=9, color=TINTA2)
    ax1.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    rotulo_y = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax1.yaxis.set_major_formatter(rotulo_y)
    ax1.yaxis.set_minor_formatter(rotulo_y)
    ax1.set_ylabel("k da partícula (W/mK)")
    ax1.set_title(f"{ox.nome}: efeito do tamanho de grão (ou domínio)", loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.axhline(1, color=TINTA2, lw=1)
    for nome, marcador in (("uma", "o"), ("muitas", "s")):
        sufixo = f"_{args.material}" + (f"_k0_{args.k0:g}" if args.k0 else "")
        pts = ler_fem(PASTA / f"nc_{nome}{sufixo}.csv", PASTA / f"nc_mapa{sufixo}.csv", args.material)
        for rgb, _, cor, _ in RGBS:
            sel = sorted((x, k) for r, x, k, _ in pts if r == rgb and np.isfinite(x))
            if sel:
                x, y = zip(*sel)
                ax2.plot(x, y, marcador, ms=7, color=cor, mfc="white", mew=1.6)
    ax2.plot([], [], "o", color=TINTA2, mfc="white", label="FEM, uma esfera")
    ax2.plot([], [], "s", color=TINTA2, mfc="white", label="FEM, 80 esferas")
    ax2.set_ylabel("k_eff / k_m")
    ax2.set_title(f"Compósito a 5 % em massa (φ = {phi:.1%}), Rₖ = {args.rk:g}".replace(".", ","), loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("tamanho do cristalito d (nm)")
        ax.set_xticks(CRISTALITOS + [50, 100], [str(x) for x in CRISTALITOS + [50, 100]])
        ax.grid(axis="y", color=GRADE, lw=0.8)
        for x in CRISTALITOS:
            ax.axvline(x, color=GRADE, lw=0.6, zorder=0)
    (PASTA / "figuras").mkdir(exist_ok=True)
    nome = f"nanocristal_{args.material}" + (f"_k0_{args.k0:g}" if args.k0 else "") + ".png"
    fig.savefig(PASTA / "figuras" / nome)
    print(f"Figura: {PASTA / 'figuras' / nome}")


if __name__ == "__main__":
    main()
