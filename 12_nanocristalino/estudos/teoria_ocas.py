"""Esferas ocas nanocristalinas: teoria, com os pontos do FEM quando existirem.

A parede (espessura t = (t/R)·R) é nanocristalina, com k_nc(d) de nanocristal.py,
e o núcleo é ar (k_n = 0,026). Só valem cristalitos d <= t: o grão não pode ser
maior que a parede. A esfera oca vira uma esfera equivalente (esfera revestida):
    k_eq = k_nc [(k_n + 2k_nc) + 2c(k_n − k_nc)] / [(k_n + 2k_nc) − c(k_n − k_nc)],
    c = (1 − t/R)³,
e o compósito sai de Hasselman–Johnson com φ_externo = φ_sólido/(1 − c)
(mesma massa de partícula, 5 % em massa) e R_K na superfície externa.

Figura (estudos/figuras/ocas_nc_<material>[_k0_X].png):
  esquerda: k_eq da partícula oca contra d, uma curva por t/R;
  direita:  k_eff/k_m contra d, com os pontos do FEM de rodar.py --espessura.

Rodar de 12_nanocristalino:
    python3 estudos/teoria_ocas.py --material sio2
    python3 estudos/teoria_ocas.py --material gama --k0 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent))
sys.path.insert(0, str(PASTA))
from nanocristal import MATERIAIS, com_k0, fracao_volumetrica, hasselman_johnson, k_nanocristalino  # noqa: E402
from teoria import GRADE, K_M, TINTA, TINTA2, ler_fem  # noqa: E402

K_AR = 0.026
CORES = {1.0: "#1f2a44", 0.2: "#2f5d9b", 0.1: "#c2410c", 0.05: "#7c3aed"}


def k_equivalente(k_p, k_n, c):
    a = k_n + 2 * k_p
    b = c * (k_n - k_p)
    return k_p * (a + 2 * b) / (a - b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", choices=sorted(MATERIAIS), default="sio2")
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--espessuras", type=float, nargs="+", default=[1.0, 0.2, 0.1], help="t/R")
    parser.add_argument("--rgb", type=float, default=1e-9)
    parser.add_argument("--diametro", type=float, default=100.0)
    parser.add_argument("--rk", type=float, default=1e-8)
    args = parser.parse_args()
    ox = com_k0(MATERIAIS[args.material], args.k0)
    raio_nm = args.diametro / 2
    alfa = args.rk * K_M / (raio_nm * 1e-9)
    phi_sol = fracao_volumetrica(0.05, ox.densidade)
    base = f"_{args.material}" + (f"_k0_{args.k0:g}" if args.k0 else "")

    def curva(t_R, d):
        c = (1 - t_R) ** 3
        kp = np.array([k_nanocristalino(x, ox, args.rgb) for x in d])
        keq = k_equivalente(kp, K_AR, c) if t_R < 1 else kp
        phi = phi_sol / (1 - c)
        return keq, np.array([hasselman_johnson(phi, alfa, K_M, k) / K_M for k in keq]), phi

    print(f"{ox.nome}: k0 = {ox.k0:g}, ℓ = {ox.ell_nm:g} nm, R_gb = {args.rgb:g}, R_K = {args.rk:g}, "
          f"5 % em massa → φ_sólido = {phi_sol:.2%}")
    for t_R in args.espessuras:
        t_nm = t_R * raio_nm
        ds = [d for d in (1, 2, 3, 5, 10, 15, 25) if d <= t_nm]
        keq, kk, phi = curva(t_R, np.array(ds, float))
        _, denso, _ = curva(t_R, np.array([1e9]))
        print(f"t/R = {t_R:g} (parede {t_nm:g} nm, φ_ext = {phi:.2%}); sem contornos: k/k_m = {denso[0]:.4f}")
        for d, a, b in zip(ds, keq, kk):
            print(f"   d = {d:>2} nm: k_eq = {a:.3f} W/mK, k/k_m = {b:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for t_R in args.espessuras:
        t_nm = t_R * raio_nm
        d = np.logspace(0, np.log10(t_nm), 150)
        keq, kk, phi = curva(t_R, d)
        cor = CORES.get(t_R, TINTA2)
        rot = "maciça" if t_R >= 1 else f"oca, t/R = {t_R:g} (parede {t_nm:g} nm)"
        ax1.plot(d, keq, color=cor, lw=2, label=rot.replace(".", ","))
        ax2.plot(d, kk, color=cor, lw=2, label=f"{rot}, φ = {phi:.1%}".replace(".", ","))
        suf = base + (f"_oca_tR_{t_R:g}" if t_R < 1 else "")
        for nome, marcador in (("uma", "o"), ("muitas", "s")):
            pts = ler_fem(PASTA / f"nc_{nome}{suf}.csv", PASTA / f"nc_mapa{suf}.csv", args.material)
            sel = sorted((x, k) for r, x, k, _ in pts if r == args.rgb and np.isfinite(x))
            if sel:
                x, y = zip(*sel)
                ax2.plot(x, y, marcador, ms=7, color=cor, mfc="white", mew=1.6)
    ax1.axhline(K_M, color=TINTA2, lw=1)
    ax1.text(1.05, K_M * 1.06, "matriz (epóxi)", fontsize=9, color=TINTA2)
    ax1.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax1.yaxis.set_major_formatter(fmt)
    ax1.yaxis.set_minor_formatter(fmt)
    ax1.set_ylabel("k equivalente da partícula (W/mK)")
    ax1.set_title(f"Parede de {ox.nome}, R_gb = {args.rgb:g}", loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.axhline(1, color=TINTA2, lw=1)
    ax2.plot([], [], "o", color=TINTA2, mfc="white", label="FEM, uma esfera")
    ax2.plot([], [], "s", color=TINTA2, mfc="white", label="FEM, várias esferas")
    ax2.set_ylabel("k_eff / k_m")
    ax2.set_title(f"Compósito, 5 % em massa, Rₖ = {args.rk:g}", loc="left")
    ax2.legend(frameon=False, fontsize=8.5, loc="best")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("tamanho do cristalito d (nm)")
        ax.set_xticks([1, 2, 3, 5, 10, 25, 50], ["1", "2", "3", "5", "10", "25", "50"])
        ax.grid(axis="y", color=GRADE, lw=0.8)
    (PASTA / "figuras").mkdir(exist_ok=True)
    nome = f"ocas_nc{base}.png"
    fig.savefig(PASTA / "figuras" / nome)
    print(f"Figura: {PASTA / 'figuras' / nome}")


if __name__ == "__main__":
    main()
