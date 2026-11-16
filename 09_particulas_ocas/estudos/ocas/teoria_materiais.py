"""Ocas contra maciças para dois materiais: SiO₂ (k = 1,36) e γ-Al₂O₃ (k = 5).

Mesma teoria de ``teoria.py`` (HJ com a esfera oca equivalente), sem mudar
aquele arquivo. Duas comparações, como antes:
  * mesmo φ externo (--phi-externo, 5 %);
  * mesma massa de óxido: 5 % em massa da tinta dá um φ sólido diferente para
    cada material, porque as densidades são diferentes. Com a tinta a
    ~1,5 g/cm³ (ainda falta o valor da N2680): γ-Al₂O₃ (3,65 g/cm³) → ~2,1 %,
    SiO₂ amorfa (2,2 g/cm³) → ~3,4 %. Mude com --rho-tinta.

Se existirem ``ocas_sio2.csv`` e ``ocas_gama.csv`` os pontos do FEM entram na figura.

Rodar de 09_particulas_ocas:  python3 estudos/ocas/teoria_materiais.py
Saídas: figuras/teoria_materiais.png e teoria_materiais.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from teoria import K_M, k_oca  # noqa: E402

TINTA, TINTA_SECUNDARIA, GRADE = "#1f2a44", "#5b6474", "#d9dde5"
# (nome, k_p, densidade g/cm³, cor, arquivos do FEM)
MATERIAIS = [
    ("SiO₂ (k = 1,36)", 1.36, 2.20, "#c2410c", ["ocas_sio2.csv"]),
    ("γ-Al₂O₃ (k = 5)", 5.0, 3.65, "#2f5d9b", ["ocas_gama.csv"]),
]


def fracao_volumetrica(w, rho_p, rho_m):
    return (w / rho_p) / (w / rho_p + (1 - w) / rho_m)


def ler_fem(nomes, k_p):
    pontos = []
    for nome in nomes:
        arquivo = PASTA / nome
        if not arquivo.exists():
            continue
        with arquivo.open(encoding="utf-8") as f:
            for l in csv.DictReader(f):
                if abs(float(l["k_p"]) - k_p) < 1e-9:
                    pontos.append((l["modo"], float(l["R_K"]), float(l["t_R"]),
                                   float(l["phi_solido"]), float(l["k_sobre_km"]), float(l["phi_externo"])))
    return pontos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phi-externo", type=float, nargs="+", default=[0.05, 0.20])
    parser.add_argument("--fracao-massica", type=float, default=0.05)
    parser.add_argument("--rho-tinta", type=float, default=1.5, help="g/cm³")
    parser.add_argument("--rk", type=float, nargs="+", default=[1e-8])
    parser.add_argument("--k-nucleo", type=float, default=0.026)
    args = parser.parse_args()

    t = np.logspace(-2, 0, 200)
    estilos = ["-", "--", ":", "-."]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": TINTA_SECUNDARIA, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
                         "text.color": TINTA, "savefig.dpi": 200})
    n = len(args.phi_externo) + 1
    fig, eixos = plt.subplots(1, n, figsize=(6.2 * n, 5.2), layout="constrained")
    linhas = []
    for nome, k_p, rho, cor, arquivos in MATERIAIS:
        phi_sol = fracao_volumetrica(args.fracao_massica, rho, args.rho_tinta)
        fem = ler_fem(arquivos, k_p)
        print(f"{nome}: 5 % em massa → φ sólido = {phi_sol:.2%}")
        for rk, ls in zip(args.rk, estilos):
            phi_ext = phi_sol / (1 - (1 - t) ** 3)
            y2 = np.array([k_oca(p, rk, x, k_p, args.k_nucleo) if p < 0.3 else np.nan
                           for p, x in zip(phi_ext, t)]) / K_M
            rotulo = f"{nome}, Rₖ = {rk:g}"
            for j, pe in enumerate(args.phi_externo):
                y1 = np.array([k_oca(pe, rk, x, k_p, args.k_nucleo) for x in t]) / K_M
                eixos[j].plot(t, y1, ls, color=cor, lw=2, label=rotulo)
            eixos[-1].plot(t, y2, ls, color=cor, lw=2, label=rotulo)
            paineis = [("mesmo_phi", pe, eixos[j]) for j, pe in enumerate(args.phi_externo)]
            for modo, pe, eixo in paineis + [("mesma_massa", None, eixos[-1])]:
                pts = [(tr, k) for m, r, tr, ps, k, px in fem if m == modo and r == rk
                       and (abs(px - pe) < 1e-6 if modo == "mesmo_phi" else abs(ps - phi_sol) < 1e-4)]
                if pts:
                    x, y = zip(*pts)
                    eixo.plot(x, y, "o", ms=7, color=cor, mfc="white", mew=1.6)
            for x in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
                pe = phi_sol / (1 - (1 - x) ** 3)
                linhas.append([nome, k_p, rk, x, args.phi_externo[0],
                               f"{k_oca(args.phi_externo[0], rk, x, k_p, args.k_nucleo) / K_M:.5f}",
                               f"{phi_sol:.5f}", f"{pe:.5f}",
                               f"{k_oca(pe, rk, x, k_p, args.k_nucleo) / K_M:.5f}" if pe < 0.3 else ""])
    titulos = [(eixos[j], f"Mesmo φ externo ({pe:.0%})") for j, pe in enumerate(args.phi_externo)]
    for eixo, titulo in titulos + [(eixos[-1], f"Mesma massa ({args.fracao_massica:.0%} em massa, tinta {args.rho_tinta:g} g/cm³)")]:
        eixo.axhline(1, color=TINTA_SECUNDARIA, lw=1)
        eixo.set_xscale("log")
        eixo.set_xlabel("espessura da parede t/R  (t/R = 1: maciça)")
        eixo.set_ylabel("k_eff / k_m")
        eixo.set_title(titulo.replace(".", ","))
        eixo.grid(axis="y", color=GRADE, lw=0.8)
        eixo.spines[["top", "right"]].set_visible(False)
    eixos[0].legend(frameon=False, fontsize=9, loc="upper left")
    eixos[-1].text(0.99, 0.02, "círculos: FEM", transform=eixos[-1].transAxes, ha="right", fontsize=9,
                  color=TINTA_SECUNDARIA)
    fig.suptitle("Ocas (núcleo de ar) contra maciças: SiO₂ e γ-Al₂O₃, R = 50 nm", fontsize=12)
    (PASTA / "figuras").mkdir(exist_ok=True)
    fig.savefig(PASTA / "figuras" / "teoria_materiais.png")
    with (PASTA / "teoria_materiais.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["material", "k_p", "R_K", "t_R", "phi_externo", "k_sobre_km_mesmo_phi",
                    "phi_solido_massa", "phi_externo_massa", "k_sobre_km_mesma_massa"])
        w.writerows(linhas)
    print("\nmaterial            R_K    t/R   mesmo φ   mesma massa")
    for l in linhas:
        print(f"{l[0]:18s} {l[2]:6g} {l[3]:5g}   {l[5]}   {l[8]}")
    print(f"\nFigura: {PASTA / 'figuras' / 'teoria_materiais.png'}")


if __name__ == "__main__":
    main()
