"""k_eff/k_m contra a concentração de partículas, ocas e maciças, SiO₂ e γ-Al₂O₃.

Um painel por material. As linhas são a teoria (HJ com a esfera oca
equivalente, ``teoria.k_oca``) para cada espessura de parede; os círculos são
o FEM de ``rodar_materiais.py --etapa concentracao`` (ocas_sio2.csv,
ocas_gama.csv), quando existem. Como HJ é uma teoria diluída, a diferença entre
linha e círculo em φ alto mostra o efeito das interações entre partículas.

Rodar de 09_particulas_ocas:  python3 estudos/ocas/concentracao_materiais.py
Saída: figuras/concentracao_materiais.png
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from teoria import K_M, k_oca  # noqa: E402

TINTA, TINTA2, GRADE = "#1f2a44", "#5b6474", "#d9dde5"
MATERIAIS = [("SiO₂ (k = 1,36)", 1.36, "ocas_sio2.csv"), ("γ-Al₂O₃ (k = 5)", 5.0, "ocas_gama.csv")]
# parede: (rótulo, cor) — do mais escuro (maciça) ao mais claro (parede fina)
PAREDES = [(1.0, "maciça", "#1f2a44"), (0.2, "oca, t/R = 0,2", "#2f5d9b"),
           (0.1, "oca, t/R = 0,1", "#d08a1e"), (0.05, "oca, t/R = 0,05", "#c2410c")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rk", type=float, default=1e-8)
    parser.add_argument("--k-nucleo", type=float, default=0.026)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained", sharey=True)
    phi = np.linspace(0, 0.30, 121)
    for eixo, (nome, k_p, arquivo) in zip(eixos, MATERIAIS):
        fem = []
        if (PASTA / arquivo).exists():
            with (PASTA / arquivo).open(encoding="utf-8") as f:
                fem = [l for l in csv.DictReader(f) if l["modo"] == "mesmo_phi"
                       and float(l["R_K"]) == args.rk and abs(float(l["k_p"]) - k_p) < 1e-9]
        print(nome)
        for t_R, rotulo, cor in PAREDES:
            eixo.plot(phi * 100, [k_oca(p, args.rk, t_R, k_p, args.k_nucleo) / K_M for p in phi],
                      color=cor, lw=2, label=f"{rotulo} (teoria)")
            pts = sorted((float(l["phi_externo"]), float(l["k_sobre_km"])) for l in fem
                         if abs(float(l["t_R"]) - t_R) < 1e-6)
            if pts:
                x, y = zip(*pts)
                eixo.plot(np.array(x) * 100, y, "o", ms=7, color=cor, mfc="white", mew=1.6)
                for p, k in pts:
                    print(f"  t/R = {t_R:g}, φ = {p:.0%}: FEM {k:.4f}, teoria "
                          f"{k_oca(p, args.rk, t_R, k_p, args.k_nucleo) / K_M:.4f}")
        eixo.axhline(1, color=TINTA2, lw=1)
        eixo.set_title(nome, loc="left")
        eixo.set_xlabel("partículas, φ externo (% em volume)")
        eixo.grid(axis="y", color=GRADE, lw=0.8)
        eixo.set_xlim(0, 30)
    eixos[0].set_ylabel("k_eff / k_m")
    eixos[0].legend(frameon=False, fontsize=9, loc="upper left")
    eixos[1].text(0.99, 0.02, "círculos: FEM (uma esfera por célula)", transform=eixos[1].transAxes,
                  ha="right", fontsize=9, color=TINTA2)
    fig.suptitle(f"Concentração de partículas ocas (núcleo de ar) e maciças · Rₖ = {args.rk:g} m²K/W, R = 50 nm",
                 fontsize=12, color=TINTA2)
    (PASTA / "figuras").mkdir(exist_ok=True)
    destino = PASTA / "figuras" / "concentracao_materiais.png"
    fig.savefig(destino)
    print(f"Figura: {destino}")


if __name__ == "__main__":
    main()
