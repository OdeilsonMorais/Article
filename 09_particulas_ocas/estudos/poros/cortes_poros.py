"""Cortes de temperatura e fluxo com um bolsão de ar (sem partícula) no centro da célula.

Usa o mesmo cálculo de ``../ocas/cortes_ocas.py``: uma esfera no centro, G ao
longo de x, corte no plano xy em z = L/2. Aqui a esfera é de ar (k = k_ar) e
não há Kapitza (R_K desprezível, 1e-12). Cada coluna é uma porosidade:
  * em cima, T/(G R) com isotermas;
  * embaixo, |q|/(k_m G) em escala log com as linhas de fluxo.
As escalas de cor são as mesmas em todas as colunas.

Duas etapas:
1. Calcular (FEniCSx, em série; pula o que já existe em dados/):
       python3 estudos/poros/cortes_poros.py --calcular
2. Redesenhar (só Matplotlib):
       python3 estudos/poros/cortes_poros.py

Padrão: porosidades 5, 10 e 20 %, h_bulk = 0,25R, 400 pontos por lado.
Rodar de 09_particulas_ocas.
"""

import argparse
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parents[1]))           # 09_particulas_ocas
sys.path.insert(0, str(PASTA.parent / "ocas"))      # cortes_ocas.calcular

import numpy as np  # noqa: E402

from cortes_ocas import TINTA, TINTA_SECUNDARIA, calcular  # noqa: E402

RK_NULO = 1e-12
Q_MIN = 0.01


def desenhar(casos, destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "axes.edgecolor": TINTA_SECUNDARIA, "axes.labelcolor": TINTA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
        "text.color": TINTA, "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200,
    })
    n = len(casos)
    q_max = max(float(np.nanpercentile(d["q_modulo"], 99.5)) for d in casos)
    norma_q = LogNorm(vmin=Q_MIN, vmax=max(q_max, 1.5))
    limite = 0.5 * max(float(d["lado"]) for d in casos)  # mesma escala de T em todas as colunas
    figura, eixos = plt.subplots(2, n, figsize=(5.2 * n + 1.2, 10), squeeze=False, layout="constrained")
    for j, d in enumerate(casos):
        # Cada porosidade tem um lado de célula; T é mostrado em relação à média do corte.
        T = d["T"] - float(np.mean(d["T"]))
        extensao = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
        meio = float(d["lado"]) / 2
        eixo_T, eixo_q = eixos[0, j], eixos[1, j]
        imagem_T = eixo_T.imshow(T, origin="lower", extent=extensao, cmap="inferno",
                                 vmin=-limite, vmax=limite, interpolation="bilinear")
        eixo_T.contour(d["U"], d["V"], T, levels=np.linspace(-limite, limite, 17), colors="white",
                       linewidths=0.6, alpha=0.75, linestyles="solid")
        imagem_q = eixo_q.imshow(np.clip(d["q_modulo"], Q_MIN, None), origin="lower", extent=extensao,
                                 cmap="magma", norm=norma_q, interpolation="bilinear")
        eixo_q.streamplot(d["U"][0, :], d["V"][:, 0], d["q_u"], d["q_v"], color=(1, 1, 1, 0.6),
                          linewidth=0.7, density=1.2, arrowsize=0.7, broken_streamlines=False)
        for eixo in (eixo_T, eixo_q):
            eixo.add_patch(Circle((meio, meio), 1.0, fill=False, lw=1.2, color="white"))
            eixo.set_xlabel("x/R"), eixo.set_ylabel("y/R"), eixo.set_aspect("equal")
            eixo.set_xlim(extensao[:2]), eixo.set_ylim(extensao[2:])
        k_rel = float(d["k_xx"]) / float(d["k_m"])
        eixo_T.set_title(f"porosidade {100 * float(d['phi']):.3g} %\nk_eff/k_m = {k_rel:.4f}".replace(".", ","))
        eixo_q.set_title(f"⟨q_x⟩ = {k_rel:.4f} k_m G".replace(".", ","))
    figura.colorbar(imagem_T, ax=eixos[0, :].tolist(), shrink=0.85,
                    label="(T − T_médio) / (G R)")
    barra = figura.colorbar(imagem_q, ax=eixos[1, :].tolist(), shrink=0.85, label="|q| / (k_m G)")
    barra.ax.axhline(1, color="white", lw=1.5)
    d = casos[0]
    figura.suptitle(f"Bolsão de ar na epóxi, sem partícula · k_ar = {float(d['k_p']):g}, "
                    f"k_m = {float(d['k_m']):g} W/mK · corte z = L/2, G ao longo de x".replace(".", ","),
                    fontsize=13, color=TINTA_SECUNDARIA)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    print(f"Figura: {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calcular", action="store_true")
    parser.add_argument("--porosidades", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--k-ar", type=float, default=0.026)
    parser.add_argument("--h-bulk", type=float, default=0.25)
    parser.add_argument("--pontos", type=int, default=400)
    args = parser.parse_args()

    dados_dir = PASTA / "dados"
    casos = []
    for phi in args.porosidades:
        arquivo = dados_dir / f"poro_phi{phi:.4g}_kar{args.k_ar:g}.npz"
        if args.calcular and not arquivo.exists():
            dados_dir.mkdir(exist_ok=True)
            # Esfera maciça (t/R = 1) com k_p = k_ar e sem Kapitza: um poro.
            d = calcular(phi, RK_NULO, 1.0, args.k_ar, args.k_ar, args.h_bulk, args.pontos)
            np.savez_compressed(arquivo, **d)
            print(f"porosidade {phi:.0%}: k_xx/k_m = {d['k_xx'] / d['k_m']:.5f} ({d['nos']} nós)", flush=True)
        if not arquivo.exists():
            raise SystemExit(f"Falta {arquivo.name}: rode com --calcular.")
        casos.append(dict(np.load(arquivo)))
    nomes = "_".join(f"{p:g}" for p in args.porosidades)
    desenhar(casos, PASTA / "figuras" / f"cortes_poros_phi{nomes}_kar{args.k_ar:g}.png")


if __name__ == "__main__":
    main()
