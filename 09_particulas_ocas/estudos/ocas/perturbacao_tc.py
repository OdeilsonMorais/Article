"""Figura da espessura crítica pela perturbação de temperatura.

Lê os mesmos dados de cortes_tc.py (estudos/ocas/dados/tc_*.npz; não roda
FEniCSx) e desenha θ = T + G x, ou seja, o quanto a partícula desvia o campo
linear da matriz pura. Acima de t_c a partícula puxa calor (dipolo positivo,
isotermas afastadas na frente e atrás); abaixo, empurra (dipolo negativo).
Em t_c o conjunto partícula + R_K é invisível de fora: θ ≈ 0 na matriz.
Embaixo, |q|/(k_m G) − 1 na matriz, mesma ideia para o fluxo.

Rodar de 09_particulas_ocas (depois de cortes_tc.py --calcular):
    python3 estudos/ocas/perturbacao_tc.py
    python3 estudos/ocas/perturbacao_tc.py --k-particula 1.36 --espessuras 0.2 0.068 0.03
"""

import argparse
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
TINTA, TINTA2 = "#1f2a44", "#5b6474"


def nome(phi, rk, t_R, k_p, k_n):
    return f"tc_phi{phi:.4g}_rk{rk:g}_t{t_R:.4g}_kp{k_p:g}_kn{k_n:g}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phi", type=float, default=0.10)
    p.add_argument("--rk", type=float, default=1e-8)
    p.add_argument("--espessuras", type=float, nargs="+", default=[0.5, 0.3187, 0.15])
    p.add_argument("--k-particula", type=float, default=0.33)
    p.add_argument("--k-nucleo", type=float, default=0.026)
    p.add_argument("--t-critico", type=float, default=None, help="t_c/R para o rótulo (padrão: o do meio)")
    a = p.parse_args()

    casos = []
    for t in a.espessuras:
        arq = PASTA / "dados" / f"{nome(a.phi, a.rk, t, a.k_particula, a.k_nucleo)}.npz"
        if not arq.exists():
            raise SystemExit(f"Falta {arq.name}: rode cortes_tc.py --calcular.")
        casos.append(dict(np.load(arq)))
    tc = a.t_critico if a.t_critico else sorted(a.espessuras)[len(a.espessuras) // 2]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                         "savefig.dpi": 200, "figure.facecolor": "white"})

    dados = []
    for d in casos:
        meio = float(d["lado"]) / 2
        r = np.hypot(d["U"] - meio, d["V"] - meio)
        matriz = r > 1.0
        theta = d["T"] + d["U"]                      # θ/(G R), com constante arbitrária
        theta = theta - np.mean(theta[matriz])       # zero na média da matriz no corte
        dq = d["q_modulo"] - 1.0
        dados.append((d, meio, r, matriz, theta, dq))
    lim_t = max(np.percentile(np.abs(th[m]), 99.5) for _, _, _, m, th, _ in dados)
    lim_q = max(np.percentile(np.abs(q[m]), 99.5) for _, _, _, m, _, q in dados)

    n = len(casos)
    fig, eixos = plt.subplots(2, n, figsize=(5.0 * n + 1.4, 9.6), squeeze=False, layout="constrained")
    for j, (d, meio, r, matriz, theta, dq) in enumerate(dados):
        ext = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
        t_R = float(d["t_R"])
        kk = float(d["k_xx"]) / float(d["k_m"])
        e1, e2 = eixos[0, j], eixos[1, j]
        im1 = e1.imshow(np.where(matriz, theta, np.nan), origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-lim_t, vmax=lim_t, interpolation="bilinear")
        e1.contour(d["U"], d["V"], np.where(matriz, theta, np.nan),
                   levels=np.linspace(-lim_t, lim_t, 13), colors=TINTA2, linewidths=0.5)
        im2 = e2.imshow(np.where(matriz, dq, np.nan), origin="lower", extent=ext, cmap="PuOr_r",
                        vmin=-lim_q, vmax=lim_q, interpolation="bilinear")
        e2.streamplot(d["U"][0, :], d["V"][:, 0], d["q_u"], d["q_v"], color=(0.15, 0.17, 0.27, 0.5),
                      linewidth=0.6, density=1.1, arrowsize=0.6, broken_streamlines=False)
        for e in (e1, e2):
            e.add_patch(Circle((meio, meio), 1.0, facecolor="#e8e6e1", edgecolor=TINTA, lw=1.0, zorder=3))
            e.add_patch(Circle((meio, meio), float(d["r_i"]), facecolor="white", edgecolor=TINTA2,
                               lw=0.8, ls="--", zorder=4))
            e.set_xlim(ext[:2]), e.set_ylim(ext[2:]), e.set_aspect("equal")
            e.set_xlabel("x/R"), e.set_ylabel("y/R")
        if abs(t_R - tc) < 1e-9:
            rot = f"t = t_c = {t_R:g} R"
        else:
            rot = f"t = {t_R:g} R ({'>' if t_R > tc else '<'} t_c)"
        e1.set_title(f"{rot}\nk_eff/k_m = {kk:.4f}".replace(".", ","))
    fig.colorbar(im1, ax=eixos[0, :].tolist(), shrink=0.85, label="θ / (G R)  (perturbação de T na matriz)")
    fig.colorbar(im2, ax=eixos[1, :].tolist(), shrink=0.85, label="|q| / (k_m G) − 1  (matriz)")
    fig.suptitle((f"kₚ = {a.k_particula:g} W/mK, núcleo {a.k_nucleo:g} W/mK, φ externo {100 * a.phi:.3g} %, "
                  f"Rₖ = {a.rk:g} m²K/W · corte z = L/2, G ao longo de x").replace(".", ","),
                 fontsize=12, color=TINTA2)
    esp = "_".join(f"{t:g}" for t in a.espessuras)
    destino = PASTA / "figuras" / f"perturbacao_tc_kp{a.k_particula:g}_phi{a.phi:.4g}_rk{a.rk:g}_t{esp}.png"
    destino.parent.mkdir(exist_ok=True)
    fig.savefig(destino)
    print(f"Figura: {destino}")
    for d, _, _, m, th, _ in dados:
        print(f"t/R = {float(d['t_R']):g}: max|θ| na matriz = {np.abs(th[m]).max():.4f} G R")


if __name__ == "__main__":
    main()
