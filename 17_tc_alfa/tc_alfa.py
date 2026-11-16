"""Espessura crítica da parede contra alfa = R_K k_m / R (reforço 1 do artigo).

A parede crítica t_c é a espessura em que a esfera oca (núcleo de ar) com
resistência de Kapitza na superfície externa deixa k_eff = k_m. Sem R_K é o
resultado de Felske (2004); com R_K a esfera equivalente precisa valer
k* = k_m/(1 − alfa), então

    delta_c = (2k_p + k_c)(k_p − k*) / [(k_p − k_c)(2k_p + k*)],
    t_c/R   = 1 − delta_c^(1/3).

Para alfa >= alfa_c = 1 − k_m/k_p não existe parede crítica: a partícula
maciça já baixa k. A correção cresce como ~1/(1 − alfa) (limite de parede fina).

Não roda FEM. Lê os CSVs de fem/ (gerados por rodar_fem.sh) se existirem e
marca na figura o t_c obtido por interpolação dos três t/R de cada caso.
Uso (de qualquer lugar):
    python3 17_tc_alfa/tc_alfa.py
Saída: 17_tc_alfa/teoria_tc_alfa.csv e 17_tc_alfa/figuras/fig_tc_alfa.pdf/.png
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

PASTA = Path(__file__).resolve().parent
K_M, K_AR = 0.2, 0.026
AZUL, LARANJA, VERDE, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
CINZA = "#52514e"

# (rótulo, k_p da casca, cor, marcador). Sílica granular: d = 3 nm, R_gb = 1e-9 (pasta 12/14).
MATERIAIS = [
    ("SiO$_2$ dense", 1.36, AZUL, "o"),
    ("SiO$_2$ granular (3 nm)", 0.8226, LARANJA, "s"),
    ("$\\gamma$-Al$_2$O$_3$ ($k_p$ = 5)", 5.0, VERDE, "^"),
    ("ZIF-8", 0.33, VIOLETA, "D"),
]

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 7,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "savefig.dpi": 600,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "figure.constrained_layout.use": True,
})


def t_critico(kp, alfa, kc=K_AR, km=K_M):
    """t_c/R da esfera oca com núcleo kc; NaN quando alfa >= alfa_c."""
    alfa = np.asarray(alfa, dtype=float)
    ks = km / (1 - alfa)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (2 * kp + kc) * (kp - ks) / ((kp - kc) * (2 * kp + ks))
        t = 1 - np.cbrt(d)
    return np.where(ks < kp, t, np.nan)


def tc_fem(arquivo):
    """t_c por FEM: para cada alfa, raiz de k/k_m = 1 numa parábola pelos três t/R rodados."""
    casos = {}
    with open(arquivo, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            casos.setdefault(round(float(linha["alfa"]), 6), []).append(
                (float(linha["t_R"]), float(linha["k_sobre_km"])))
    pontos = []
    for alfa, lista in sorted(casos.items()):
        lista.sort()
        t, k = np.array(lista).T
        if len(t) < 3 or (k.min() - 1) * (k.max() - 1) > 0:
            continue
        # parábola pelos três pontos e raiz de k/k_m = 1 dentro do intervalo rodado
        raizes = np.roots(np.polyfit(t, k - 1, 2))
        raizes = [r.real for r in raizes if abs(r.imag) < 1e-12 and t[0] <= r.real <= t[-1]]
        if raizes:
            pontos.append((alfa, raizes[0]))
    return pontos


def main():
    alfas = np.linspace(0, 0.96, 961)
    with open(PASTA / "teoria_tc_alfa.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alfa"] + [f"tc_R_kp{kp:g}" for _, kp, _, _ in MATERIAIS])
        tabela = [t_critico(kp, alfas) for _, kp, _, _ in MATERIAIS]
        for i, a in enumerate(alfas):
            w.writerow([f"{a:.3f}"] + [("" if np.isnan(t[i]) else f"{t[i]:.6f}") for t in tabela])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for rotulo, kp, cor, marc in MATERIAIS:
        t = t_critico(kp, alfas)
        ac = 1 - K_M / kp
        ax1.plot(alfas, t, color=cor, label=rotulo)
        ax2.plot(alfas, t / t_critico(kp, 0.0), color=cor, label=rotulo)
        if ac < 0.96:
            ax1.axvline(ac, color=cor, lw=0.7, ls=":")
        ax1.plot([0], [t_critico(kp, 0.0)], marker=marc, color=cor, ms=4, clip_on=False)
    # limite de parede fina
    a = np.linspace(0, 0.75, 200)
    ax2.plot(a, 1 / (1 - a), color=CINZA, ls="--", lw=1.0, label="1/(1 − α), thin wall")

    # FEM (pasta fem/): um arquivo por material, nome fem_<chave>.csv
    chaves = {"sio2": 0, "sio2_gran": 1, "gama": 2, "zif8": 3}
    rotulado = False
    for chave, i in chaves.items():
        arq = PASTA / "fem" / f"fem_{chave}.csv"
        if not arq.exists():
            continue
        _, kp, cor, marc = MATERIAIS[i]
        for alfa, t in tc_fem(arq):
            for ax, y in ((ax1, t), (ax2, t / t_critico(kp, 0.0))):
                ax.plot(alfa, y, marker=marc, mfc="white", mec=cor, mew=1.2, ms=6, ls="none",
                        label=None if rotulado or ax is ax1 else "FEM")
            rotulado = True

    ax1.set_xlim(0, 0.96)
    ax1.set_ylim(0, 0.8)
    ax1.set_xlabel("α = $R_K k_m / R$")
    ax1.set_ylabel("Critical wall $t_c/R$")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes, va="top", fontweight="bold")
    # eixo de cima: raio para R_K = 1e-8 m²K/W (R = R_K k_m / alfa)
    topo = ax1.secondary_xaxis("top", functions=(lambda x: 2.0 / np.maximum(x, 1e-3),
                                                 lambda r: 2.0 / np.maximum(r, 1e-3)))
    topo.set_xticks([20, 10, 5, 4, 3, 2.5])
    topo.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    topo.set_xlabel("$R$ (nm) for $R_K$ = 10$^{-8}$ m$^2$K/W", fontsize=7)

    ax2.set_xlim(0, 0.75)
    ax2.set_ylim(1, 4)
    ax2.set_xlabel("α = $R_K k_m / R$")
    ax2.set_ylabel("$t_c(α)\\,/\\,t_c(0)$")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, va="top", fontweight="bold")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0))

    saida = PASTA / "figuras"
    saida.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(saida / f"fig_tc_alfa.{ext}")
    print("Figura em", saida / "fig_tc_alfa.pdf")


if __name__ == "__main__":
    main()
