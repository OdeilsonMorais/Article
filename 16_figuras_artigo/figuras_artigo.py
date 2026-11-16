"""Figuras do artigo (IJHMT), geradas só a partir dos CSVs já gravados nas pastas.

Não roda FEM. Uso (da raiz do repositório ou de qualquer lugar):
    python3 16_figuras_artigo/figuras_artigo.py            # todas
    python3 16_figuras_artigo/figuras_artigo.py 7 12       # só as figuras 7 e 12
Saída: 16_figuras_artigo/figuras/figNN_nome.pdf e .png (600 dpi).
Uma figura cujo CSV ainda não existe é pulada com um aviso.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = Path(__file__).resolve().parent / "figuras"

K_M, R_NM = 0.2, 50.0
K_AR = 0.026
# cores (paleta categórica validada para daltonismo) e marcadores: a identidade
# nunca depende só da cor
AZUL, LARANJA, VERDE, AMARELO, VIOLETA, VERMELHO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
CINZA = "#52514e"
RAMPA = ["#9ec5f4", "#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
L1, L2 = 3.5, 7.0  # larguras de uma e duas colunas (polegadas)

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "lines.linewidth": 1.3, "lines.markersize": 4.5, "legend.frameon": False, "legend.fontsize": 7,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "savefig.dpi": 600,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "figure.constrained_layout.use": True,
})


# ---------------------------------------------------------------- teoria
def hj(phi, alfa, kp, km=K_M):
    """Hasselman-Johnson, devolve k_eff/k_m."""
    a = kp * (1 + 2 * alfa) + 2 * km
    b = kp * (1 - alfa) - km
    return (a + 2 * phi * b) / (a - phi * b)


def k_eq(kp, kc, delta):
    """Esfera revestida: casca kp, núcleo kc, delta = (r_i/R)^3."""
    return kp * ((kc + 2 * kp) + 2 * delta * (kc - kp)) / ((kc + 2 * kp) - delta * (kc - kp))


def oca_mesma_massa(tR, phi_s, kp, rk, kc=K_AR):
    delta = (1 - tR) ** 3
    return hj(phi_s / (1 - delta), rk * K_M / (R_NM * 1e-9), k_eq(kp, kc, delta))


def oca_mesmo_phi(tR, phi, kp, rk, kc=K_AR):
    return hj(phi, rk * K_M / (R_NM * 1e-9), k_eq(kp, kc, (1 - tR) ** 3))


def k_nc(d_nm, k0, ell_nm, rgb=1e-9):
    kg = k0 / (1 + ell_nm / d_nm)
    return kg / (1 + rgb * kg / (d_nm * 1e-9))


def k_knudsen(d_nm, beta=1.64, ell=68.0):
    return K_AR / (1 + 2 * beta * ell / d_nm)


def t_critico(kp, rk=1e-8, kc=K_AR):
    """t_c/R; nan se a casca nunca cruza k_m."""
    ks = K_M / (1 - rk * K_M / (R_NM * 1e-9))
    d = (2 * kp + kc) * (kp - ks) / ((kp - kc) * (2 * kp + ks))
    return 1 - d ** (1 / 3) if 0 < d < 1 else float("nan")


# ---------------------------------------------------------------- utilitários
def ler(rel):
    p = RAIZ / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(linhas, *cols):
    return [np.array([float(x[c]) for x in linhas]) for c in cols]


def salvar(fig, nome):
    SAIDA.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(SAIDA / f"{nome}.{ext}")
    plt.close(fig)
    print("  gravada", nome)


def rk_txt(rk):
    e = int(round(np.log10(rk)))
    m = rk / 10 ** e
    return (rf"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else rf"${m:g}\times10^{{{e}}}$")


# ---------------------------------------------------------------- figuras
def fig02():
    """Convergência de malha: diferença de cada nível para o valor extrapolado."""
    d = ler("04_duas_particulas/estudos/uma_esfera/uma_esfera.csv")
    fig, ax = plt.subplots(figsize=(L1, 2.5))
    phis = sorted({float(x["phi_percent"]) for x in d})
    for cor, p in zip(RAMPA, phis):
        s = [x for x in d if float(x["phi_percent"]) == p]
        k1, k2, k3, ke = num(s, "k_1", "k_2", "k_3", "k_extrapolado")
        y = [np.median(np.abs(k / ke - 1)) * 100 for k in (k1, k2, k3)]
        ax.semilogy([1, 2, 3], y, "o-", color=cor, label=f"{p:.2g}")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Mesh level")
    ax.set_ylabel(r"$|k_i/k_\mathrm{extr}-1|$ (%)")
    ax.legend(title=r"$\phi$ (%)", ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    salvar(fig, "fig02_convergence")


def fig03():
    """k_eff/k_m contra alfa para as sete frações; cruzamento em alfa_c."""
    d = ler("04_duas_particulas/estudos/uma_esfera/uma_esfera.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(L2, 2.6))
    phis = sorted({float(x["phi_percent"]) for x in d})
    for cor, p in zip(RAMPA, phis):
        s = sorted((x for x in d if float(x["phi_percent"]) == p), key=lambda x: float(x["alpha"]))
        a, ke, kh, er = num(s, "alpha", "k_extrapolado", "k_HJ", "erro_HJ_percent")
        ax.semilogx(a, kh / K_M, "-", color=cor, lw=0.9)
        ax.semilogx(a, ke / K_M, "o", color=cor, ms=3, label=f"{p:.2g}")
        bx.semilogx(a, er, "o-", color=cor, ms=3, lw=0.8)
    ax.axhline(1, color=CINZA, lw=0.6)
    ax.axvline(0.96, color=CINZA, lw=0.6, ls="--")
    ax.annotate(r"$\alpha_c = 1-k_m/k_p$", (0.96, 1), xytext=(5, 70), textcoords="offset points",
                ha="left", color=CINZA)
    ax.set_xlabel(r"$\alpha = R_K k_m/R$")
    ax.set_ylabel(r"$k_\mathrm{eff}/k_m$")
    ax.legend(title=r"$\phi$ (%)  (symbols FEM, lines HJ)", ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    bx.set_xlabel(r"$\alpha$")
    bx.set_ylabel(r"$(k_\mathrm{FEM}-k_\mathrm{HJ})/k_\mathrm{HJ}$ (%)")
    ax.set_title("(a)", loc="left"); bx.set_title("(b)", loc="left")
    salvar(fig, "fig03_critical_alpha")


def fig04():
    """R_K (AMM e DMM) contra a temperatura."""
    d = ler("07_resistencia_interface/resultados/rk_amm_dmm.csv")
    nomes = {"anatase": (r"TiO$_2$ anatase", AZUL), "rutilo": (r"TiO$_2$ rutile", LARANJA),
             "gama_alumina": (r"$\gamma$-Al$_2$O$_3$", VERDE), "alfa_alumina": (r"$\alpha$-Al$_2$O$_3$", VIOLETA)}
    fig, ax = plt.subplots(figsize=(L1, 2.6))
    for p, (rot, cor) in nomes.items():
        for mod, ls in (("DMM", "-"), ("AMM", "--")):
            s = [x for x in d if x["particula"] == p and x["modelo"] == mod]
            T, r = num(s, "T_K", "R_K_m2K_W")
            ax.plot(T, r * 1e8, ls, color=cor, label=rot if mod == "DMM" else None)
    ax.plot([], [], "-", color=CINZA, label="DMM"); ax.plot([], [], "--", color=CINZA, label="AMM")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$R_K$ ($10^{-8}$ m$^2$ K W$^{-1}$)")
    ax.set_ylim(0)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    salvar(fig, "fig04_rk_mismatch")


def fig05():
    """Par de esferas: desvio de k_par e k_perp em relação a HJ contra s/R."""
    d = ler("04_duas_particulas/estudos/varredura/varredura_rks_L7R.csv")
    fig, ax = plt.subplots(figsize=(L1, 2.7))
    for rk, cor, mk in ((1e-10, AZUL, "o"), (1e-8, LARANJA, "s"), (1e-6, VERDE, "^")):
        for ori, ls, cheio in (("longitudinal", "-", True), ("transversal", "--", False)):
            s = sorted((x for x in d if x["orientacao"] == ori and abs(float(x["R_K"]) / rk - 1) < 1e-6),
                       key=lambda x: float(x["s_R"]))
            if not s:
                continue
            sR, k, phi = num(s, "s_R", "k_extrapolado", "phi")
            ref = K_M * hj(phi, rk * K_M / (R_NM * 1e-9), 5.0)
            ax.semilogx(sR, (k / ref - 1) * 100, ls, marker=mk, color=cor,
                        mfc=cor if cheio else "white", label=rk_txt(rk) if cheio else None)
    ax.plot([], [], "-", color=CINZA, label=r"$k_\parallel$"); ax.plot([], [], "--", color=CINZA, label=r"$k_\perp$")
    ax.axhline(0, color=CINZA, lw=0.6)
    ax.set_xlabel(r"Surface separation $s/R$")
    ax.set_ylabel(r"Deviation from $k_\mathrm{HJ}$ (%)")
    ax.legend(title=r"$R_K$ (m$^2$ K W$^{-1}$)", ncol=2)
    salvar(fig, "fig05_pair")


def fig06():
    """Aglomerados: desvio médio em relação a HJ (phi = 2,44 %, s = 0,05R)."""
    d = ler("05_N_particulas/estudos/aglomerados/aglomerados.csv")
    ordem = [("par", "Pair"), ("triangulo", "Triangle"), ("quadrado", "Square"), ("tetraedro", "Tetrahedron"),
             ("octaedro", "Octahedron"), ("cubo", "Cube"), ("cadeia3", "Chain of 3"), ("cadeia4", "Chain of 4")]
    fig, ax = plt.subplots(figsize=(L1, 2.5))
    for rk, cor, mk in ((1e-10, AZUL, "o"), (1e-8, LARANJA, "s")):
        y, v = [], []
        for i, (c, _) in enumerate(ordem):
            s = [x for x in d if x["configuracao"] == c and abs(float(x["phi"]) - 0.02442443) < 1e-6
                 and float(x["s_R"]) == 0.05 and abs(float(x["R_K"]) / rk - 1) < 1e-6]
            if s:
                y.append(i); v.append(float(s[0]["desvio_HJ_percent"]))
        ax.plot(v, y, mk, color=cor, ms=6, label=rk_txt(rk))
    ax.set_yticks(range(len(ordem))); ax.set_yticklabels([r for _, r in ordem]); ax.invert_yaxis()
    ax.set_xlim(0)
    ax.set_xlabel(r"Deviation of tr$(K_\mathrm{eff})/3$ from $k_\mathrm{HJ}$ (%)")
    ax.legend(title=r"$R_K$ (m$^2$ K W$^{-1}$)", loc="upper right")
    salvar(fig, "fig06_clusters")


def _liquido(arq, n_sel):
    """Desvio líquido (dispersão menos esfera isolada na mesma malha): {(rk, phi): (média, erro padrão)}."""
    d = ler(arq)
    res = {}
    for rk in (1e-10, 1e-8):
        for phi in sorted({float(x["phi"]) for x in d}):
            s = [x for x in d if float(x["phi"]) == phi and abs(float(x["R_K"]) / rk - 1) < 1e-6]
            um = [float(x["desvio_HJ_percent"]) for x in s if x["N"] == "1"]
            v = np.array([float(x["desvio_HJ_percent"]) for x in s if int(x["N"]) in n_sel])
            if len(um) and len(v):
                res[(rk, phi)] = (v.mean() - um[0], v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return res


def fig07():
    """Dispersões aleatórias: desvio líquido contra phi, lei c phi^2 e teoria de pares."""
    baixo = _liquido("06_dispersao_aleatoria/estudos/aleatorio/aleatorio.csv", {8})
    try:
        alto = _liquido("06_dispersao_aleatoria/estudos/aleatorio_n/aleatorio_alto.csv", {27, 64})
    except FileNotFoundError:
        alto = {}
    fig, ax = plt.subplots(figsize=(L1, 2.8))
    x = np.logspace(-2, np.log10(0.3), 50)
    for rk, cor, mk, c_fem, c_par in ((1e-10, AZUL, "o", 0.78, 0.87), (1e-8, LARANJA, "s", 0.50, 0.57)):
        ax.loglog(x * 100, c_fem * x ** 2 * 100, "-", color=cor, lw=1.0)
        ax.loglog(x * 100, c_par * x ** 2 * 100, ":", color=cor, lw=1.0)
        for dados, cheio in ((baixo, True), (alto, False)):
            pts = sorted((p, v) for (r, p), v in dados.items() if r == rk)
            if pts:
                ax.errorbar([p * 100 for p, _ in pts], [v[0] for _, v in pts], yerr=[v[1] for _, v in pts],
                            fmt=mk, color=cor, mfc=cor if cheio else "white", capsize=2, elinewidth=0.7,
                            label=rk_txt(rk) if cheio else None)
    ax.plot([], [], "o", color=CINZA, label="N = 8, fine meshes")
    ax.plot([], [], "o", color=CINZA, mfc="white", label="N = 27, 64, coarse meshes")
    ax.plot([], [], "-", color=CINZA, label=r"$c\phi^2$ (fit, $\phi\leq10\%$)")
    ax.plot([], [], ":", color=CINZA, label="pair theory")
    ax.set_xlabel(r"$\phi$ (%)")
    ax.set_ylabel(r"Net deviation from $k_\mathrm{HJ}$ (%)")
    ax.legend(title=r"$R_K$ (m$^2$ K W$^{-1}$)", loc="upper left")
    salvar(fig, "fig07_random")


def fig08():
    """Esferas ocas com a mesma massa: k_eff/k_m contra t/R, teoria e FEM."""
    casos = [("SiO$_2$", 1.36, "09_particulas_ocas/estudos/ocas/ocas_sio2.csv", AZUL, "o", 0.035),
             (r"$\gamma$-Al$_2$O$_3$ ($k_p$ = 5)", 5.0, "09_particulas_ocas/estudos/ocas/ocas_gama.csv", LARANJA, "s", 0.02),
             ("ZIF-8", 0.33, "09_particulas_ocas/estudos/ocas/zif8.csv", VERDE, "^", 0.07)]
    fig, ax = plt.subplots(figsize=(L1, 2.8))
    for rot, kp, arq, cor, mk, tmin in casos:
        try:
            s = [x for x in ler(arq) if x["modo"] == "mesma_massa" and abs(float(x["R_K"]) / 1e-8 - 1) < 1e-6]
        except FileNotFoundError:
            print("  (sem", arq, ")"); continue
        phi_s = float(s[0]["phi_solido"])
        t = np.logspace(np.log10(tmin), 0, 200)
        ax.semilogx(t, oca_mesma_massa(t, phi_s, kp, 1e-8), "-", color=cor)
        tR, k = num(s, "t_R", "k_sobre_km")
        ax.semilogx(tR, k, mk, color=cor, label=rot)
        tc = t_critico(kp)
        if tc == tc and tc > tmin:
            ax.plot([tc], [1], "|", color=cor, ms=12, mew=1.2)
    # gás de Knudsen na cavidade (sílica)
    t = np.logspace(np.log10(0.035), np.log10(0.6), 100)
    kn = k_knudsen(2 * R_NM * (1 - t))
    ax.semilogx(t, oca_mesma_massa(t, 0.0346, 1.36, 1e-8, kn), "--", color=AZUL, lw=1.0)
    pts = []
    for f in ("t020", "t010", "t005"):
        try:
            pts += ler(f"09_particulas_ocas/estudos/ocas/knudsen_sio2_{f}.csv")
        except FileNotFoundError:
            pass
    if pts:
        tR, k = num(pts, "t_R", "k_sobre_km")
        ax.semilogx(tR, k, "o", color=AZUL, mfc="white", label=r"SiO$_2$, Knudsen gas")
    ax.axhline(1, color=CINZA, lw=0.6)
    ax.set_xlabel(r"Relative wall thickness $t/R$")
    ax.set_ylabel(r"$k_\mathrm{eff}/k_m$")
    ax.set_ylim(0.78, 1.16)
    ax.legend(title="Symbols FEM, lines theory; vertical ticks: $t_c$", ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    salvar(fig, "fig08_hollow_fixed_mass")


def _nc(arq_uma, arq_mapa):
    mapa = {round(float(x["k_p"]), 4): float(x["d_nm"]) for x in ler(arq_mapa)}
    s = ler(arq_uma)
    d = np.array([mapa[round(float(x["k_p"]), 4)] for x in s])
    return d, num(s, "k_sobre_km")[0], float(s[0]["phi_solido"])


def fig09():
    """Partículas nanocristalinas maciças e ocas: k_eff/k_m contra o tamanho do grão."""
    P = "12_nanocristalino/estudos/"
    casos = [("SiO$_2$ solid", "sio2", 1.36, 0.6, 1.0, AZUL, "o", True),
             (r"$\gamma$-Al$_2$O$_3$ solid", "gama_k0_5", 5.0, 1.0, 1.0, LARANJA, "s", True),
             ("Quartz solid", "quartzo", 8.0, 3.0, 1.0, VERDE, "^", True),
             ("SiO$_2$ hollow, $t/R$ = 0.2", "sio2_oca_tR_0.2", 1.36, 0.6, 0.2, AZUL, "o", False),
             ("SiO$_2$ hollow, $t/R$ = 0.1", "sio2_oca_tR_0.1", 1.36, 0.6, 0.1, VIOLETA, "D", False)]
    fig, ax = plt.subplots(figsize=(L1, 2.8))
    for rot, chave, k0, ell, tR, cor, mk, cheio in casos:
        try:
            d, k, phi_s = _nc(P + f"nc_uma_{chave}.csv", P + f"nc_mapa_{chave}.csv")
        except FileNotFoundError:
            print("  (sem nc_uma_" + chave + ")"); continue
        dd = np.logspace(0, np.log10(25 if tR == 1 else tR * R_NM), 100)
        ax.semilogx(dd, oca_mesma_massa(tR, phi_s, k_nc(dd, k0, ell), 1e-8), "-" if cheio else "--", color=cor, lw=1.0)
        ax.semilogx(d, k, mk, color=cor, mfc=cor if cheio else "white", label=rot)
        teo = oca_mesma_massa(tR, phi_s, k_nc(d, k0, ell), 1e-8)
        print(f"    {chave}: maior diferença FEM-teoria {np.max(np.abs(k / teo - 1)) * 100:.3f} %")
    ax.axhline(1, color=CINZA, lw=0.6)
    ax.set_xlabel("Grain size $d$ (nm)")
    ax.set_ylabel(r"$k_\mathrm{eff}/k_m$")
    ax.legend(loc="lower right")
    salvar(fig, "fig09_nanocrystalline")


def fig10():
    """Mapa de projeto: k_eff/k_m no plano (t/R, d) para sílica, carga de referência."""
    phi_s = 0.0346
    t = np.logspace(np.log10(0.035), 0, 240)
    d = np.logspace(0, np.log10(50), 200)
    T, D = np.meshgrid(t, d)
    Z = oca_mesma_massa(T, phi_s, k_nc(D, 1.36, 0.6), 1e-8)
    Z = np.ma.masked_where(D > T * R_NM, Z)  # o grão não pode ser maior que a parede
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#86b6ef", "#f0efec", "#f3a583", "#b23f12"])
    fig, ax = plt.subplots(figsize=(L1, 2.8))
    pc = ax.pcolormesh(T, D, Z, cmap=cmap, norm=TwoSlopeNorm(1.0, 0.80, 1.08), shading="auto", rasterized=True)
    cs = ax.contour(T, D, Z, levels=[0.85, 0.90, 0.95, 1.0, 1.03, 1.06], colors="#0b0b0b",
                    linewidths=[0.5, 0.5, 0.5, 1.4, 0.5, 0.5])
    ax.clabel(cs, fmt="%.2f", fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(False)
    ax.text(0.04, 30, "grain larger\nthan wall", color=CINZA, fontsize=7, va="center")
    ax.set_xlabel(r"Relative wall thickness $t/R$")
    ax.set_ylabel("Grain size $d$ (nm)")
    fig.colorbar(pc, ax=ax, label=r"$k_\mathrm{eff}/k_m$", extend="both")
    salvar(fig, "fig10_design_map")


def fig11():
    """Sensibilidade a k_p (maciça, phi = 2,1 %), com os valores da literatura marcados."""
    fig, ax = plt.subplots(figsize=(L1, 2.7))
    kp = np.logspace(-1.1, 2, 200)
    pts = []
    for v in ("0.1", "0.3", "1", "2", "10", "35"):
        try:
            pts += ler(f"09_particulas_ocas/estudos/ocas/varredura_kp_{v}.csv")
        except FileNotFoundError:
            pass
    phi = float(pts[0]["phi_externo"]) if pts else 0.0212
    for rk, cor, mk in ((1e-10, AZUL, "o"), (1e-8, LARANJA, "s")):
        ax.semilogx(kp, hj(phi, rk * K_M / (R_NM * 1e-9), kp), "-", color=cor)
        s = [x for x in pts if abs(float(x["R_K"]) / rk - 1) < 1e-6]
        if s:
            x, y = num(s, "k_p", "k_sobre_km")
            ax.semilogx(x, y, mk, color=cor, label=rk_txt(rk))
    ax.axhline(1, color=CINZA, lw=0.6)
    y0 = 0.972
    for rot, a, b in (("ZIF-8", 0.33, None), ("SiO$_2$", 1.36, None), (r"$\gamma$-Al$_2$O$_3$", 1, 5),
                      ("TiO$_2$", 8.5, None), (r"$\alpha$-Al$_2$O$_3$", 35, None)):
        if b:
            ax.plot([a, b], [y0, y0], "-", color=CINZA, lw=3, solid_capstyle="butt")
        else:
            ax.plot([a], [y0], "|", color=CINZA, ms=8, mew=1.2)
        ax.text(np.sqrt(a * b) if b else a, y0 - 0.004 if b else y0 + 0.004, rot, ha="center",
                va="top" if b else "bottom", fontsize=6, color=CINZA)
    ax.set_ylim(0.958, 1.07)
    ax.set_xlabel(r"Particle conductivity $k_p$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_ylabel(r"$k_\mathrm{eff}/k_m$")
    ax.legend(title=r"$R_K$ (m$^2$ K W$^{-1}$)", loc="center right")
    salvar(fig, "fig11_kp_sensitivity")


def fig12():
    """Verificação da espessura crítica: o cruzamento não depende de phi."""
    fig, eixos = plt.subplots(1, 2, figsize=(L2, 2.7))
    for ax, (rot, chave, kp, lim) in zip(eixos, (("ZIF-8", "zif8", 0.33, (0.2, 0.5)), ("SiO$_2$", "sio2", 1.36, (0.04, 0.13)))):
        tc = t_critico(kp)
        t = np.linspace(*lim, 100)
        tem = False
        for p, cor, mk in (("0.05", RAMPA[1], "o"), ("0.10", RAMPA[3], "s"), ("0.20", RAMPA[6], "^")):
            ax.plot(t, oca_mesmo_phi(t, float(p), kp, 1e-8), "-", color=cor, lw=1.0)
            try:
                s = ler(f"09_particulas_ocas/estudos/ocas/tc_{chave}_phi{p}.csv")
            except FileNotFoundError:
                continue
            tR, k = num(s, "t_R", "k_sobre_km")
            ax.plot(tR, k, mk, color=cor, label=f"{float(p) * 100:.0f}")
            tem = True
        ax.axhline(1, color=CINZA, lw=0.6); ax.axvline(tc, color=CINZA, lw=0.6, ls="--")
        ax.annotate(rf"$t_c/R$ = {tc:.3f}", (tc, 1), xytext=(5, -28), textcoords="offset points", color=CINZA)
        ax.set_xlabel(r"Relative wall thickness $t/R$"); ax.set_ylabel(r"$k_\mathrm{eff}/k_m$")
        ax.set_title(rot, loc="left")
        if tem:
            ax.legend(title=r"$\phi$ (%)  (symbols FEM)", loc="upper left")
        else:
            print(f"  (ainda sem FEM de t_c para {chave})")
    salvar(fig, "fig12_critical_thickness")


def fig13():
    """Campo em torno da esfera oca acima, em e abaixo de t_c (dados de 09/estudos/ocas/cortes_tc.py)."""
    from matplotlib.patches import Circle
    P = "09_particulas_ocas/estudos/ocas/dados/"
    casos = []
    for t in ("0.5", "0.3187", "0.15"):
        arq = RAIZ / P / f"tc_phi0.1_rk1e-08_t{t}_kp0.33_kn0.026.npz"
        if not arq.exists():
            raise FileNotFoundError(arq.name)
        casos.append(dict(np.load(arq)))
    borda = 1.035  # esconde a primeira camada de células (DG0 serrilhado) junto à interface
    prep = []
    for d in casos:
        meio = float(d["lado"]) / 2
        X, Y = d["U"] - meio, d["V"] - meio
        r = np.hypot(X, Y)
        theta = d["T"] + d["U"]
        theta = theta - np.mean(theta[r > 1])
        dq = d["q_modulo"] - 1.0
        # na coroa 1 < r < borda repete o valor em r = borda na mesma direção
        x0, dx = d["U"][0, 0], d["U"][0, 1] - d["U"][0, 0]
        y0, dy = d["V"][0, 0], d["V"][1, 0] - d["V"][0, 0]
        f = np.where((r > 0.5) & (r < borda), borda / np.maximum(r, 1e-9), 1.0)
        j = np.clip(np.rint((meio + X * f - x0) / dx).astype(int), 0, r.shape[1] - 1)
        i = np.clip(np.rint((meio + Y * f - y0) / dy).astype(int), 0, r.shape[0] - 1)
        prep.append((d, meio, r, theta[i, j], dq[i, j]))
    lim_t = max(np.percentile(np.abs(th[r > borda]), 99.5) for _, _, r, th, _ in prep)
    lim_q = max(np.percentile(np.abs(q[r > borda]), 99.5) for _, _, r, _, q in prep)
    fig, eixos = plt.subplots(2, 3, figsize=(L2, 4.4))
    rot = [r"$t = 0.50R > t_c$", r"$t = t_c = 0.319R$", r"$t = 0.15R < t_c$"]
    for k, (d, meio, r, theta, dq) in enumerate(prep):
        ext = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
        m = r > 1
        e1, e2 = eixos[0, k], eixos[1, k]
        im1 = e1.imshow(np.where(m, theta, np.nan), origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-lim_t, vmax=lim_t, interpolation="bilinear", rasterized=True)
        e1.contour(d["U"], d["V"], np.where(r > borda, theta, np.nan), levels=np.linspace(-lim_t, lim_t, 13),
                   colors=CINZA, linewidths=0.3)
        im2 = e2.imshow(np.where(m, dq, np.nan), origin="lower", extent=ext, cmap="PuOr_r",
                        vmin=-lim_q, vmax=lim_q, interpolation="bilinear", rasterized=True)
        e2.streamplot(d["U"][0, :], d["V"][:, 0], d["q_u"], d["q_v"], color=(0.15, 0.17, 0.27, 0.5),
                      linewidth=0.4, density=0.9, arrowsize=0.4, broken_streamlines=False)
        for e in (e1, e2):
            e.add_patch(Circle((meio, meio), 1.0, facecolor="#e8e6e1", edgecolor="#0b0b0b", lw=0.6, zorder=3))
            e.add_patch(Circle((meio, meio), float(d["r_i"]), facecolor="white", edgecolor=CINZA, lw=0.5, ls="--", zorder=4))
            e.set_xlim(ext[:2]); e.set_ylim(ext[2:]); e.set_aspect("equal"); e.grid(False)
            e.set_xticks([0, 1, 2, 3]); e.set_yticks([0, 1, 2, 3])
            for lado in ("top", "right"):
                e.spines[lado].set_visible(True)
        e2.set_xlabel("$x/R$")
        if k == 0:
            e1.set_ylabel("$y/R$"); e2.set_ylabel("$y/R$")
        e1.set_title(f"{rot[k]}\n" + rf"$k_\mathrm{{eff}}/k_m$ = {float(d['k_xx']) / float(d['k_m']):.4f}", fontsize=8)
    fig.colorbar(im1, ax=eixos[0, :].tolist(), shrink=0.9, label=r"$(T + Gx)/(GR)$")
    fig.colorbar(im2, ax=eixos[1, :].tolist(), shrink=0.9, label=r"$|q|/(k_m G) - 1$")
    salvar(fig, "fig13_field_perturbation")


def fig01():
    """Esquema: célula periódica, arranjos estudados e arquiteturas de partícula."""
    from matplotlib.patches import Circle, Polygon, Wedge, FancyArrowPatch
    csv.field_size_limit(10 ** 9)
    ESC, PART, BORDA, FUNDO, GAS = "#0b0b0b", "#86b6ef", "#184f95", "#f0efec", "#fbfbfa"
    fig = plt.figure(figsize=(L2, 2.45), layout="none")

    def eixo(ret):
        ax = fig.add_axes(ret)
        ax.set_aspect("equal"); ax.axis("off")
        return ax

    def seta(ax, a, b, txt=None, pos=None, **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=kw.pop("estilo", "-|>"), mutation_scale=7,
                                     lw=0.7, color=kw.pop("cor", ESC), shrinkA=0, shrinkB=0))
        if txt:
            ax.text(*pos, txt, ha=kw.get("ha", "center"), va=kw.get("va", "center"), fontsize=kw.get("fs", 7))

    def esfera(ax, c, r, z=1.0, cor=PART):
        ax.add_patch(Circle(c, r, facecolor=cor, edgecolor=BORDA, lw=0.6, alpha=z, zorder=3))
        ax.add_patch(Circle((c[0] - 0.35 * r, c[1] + 0.35 * r), 0.3 * r, facecolor="white", edgecolor="none",
                            alpha=0.45 * z, zorder=3))

    # ---- (a) célula periódica
    ax = eixo([0.0, 0.04, 0.30, 0.86])
    dx, dy = 0.38, 0.28
    frente = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    ax.add_patch(Polygon(frente + [dx, dy], closed=True, facecolor=FUNDO, edgecolor=CINZA, lw=0.5, ls="--"))
    ax.add_patch(Polygon(frente, closed=True, facecolor=FUNDO, edgecolor=ESC, lw=0.7, alpha=0.55))
    for p in ([1, 0], [1, 1], [0, 1]):
        ax.plot([p[0], p[0] + dx], [p[1], p[1] + dy], color=ESC, lw=0.7)
    ax.plot([0, dx], [0, dy], color=CINZA, lw=0.5, ls="--")
    ax.plot([1 + dx, 1 + dx, dx], [dy, 1 + dy, 1 + dy], color=ESC, lw=0.7)
    c = (0.5 + dx / 2, 0.5 + dy / 2)
    esfera(ax, c, 0.27)
    ax.text(*c, "$k_p$", ha="center", va="center", fontsize=8, zorder=5)
    ax.text(0.72, 0.07, "$k_m$", fontsize=8)
    ax.text(0.5, -0.40, r"interface: $T_m - T_p = R_K\,q_n$", ha="center", va="top", fontsize=6.5)
    seta(ax, (0.0, -0.17), (1.0, -0.17), estilo="<|-|>")
    ax.text(0.5, -0.27, "$L$", ha="center", va="top", fontsize=7)
    seta(ax, (0.05, 1.5), (0.75, 1.5))
    ax.text(0.80, 1.5, r"$-\nabla T = G\,\mathbf{e}_x$", va="center", fontsize=7)
    ax.text(0.0, 1.30, r"periodic: $T(x+L) = T(x) - GL$", fontsize=6.5)
    ax.set_xlim(-0.05, 1.75); ax.set_ylim(-0.62, 1.75)
    ax.text(-0.05, 1.75, "(a)", fontsize=8, fontweight="bold", va="top")

    # ---- (b) arranjos
    ax = eixo([0.31, 0.53, 0.15, 0.36])  # par
    esfera(ax, (0, 0), 1); esfera(ax, (2.35, 0), 1)
    seta(ax, (1.0, -1.25), (1.35, -1.25), estilo="|-|")
    ax.text(1.175, -1.45, "$s$", ha="center", va="top", fontsize=7)
    ax.text(1.175, 1.35, "pair", ha="center", fontsize=7)
    ax.set_xlim(-1.2, 3.55); ax.set_ylim(-2.0, 1.9)
    ax.text(-1.2, 2.6, "(b)", fontsize=8, fontweight="bold", va="top")

    ax = eixo([0.31, 0.06, 0.15, 0.40])  # aglomerado (tetraedro projetado)
    a = 2.1
    pts = np.array([[0, 0, 0], [a, 0, 0], [a / 2, a * 0.866, 0], [a / 2, a * 0.289, a * 0.816]])
    pts = pts - pts.mean(0)
    rot = np.array([[1, 0, 0], [0, np.cos(1.0), -np.sin(1.0)], [0, np.sin(1.0), np.cos(1.0)]])
    p3 = pts @ rot.T
    for i in np.argsort(p3[:, 2]):
        esfera(ax, (p3[i, 0], p3[i, 1]), 1)
    ax.text(0, -2.6, "clusters, $N$ = 3 to 8", ha="center", fontsize=7)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-3.0, 2.4)

    ax = eixo([0.465, 0.06, 0.215, 0.84])  # dispersão aleatória
    try:
        lin = [x for x in ler("06_dispersao_aleatoria/estudos/aleatorio_n/aleatorio_n.csv")
               if x["N"] == "27" and float(x["phi"]) == 0.10][0]
        L = float(lin["lado_R"]); cs = np.array(eval(lin["centros_R"]))
    except (FileNotFoundError, IndexError):
        L, cs = 10.4, np.random.default_rng(0).uniform(1.3, 9.1, (27, 3))
    ax.add_patch(Polygon([[0, 0], [L, 0], [L, L], [0, L]], closed=True, facecolor=FUNDO, edgecolor=ESC, lw=0.7))
    for i in np.argsort(cs[:, 2]):
        z = 0.35 + 0.65 * cs[i, 2] / L
        esfera(ax, (cs[i, 0], cs[i, 1]), 1, z=z)
    ax.text(L / 2, -0.6, rf"random, $N$ = 27, $\phi$ = 10%", ha="center", va="top", fontsize=7)
    ax.set_xlim(-0.3, L + 0.3); ax.set_ylim(-2.0, L + 0.3)

    # ---- (c) arquiteturas
    ax = eixo([0.69, 0.04, 0.31, 0.86])
    xs = (0, 2.7, 5.4)
    esfera(ax, (xs[0], 0), 1)
    ax.text(xs[0], 0, "$k_p$", ha="center", va="center", fontsize=7, zorder=5)
    ax.text(xs[0], -1.35, "solid", ha="center", va="top", fontsize=7)
    t = 0.3
    for x0, granular in ((xs[1], False), (xs[2], True)):
        ax.add_patch(Circle((x0, 0), 1, facecolor=PART, edgecolor=BORDA, lw=0.6, zorder=3))
        ax.add_patch(Circle((x0, 0), 1 - t, facecolor=GAS, edgecolor=BORDA, lw=0.5, zorder=4))
        if granular:
            n = int(round(2 * np.pi * (1 - t / 2) / t))
            for k in range(n):
                th = 2 * np.pi * k / n + 0.2
                ax.plot([x0 + (1 - t) * np.cos(th), x0 + np.cos(th)], [(1 - t) * np.sin(th), np.sin(th)],
                        color=BORDA, lw=0.5, zorder=5)
            ax.text(x0, 0.05, "$k_c$", ha="center", va="center", fontsize=7, zorder=6)
            th = 2 * np.pi * 2 / n + 0.2
            seta(ax, (x0 + 1.25 * np.cos(th) + 0.25, 1.25 * np.sin(th) + 0.35),
                 (x0 + (1 - t / 2) * np.cos(th), (1 - t / 2) * np.sin(th)), estilo="-")
            ax.text(x0 + 1.25 * np.cos(th) + 0.25, 1.25 * np.sin(th) + 0.4, "$R_{gb}$", fontsize=7, va="bottom")
            ax.text(x0 + 1.15, -0.55, "grain\n$d \\approx t$", fontsize=6.5, va="center")
            ax.text(x0, -1.35, "granular shell", ha="center", va="top", fontsize=7)
        else:
            ax.text(x0, 0.05, "$k_c$", ha="center", va="center", fontsize=7, zorder=6)
            th = np.deg2rad(140)
            seta(ax, (x0, 0), (x0 + np.cos(th), np.sin(th)), cor=ESC)
            ax.text(x0 - 0.45, 0.22, "$R$", fontsize=7, ha="center", va="bottom", zorder=6)
            th = np.deg2rad(-35)
            seta(ax, (x0 + 1.45 * np.cos(th), 1.45 * np.sin(th)), (x0 + (1 - t / 2) * np.cos(th), (1 - t / 2) * np.sin(th)),
                 estilo="-")
            ax.text(x0 + 1.5 * np.cos(th), 1.5 * np.sin(th) - 0.05, "$t$", fontsize=7, va="top")
            ax.text(x0, -1.35, "hollow", ha="center", va="top", fontsize=7)
    ax.text(xs[1], 1.25, "shell $k_p$, core gas $k_c$", ha="center", va="bottom", fontsize=6.5)
    ax.set_xlim(-1.2, 7.0); ax.set_ylim(-2.2, 2.6)
    ax.text(-1.2, 2.6, "(c)", fontsize=8, fontweight="bold", va="top")
    salvar(fig, "fig01_schematic")


FIGURAS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10, 11: fig11, 12: fig12, 13: fig13}

if __name__ == "__main__":
    quais = [int(a) for a in sys.argv[1:]] or sorted(FIGURAS)
    for n in quais:
        print(f"Figura {n}: {FIGURAS[n].__doc__.splitlines()[0]}")
        try:
            FIGURAS[n]()
        except FileNotFoundError as e:
            print("  pulada, falta o arquivo", e)
