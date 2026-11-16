"""Gera os gráficos de todos os estudos do 04_duas_particulas e dos aglomerados do 05.

Lê os CSVs e logs em estudos/ e PNG (para slides)
em estudos/graficos/figuras/. Um gráfico cujo arquivo de dados ainda não
existe é pulado com um aviso, então o script pode ser rodado a qualquer
momento. Não precisa de FEniCSx, só de matplotlib e numpy:

    cd ~/fenics/kapitza/04_duas_particulas
    python3 estudos/graficos/graficos.py
"""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# A pasta kapitza é o primeiro ancestral que contém 04_duas_particulas, então o
# script funciona tanto em 04/estudos/graficos quanto numa cópia no 05.
KAPITZA = next(p for p in Path(__file__).resolve().parents if (p / "04_duas_particulas").is_dir())
ESTUDOS = KAPITZA / "04_duas_particulas" / "estudos"
AGLOMERADOS = KAPITZA / "05_N_particulas" / "estudos" / "aglomerados" / "aglomerados.csv"
# Pares de esferas vizinhas (a lacuna s) em cada aglomerado.
PARES_VIZINHOS = {"par": 1, "cadeia3": 2, "cadeia4": 3, "triangulo": 3, "quadrado": 4,
                  "tetraedro": 6, "octaedro": 12, "cubo": 12}
NOMES_AGLOMERADOS = {"par": "par", "cadeia3": "cadeia 3", "cadeia4": "cadeia 4",
                     "triangulo": "triângulo", "quadrado": "quadrado", "tetraedro": "tetraedro",
                     "octaedro": "octaedro", "cubo": "cubo"}
FIGURAS = Path(__file__).resolve().parent / "figuras"

K_M, K_P, RAIO = 0.2, 5.0, 50e-9
ALFA_CRITICO = 1 - K_M / K_P
PARALELO, PERPENDICULAR, REFERENCIA = "#c2410c", "#2f5d9b", "#6b7385"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e2dccf", "grid.linewidth": 0.8,
    "lines.linewidth": 1.8, "lines.markersize": 6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})


def virgula(*eixos, x=True, y=True):
    """Separador decimal com vírgula nos eixos lineares."""
    formatador = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for eixo in eixos:
        if x and eixo.get_xscale() == "linear":
            eixo.xaxis.set_major_formatter(formatador)
        if y and eixo.get_yscale() == "linear":
            eixo.yaxis.set_major_formatter(formatador)


def salvar(figura, nome):
    FIGURAS.mkdir(exist_ok=True)
    for extensao in ("png",):
        figura.savefig(FIGURAS / f"{nome}.{extensao}")
    plt.close(figura)
    print(f"  {nome} .png")


def ler_csv(caminho):
    if not caminho.exists():
        print(f"  pulado: {caminho} não existe")
        return None
    with caminho.open(encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def hasselman_johnson(phi, alfa):
    a = K_P * (1 + 2 * alfa) + 2 * K_M
    b = K_P * (1 - alfa) - K_M
    return K_M * (a + 2 * phi * b) / (a - phi * b)


def aitken(k1, k2, k3):
    d1, d2 = k2 - k1, k3 - k2
    return k3 - d2 * d2 / (d2 - d1)


def ler_log(caminho):
    """Pares (h_bulk/R, k_eff) de um log de convergência."""
    if not caminho.exists():
        return []
    pares, h = [], None
    for linha in caminho.read_text(encoding="utf-8", errors="ignore").splitlines():
        if linha.startswith("== "):
            h = float(re.search(r"h_bulk=([\d.]+)", linha).group(1))
        elif linha.startswith("k_eff,xx") and h is not None:
            pares.append((h, float(linha.split(":")[1])))
            h = None
    return pares


def convergencia():
    pasta = ESTUDOS / "convergencia"
    bulk = ler_log(pasta / "conv_bulk.txt")
    globais = list(dict(ler_log(pasta / "conv_global.txt")).items())
    if not bulk:
        print("  pulado: convergencia/conv_bulk.txt não existe")
        return
    # O nível h_bulk = 0,5 da série do volume é o primeiro caso do refino global.
    if globais and all(h != globais[0][0] for h, _ in bulk):
        bulk = [globais[0]] + bulk
    bulk.sort(reverse=True)
    h, k = np.array(bulk).T
    k_ext = aitken(*k[-3:])

    figura, eixo = plt.subplots(figsize=(6, 4))
    eixo.plot(h**2, k, "o-", color=PERPENDICULAR, label="só o volume (interface R/10, lacuna R/80)")
    if len(globais) > 1:
        hg, kg = np.array(sorted(globais, reverse=True)).T
        eixo.plot(hg**2, kg, "s--", color=REFERENCIA, mfc="white",
                  label="refino global (todos os tamanhos × 0,8)")
    eixo.axhline(k_ext, color=PARALELO, ls=":", lw=1.2)
    eixo.plot([0], [k_ext], "o", color=PARALELO, ms=8, label=f"extrapolado: {k_ext:.4f}".replace(".", ","))
    eixo.set_xlim(left=-0.005)
    eixo.set_xlabel("(h$_{bulk}$/R)²")
    eixo.set_ylabel("k$_{eff}$ (W/mK)")
    eixo.set_title("Convergência de malha: s/R = 0,05, R$_K$ = 10⁻⁸ m²K/W, φ = 5 %", fontsize=11)
    eixo.legend(fontsize=9)
    virgula(eixo)
    salvar(figura, "01_convergencia")


def uma_esfera():
    linhas = ler_csv(ESTUDOS / "uma_esfera" / "uma_esfera.csv")
    if not linhas:
        return
    phis = sorted({float(l["phi_percent"]) for l in linhas})
    cores = plt.cm.viridis(np.linspace(0.05, 0.85, len(phis)))
    alfa_fino = np.logspace(np.log10(1.5e-4), np.log10(30), 300)

    figura, eixo = plt.subplots(figsize=(6.5, 4.5))
    for phi, cor in zip(phis, cores):
        dados = sorted((float(l["alpha"]), float(l["k_extrapolado"])) for l in linhas
                       if float(l["phi_percent"]) == phi)
        a, k = np.array(dados).T
        eixo.plot(alfa_fino, hasselman_johnson(phi / 100, alfa_fino) / K_M, "-", color=cor, lw=1)
        eixo.plot(a, k / K_M, "o", color=cor, ms=4, label=f"{phi:.2g} %".replace(".", ","))
    eixo.axhline(1, color=REFERENCIA, lw=1)
    eixo.axvline(ALFA_CRITICO, color=REFERENCIA, ls=":", lw=1.2)
    eixo.text(ALFA_CRITICO * 1.1, eixo.get_ylim()[1] * 0.995, "α$_c$ = 0,96", va="top", fontsize=9,
              color=REFERENCIA)
    eixo.set_xscale("log")
    eixo.set_xlabel("α = R$_K$ k$_m$ / R")
    eixo.set_ylabel("k$_{eff}$ / k$_m$")
    eixo.set_title("Uma esfera: FEM (pontos) e Hasselman–Johnson (linhas)", fontsize=11)
    eixo.legend(title="φ", fontsize=9, title_fontsize=9, loc="upper right")
    virgula(eixo)
    salvar(figura, "02_uma_esfera_k_alfa")

    comparacao = ler_csv(ESTUDOS / "uma_esfera" / "comparacao_462.csv")
    figura, eixos = plt.subplots(1, 2 if comparacao else 1, figsize=(11 if comparacao else 6, 4),
                                 sharey=False, squeeze=False)
    eixo = eixos[0, 0]
    for phi, cor in zip(phis, cores):
        dados = sorted((float(l["alpha"]), float(l["erro_HJ_percent"])) for l in linhas
                       if float(l["phi_percent"]) == phi)
        a, e = np.array(dados).T
        eixo.plot(a, e, "o-", color=cor, ms=3, lw=1, label=f"{phi:.2g} %".replace(".", ","))
    eixo.axhline(0, color=REFERENCIA, lw=1)
    eixo.set_xscale("log")
    eixo.set_xlabel("α = R$_K$ k$_m$ / R")
    eixo.set_ylabel("(k$_{FEM}$ − k$_{HJ}$) / k$_{HJ}$ (%)")
    eixo.set_title("Código novo", fontsize=11)
    baixo, cima = eixo.get_ylim()
    eixo.set_ylim(baixo - 0.45 * (cima - baixo), cima)
    eixo.legend(title="φ", fontsize=8, title_fontsize=8, ncol=4, loc="lower left")
    virgula(eixo)
    if comparacao:
        eixo = eixos[0, 1]
        antigo = np.array([abs(float(l["erro_antigo_percent"])) for l in comparacao])
        novo = np.array([abs(float(l["erro_novo_percent"])) for l in comparacao])
        caixas = eixo.boxplot([antigo, novo], tick_labels=["código antigo", "código novo"],
                              widths=0.5, patch_artist=True, showfliers=True,
                              medianprops=dict(color="#1b2333", lw=1.5),
                              flierprops=dict(marker="o", ms=4, mfc="none", mec=REFERENCIA))
        for caixa, cor in zip(caixas["boxes"], (REFERENCIA, PERPENDICULAR)):
            caixa.set_facecolor(cor)
            caixa.set_alpha(0.35)
        eixo.set_yscale("log")
        eixo.set_ylabel("|desvio de HJ| (%)")
        eixo.set_title(f"Os 462 casos: máximo {antigo.max():.2f} % → {novo.max():.3f} %"
                       .replace(".", ","), fontsize=11)
        eixo.grid(axis="x", visible=False)
    salvar(figura, "03_uma_esfera_erro")


def duas_particulas(arquivo, nome, titulo, espelho=None):
    linhas = ler_csv(ESTUDOS / "varredura" / arquivo)
    if not linhas:
        return
    phi = float(linhas[0]["phi"])
    alfa = float(linhas[0]["R_K"]) * K_M / RAIO
    figura, eixo = plt.subplots(figsize=(6, 4))
    if espelho is not None:
        eixo.axvspan(espelho, max(float(l["s_R"]) for l in linhas) * 1.05, color="#ece7dc", lw=0)
        eixo.text(espelho * 1.02, 0.02, "espelhado", transform=eixo.get_xaxis_transform(),
                  fontsize=9, color=REFERENCIA)
    medias = {}
    for orientacao, cor, rotulo in (("longitudinal", PARALELO, "k$_∥$ (ao longo do eixo)"),
                                    ("transversal", PERPENDICULAR, "k$_⊥$ (perpendicular)")):
        dados = sorted((float(l["s_R"]), float(l["k_extrapolado"])) for l in linhas
                       if l["orientacao"] == orientacao)
        s, k = np.array(dados).T
        medias[orientacao] = dict(dados)
        eixo.plot(s, k, "o-", color=cor, label=rotulo)
    comuns = sorted(set(medias["longitudinal"]) & set(medias["transversal"]))
    if comuns:
        media = [(medias["longitudinal"][s] + 2 * medias["transversal"][s]) / 3 for s in comuns]
        eixo.plot(comuns, media, "^:", color="#1b2333", ms=5, lw=1.2, label="média (k$_∥$ + 2k$_⊥$)/3")
    eixo.axhline(hasselman_johnson(phi, alfa), color=REFERENCIA, ls="--", lw=1.2,
                 label="Hasselman–Johnson")
    eixo.set_xlim(left=0)
    eixo.set_xlabel("s/R (distância entre as superfícies)")
    eixo.set_ylabel("k$_{eff}$ (W/mK)")
    eixo.set_title(titulo, fontsize=11)
    eixo.legend(fontsize=9)
    virgula(eixo)
    salvar(figura, nome)


def varredura_rk():
    linhas = ler_csv(ESTUDOS / "varredura" / "varredura_rks_L7R.csv")
    if not linhas:
        return
    rks = sorted({float(l["R_K"]) for l in linhas})
    cores = plt.cm.viridis(np.linspace(0.05, 0.9, len(rks)))
    k = {(l["orientacao"], float(l["s_R"]), float(l["R_K"])): float(l["k_extrapolado"]) for l in linhas}
    s_max = max(s for _, s, _ in k)
    if len({s for _, s, _ in k}) < 2:
        print("  (varredura de R_K: só uma distância até agora, gráficos 06 e 07 pulados)")
        return

    figura, eixos = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for eixo, orientacao, simbolo in ((eixos[0], "longitudinal", "∥"), (eixos[1], "transversal", "⊥")):
        for rk, cor in zip(rks, cores):
            referencia = k.get((orientacao, s_max, rk))
            pontos = sorted((s, 100 * (v / referencia - 1)) for (o, s, r), v in k.items()
                            if o == orientacao and r == rk and referencia)
            if len(pontos) < 2:
                continue
            s, d = np.array(pontos).T
            eixo.plot(s, d, "o-", color=cor, ms=4, lw=1.4,
                      label=f"α = {rk * K_M / RAIO:.2g}".replace(".", ","))
        eixo.axhline(0, color=REFERENCIA, lw=1)
        eixo.set_xlim(left=0)
        eixo.set_xlabel("s/R")
        eixo.set_title(f"k$_{simbolo}$ relativo a s/R = {s_max:g}".replace(".", ","), fontsize=11)
        virgula(eixo)
    eixos[0].set_ylabel("variação de k$_{eff}$ (%)")
    eixos[1].legend(fontsize=8, ncol=2)
    figura.suptitle("Par isolado (L = 7R): efeito da aproximação para cada α", fontsize=11)
    salvar(figura, "06_varredura_rk_aproximacao")

    s_min = min(s for _, s, _ in k)
    pontos = [(rk * K_M / RAIO, 100 * (k[("longitudinal", s_min, rk)] / k[("transversal", s_min, rk)] - 1))
              for rk in rks if ("longitudinal", s_min, rk) in k and ("transversal", s_min, rk) in k]
    if len(pontos) >= 2:
        a, anis = np.array(pontos).T
        figura, eixo = plt.subplots(figsize=(6, 4))
        eixo.plot(a, anis, "o-", color=PARALELO)
        eixo.axhline(0, color=REFERENCIA, lw=1)
        eixo.axvline(ALFA_CRITICO, color=REFERENCIA, ls=":", lw=1.2)
        eixo.set_xscale("log")
        eixo.set_xlabel("α = R$_K$ k$_m$ / R")
        eixo.set_ylabel("k$_∥$ / k$_⊥$ − 1 (%)")
        eixo.set_title(f"Anisotropia com as partículas a s/R = {s_min:g}".replace(".", ","), fontsize=11)
        virgula(eixo)
        salvar(figura, "07_varredura_rk_anisotropia")


def aglomerados():
    """Aglomerados do 05: efeito por 1 % de phi e comparação com a soma de pares.

    O desvio de cada aglomerado em relação a HJ tem descontado o desvio da
    esfera isolada no mesmo phi e R_K (o erro numérico de fundo, ~0,02 %).
    """
    linhas = ler_csv(AGLOMERADOS)
    if not linhas:
        return
    base = {(round(float(l["phi"]), 6), float(l["R_K"])): float(l["desvio_HJ_percent"])
            for l in linhas if l["configuracao"] == "isolada"}
    dados = {}
    for l in linhas:
        nome = l["configuracao"]
        chave_base = (round(float(l["phi"]), 6), float(l["R_K"]))
        if nome == "isolada" or chave_base not in base:
            continue
        excesso = float(l["desvio_HJ_percent"]) - base[chave_base]
        dados[(nome, chave_base[0], float(l["s_R"]), chave_base[1])] = (
            excesso, int(l["N"]), float(l["lacuna_imagem_R"]))
    if not dados:
        print("  pulado: aglomerados.csv sem a esfera isolada para descontar")
        return
    nomes = [n for n in PARES_VIZINHOS if any(c[0] == n for c in dados)]
    phis = sorted({c[1] for c in dados})
    separacoes = sorted({c[2] for c in dados})
    rks = sorted({c[3] for c in dados})
    cores = plt.cm.viridis(np.linspace(0.1, 0.8, len(phis)))
    x = np.arange(len(nomes))

    figura, eixos = plt.subplots(len(rks), len(separacoes), figsize=(5.2 * len(separacoes), 3.6 * len(rks)),
                                 squeeze=False, sharex=True)
    for i, rk in enumerate(rks):
        for j, s in enumerate(separacoes):
            eixo = eixos[i, j]
            for phi, cor in zip(phis, cores):
                y = [dados[(n, phi, s, rk)][0] / (100 * phi) if (n, phi, s, rk) in dados else np.nan
                     for n in nomes]
                eixo.plot(x, y, "o-", color=cor, lw=1.2, ms=6,
                          label=f"φ = {100 * phi:.3g} %".replace(".", ","))
            eixo.set_title(f"s/R = {s:g}, α = {rk * K_M / RAIO:.2g}".replace(".", ","), fontsize=11)
            eixo.set_xticks(x, [NOMES_AGLOMERADOS[n] for n in nomes], rotation=30, ha="right")
            eixo.set_ylim(bottom=0)
            virgula(eixo, x=False)
        eixos[i, 0].set_ylabel("(k/k$_{HJ}$ − 1) por 1 % de φ (%)")
    eixos[0, -1].legend(fontsize=9)
    figura.suptitle("Aglomerados: efeito por 1 % de φ (curvas de φ diferentes coincidem "
                    "quando o aglomerado está isolado)", fontsize=11)
    salvar(figura, "08_aglomerados_por_phi")

    # Soma de pares: o efeito por ligação vizinha e por partícula, relativo ao par.
    phi = phis[0]  # o menor phi: aglomerados mais longe das imagens periódicas
    s = separacoes[0]
    figura, eixo = plt.subplots(figsize=(6.5, 4))
    largura = 0.8 / len(rks)
    for k, (rk, cor) in enumerate(zip(rks, (PARALELO, PERPENDICULAR, REFERENCIA))):
        par = dados.get(("par", phi, s, rk))
        if not par:
            continue
        unidade = par[0] / (PARES_VIZINHOS["par"] / par[1])
        y = [dados[(n, phi, s, rk)][0] / (PARES_VIZINHOS[n] / dados[(n, phi, s, rk)][1]) / unidade
             if (n, phi, s, rk) in dados else np.nan for n in nomes]
        eixo.bar(x + (k - (len(rks) - 1) / 2) * largura, y, largura, color=cor, alpha=0.85,
                 label=f"α = {rk * K_M / RAIO:.2g}".replace(".", ","))
    eixo.axhline(1, color="#1b2333", ls="--", lw=1.2, label="soma de pares independentes")
    eixo.set_xticks(x, [NOMES_AGLOMERADOS[n] for n in nomes], rotation=30, ha="right")
    eixo.set_ylabel("efeito por par vizinho e por partícula\n(relativo ao par)")
    eixo.set_title(f"Aditividade: φ = {100 * phi:.3g} %, s/R = {s:g}".replace(".", ","), fontsize=11)
    eixo.legend(fontsize=9)
    eixo.grid(axis="x", visible=False)
    virgula(eixo, x=False)
    salvar(figura, "09_aglomerados_aditividade")


def main():
    print(f"Figuras em {FIGURAS}")
    convergencia()
    uma_esfera()
    lado_phi5 = (8 * np.pi / (3 * 0.05)) ** (1 / 3)
    duas_particulas("varredura_rk_1e-08.csv", "04_duas_particulas_phi5",
                    "Duas partículas, φ = 5 %, R$_K$ = 10⁻⁸ m²K/W", espelho=(lado_phi5 - 4) / 2)
    duas_particulas("varredura_rk_1e-08_L7R.csv", "05_par_isolado",
                    "Par isolado (L = 7R, φ = 2,4 %), R$_K$ = 10⁻⁸ m²K/W")
    varredura_rk()
    aglomerados()


if __name__ == "__main__":
    main()
