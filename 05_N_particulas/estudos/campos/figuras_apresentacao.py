"""Figuras de apresentação: cortes do campo de temperatura para todos os aglomerados.

Duas etapas:

1. Calcular (FEniCSx, em série; pula os casos cujo .npz já existe):
       python3 estudos/campos/figuras_apresentacao.py --calcular
   Para cada configuração resolve um caso (um nível de malha) com G na
   direção x e amostra T, θ e o fluxo q num plano z que passa pela primeira
   esfera. Os dados vão para estudos/campos/dados/, com os mesmos nomes de
   corte_temperatura.py.

2. Desenhar (só Matplotlib, roda em qualquer lugar):
       python3 estudos/campos/figuras_apresentacao.py
   Gera em estudos/campos/figuras/apresentacao/:
     <configuracao>.png    temperatura com isotermas e |q| com linhas de fluxo,
                           16:9, para um slide;
     galeria_fluxo.png     |q| de todas as configurações na mesma escala;
     galeria_theta.png     a perturbação θ de todas, na mesma escala.
   Cada painel traz o fluxo médio ⟨q_x⟩ = k_xx G (em k_m G) e o desvio de
   Hasselman–Johnson, lidos de estudos/aglomerados/aglomerados.csv.

Por padrão: phi = 2,44 % (a do par com L = 7R), s/R = 0,05 e R_K = 1e-10,
o caso em que o efeito é mais visível. Mude com --phi, --separacao e --rk
(os mesmos valores nas duas etapas).
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]  # 05_N_particulas
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "estudos" / "aglomerados"))
sys.path.insert(0, str(PASTA))

import numpy as np  # noqa: E402

from corte_temperatura import ler_npz, nome_do_caso  # noqa: E402

ORDEM = ["isolada", "par", "cadeia3", "cadeia4", "triangulo", "quadrado",
         "tetraedro", "octaedro", "cubo"]
NOMES = {"isolada": "Esfera isolada", "par": "Par", "cadeia3": "Cadeia de 3",
         "cadeia4": "Cadeia de 4", "triangulo": "Triângulo", "quadrado": "Quadrado",
         "tetraedro": "Tetraedro", "octaedro": "Octaedro", "cubo": "Cubo"}
PHI_PADRAO = 0.02442443  # 8 pi / (3 * 7^3), o par com L = 7R
TINTA = "#1f2a44"
TINTA_SECUNDARIA = "#5b6474"


def calcular(args) -> None:
    from corte_temperatura import amostrar

    (PASTA / "dados").mkdir(exist_ok=True)
    for nome in args.configuracoes:
        arquivo = PASTA / "dados" / f"{nome_do_caso(nome, args.phi, args.separacao, args.rk)}.npz"
        if arquivo.exists() and not args.refazer:
            print(f"{nome}: já existe ({arquivo.name})", flush=True)
            continue
        caso = Namespace(configuracao=nome, phi=args.phi, separacao=args.separacao, rk=args.rk,
                         direcao="x", normal="z", h_bulk=args.h_bulk, pontos=args.pontos)
        try:
            d = amostrar(caso)
        except ValueError as erro:  # a geometria não cabe na célula com esse phi
            print(f"{nome}: pulado ({erro})", flush=True)
            continue
        np.savez_compressed(arquivo, **d)
        print(f"{nome}: {arquivo.name}", flush=True)


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
        "axes.edgecolor": TINTA_SECUNDARIA, "axes.labelcolor": TINTA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
        "text.color": TINTA, "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })
    return plt


def _circulos(eixo, d, cor, largura=1.2):
    from matplotlib.patches import Circle
    for cx, cy, r in d["circulos"]:
        if r > 0:
            eixo.add_patch(Circle((cx, cy), r, fill=False, lw=largura, color=cor))


def _limpar(eixo, d, rotulos=True):
    eixo.set_aspect("equal")
    eixo.set_xlim(d["U"].min(), d["U"].max())
    eixo.set_ylim(d["V"].min(), d["V"].max())
    if rotulos:
        eixo.set_xlabel("x / R")
        eixo.set_ylabel("y / R")
    else:
        eixo.set_xticks([])
        eixo.set_yticks([])


def _norma_fluxo(maximo):
    from matplotlib.colors import LogNorm
    return LogNorm(vmin=0.1, vmax=max(maximo, 1.5))


def _mapa_fluxo(eixo, d, norma, linhas=True, densidade=1.2):
    """|q| / (k_m G) em escala log, com linhas de fluxo. Sem fluxo no .npz, None."""
    if "q_modulo" not in d:
        return None
    extensao = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
    imagem = eixo.imshow(np.clip(d["q_modulo"], norma.vmin, None), origin="lower",
                         extent=extensao, cmap="magma", norm=norma, interpolation="bilinear")
    if linhas:
        u = d["U"][0, :]
        v = d["V"][:, 0]
        eixo.streamplot(u, v, d["q_u"], d["q_v"], color=(1, 1, 1, 0.55), linewidth=0.7,
                        density=densidade, arrowsize=0.7, broken_streamlines=False)
    return imagem


def _cientifico(valor: float) -> str:
    """1e-10 -> 10⁻¹⁰; 2.5e-9 -> 2,5×10⁻⁹."""
    mantissa, expoente = f"{valor:.1e}".split("e")
    sobrescrito = str(int(expoente)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"10{sobrescrito}" if mantissa == "1" else f"{mantissa}×10{sobrescrito}"


def figura_slide(plt, d, destino: Path, norma_fluxo) -> None:
    """Um slide 16:9: temperatura e fluxo (ou θ, se o .npz não tiver fluxo)."""
    figura, (eixo_T, eixo_q) = plt.subplots(1, 2, figsize=(16, 7.2), layout="constrained")
    extensao = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())

    imagem = eixo_T.imshow(d["T"], origin="lower", extent=extensao, cmap="inferno",
                           interpolation="bilinear")
    eixo_T.contour(d["U"], d["V"], d["T"], levels=18, colors="white", linewidths=0.6, alpha=0.7,
                   linestyles="solid")
    _circulos(eixo_T, d, "white")
    figura.colorbar(imagem, ax=eixo_T, shrink=0.82, label="T / (G R)")
    eixo_T.set_title("Temperatura e isotermas")
    _limpar(eixo_T, d)

    imagem = _mapa_fluxo(eixo_q, d, norma_fluxo)
    if imagem is not None:
        _circulos(eixo_q, d, "white", 1.0)
        barra = figura.colorbar(imagem, ax=eixo_q, shrink=0.82, label="|q| / (k_m G)")
        barra.ax.axhline(1, color="white", lw=1.5)
        eixo_q.set_title("Fluxo de calor" + _rotulo_fluxo_medio(d, "   "))
    else:
        limite = np.nanmax(np.abs(d["theta"]))
        imagem = eixo_q.imshow(d["theta"], origin="lower", extent=extensao, cmap="RdBu_r",
                               vmin=-limite, vmax=limite, interpolation="bilinear")
        _circulos(eixo_q, d, TINTA, 1.0)
        figura.colorbar(imagem, ax=eixo_q, shrink=0.82, label="θ / (G R)")
        eixo_q.set_title("Perturbação θ = T + G x")
    _limpar(eixo_q, d)

    rk = _cientifico(float(d["rk"]))
    figura.suptitle(f"{NOMES.get(d['configuracao'], d['configuracao'])}   ·   "
                    f"φ = {100 * float(d['phi']):.2f} %   ·   s = {float(d['separacao']):g} R   ·   "
                    f"R_K = {rk} m² K/W   ·   G ao longo de x".replace(".", ","),
                    fontsize=15, color=TINTA_SECUNDARIA)
    figura.savefig(destino)
    plt.close(figura)
    print(f"Figura: {destino}")


def _rotulo_fluxo_medio(d, separador: str) -> str:
    """'⟨q_x⟩ = 1,0938 k_m G (+0,49 % vs HJ)', ou '' sem o valor no CSV.

    Com G ao longo de x, o fluxo médio na célula é ⟨q_x⟩ = k_xx G, então
    ⟨q_x⟩ / (k_m G) = k_xx / k_m. O valor vem do aglomerados.csv (três
    níveis de malha + Aitken), mais preciso que a malha única do corte.
    """
    if "k_xx_rel" not in d:
        return ""
    texto = f"⟨q_x⟩ = {d['k_xx_rel']:.4f} k_m G ({d['desvio_xx']:+.2f} % vs HJ)"
    return separador + texto.replace(".", ",")


def _fluxos_medios(phi, separacao, rk):
    """{configuracao: (k_xx / k_m, desvio de k_xx em relação a HJ em %)} do aglomerados.csv."""
    import csv
    arquivo = RAIZ / "estudos" / "aglomerados" / "aglomerados.csv"
    if not arquivo.exists():
        return {}
    try:
        from parametros import Parametros
        k_m = Parametros().k_matriz
    except Exception:
        k_m = 0.2
    valores = {}
    with arquivo.open(encoding="utf-8") as entrada:
        for linha in csv.DictReader(entrada):
            if (abs(float(linha["phi"]) - phi) < 1e-6 and abs(float(linha["s_R"]) - separacao) < 1e-9
                    and abs(float(linha["R_K"]) / rk - 1) < 1e-6) or (
                    linha["configuracao"] == "isolada" and abs(float(linha["phi"]) - phi) < 1e-6
                    and abs(float(linha["R_K"]) / rk - 1) < 1e-6):
                k_xx, k_hj = float(linha["k_xx"]), float(linha["k_HJ"])
                valores[linha["configuracao"]] = (k_xx / k_m, 100 * (k_xx / k_hj - 1))
    return valores


def galeria(plt, casos, destino: Path, tipo: str, norma_fluxo) -> None:
    """Todas as configurações lado a lado, na mesma escala de cor."""
    from matplotlib import colors
    n = len(casos)
    colunas = 3 if n > 4 else n
    linhas = -(-n // colunas)
    figura, eixos = plt.subplots(linhas, colunas, figsize=(5.2 * colunas, 5.0 * linhas),
                                 layout="constrained", squeeze=False)
    if tipo == "theta":
        limite = max(np.nanmax(np.abs(d["theta"])) for d in casos)
        norma = colors.Normalize(-limite, limite)
    imagem = None
    for eixo, d in zip(eixos.flat, casos):
        if tipo == "fluxo":
            imagem = _mapa_fluxo(eixo, d, norma_fluxo, densidade=0.8)
            _circulos(eixo, d, "white", 0.9)
        else:
            extensao = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
            imagem = eixo.imshow(d["theta"], origin="lower", extent=extensao, cmap="RdBu_r",
                                 norm=norma, interpolation="bilinear")
            _circulos(eixo, d, TINTA, 0.9)
        _limpar(eixo, d, rotulos=False)
        eixo.set_title(NOMES.get(d["configuracao"], d["configuracao"]) + _rotulo_fluxo_medio(d, "\n"),
                       fontsize=14)
    for eixo in list(eixos.flat)[n:]:
        eixo.set_visible(False)
    rotulo = "|q| / (k_m G)" if tipo == "fluxo" else "θ / (G R)"
    figura.colorbar(imagem, ax=eixos, shrink=0.6, label=rotulo)
    figura.savefig(destino)
    plt.close(figura)
    print(f"Figura: {destino}")


def desenhar(args) -> None:
    plt = _estilo()
    casos = []
    for nome in args.configuracoes:
        arquivo = PASTA / "dados" / f"{nome_do_caso(nome, args.phi, args.separacao, args.rk)}.npz"
        if not arquivo.exists():
            print(f"{nome}: sem dados ({arquivo.name}); rode antes com --calcular")
            continue
        d = ler_npz(arquivo)
        d.setdefault("configuracao", nome)
        d.setdefault("phi", args.phi)
        d.setdefault("separacao", args.separacao)
        d.setdefault("rk", args.rk)
        casos.append(d)
    if not casos:
        return
    medios = _fluxos_medios(args.phi, args.separacao, args.rk)
    for d in casos:
        if d["configuracao"] in medios:
            d["k_xx_rel"], d["desvio_xx"] = medios[d["configuracao"]]
        else:
            print(f"{d['configuracao']}: sem fluxo médio no aglomerados.csv para este caso")
    destino = PASTA / "figuras" / "apresentacao"
    destino.mkdir(parents=True, exist_ok=True)
    com_fluxo = [d for d in casos if "q_modulo" in d]
    # Mesma escala de |q| em todas as figuras: comparar slides lado a lado vale.
    maximo = max((np.nanpercentile(d["q_modulo"], 99.5) for d in com_fluxo), default=1.0)
    norma = _norma_fluxo(maximo)
    for d in casos:
        figura_slide(plt, d, destino / f"{d['configuracao']}.png", norma)
    if com_fluxo:
        galeria(plt, com_fluxo, destino / "galeria_fluxo.png", "fluxo", norma)
    galeria(plt, casos, destino / "galeria_theta.png", "theta", norma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calcular", action="store_true", help="Resolver e amostrar (FEniCSx).")
    parser.add_argument("--refazer", action="store_true", help="Com --calcular: refazer os .npz existentes.")
    parser.add_argument("--configuracoes", nargs="+", default=ORDEM, choices=ORDEM)
    parser.add_argument("--phi", type=float, default=PHI_PADRAO)
    parser.add_argument("--separacao", type=float, default=0.05)
    parser.add_argument("--rk", type=float, default=1e-10)
    parser.add_argument("--h-bulk", type=float, default=0.25, help="h_bulk/R (um nível só).")
    parser.add_argument("--pontos", type=int, default=500, help="Pontos por lado do corte.")
    args = parser.parse_args()
    if args.calcular:
        calcular(args)
    desenhar(args)


if __name__ == "__main__":
    main()
