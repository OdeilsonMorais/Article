"""Mapas das dispersões aleatórias: posições em 3D e cortes de temperatura e fluxo.

Lê os centros de cada sorteio em aleatorio.csv, então reproduz exatamente as
geometrias já calculadas. Duas etapas, como em 05/estudos/campos:

1. Calcular (FEniCSx, em série; pula o que já existe em dados/):
       python3 estudos/aleatorio/cortes_aleatorios.py --calcular --phi 0.1 --sementes 0 3 6
   Resolve cada sorteio num nível de malha e amostra T, θ e q num plano que
   passa pelo par de esferas mais próximo. G é sempre x (face "quente" em
   x = 0, como nos aglomerados) e o plano é xy ou xz, o que passar mais perto
   do centro das duas esferas do par.

2. Desenhar (só Matplotlib):
       python3 estudos/aleatorio/cortes_aleatorios.py --phi 0.1 --sementes 0 3 6
   Um slide 16:9 por sorteio em estudos/aleatorio/figuras/: as esferas em 3D
   com o plano do corte, a temperatura com isotermas e |q| com linhas de
   fluxo. O título traz o desvio de HJ do sorteio (média dos três eixos) e a
   componente de k na direção de G.
   Sem os dados do corte, desenha só a vista 3D (útil para escolher sorteios).

Rodar de 06_dispersao_aleatoria.
"""

import argparse
import csv
import json
import sys
from math import dist
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]  # 06_dispersao_aleatoria
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

EIXOS = "xyz"
TINTA = "#1f2a44"
TINTA_SECUNDARIA = "#5b6474"


def ler_sorteios(arquivo: Path, phi: float, rk: float):
    """{semente: linha do CSV} para o phi e o R_K pedidos (só N > 1)."""
    sorteios = {}
    with arquivo.open(encoding="utf-8") as entrada:
        for linha in csv.DictReader(entrada):
            if (int(linha["semente"]) >= 0 and abs(float(linha["phi"]) - phi) < 1e-6
                    and abs(float(linha["R_K"]) / rk - 1) < 1e-6):
                linha["centros"] = [tuple(c) for c in json.loads(linha["centros_R"])]
                sorteios[int(linha["semente"])] = linha
    return sorteios


def escolher_corte(centros):
    """Par mais próximo, normal do plano, posição do plano e direção de G (índices 0-2).

    G fica sempre ao longo de x, como nos aglomerados (face "quente" em x = 0), e
    o plano é y ou z constante, o que cortar o par mais perto do centro das duas.
    """
    pares = [(dist(a, b), i, j) for i, a in enumerate(centros) for j, b in enumerate(centros) if i < j]
    _, i, j = min(pares)
    a, b = np.array(centros[i]), np.array(centros[j])
    normal = 1 if abs(b[1] - a[1]) < abs(b[2] - a[2]) else 2
    plano = float((a[normal] + b[normal]) / 2)
    eixo = 0
    return (i, j), normal, plano, eixo


def nome_do_caso(phi, semente, rk):
    return f"aleatorio_phi{phi:.4g}_semente{semente}_rk{rk:g}"


def resolver(centros_R, lado_R, phi, rk, h_bulk, eixo=0):
    """Resolver um sorteio com G ao longo de ``eixo`` e devolver o que a amostragem precisa.

    Uma malha só (h_bulk), em série. O fluxo q = k (G - grad θ) vai para uma
    função DG0 por fase. Unidades: G = 1 K/m, então T sai em G R ao dividir por R.
    """
    from mpi4py import MPI
    import ufl
    from dolfinx import fem, geometry

    from geometria import geometria_de_centros
    from malha import MATRIZ, etiquetas_presentes, gerar_malha
    from parametros import Parametros
    from periodicidade import preparar_periodicidade_kapitza
    from solver import resolver_kapitza_gradientes

    if MPI.COMM_WORLD.size > 1:
        raise SystemExit("Rode em série (sem mpirun).")
    base = Parametros(fracao_volumetrica=phi, resistencia_kapitza=rk)
    raio = base.raio
    geometria = geometria_de_centros(lado_R * raio, raio, [tuple(x * raio for x in c) for c in centros_R])
    dados = gerar_malha(geometria, tamanho_elemento=h_bulk * raio, numero_threads=1,
                        malha_adaptativa=True, tamanho_interface=0.1 * raio,
                        alcance_refino=raio / 2, engrossar_longe=True)
    sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
    gradiente = tuple(1.0 if i == eixo else 0.0 for i in range(3))  # G = 1 K/m
    solucao = resolver_kapitza_gradientes(base, dados, sistema, [gradiente])[0]
    theta_m = solucao.theta.sub(0).collapse()
    theta_p = solucao.theta.sub(1).collapse()

    malha = dados.mesh
    tdim = malha.topology.dim
    V = fem.functionspace(malha, ("DG", 0, (3,)))
    q = fem.Function(V)
    pontos_interp = V.element.interpolation_points
    pontos_interp = pontos_interp() if callable(pontos_interp) else pontos_interp
    particulas = etiquetas_presentes(dados.cell_tags, MATRIZ, malha.comm)
    celulas_p = np.concatenate([dados.cell_tags.find(t) for t in particulas]).astype(np.int32)
    celulas_m = dados.cell_tags.find(MATRIZ).astype(np.int32)
    G = ufl.as_vector(gradiente)
    for k, theta, celulas in ((base.k_matriz, theta_m, celulas_m), (base.k_particula, theta_p, celulas_p)):
        q.interpolate(fem.Expression(k * (G - ufl.grad(theta)), pontos_interp), cells0=celulas)
    q.x.scatter_forward()

    numero_celulas = malha.topology.index_map(tdim).size_local + malha.topology.index_map(tdim).num_ghosts
    fase = np.zeros(numero_celulas, dtype=np.int32)
    fase[dados.cell_tags.indices] = dados.cell_tags.values
    return dict(malha=malha, theta_m=theta_m, theta_p=theta_p, q=q, fase=fase, matriz=MATRIZ,
                arvore=geometry.bb_tree(malha, tdim), raio=raio, lado=lado_R * raio,
                k_matriz=base.k_matriz, eixo=eixo)


def amostrar_plano(ctx, normal, plano_R, pontos):
    """T, θ e q num plano normal a ``normal`` (índice 0-2) na posição plano_R (em R)."""
    from dolfinx import geometry

    raio, lado = ctx["raio"], ctx["lado"]
    eixos_plano = [i for i in range(3) if i != normal]
    margem = 1e-9 * lado
    u = np.linspace(margem, lado - margem, pontos)
    U, W = np.meshgrid(u, u, indexing="xy")
    xyz = np.zeros((U.size, 3))
    xyz[:, eixos_plano[0]] = U.ravel()
    xyz[:, eixos_plano[1]] = W.ravel()
    xyz[:, normal] = min(max(plano_R * raio, margem), lado - margem)
    candidatos = geometry.compute_collisions_points(ctx["arvore"], xyz)
    colisoes = geometry.compute_colliding_cells(ctx["malha"], candidatos, xyz)
    inicio, quantos = colisoes.offsets[:-1], np.diff(colisoes.offsets)
    if np.any(quantos == 0):
        raise RuntimeError("Há pontos de amostragem fora da malha.")
    celulas = colisoes.array[inicio].astype(np.int32)
    na_particula = ctx["fase"][celulas] != ctx["matriz"]
    theta = np.empty(len(xyz))
    for funcao, mascara in ((ctx["theta_m"], ~na_particula), (ctx["theta_p"], na_particula)):
        if mascara.any():
            theta[mascara] = funcao.eval(xyz[mascara], celulas[mascara])[:, 0]
    qv = ctx["q"].eval(xyz, celulas) / ctx["k_matriz"]
    T = -xyz[:, ctx["eixo"]] + theta
    forma = U.shape
    return dict(U=U / raio, V=W / raio, T=T.reshape(forma) / raio, theta=theta.reshape(forma) / raio,
                q_u=qv[:, eixos_plano[0]].reshape(forma), q_v=qv[:, eixos_plano[1]].reshape(forma),
                q_n=qv[:, normal].reshape(forma), q_modulo=np.linalg.norm(qv, axis=1).reshape(forma))


def amostrar(centros_R, lado_R, phi, rk, h_bulk, pontos):
    """Resolver um sorteio e amostrar T, θ e q no plano do par mais próximo (unidades de R, G R e k_m G)."""
    par, normal, plano_R, eixo = escolher_corte(centros_R)
    ctx = resolver(centros_R, lado_R, phi, rk, h_bulk, eixo)
    d = amostrar_plano(ctx, normal, plano_R, pontos)
    d.update(normal=normal, plano=plano_R, eixo=eixo, par=np.array(par))
    return d


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "axes.edgecolor": TINTA_SECUNDARIA, "axes.labelcolor": TINTA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
        "text.color": TINTA, "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200,
    })
    return plt


def vista_3d(eixo, centros, lado, corte=None):
    """Esferas (superfícies) na célula, com o par mais próximo destacado e o plano do corte."""
    par, normal, plano, _ = corte if corte is not None else escolher_corte(centros)
    um, vm = np.mgrid[0:2 * np.pi:28j, 0:np.pi:16j]
    esfera = np.array([np.cos(um) * np.sin(vm), np.sin(um) * np.sin(vm), np.cos(vm)])
    for k, c in enumerate(centros):
        cor = "#c2410c" if k in tuple(par) else "#8fa6c8"
        eixo.plot_surface(c[0] + esfera[0], c[1] + esfera[1], c[2] + esfera[2],
                          color=cor, linewidth=0, antialiased=True, shade=True, alpha=0.95)
    # Arestas do cubo.
    for a in (0, lado):
        for b in (0, lado):
            eixo.plot([0, lado], [a, a], [b, b], color=TINTA_SECUNDARIA, lw=0.6)
            eixo.plot([a, a], [0, lado], [b, b], color=TINTA_SECUNDARIA, lw=0.6)
            eixo.plot([a, a], [b, b], [0, lado], color=TINTA_SECUNDARIA, lw=0.6)
    # Plano do corte.
    s = np.array([[0, lado], [0, lado]])
    t = s.T
    p = np.full((2, 2), plano)
    coords = {0: (p, s, t), 1: (s, p, t), 2: (s, t, p)}[int(normal)]
    eixo.plot_surface(*coords, color="#f2c94c", alpha=0.18, linewidth=0)
    eixo.set_xlim(0, lado), eixo.set_ylim(0, lado), eixo.set_zlim(0, lado)
    eixo.set_box_aspect((1, 1, 1))
    eixo.set_xlabel("x/R"), eixo.set_ylabel("y/R"), eixo.set_zlabel("z/R")
    eixo.tick_params(labelsize=9)


def _circulos(eixo, centros, d, cor):
    from matplotlib.patches import Circle
    normal = int(d["normal"])
    a, b = [i for i in range(3) if i != normal]
    for c in centros:
        r2 = 1 - (c[normal] - float(d["plano"])) ** 2
        if r2 > 0:
            eixo.add_patch(Circle((c[a], c[b]), np.sqrt(r2), fill=False, lw=1.1, color=cor))


def slide(plt, linha, d, destino: Path) -> None:
    from matplotlib.colors import LogNorm
    centros = linha["centros"]
    lado = float(linha["lado_R"])
    tem_corte = d is not None
    figura = plt.figure(figsize=(16 if tem_corte else 7, 6.4), layout="constrained")
    grade = figura.add_gridspec(1, 3 if tem_corte else 1, width_ratios=(1, 1.05, 1.05) if tem_corte else None)
    eixo3d = figura.add_subplot(grade[0, 0], projection="3d")
    corte = None
    if tem_corte:
        corte = (tuple(int(i) for i in d["par"]), int(d["normal"]), float(d["plano"]), int(d["eixo"]))
    vista_3d(eixo3d, centros, lado, corte)
    eixo3d.set_title("Posições (laranja: par mais próximo)")

    phi = 100 * float(linha["phi"])
    quebra = "   ·   " if tem_corte else "\n"
    titulo = (f"Sorteio {linha['semente']}   ·   φ = {phi:.3g} %   ·   {len(centros)} esferas{quebra}"
              f"menor lacuna {float(linha['lacuna_interna_R']):.2f} R   ·   "
              f"k médio {float(linha['desvio_HJ_percent']):+.2f} % vs HJ")
    if tem_corte:
        normal, eixo_g = int(d["normal"]), int(d["eixo"])
        a, b = [EIXOS[i] for i in range(3) if i != normal]
        extensao = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
        eixo_T = figura.add_subplot(grade[0, 1])
        imagem = eixo_T.imshow(d["T"], origin="lower", extent=extensao, cmap="inferno", interpolation="bilinear")
        eixo_T.contour(d["U"], d["V"], d["T"], levels=18, colors="white", linewidths=0.6, alpha=0.7,
                       linestyles="solid")
        _circulos(eixo_T, centros, d, "white")
        figura.colorbar(imagem, ax=eixo_T, shrink=0.8, label="T / (G R)")
        eixo_T.set_title(f"Temperatura (corte {EIXOS[normal]} = {float(d['plano']):.2f} R)".replace(".", ","))

        eixo_q = figura.add_subplot(grade[0, 2])
        norma = LogNorm(vmin=0.1, vmax=max(float(np.nanpercentile(d["q_modulo"], 99.5)), 1.5))
        imagem = eixo_q.imshow(np.clip(d["q_modulo"], 0.1, None), origin="lower", extent=extensao,
                               cmap="magma", norm=norma, interpolation="bilinear")
        eixo_q.streamplot(d["U"][0, :], d["V"][:, 0], d["q_u"], d["q_v"], color=(1, 1, 1, 0.55),
                          linewidth=0.7, density=1.1, arrowsize=0.7, broken_streamlines=False)
        _circulos(eixo_q, centros, d, "white")
        barra = figura.colorbar(imagem, ax=eixo_q, shrink=0.8, label="|q| / (k_m G)")
        barra.ax.axhline(1, color="white", lw=1.5)
        k_ii = float(linha[f"k_{EIXOS[eixo_g]}{EIXOS[eixo_g]}"]) / 0.2
        eixo_q.set_title(f"Fluxo de calor   ⟨q_{EIXOS[eixo_g]}⟩ = {k_ii:.4f} k_m G".replace(".", ","))
        for eixo in (eixo_T, eixo_q):
            eixo.set_xlabel(f"{a}/R"), eixo.set_ylabel(f"{b}/R"), eixo.set_aspect("equal")
        titulo += f"   ·   G ao longo de {EIXOS[eixo_g]}"
    figura.suptitle(titulo.replace(".", ","), fontsize=14, color=TINTA_SECUNDARIA)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    print(f"Figura: {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calcular", action="store_true", help="Resolver e amostrar (FEniCSx).")
    parser.add_argument("--phi", type=float, default=0.1)
    parser.add_argument("--sementes", type=int, nargs="+", default=None,
                        help="Padrão: todas as sementes do CSV para esse phi.")
    parser.add_argument("--rk", type=float, default=1e-10)
    parser.add_argument("--h-bulk", type=float, default=0.25)
    parser.add_argument("--pontos", type=int, default=450)
    parser.add_argument("--csv", default=str(PASTA / "aleatorio.csv"))
    args = parser.parse_args()

    sorteios = ler_sorteios(Path(args.csv), args.phi, args.rk)
    sementes = args.sementes if args.sementes is not None else sorted(sorteios)
    faltando = [s for s in sementes if s not in sorteios]
    if faltando:
        print(f"Sementes sem linha no CSV para phi={args.phi:g}, R_K={args.rk:g}: {faltando}")
    dados_dir = PASTA / "dados"
    if args.calcular:
        dados_dir.mkdir(exist_ok=True)
        for semente in sementes:
            if semente not in sorteios:
                continue
            arquivo = dados_dir / f"{nome_do_caso(args.phi, semente, args.rk)}.npz"
            if arquivo.exists():
                print(f"semente {semente}: já existe ({arquivo.name})")
                continue
            linha = sorteios[semente]
            d = amostrar(linha["centros"], float(linha["lado_R"]), args.phi, args.rk, args.h_bulk, args.pontos)
            np.savez_compressed(arquivo, **d)
            print(f"semente {semente}: {arquivo.name}", flush=True)

    plt = _estilo()
    for semente in sementes:
        if semente not in sorteios:
            continue
        arquivo = dados_dir / f"{nome_do_caso(args.phi, semente, args.rk)}.npz"
        d = dict(np.load(arquivo)) if arquivo.exists() else None
        slide(plt, sorteios[semente], d,
              PASTA / "figuras" / f"{nome_do_caso(args.phi, semente, args.rk)}.png")


if __name__ == "__main__":
    main()
