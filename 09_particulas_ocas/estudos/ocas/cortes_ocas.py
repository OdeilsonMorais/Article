"""Cortes de temperatura e fluxo: partícula oca contra maciça, lado a lado.

Uma esfera no centro da célula, G ao longo de x (face "quente" em x = 0), e o
corte é o plano xy que passa pelo centro (z = L/2). Cada coluna da figura é
uma espessura de parede t/R (1 = maciça):
  * em cima, T/(G R) com isotermas;
  * embaixo, |q|/(k_m G) em escala log com as linhas de fluxo.
As escalas de cor são as mesmas em todas as colunas. O título de cada coluna
traz ⟨q_x⟩ = k_xx G, calculado na própria malha do corte (um nível só).

Duas etapas, como em 05/estudos/campos e 06/estudos/aleatorio:
1. Calcular (FEniCSx, em série; pula o que já existe em dados/):
       python3 estudos/ocas/cortes_ocas.py --calcular
2. Desenhar (só Matplotlib):
       python3 estudos/ocas/cortes_ocas.py

Padrão: φ externo 5 %, R_K = 1e-10, t/R = 1, 0,3 e 0,1, h_bulk = 0,25R,
400 pontos por lado. Rodar de 09_particulas_ocas.
"""

import argparse
import sys
from math import pi
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]  # 09_particulas_ocas
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

TINTA = "#1f2a44"
TINTA_SECUNDARIA = "#5b6474"
Q_MIN = 0.01  # o ar do núcleo conduz ~10x menos que a matriz


def nome_do_caso(phi, rk, t_R, k_nucleo):
    return f"oca_phi{phi:.4g}_rk{rk:g}_t{t_R:.3g}_kn{k_nucleo:g}"


def calcular(phi, rk, t_R, k_p, k_nucleo, h_bulk, pontos):
    from mpi4py import MPI
    import ufl
    from dolfinx import fem, geometry

    from geometria import geometria_de_centros
    from malha import MATRIZ, etiquetas_de_nucleo, etiquetas_de_particula, gerar_malha
    from parametros import Parametros
    from periodicidade import preparar_periodicidade_kapitza
    from pos_processamento import calcular_condutividade
    from solver import resolver_kapitza_gradientes

    if MPI.COMM_WORLD.size > 1:
        raise SystemExit("Rode em série (sem mpirun).")
    base = Parametros(fracao_volumetrica=phi, resistencia_kapitza=rk, k_particula=k_p,
                      k_nucleo=k_nucleo)
    raio = base.raio
    r_i = 1 - t_R
    lado_R = (4 * pi / (3 * phi)) ** (1 / 3)
    lado = lado_R * raio
    geometria = geometria_de_centros(lado, raio, [(lado / 2,) * 3], r_i * raio)
    h_int = min(0.10, t_R / 3) if r_i > 0 else 0.10
    dados = gerar_malha(geometria, tamanho_elemento=h_bulk * raio, numero_threads=1,
                        malha_adaptativa=True, tamanho_interface=h_int * raio,
                        alcance_refino=raio / 2, engrossar_longe=True)
    sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
    gradiente = (1.0, 0.0, 0.0)
    (solucao,) = resolver_kapitza_gradientes(base, dados, sistema, [gradiente])
    k_xx = calcular_condutividade(solucao, base, gradiente).fluxo_medio[0]
    theta_m = solucao.theta.sub(0).collapse()
    theta_p = solucao.theta.sub(1).collapse()

    malha = dados.mesh
    tdim = malha.topology.dim
    comm = malha.comm
    # Fluxo q = k (G - grad θ) em DG0, por fase (matriz, casca, núcleo).
    V = fem.functionspace(malha, ("DG", 0, (3,)))
    q = fem.Function(V)
    pontos_interp = V.element.interpolation_points
    pontos_interp = pontos_interp() if callable(pontos_interp) else pontos_interp
    G = ufl.as_vector(gradiente)
    fases = [(base.k_matriz, theta_m, dados.cell_tags.find(MATRIZ))]
    cascas = etiquetas_de_particula(dados.cell_tags, comm)
    fases.append((k_p, theta_p, np.concatenate([dados.cell_tags.find(t) for t in cascas])))
    nucleos = etiquetas_de_nucleo(dados.cell_tags, comm)
    if nucleos:
        fases.append((k_nucleo, theta_p, np.concatenate([dados.cell_tags.find(t) for t in nucleos])))
    for k, theta, celulas in fases:
        q.interpolate(fem.Expression(k * (G - ufl.grad(theta)), pontos_interp),
                      cells0=celulas.astype(np.int32))
    q.x.scatter_forward()

    numero_celulas = malha.topology.index_map(tdim).size_local + malha.topology.index_map(tdim).num_ghosts
    fase = np.zeros(numero_celulas, dtype=np.int32)
    fase[dados.cell_tags.indices] = dados.cell_tags.values
    arvore = geometry.bb_tree(malha, tdim)

    margem = 1e-9 * lado
    u = np.linspace(margem, lado - margem, pontos)
    U, W = np.meshgrid(u, u, indexing="xy")
    xyz = np.zeros((U.size, 3))
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = U.ravel(), W.ravel(), lado / 2
    candidatos = geometry.compute_collisions_points(arvore, xyz)
    colisoes = geometry.compute_colliding_cells(malha, candidatos, xyz)
    quantos = np.diff(colisoes.offsets)
    if np.any(quantos == 0):
        raise RuntimeError("Há pontos de amostragem fora da malha.")
    celulas = colisoes.array[colisoes.offsets[:-1]].astype(np.int32)
    lado_p = fase[celulas] != MATRIZ  # casca e núcleo usam theta_p
    theta = np.empty(len(xyz))
    for funcao, mascara in ((theta_m, ~lado_p), (theta_p, lado_p)):
        if mascara.any():
            theta[mascara] = funcao.eval(xyz[mascara], celulas[mascara])[:, 0]
    qv = q.eval(xyz, celulas) / base.k_matriz
    T = -xyz[:, 0] + theta
    forma = U.shape
    return dict(U=U / raio, V=W / raio, T=T.reshape(forma) / raio,
                q_u=qv[:, 0].reshape(forma), q_v=qv[:, 1].reshape(forma),
                q_modulo=np.linalg.norm(qv, axis=1).reshape(forma),
                lado=lado_R, r_i=r_i, t_R=t_R, k_xx=k_xx, k_m=base.k_matriz, phi=phi, rk=rk,
                k_p=k_p, k_nucleo=k_nucleo, nos=malha.topology.index_map(0).size_global)


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
    T_min = min(float(d["T"].min()) for d in casos)
    T_max = max(float(d["T"].max()) for d in casos)
    niveis = np.linspace(T_min, T_max, 20)
    q_max = max(float(np.nanpercentile(d["q_modulo"], 99.5)) for d in casos)
    norma_q = LogNorm(vmin=Q_MIN, vmax=max(q_max, 1.5))
    figura, eixos = plt.subplots(2, n, figsize=(5.2 * n + 1.2, 10), squeeze=False,
                                 layout="constrained")
    for j, d in enumerate(casos):
        extensao = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
        meio = float(d["lado"]) / 2
        eixo_T, eixo_q = eixos[0, j], eixos[1, j]
        imagem_T = eixo_T.imshow(d["T"], origin="lower", extent=extensao, cmap="inferno",
                                 vmin=T_min, vmax=T_max, interpolation="bilinear")
        eixo_T.contour(d["U"], d["V"], d["T"], levels=niveis, colors="white", linewidths=0.6,
                       alpha=0.75, linestyles="solid")
        imagem_q = eixo_q.imshow(np.clip(d["q_modulo"], Q_MIN, None), origin="lower", extent=extensao,
                                 cmap="magma", norm=norma_q, interpolation="bilinear")
        eixo_q.streamplot(d["U"][0, :], d["V"][:, 0], d["q_u"], d["q_v"], color=(1, 1, 1, 0.6),
                          linewidth=0.7, density=1.2, arrowsize=0.7, broken_streamlines=False)
        t_R = float(d["t_R"])
        for eixo in (eixo_T, eixo_q):
            eixo.add_patch(Circle((meio, meio), 1.0, fill=False, lw=1.2, color="white"))
            if float(d["r_i"]) > 0:
                eixo.add_patch(Circle((meio, meio), float(d["r_i"]), fill=False, lw=1.0,
                                      color="white", ls="--"))
            eixo.set_xlabel("x/R"), eixo.set_ylabel("y/R"), eixo.set_aspect("equal")
            eixo.set_xlim(extensao[:2]), eixo.set_ylim(extensao[2:])
        rotulo = "maciça" if t_R >= 1 else f"oca, parede t = {t_R:g} R"
        k_rel = float(d["k_xx"]) / float(d["k_m"])
        eixo_T.set_title(f"{rotulo}\nk_eff/k_m = {k_rel:.4f}".replace(".", ","))
        eixo_q.set_title(f"⟨q_x⟩ = {k_rel:.4f} k_m G".replace(".", ","))
    figura.colorbar(imagem_T, ax=eixos[0, :].tolist(), shrink=0.85, label="T / (G R)")
    barra = figura.colorbar(imagem_q, ax=eixos[1, :].tolist(), shrink=0.85, label="|q| / (k_m G)")
    barra.ax.axhline(1, color="white", lw=1.5)
    d = casos[0]
    figura.suptitle(
        f"Partícula oca contra maciça · φ externo = {100 * float(d['phi']):.3g} % · "
        f"Rₖ = {float(d['rk']):g} m²K/W · kₚ = {float(d['k_p']):g}, k_ar = {float(d['k_nucleo']):g} W/mK · "
        "corte z = L/2, G ao longo de x".replace(".", ","), fontsize=13, color=TINTA_SECUNDARIA)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    print(f"Figura: {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calcular", action="store_true")
    parser.add_argument("--phi", type=float, default=0.05, help="φ externo.")
    parser.add_argument("--rk", type=float, default=1e-10)
    parser.add_argument("--espessuras", type=float, nargs="+", default=[1.0, 0.3, 0.1])
    parser.add_argument("--k-particula", type=float, default=5.0)
    parser.add_argument("--k-nucleo", type=float, default=0.026)
    parser.add_argument("--h-bulk", type=float, default=0.25)
    parser.add_argument("--pontos", type=int, default=400)
    args = parser.parse_args()

    dados_dir = PASTA / "dados"
    casos = []
    for t_R in args.espessuras:
        arquivo = dados_dir / f"{nome_do_caso(args.phi, args.rk, t_R, args.k_nucleo)}.npz"
        if args.calcular and not arquivo.exists():
            dados_dir.mkdir(exist_ok=True)
            d = calcular(args.phi, args.rk, t_R, args.k_particula, args.k_nucleo, args.h_bulk, args.pontos)
            np.savez_compressed(arquivo, **d)
            print(f"t/R = {t_R:g}: k_xx/k_m = {d['k_xx'] / d['k_m']:.5f} ({d['nos']} nós) -> {arquivo.name}",
                  flush=True)
        if not arquivo.exists():
            raise SystemExit(f"Falta {arquivo.name}: rode com --calcular.")
        casos.append(dict(np.load(arquivo)))
    espessuras = "_".join(f"{t:g}" for t in args.espessuras)
    desenhar(casos, PASTA / "figuras" /
             f"cortes_ocas_phi{args.phi:.4g}_rk{args.rk:g}_t{espessuras}_kn{args.k_nucleo:g}.png")


if __name__ == "__main__":
    main()
