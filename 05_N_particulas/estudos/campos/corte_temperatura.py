"""Mapa de temperatura num corte da célula e perfil do gradiente ao longo do eixo.

Resolve um caso (um aglomerado, um phi, um s/R, um R_K, um nível de malha)
com o gradiente macroscópico G na direção escolhida e amostra
T(x) = -G·x + θ(x) num plano que passa pelo centro da primeira esfera. Em
cada ponto usa θ_m na matriz e θ_p nas partículas, então o salto de Kapitza
aparece no mapa e no perfil.

Figura (PNG em estudos/campos/figuras/):
  1. T / (G R) no corte, com isotermas;
  2. θ / (G R), a perturbação causada pelas partículas;
  3. perfil ao longo do eixo do gradiente pela primeira esfera: T e o
     gradiente local -dT/dx dividido por G (vale 1 longe das partículas).
Os pontos amostrados (T, θ e o fluxo q no plano) ficam num .npz, para refazer
a figura sem resolver de novo. As figuras para apresentação saem desses .npz
com figuras_apresentacao.py.
    python3 estudos/campos/corte_temperatura.py --replotar <arquivo.npz>

Rodar de 05_N_particulas, em série (um processo), por exemplo
    python3 estudos/campos/corte_temperatura.py --configuracao par --rk 1e-10
    python3 estudos/campos/corte_temperatura.py --configuracao cadeia3 --separacao 0.2 --rk 1e-8
"""

import argparse
import sys
from math import pi
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]  # 05_N_particulas
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "estudos" / "aglomerados"))

import numpy as np  # noqa: E402

EIXOS = "xyz"


def amostrar(args):
    """Resolver o caso e amostrar T, θ e a fase no plano e na linha."""
    from mpi4py import MPI
    from dolfinx import geometry

    from aglomerados import CONFIGURACOES, centros_do_aglomerado
    from geometria import geometria_de_centros
    from malha import MATRIZ, gerar_malha
    from parametros import Parametros
    from periodicidade import preparar_periodicidade_kapitza
    from solver import resolver_kapitza_gradientes

    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise SystemExit("Rode em série: python3 estudos/campos/corte_temperatura.py ...")
    base = Parametros(fracao_volumetrica=args.phi, resistencia_kapitza=args.rk)
    raio = base.raio
    n = len(CONFIGURACOES[args.configuracao][0])
    lado = raio * (4 * pi * n / (3 * args.phi)) ** (1 / 3)
    geometria = geometria_de_centros(
        lado, raio, centros_do_aglomerado(args.configuracao, args.separacao, lado, raio))
    dados = gerar_malha(geometria, tamanho_elemento=args.h_bulk * raio, numero_threads=1,
                        malha_adaptativa=True, tamanho_interface=0.1 * raio,
                        alcance_refino=raio / 2, engrossar_longe=True)
    sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
    eixo = EIXOS.index(args.direcao)
    gradiente = tuple(1.0 if i == eixo else 0.0 for i in range(3))  # G = 1 K/m
    solucao = resolver_kapitza_gradientes(base, dados, sistema, [gradiente])[0]
    theta_m = solucao.theta.sub(0).collapse()
    theta_p = solucao.theta.sub(1).collapse()
    fluxo = fluxo_dg0(dados, base, gradiente, theta_m, theta_p)

    malha = dados.mesh
    tdim = malha.topology.dim
    numero_celulas = malha.topology.index_map(tdim).size_local + malha.topology.index_map(tdim).num_ghosts
    fase = np.zeros(numero_celulas, dtype=np.int32)
    fase[dados.cell_tags.indices] = dados.cell_tags.values
    arvore = geometry.bb_tree(malha, tdim)

    def avaliar(pontos):
        """θ e indicador de partícula em cada ponto (pontos em metros, (n, 3))."""
        candidatos = geometry.compute_collisions_points(arvore, pontos)
        colisoes = geometry.compute_colliding_cells(malha, candidatos, pontos)
        celulas = np.array([colisoes.links(i)[0] if len(colisoes.links(i)) else -1
                            for i in range(len(pontos))], dtype=np.int32)
        if np.any(celulas < 0):
            raise RuntimeError("Há pontos de amostragem fora da malha.")
        na_particula = fase[celulas] != MATRIZ
        theta = np.empty(len(pontos))
        for funcao, mascara in ((theta_m, ~na_particula), (theta_p, na_particula)):
            if mascara.any():
                theta[mascara] = funcao.eval(pontos[mascara], celulas[mascara])[:, 0]
        q = fluxo.eval(pontos, celulas) if fluxo is not None else None
        return theta, na_particula, q

    # Plano normal ao eixo --normal, pela primeira esfera.
    centro = np.array(geometria.centros[0])
    normal = EIXOS.index(args.normal)
    eixos_plano = [i for i in range(3) if i != normal]
    margem = 1e-9 * lado
    u = np.linspace(margem, lado - margem, args.pontos)
    U, V = np.meshgrid(u, u, indexing="xy")
    pontos = np.zeros((U.size, 3))
    pontos[:, eixos_plano[0]] = U.ravel()
    pontos[:, eixos_plano[1]] = V.ravel()
    pontos[:, normal] = centro[normal]
    theta, na_particula, q = avaliar(pontos)
    temperatura = -pontos[:, eixo] + theta  # G = 1

    # Linha ao longo do gradiente, pelo centro da primeira esfera.
    s = np.linspace(margem, lado - margem, 4 * args.pontos)
    linha = np.tile(centro, (len(s), 1))
    linha[:, eixo] = s
    theta_linha, particula_linha, _ = avaliar(linha)

    circulos = [(c[eixos_plano[0]], c[eixos_plano[1]],
                 np.sqrt(max(raio**2 - (c[normal] - centro[normal]) ** 2, 0.0)))
                for c in geometria.centros]
    # Tudo em unidades de R (comprimentos), de G R (temperaturas) e de k_m G (fluxos).
    extras = {}
    if q is not None:
        escala_q = base.k_matriz  # G = 1 K/m
        extras = dict(q_u=q[:, eixos_plano[0]].reshape(U.shape) / escala_q,
                      q_v=q[:, eixos_plano[1]].reshape(U.shape) / escala_q,
                      q_modulo=np.linalg.norm(q, axis=1).reshape(U.shape) / escala_q)
    return dict(
        **extras, configuracao=args.configuracao, phi=args.phi, separacao=args.separacao, rk=args.rk,
        U=U / raio, V=V / raio, T=temperatura.reshape(U.shape) / raio,
        theta=theta.reshape(U.shape) / raio, particula=na_particula.reshape(U.shape),
        s=s / raio, T_linha=(-s + theta_linha) / raio, particula_linha=particula_linha,
        circulos=np.array(circulos) / raio, lado=lado / raio,
        eixos_plano=np.array(eixos_plano), eixo=eixo, normal=normal,
        titulo=(f"{args.configuracao}, φ = {100 * args.phi:.3g} %, s/R = {args.separacao:g}, "
                f"R_K = {args.rk:g} m²K/W, G na direção {args.direcao}").replace(".", ","))


def fluxo_dg0(dados, parametros, gradiente, theta_m, theta_p):
    """Fluxo q = k (G - grad θ) como função DG0 vetorial (constante em cada célula).

    Na matriz usa θ_m e k_m; nas partículas, θ_p e k_p. Devolve None (com aviso)
    se a versão do DOLFINx não aceitar a interpolação por subconjunto de células.
    """
    try:
        import ufl
        from dolfinx import fem
        from malha import MATRIZ, etiquetas_presentes

        malha = dados.mesh
        V = fem.functionspace(malha, ("DG", 0, (3,)))
        q = fem.Function(V, name="fluxo")
        G = ufl.as_vector(gradiente)
        pontos = V.element.interpolation_points
        pontos = pontos() if callable(pontos) else pontos
        particulas = etiquetas_presentes(dados.cell_tags, MATRIZ, malha.comm)
        celulas_p = np.concatenate([dados.cell_tags.find(t) for t in particulas]).astype(np.int32)
        celulas_m = dados.cell_tags.find(MATRIZ).astype(np.int32)
        for k, theta, celulas in ((parametros.k_matriz, theta_m, celulas_m),
                                  (parametros.k_particula, theta_p, celulas_p)):
            expressao = fem.Expression(k * (G - ufl.grad(theta)), pontos)
            q.interpolate(expressao, cells0=celulas)
        q.x.scatter_forward()
        return q
    except Exception as erro:  # o mapa de T e θ sai mesmo assim
        print(f"Aviso: fluxo não calculado ({type(erro).__name__}: {erro}).", flush=True)
        return None


def gradiente_por_fase(s, T, particula):
    """-dT/ds em cada trecho contínuo de uma mesma fase (sem atravessar o salto)."""
    g = np.full_like(T, np.nan)
    inicio = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or particula[i] != particula[inicio]:
            if i - inicio >= 3:
                g[inicio:i] = -np.gradient(T[inicio:i], s[inicio:i])
            inicio = i
    return g


def plotar(d, saida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    nomes = EIXOS
    figura = plt.figure(figsize=(15, 4.8), layout="constrained")
    grade = figura.add_gridspec(1, 3, width_ratios=(1, 1, 1.25))
    eixo_T, eixo_theta, eixo_perfil = (figura.add_subplot(grade[0, i]) for i in range(3))
    extensao = (d["U"].min(), d["U"].max(), d["V"].min(), d["V"].max())
    rotulos = [f"{nomes[int(i)]}/R" for i in d["eixos_plano"]]

    def circulos(eixo, cor):
        for cx, cy, r in d["circulos"]:
            if r > 0:
                eixo.add_patch(Circle((cx, cy), r, fill=False, lw=0.8, color=cor, alpha=0.8))

    imagem = eixo_T.imshow(d["T"], origin="lower", extent=extensao, cmap="inferno")
    eixo_T.contour(d["U"], d["V"], d["T"], levels=15, colors="white", linewidths=0.4, alpha=0.6)
    circulos(eixo_T, "white")
    figura.colorbar(imagem, ax=eixo_T, shrink=0.85, label="T / (G R)")
    eixo_T.set_title("Temperatura")

    limite = np.nanmax(np.abs(d["theta"]))
    imagem = eixo_theta.imshow(d["theta"], origin="lower", extent=extensao, cmap="RdBu_r",
                               vmin=-limite, vmax=limite)
    circulos(eixo_theta, "black")
    figura.colorbar(imagem, ax=eixo_theta, shrink=0.85, label="θ / (G R)")
    eixo_theta.set_title("Perturbação θ = T + G·x")
    for eixo in (eixo_T, eixo_theta):
        eixo.set_xlabel(rotulos[0])
        eixo.set_ylabel(rotulos[1])
        eixo.set_aspect("equal")

    s, T, p = d["s"], d["T_linha"], d["particula_linha"].astype(bool)
    g = gradiente_por_fase(s, T, p)
    eixo_perfil.plot(s, T, color="#c2410c", lw=1.6, label="T / (G R)")
    eixo_perfil.set_xlabel(f"{nomes[int(d['eixo'])]}/R (linha pelo centro da 1ª esfera)")
    eixo_perfil.set_ylabel("T / (G R)", color="#c2410c")
    segundo = eixo_perfil.twinx()
    segundo.plot(s, g, color="#2f5d9b", lw=1.4, label="−dT/dx / G")
    segundo.axhline(1, color="#2f5d9b", ls=":", lw=1)
    segundo.set_ylabel("gradiente local / G", color="#2f5d9b")
    for inicio, fim in _trechos(s, p):
        eixo_perfil.axvspan(inicio, fim, color="#ece7dc", lw=0)
    eixo_perfil.set_title("Perfil ao longo do gradiente (cinza: partículas)")
    figura.suptitle(d["titulo"], fontsize=11)
    saida.parent.mkdir(parents=True, exist_ok=True)
    # Sem with_suffix: o nome tem pontos (phi0.02442_s0.05) e ele cortaria no último.
    arquivo = saida.parent / f"{saida.name}.png"
    figura.savefig(arquivo, dpi=200)
    plt.close(figura)
    print(f"Figura: {arquivo}")


def _trechos(s, particula):
    trechos, inicio = [], None
    for i, dentro in enumerate(particula):
        if dentro and inicio is None:
            inicio = s[i]
        if not dentro and inicio is not None:
            trechos.append((inicio, s[i - 1]))
            inicio = None
    if inicio is not None:
        trechos.append((inicio, s[-1]))
    return trechos


def ler_npz(arquivo):
    d = dict(np.load(arquivo, allow_pickle=False))
    for chave in ("titulo", "configuracao"):
        if chave in d:
            d[chave] = str(d[chave])
    return d


def nome_do_caso(configuracao, phi, separacao, rk, direcao="x", normal="z"):
    return f"corte_{configuracao}_phi{phi:.4g}_s{separacao:g}_rk{rk:g}_G{direcao}_n{normal}"


def main() -> None:
    from aglomerados import CONFIGURACOES, PHI_PAR_ISOLADO

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configuracao", default="par", choices=list(CONFIGURACOES))
    parser.add_argument("--phi", type=float, default=PHI_PAR_ISOLADO)
    parser.add_argument("--separacao", type=float, default=0.05)
    parser.add_argument("--rk", type=float, default=1e-8)
    parser.add_argument("--direcao", default="x", choices=list(EIXOS), help="Direção de G.")
    parser.add_argument("--normal", default="z", choices=list(EIXOS), help="Normal do plano de corte.")
    parser.add_argument("--h-bulk", type=float, default=0.25, help="h_bulk/R da malha (um nível só).")
    parser.add_argument("--pontos", type=int, default=400, help="Pontos por lado do corte.")
    parser.add_argument("--replotar", default=None, help="Refazer a figura a partir de um .npz.")
    args = parser.parse_args()

    if args.replotar:
        arquivo = Path(args.replotar)
        d = ler_npz(arquivo)
        plotar(d, PASTA / "figuras" / arquivo.stem)
        return
    if args.normal == args.direcao:
        parser.error("O plano de corte deve conter a direção do gradiente (--normal diferente de --direcao).")
    nome = nome_do_caso(args.configuracao, args.phi, args.separacao, args.rk, args.direcao, args.normal)
    d = amostrar(args)
    (PASTA / "dados").mkdir(exist_ok=True)
    np.savez_compressed(PASTA / "dados" / f"{nome}.npz", **d)
    plotar(d, PASTA / "figuras" / nome)


if __name__ == "__main__":
    main()
