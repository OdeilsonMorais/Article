"""Aglomerados de N esferas iguais: o efeito do par cresce com N?

Cada configuração é um aglomerado compacto no centro da célula, com lacuna
s entre esferas vizinhas (distância entre centros d = (2 + s) R):

    isolada    1 esfera (referência: deve dar Hasselman–Johnson)
    par        2 esferas ao longo de x
    cadeia3    3 esferas ao longo de x
    cadeia4    4 esferas ao longo de x
    triangulo  3 esferas, triângulo equilátero no plano xy
    quadrado   4 esferas, quadrado no plano xy
    tetraedro  4 esferas, tetraedro regular
    octaedro   6 esferas, octaedro regular
    cubo       8 esferas nos vértices de um cubo

Com phi = 2,44 %, a cadeia de 4 fica a só 0,67R da sua imagem periódica
(s/R = 0,05; 0,22R com s/R = 0,2), então ela se comporta quase como uma
cadeia infinita. A coluna lacuna_imagem_R do CSV mostra isso em cada caso.

A fração volumétrica total é a mesma em todas (padrão: a do par isolado,
L = 7R, phi = 2,44 %), então o lado cresce com N: L = R (4 pi N / 3 phi)^(1/3).
Assim cada aglomerado é comparado com Hasselman–Johnson no mesmo phi, que
representa as mesmas esferas bem dispersas (os 462 casos mostram que uma
esfera por célula reproduz HJ em 0,03 %).

Para cada caso resolve três níveis de h_bulk e extrapola por Aitken, como em
``varredura/varredura_convergida.py``. A malha usa o engrossamento longe das
esferas (de h_bulk até 4 h_bulk entre 1,5R e 4,5R), validado em
``uma_esfera``. Os gradientes impostos são os necessários pela simetria:
x e y na cadeia e no par, x e z no triângulo e no quadrado, só x nos
aglomerados de simetria cúbica (isolada, tetraedro, octaedro, cubo), cujo
tensor é isotrópico. ``--todas-direcoes`` resolve x, y e z sempre.

Rodar de 05_N_particulas, por exemplo
    mpirun -np 2 python3 estudos/aglomerados/aglomerados.py --rk 1e-10 1e-8 \
        --configuracoes isolada par cadeia3 triangulo
O CSV é gravado ao fim de cada caso; rodar de novo pula os casos já feitos.

Vários phi de uma vez: --phi 0.0125 0.02442443 0.05 grava tudo em
aglomerados.csv (coluna phi). As geometrias que não cabem na célula com o phi
pedido (por exemplo, cadeia3 e cadeia4 com phi = 5 %) são puladas com aviso.

Verificação contra o 04: o par com --sem-engrossar e s/R = 0,05 usa a mesma
malha do 04, então k_xx deve repetir o longitudinal de
04/estudos/varredura/varredura_rks_L7R.csv (0,21554394 com R_K = 1e-8). O k_yy
deve ficar perto do transversal (0,21087327), mas não igual: lá o par estava
ao longo de y e a malha era outra.

Custo: a lacuna s/R = 0,05 é refinada até s/4 em cada par vizinho, e isso
domina a malha. No nível mais fino (h_bulk = 0,18R) o par tem ~68 mil nós,
a cadeia de 3 ~110 mil e o cubo ~420 mil (com s/R = 0,2, o cubo tem ~210 mil).
A memória do LU cresce mais que linearmente; comece pelos aglomerados menores.
O solver padrão é o iterativo (gradiente conjugado + multigrid, memória
proporcional à malha); --solver lu volta ao MUMPS, que com 16 GB de RAM não
comporta o octaedro nem o cubo. Basta um processo, por exemplo
    python3 estudos/aglomerados/aglomerados.py --rk 1e-10 1e-8 \
        --phi 0.0125 --separacoes 0.2 --configuracoes octaedro cubo \
        --saida estudos/aglomerados/aglomerados.csv
"""

import argparse
import csv
import resource
import sys
import time
from dataclasses import replace
from math import pi, sqrt
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parents[1]))  # módulos de 04_duas_particulas

from mpi4py import MPI  # noqa: E402

from geometria import geometria_de_centros  # noqa: E402
from malha import gerar_malha  # noqa: E402
from parametros import Parametros  # noqa: E402
from periodicidade import preparar_periodicidade_kapitza  # noqa: E402
from pos_processamento import calcular_condutividade  # noqa: E402
import solver  # noqa: E402
from solver import resolver_kapitza_gradientes  # noqa: E402

# Posições em unidades da distância entre centros vizinhos d, e os eixos
# resolvidos: os demais componentes da diagonal saem por simetria.
_A = 1 / (2 * sqrt(2))
CONFIGURACOES = {
    "isolada": ([(0, 0, 0)], "x"),
    "par": ([(-0.5, 0, 0), (0.5, 0, 0)], "xy"),
    "cadeia3": ([(-1, 0, 0), (0, 0, 0), (1, 0, 0)], "xy"),
    "cadeia4": ([(-1.5, 0, 0), (-0.5, 0, 0), (0.5, 0, 0), (1.5, 0, 0)], "xy"),
    "triangulo": ([(1 / sqrt(3), 0, 0), (-1 / (2 * sqrt(3)), 0.5, 0),
                   (-1 / (2 * sqrt(3)), -0.5, 0)], "xz"),
    "quadrado": ([(sx * 0.5, sy * 0.5, 0) for sx in (-1, 1) for sy in (-1, 1)], "xz"),
    "tetraedro": ([(_A, _A, _A), (_A, -_A, -_A), (-_A, _A, -_A), (-_A, -_A, _A)], "x"),
    "octaedro": ([(sinal * (i == 0) / sqrt(2), sinal * (i == 1) / sqrt(2), sinal * (i == 2) / sqrt(2))
                  for i in range(3) for sinal in (-1, 1)], "x"),
    "cubo": ([(sx * 0.5, sy * 0.5, sz * 0.5)
              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], "x"),
}
EIXOS = {"x": 0, "y": 1, "z": 2}
PHI_PAR_ISOLADO = 8 * pi / (3 * 7**3)

COLUNAS = ["configuracao", "N", "s_R", "phi", "lado_R", "lacuna_imagem_R", "R_K", "alfa",
           "k_xx", "k_yy", "k_zz", "k_media", "k_HJ", "desvio_HJ_percent",
           "anisotropia_percent", "razao_xx", "razao_yy", "razao_zz",
           "k3_xx", "k3_yy", "k3_zz", "fora_diagonal_max", "nos_nivel_3", "tempo_s"]


def hasselman_johnson(phi: float, alfa: float, k_m: float, k_p: float) -> float:
    a = k_p * (1 + 2 * alfa) + 2 * k_m
    b = k_p * (1 - alfa) - k_m
    return k_m * (a + 2 * phi * b) / (a - phi * b)


def extrapolar(k1: float, k2: float, k3: float) -> tuple[float, float]:
    """Aitken (Δ²), igual ao de varredura_convergida.py."""
    d1, d2 = k2 - k1, k3 - k2
    if d1 == 0:
        return k3, float("nan")
    razao = d2 / d1
    if not 0 < razao < 1:
        return k3, razao
    return k3 - d2 * d2 / (d2 - d1), razao


def centros_do_aglomerado(nome: str, separacao_relativa: float, lado: float, raio: float):
    """Centros em metros, com o aglomerado centrado na célula."""
    posicoes, _ = CONFIGURACOES[nome]
    d = (2 + separacao_relativa) * raio
    return [tuple(lado / 2 + d * x for x in posicao) for posicao in posicoes]


def diagonal_completa(nome: str, k: dict[int, float]) -> tuple[float, float, float]:
    """Completar k_xx, k_yy, k_zz a partir dos eixos resolvidos, pela simetria."""
    if len(k) == 3:
        return k[0], k[1], k[2]
    eixos = CONFIGURACOES[nome][1]
    if eixos == "x":
        return k[0], k[0], k[0]
    if eixos == "xy":
        return k[0], k[1], k[1]
    return k[0], k[0], k[2]  # "xz": plano xy isotrópico


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rk", type=float, nargs="+", required=True,
                        help="R_K em m² K/W (um ou vários; a malha é reaproveitada).")
    parser.add_argument("--configuracoes", nargs="+", default=list(CONFIGURACOES),
                        choices=list(CONFIGURACOES))
    parser.add_argument("--separacoes", type=float, nargs="+", default=(0.05, 0.2))
    parser.add_argument("--phi", type=float, nargs="+", default=[PHI_PAR_ISOLADO],
                        help="Fração(ões) volumétrica(s) total(is) (padrão: 2,44 %%, a do par com L = 7R).")
    parser.add_argument("--niveis", type=float, nargs=3, default=(0.35, 0.25, 0.18),
                        help="h_bulk/R dos três níveis, do mais grosso ao mais fino.")
    parser.add_argument("--h-interface-relativo", type=float, default=0.10)
    parser.add_argument("--h-lacuna-relativo", type=float, default=None,
                        help="Tamanho na lacuna, h_lac/R. Omitido: min(h_int, s/4) em cada par.")
    parser.add_argument("--sem-engrossar", action="store_true",
                        help="Desliga o engrossamento longe das esferas (malha como no par).")
    parser.add_argument("--todas-direcoes", action="store_true",
                        help="Resolver x, y e z em todos os casos, sem usar a simetria.")
    parser.add_argument("--saida", default=None, help="CSV de saída.")
    parser.add_argument("--solver", choices=("lu", "iterativo"), default="iterativo",
                        help="iterativo (CG + multigrid, padrão) ou lu (MUMPS).")
    args = parser.parse_args()
    solver.SOLVER_LINEAR = args.solver

    comm = MPI.COMM_WORLD
    if args.saida:
        saida = Path(args.saida)
    elif len(args.phi) == 1:
        saida = PASTA / f"aglomerados_phi{args.phi[0]:.4g}.csv"
    else:
        saida = PASTA / "aglomerados.csv"
    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as arquivo:
            feitos = {(linha["configuracao"], round(float(linha["phi"]), 6),
                       float(linha["s_R"]), float(linha["R_K"]))
                      for linha in csv.DictReader(arquivo)}
    elif comm.rank == 0:
        with saida.open("w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(COLUNAS)

    # phi por fora: cada phi fica completo antes do próximo.
    for phi in args.phi:
        base = Parametros(fracao_volumetrica=phi)
        for nome in args.configuracoes:
            n = len(CONFIGURACOES[nome][0])
            separacoes = args.separacoes if n > 1 else args.separacoes[:1]
            eixos = "xyz" if args.todas_direcoes else CONFIGURACOES[nome][1]
            for s in separacoes:
                rks = [rk for rk in args.rk if (nome, round(phi, 6), s, rk) not in feitos]
                if not rks:
                    continue
                inicio = time.time()
                lado = base.raio * (4 * pi * n / (3 * phi)) ** (1 / 3)
                try:
                    geometria = geometria_de_centros(
                        lado, base.raio, centros_do_aglomerado(nome, s, lado, base.raio))
                except ValueError as erro:
                    # Ex.: cadeia de 4 com phi alto não cabe na célula.
                    if comm.rank == 0:
                        print(f"--- {nome} phi={phi:g} s/R={s:g} pulado: {erro}", flush=True)
                    continue
                # valores[rk][eixo] = [fluxo médio nos três níveis]
                valores = {rk: {EIXOS[e]: [] for e in eixos} for rk in rks}
                for h_bulk in args.niveis:
                    dados = gerar_malha(geometria, tamanho_elemento=h_bulk * base.raio,
                                        numero_threads=1, malha_adaptativa=True,
                                        tamanho_interface=args.h_interface_relativo * base.raio,
                                        tamanho_lacuna=(None if args.h_lacuna_relativo is None
                                                        else args.h_lacuna_relativo * base.raio),
                                        alcance_refino=base.raio / 2,
                                        engrossar_longe=not args.sem_engrossar)
                    sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
                    nos = dados.mesh.topology.index_map(0).size_global
                    gradientes = [tuple(1.0 if i == EIXOS[e] else 0.0 for i in range(3)) for e in eixos]
                    for rk in rks:
                        caso = replace(base, resistencia_kapitza=rk)
                        # Uma fatoração para todos os gradientes deste R_K.
                        solucoes = resolver_kapitza_gradientes(caso, dados, sistema, gradientes)
                        for e, gradiente, solucao in zip(eixos, gradientes, solucoes):
                            j = EIXOS[e]
                            resultado = calcular_condutividade(solucao, caso, gradiente)
                            valores[rk][j].append(resultado.fluxo_medio)
                            if comm.rank == 0:
                                iteracoes = f", {solucao.iteracoes} iterações" if args.solver == "iterativo" else ""
                                print(f"{nome} s/R={s:g} R_K={rk:g} h_bulk={h_bulk:g} G_{e}: "
                                      f"k_{e}{e}={resultado.fluxo_medio[j]:.8f} ({nos} nós{iteracoes})", flush=True)
                tempo = (time.time() - inicio) / len(rks)
                # Pico de memória (RSS) desde o início do processo, somado entre
                # os processos MPI. No Linux, ru_maxrss vem em kB.
                pico_gb = comm.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                                      op=MPI.SUM, root=0)
                if comm.rank != 0:
                    continue
                print(f"--- {nome} phi={phi:g} s/R={s:g}: {tempo * len(rks):.0f} s, "
                      f"pico de memória até aqui {pico_gb:.1f} GB", flush=True)
                with saida.open("a", newline="", encoding="utf-8") as arquivo:
                    escritor = csv.writer(arquivo)
                    for rk in rks:
                        extrapolados, razoes, finos, fora = {}, {}, {}, 0.0
                        for j, fluxos in valores[rk].items():
                            k1, k2, k3 = (fluxo[j] for fluxo in fluxos)
                            extrapolados[j], razoes[j] = extrapolar(k1, k2, k3)
                            finos[j] = k3
                            fora = max(fora, max(abs(fluxos[-1][i]) for i in range(3) if i != j) / k3)
                        k_xx, k_yy, k_zz = diagonal_completa(nome, extrapolados)
                        k3_xx, k3_yy, k3_zz = diagonal_completa(nome, finos)
                        media = (k_xx + k_yy + k_zz) / 3
                        alfa = rk * base.k_matriz / base.raio
                        k_hj = hasselman_johnson(geometria.fracao_volumetrica, alfa,
                                                 base.k_matriz, base.k_particula)
                        escritor.writerow([
                            nome, n, s, f"{geometria.fracao_volumetrica:.8f}",
                            f"{lado / base.raio:.6f}",
                            f"{geometria.lacuna_imagem_periodica / base.raio:.4f}", rk, f"{alfa:.6g}",
                            f"{k_xx:.8f}", f"{k_yy:.8f}", f"{k_zz:.8f}", f"{media:.8f}",
                            f"{k_hj:.8f}", f"{100 * (media / k_hj - 1):.4f}",
                            f"{100 * (max(k_xx, k_yy, k_zz) / min(k_xx, k_yy, k_zz) - 1):.4f}",
                            *(f"{razoes[j]:.4f}" if j in razoes else "" for j in range(3)),
                            f"{k3_xx:.8f}", f"{k3_yy:.8f}", f"{k3_zz:.8f}",
                            f"{fora:.2e}", nos, f"{tempo:.1f}"])
                        print(f"==> {nome} s/R={s:g} R_K={rk:g}: k_media={media:.6f}, "
                              f"desvio de HJ {100 * (media / k_hj - 1):+.3f} %, "
                              f"diagonal ({k_xx:.6f}, {k_yy:.6f}, {k_zz:.6f})", flush=True)


if __name__ == "__main__":
    main()
