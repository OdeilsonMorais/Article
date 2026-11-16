"""Partículas ocas contra maciças: k_eff por FEM, comparado com a teoria.

Uma esfera por célula (no centro), raio R = 50 nm, casca de óxido com
espessura t e núcleo de ar de raio r_i = R − t. t/R = 1 é a partícula maciça,
a mesma das pastas 03–06. O salto de Kapitza fica na superfície externa; a
temperatura é contínua na parede interna (casca/ar).

Para cada t/R e R_K resolve três níveis de h_bulk e extrapola com Aitken,
como nos estudos anteriores. A célula com uma esfera no centro tem simetria
cúbica, então k é isotrópico e basta o gradiente em x. Compara com
``teoria.k_oca`` (HJ com a esfera oca equivalente).

Duas comparações (--modo):
  mesmo_phi   φ externo fixo (--phi): mesmo número e tamanho de partículas;
  mesma_massa φ sólido fixo (--phi): a mesma massa de óxido, como os 5 % em
              massa da tinta. A oca ocupa φ_externo = φ_sólido/(1 − (r_i/R)³).

A malha refina a parede: h_interface = min(R/10, t/3). Paredes finas (t/R =
0,05 → h = R/60) deixam a malha bem maior; veja os nós no terminal.

Rodar de 09_particulas_ocas, em série:
    python3 estudos/ocas/ocas.py                       # mesmo φ = 5 %
    python3 estudos/ocas/ocas.py --modo mesma_massa --phi 0.02
O CSV é gravado ao fim de cada caso e rodar de novo pula o que já foi feito.
"""

import argparse
import csv
import resource
import sys
import time
from dataclasses import replace
from math import pi
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]  # 09_particulas_ocas
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(PASTA))

from teoria import hasselman_johnson, k_oca  # noqa: E402

COLUNAS = ["modo", "phi_externo", "phi_solido", "t_R", "r_i_R", "R_K", "alfa", "k_p", "k_nucleo",
           "k_xx", "razao_aitken", "k_teoria_oca", "desvio_teoria_percent", "k_HJ_macica_mesmo_phi",
           "k_sobre_km", "h_interface_R", "nos_nivel_3", "tempo_s"]


def extrapolar(k1, k2, k3):
    """Aitken (Δ²), igual ao dos estudos anteriores."""
    d1, d2 = k2 - k1, k3 - k2
    if d1 == 0:
        return k3, float("nan")
    razao = d2 / d1
    if not 0 < razao < 1:
        return k3, razao
    return k3 - d2 * d2 / (d2 - d1), razao


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modo", choices=("mesmo_phi", "mesma_massa"), default="mesmo_phi")
    parser.add_argument("--phi", type=float, default=0.05,
                        help="φ externo (mesmo_phi) ou φ sólido (mesma_massa).")
    parser.add_argument("--espessuras", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2, 0.1],
                        help="t/R da parede; 1 = maciça. 0,05 funciona, mas a malha cresce bastante.")
    parser.add_argument("--rk", type=float, nargs="+", default=[1e-10, 1e-8])
    parser.add_argument("--k-particula", type=float, default=1.36)
    parser.add_argument("--k-nucleo", type=float, default=0.026,
                        help="k do gás no núcleo (ar 0,026; em regime de Knudsen, menos).")
    parser.add_argument("--niveis", type=float, nargs=3, default=(0.35, 0.25, 0.18),
                        help="h_bulk/R dos três níveis (Aitken).")
    parser.add_argument("--saida", default=str(PASTA / "ocas.csv"))
    args = parser.parse_args()

    from mpi4py import MPI

    import solver
    from geometria import geometria_de_centros
    from malha import gerar_malha
    from parametros import Parametros
    from periodicidade import preparar_periodicidade_kapitza
    from pos_processamento import calcular_condutividade

    comm = MPI.COMM_WORLD
    saida = Path(args.saida)
    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as arquivo:
            feitos = {(l["modo"], round(float(l["phi_externo"]), 6), round(float(l["t_R"]), 6),
                       float(l["R_K"]), round(float(l["k_p"]), 6), round(float(l["k_nucleo"]), 6))
                      for l in csv.DictReader(arquivo)}
    elif comm.rank == 0:
        with saida.open("w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(COLUNAS)

    gradiente = (1.0, 0.0, 0.0)
    for t_R in args.espessuras:
        if not 0 < t_R <= 1:
            raise SystemExit("Cada espessura t/R deve estar em (0, 1].")
        r_i = 1 - t_R
        c = r_i**3
        phi_ext = args.phi if args.modo == "mesmo_phi" else args.phi / (1 - c)
        phi_sol = phi_ext * (1 - c)
        if phi_ext >= 0.30:
            if comm.rank == 0:
                print(f"--- t/R = {t_R:g} pulado: φ externo = {phi_ext:.2f} não cabe numa esfera por célula.")
            continue
        base = Parametros(fracao_volumetrica=phi_ext, k_particula=args.k_particula,
                          k_nucleo=args.k_nucleo)
        raio = base.raio
        rks = [rk for rk in args.rk
               if (args.modo, round(phi_ext, 6), round(t_R, 6), rk, round(args.k_particula, 6),
                   round(args.k_nucleo, 6)) not in feitos]
        if not rks:
            continue
        lado = (4 * pi / (3 * phi_ext)) ** (1 / 3)
        geometria = geometria_de_centros(lado * raio, raio, [(lado * raio / 2,) * 3], r_i * raio)
        h_int = min(0.10, t_R / 3) if r_i > 0 else 0.10
        inicio = time.time()
        k_niveis = {rk: [] for rk in rks}
        for h_bulk in args.niveis:
            dados = gerar_malha(geometria, tamanho_elemento=h_bulk * raio, numero_threads=1,
                                malha_adaptativa=True, tamanho_interface=h_int * raio,
                                alcance_refino=raio / 2, engrossar_longe=True)
            sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
            nos = dados.mesh.topology.index_map(0).size_global
            for rk in rks:
                caso = replace(base, resistencia_kapitza=rk)
                (solucao,) = solver.resolver_kapitza_gradientes(caso, dados, sistema, [gradiente])
                k_niveis[rk].append(calcular_condutividade(solucao, caso, gradiente).fluxo_medio[0])
                if comm.rank == 0:
                    print(f"t/R={t_R:g} φ_ext={phi_ext:.4f} R_K={rk:g} h_bulk={h_bulk:g}: "
                          f"k_xx = {k_niveis[rk][-1]:.6f} ({nos} nós)", flush=True)
        tempo = (time.time() - inicio) / len(rks)
        pico_gb = comm.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                              op=MPI.SUM, root=0)
        if comm.rank != 0:
            continue
        print(f"--- t/R={t_R:g}: {tempo * len(rks):.0f} s, pico de memória {pico_gb:.1f} GB", flush=True)
        with saida.open("a", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo)
            for rk in rks:
                k, razao = extrapolar(*k_niveis[rk])
                alfa = rk * base.k_matriz / raio
                teoria = k_oca(phi_ext, rk, t_R, args.k_particula, args.k_nucleo, base.k_matriz, raio)
                macica = hasselman_johnson(phi_ext, alfa, base.k_matriz, args.k_particula)
                escritor.writerow([args.modo, f"{phi_ext:.8f}", f"{phi_sol:.8f}", f"{t_R:.6f}",
                                   f"{r_i:.6f}", rk, f"{alfa:.6g}", args.k_particula, args.k_nucleo,
                                   f"{k:.8f}", f"{razao:.4f}", f"{teoria:.8f}",
                                   f"{100 * (k / teoria - 1):.4f}", f"{macica:.8f}",
                                   f"{k / base.k_matriz:.6f}", f"{h_int:.4f}", nos, f"{tempo:.1f}"])
                print(f"==> t/R={t_R:g} R_K={rk:g}: k = {k:.6f} (k/k_m = {k / base.k_matriz:.4f}), "
                      f"teoria {teoria:.6f} ({100 * (k / teoria - 1):+.3f} %), "
                      f"maciça no mesmo φ {macica:.6f}", flush=True)


if __name__ == "__main__":
    main()
