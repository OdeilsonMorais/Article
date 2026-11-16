"""Varredura de duas partículas com k_eff extrapolado para malha infinitamente fina.

Para cada caso de ``varreduras.gerar_casos`` resolve o problema de Kapitza em
três níveis de h_bulk (R/4, R/5,6, R/8 por padrão), com interface R/10 e
lacuna min(R/10, s/4), e extrapola por Aitken (Richardson com ordem estimada).
Receita validada em 2026-10-28 para s/R = 0,05, R_K = 1e-8: extrapolação
0,23341 contra 0,23339 com níveis mais finos; interface e lacuna mais finas
mudam k_eff em menos de 0,02 %.

Guardar em 04_duas_particulas/estudos/varredura/ e rodar de 04_duas_particulas:
    mpirun -np 2 python3 estudos/varredura/varredura_convergida.py --rk 1e-8
O CSV é gravado ao fim de cada caso; rodar de novo pula os casos já feitos.

Vários R_K de uma vez: a malha e a periodicidade de cada nível são montadas
uma vez por caso e reaproveitadas em todos os R_K, por exemplo
    mpirun -np 2 python3 estudos/varredura/varredura_convergida.py \
        --rk 1e-10 1e-9 1e-8 5e-8 1e-7 2.4e-7 5e-7 1e-6 --lado-relativo 7 \
        --separacoes 0.05 0.1 0.2 0.5 1.0 1.5 --niveis 0.35 0.25 0.18

Par isolado (L fixo em vez de phi): passe --lado-relativo, por exemplo
    mpirun -np 2 python3 estudos/varredura/varredura_convergida.py --rk 1e-8 \
        --lado-relativo 7 --separacoes 0.05 0.1 0.2 0.5 1.0 1.5 --niveis 0.35 0.25 0.18
Com L fixo, phi = 8*pi/(3*(L/R)^3). A lacuna até a imagem periódica é
L - 4R - s, então s e (L - 4R) - s são configurações espelhadas: só
s <= (L - 4R)/2 é distinto.
"""

import argparse
import csv
import sys
import time
from math import pi
from dataclasses import replace
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parents[1]))  # módulos de 04_duas_particulas

from mpi4py import MPI  # noqa: E402

from geometria import preparar_geometria  # noqa: E402
from malha import gerar_malha  # noqa: E402
from parametros import Parametros  # noqa: E402
from periodicidade import preparar_periodicidade_kapitza  # noqa: E402
from pos_processamento import calcular_condutividade  # noqa: E402
from solver import resolver_kapitza  # noqa: E402
from varreduras import SEPARACOES_RELATIVAS  # noqa: E402

COLUNAS = ["orientacao", "s_R", "phi", "R_K", "lado_R",
           "k_1", "k_2", "k_3", "k_extrapolado", "razao_diferencas",
           "nos_nivel_3", "area_interface", "salto_rms", "tempo_s"]


def extrapolar(k1: float, k2: float, k3: float) -> tuple[float, float]:
    """Aitken (Δ²): k3 - d2²/(d2 - d1). Devolve (k_extrapolado, d2/d1).

    Fora do regime assintótico (razão fora de (0, 1)) devolve o nível mais
    fino; a razão no CSV permite identificar esses casos.
    """
    d1, d2 = k2 - k1, k3 - k2
    if d1 == 0:
        return k3, float("nan")
    razao = d2 / d1
    if not 0 < razao < 1:
        return k3, razao
    return k3 - d2 * d2 / (d2 - d1), razao


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rk", type=float, nargs="+", required=True,
                        help="R_K em m² K/W (um ou vários; a malha é reaproveitada).")
    parser.add_argument("--niveis", type=float, nargs=3, default=(0.25, 0.18, 0.125),
                        help="h_bulk/R dos três níveis, do mais grosso ao mais fino.")
    parser.add_argument("--h-interface-relativo", type=float, default=0.10)
    parser.add_argument("--lado-relativo", type=float, default=None,
                        help="Fixa L/R (par isolado); phi passa a ser 8*pi/(3*(L/R)^3).")
    parser.add_argument("--phi", type=float, default=None, help="Fração volumétrica (padrão: parametros.py).")
    parser.add_argument("--separacoes", type=float, nargs="+", default=SEPARACOES_RELATIVAS)
    parser.add_argument("--orientacoes", nargs="+", default=("longitudinal", "transversal"))
    parser.add_argument("--saida", default=None, help="CSV de saída.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    base = Parametros()
    if args.lado_relativo is not None and args.phi is not None:
        parser.error("Use --lado-relativo ou --phi, não os dois.")
    if args.lado_relativo is not None:
        base = replace(base, fracao_volumetrica=8 * pi / (3 * args.lado_relativo**3))
        sufixo = f"L{args.lado_relativo:g}R"
    elif args.phi is not None:
        base = replace(base, fracao_volumetrica=args.phi)
        sufixo = f"phi{args.phi:g}"
    else:
        sufixo = f"phi{base.fracao_volumetrica:g}"
    nome_rk = f"rk_{args.rk[0]:g}" if len(args.rk) == 1 else "rks"
    saida = Path(args.saida) if args.saida else PASTA / f"varredura_{nome_rk}_{sufixo}.csv"
    casos = [replace(base, orientacao=o, separacao_relativa=s)
             for o in args.orientacoes for s in args.separacoes]
    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as arquivo:
            feitos = {(linha["orientacao"], float(linha["s_R"]), float(linha["R_K"]))
                      for linha in csv.DictReader(arquivo)}
    elif comm.rank == 0:
        with saida.open("w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(COLUNAS)

    for caso in casos:
        rks = [rk for rk in args.rk if (caso.orientacao, caso.separacao_relativa, rk) not in feitos]
        if not rks:
            continue
        inicio = time.time()
        geometria = preparar_geometria(caso)
        valores = {rk: [] for rk in rks}
        for h_bulk in args.niveis:
            dados = gerar_malha(geometria, tamanho_elemento=h_bulk * caso.raio,
                                numero_threads=1, malha_adaptativa=True,
                                tamanho_interface=args.h_interface_relativo * caso.raio,
                                alcance_refino=caso.raio / 2)
            sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
            nos = dados.mesh.topology.index_map(0).size_global
            for rk in rks:
                caso_rk = replace(caso, resistencia_kapitza=rk)
                resultado = calcular_condutividade(resolver_kapitza(caso_rk, dados, sistema), caso_rk)
                valores[rk].append(resultado)
                if comm.rank == 0:
                    print(f"{caso.orientacao} s/R={caso.separacao_relativa:g} R_K={rk:g} "
                          f"h_bulk={h_bulk:g}: k_eff={resultado.k_efetivo_xx:.8f} ({nos} nós)", flush=True)
        tempo = (time.time() - inicio) / len(rks)
        if comm.rank == 0:
            with saida.open("a", newline="", encoding="utf-8") as arquivo:
                escritor = csv.writer(arquivo)
                for rk in rks:
                    k1, k2, k3 = (v.k_efetivo_xx for v in valores[rk])
                    k_ext, razao = extrapolar(k1, k2, k3)
                    fino = valores[rk][-1]
                    escritor.writerow([
                        caso.orientacao, caso.separacao_relativa, caso.fracao_volumetrica, rk,
                        f"{geometria.lado / geometria.raio:.6f}",
                        f"{k1:.8f}", f"{k2:.8f}", f"{k3:.8f}", f"{k_ext:.8f}", f"{razao:.4f}",
                        nos, f"{fino.area_interface:.6e}", f"{fino.salto_rms:.6e}", f"{tempo:.1f}"])
                    print(f"==> {caso.orientacao} s/R={caso.separacao_relativa:g} R_K={rk:g}: "
                          f"k_extrapolado={k_ext:.6f} (razão {razao:.3f})", flush=True)

if __name__ == "__main__":
    main()
