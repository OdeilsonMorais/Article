"""Salvar a malha de cada aglomerado em .msh, para abrir no Gmsh.

Usa a mesma geometria e o mesmo refino de ``aglomerados.py``, num único
nível de h_bulk. Rodar de 05_N_particulas, por exemplo
    python3 estudos/aglomerados/salvar_malhas.py --separacao 0.05 --h-bulk 0.35
Os arquivos vão para estudos/aglomerados/malhas/<configuracao>_s<s>.msh.

No Gmsh, Tools > Visibility mostra cada volume (1 = matriz, 2, 3, ... =
partículas) e Tools > Options > Mesh > Clipping corta a malha por um plano.
"""

import argparse
import sys
from math import pi
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parents[1]))  # módulos de 05_N_particulas
sys.path.insert(0, str(PASTA))

from aglomerados import CONFIGURACOES, PHI_PAR_ISOLADO, centros_do_aglomerado  # noqa: E402
from geometria import geometria_de_centros  # noqa: E402
from malha import gerar_malha  # noqa: E402
from parametros import Parametros  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configuracoes", nargs="+", default=list(CONFIGURACOES),
                        choices=list(CONFIGURACOES))
    parser.add_argument("--separacao", type=float, default=0.05)
    parser.add_argument("--phi", type=float, default=PHI_PAR_ISOLADO)
    parser.add_argument("--h-bulk", type=float, default=0.35, help="h_bulk/R (0,35 é o nível mais grosso).")
    parser.add_argument("--h-interface-relativo", type=float, default=0.10)
    parser.add_argument("--sem-engrossar", action="store_true")
    args = parser.parse_args()

    raio = Parametros().raio
    destino = PASTA / "malhas"
    destino.mkdir(exist_ok=True)
    for nome in args.configuracoes:
        n = len(CONFIGURACOES[nome][0])
        lado = raio * (4 * pi * n / (3 * args.phi)) ** (1 / 3)
        geometria = geometria_de_centros(lado, raio,
                                         centros_do_aglomerado(nome, args.separacao, lado, raio))
        arquivo = destino / f"{nome}_s{args.separacao:g}.msh"
        dados = gerar_malha(geometria, tamanho_elemento=args.h_bulk * raio, arquivo_msh=arquivo,
                            numero_threads=1, malha_adaptativa=True,
                            tamanho_interface=args.h_interface_relativo * raio,
                            alcance_refino=raio / 2, engrossar_longe=not args.sem_engrossar)
        if dados.mesh.comm.rank == 0:
            print(f"{arquivo.name}: {dados.mesh.topology.index_map(0).size_global} nós", flush=True)


if __name__ == "__main__":
    main()
