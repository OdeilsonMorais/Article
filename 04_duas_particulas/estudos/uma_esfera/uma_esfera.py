"""Refaz o mapa de uma esfera (03_rve_periodica/mapa_R_RK_phi) com o código novo.

O estudo antigo usava malha não periódica e dolfinx_mpc (erro de
periodicidade de 0,1 a 0,2 %) e uma única malha por caso (erro de área da
interface de até 3,2 %). Aqui cada caso usa a malha periódica do 04, a
classificação dos DOFs por cell_tags e três níveis de h_bulk com
extrapolação de Aitken, como em estudos/varredura/varredura_convergida.py.

Em unidades de R, o problema depende só de phi, k_p/k_m e alpha = R_K k_m / R.
Os 6 R × 11 R_K do estudo antigo dão 42 valores distintos de alpha, então
bastam 7 phi × 42 alpha = 294 casos. Para cada phi a malha e a
periodicidade de cada nível são montadas uma vez e reaproveitadas em todos
os alpha.

Guardar em 04_duas_particulas/estudos/uma_esfera/ e rodar de 04_duas_particulas:
    mpirun -np 2 python3 estudos/uma_esfera/uma_esfera.py
Teste curto (um phi, três alpha):
    mpirun -np 2 python3 estudos/uma_esfera/uma_esfera.py --phis 5 --alfas 0.002 0.04 1 --saida teste.csv
O CSV é gravado ao fim de cada phi; rodar de novo pula os phi já feitos.
No fim, comparacao_462.csv põe lado a lado os 462 casos antigos e os novos.
"""

import argparse
import csv
import sys
import time
from dataclasses import replace
from math import pi
from pathlib import Path
from types import SimpleNamespace

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parents[1]))  # módulos de 04_duas_particulas

from mpi4py import MPI  # noqa: E402

from malha import FACES_EXTERNAS, INTERFACE_1, MATRIZ, PARTICULA_1, _impor_malha_periodica  # noqa: E402
from parametros import Parametros  # noqa: E402
from periodicidade import preparar_periodicidade_kapitza  # noqa: E402
from pos_processamento import calcular_condutividade  # noqa: E402
from solver import resolver_kapitza  # noqa: E402

ANTIGO = PASTA.parents[2] / "03_rve_periodica" / "mapa_R_RK_phi" / "mapa_R_RK_phi_periodico.csv"

# Grade do estudo antigo.
PHIS_PERCENT = (0.41887902, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)
RAIOS_NM = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0)
RKS = (1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6)

COLUNAS = ["phi_percent", "alpha", "lado_R", "k_1", "k_2", "k_3", "k_extrapolado",
           "razao_diferencas", "k_HJ", "erro_HJ_percent", "nos_nivel_3",
           "erro_area_percent", "salto_rms", "tempo_s"]


def alfas_padrao(k_m: float) -> list[float]:
    return sorted({round(rk * k_m / (r * 1e-9), 12) for r in RAIOS_NM for rk in RKS})


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


def gerar_malha_uma_esfera(lado_R: float, raio: float, h_bulk: float, h_int: float, alcance: float):
    """Uma esfera no centro do cubo, com as mesmas etiquetas do 04.

    Comprimentos em unidades de R; a malha é devolvida em metros. Como no 04:
    faces periódicas com nós coincidentes, Delaunay (5) na esfera, HXT com
    uma thread (malha reprodutível) e refino por Distance/Threshold.
    """
    import gmsh
    from dolfinx.io import gmsh as gmshio
    from dolfinx.mesh import GhostMode, create_cell_partitioner

    comm = MPI.COMM_WORLD
    particionador = create_cell_partitioner(GhostMode.shared_facet, 2)
    if comm.rank == 0:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            gmsh.model.add("uma_esfera_periodica")
            occ = gmsh.model.occ
            caixa = occ.addBox(0, 0, 0, lado_R, lado_R, lado_R)
            esfera = occ.addSphere(lado_R / 2, lado_R / 2, lado_R / 2, 1.0)
            _, mapas = occ.fragment([(3, caixa)], [(3, esfera)])
            occ.synchronize()
            particula = [tag for dim, tag in mapas[1] if dim == 3]
            matriz = [tag for _, tag in gmsh.model.getEntities(3) if tag not in particula]
            if len(particula) != 1 or not matriz:
                raise RuntimeError("Não foi possível identificar os volumes da RVE.")
            interface = [tag for _, tag in gmsh.model.getBoundary(
                [(3, tag) for tag in particula], oriented=False, recursive=False)]
            externas = [tag for dim, tag in gmsh.model.getBoundary(
                [(3, tag) for tag in matriz], oriented=False, recursive=False)
                if dim == 2 and tag not in interface]
            gmsh.model.addPhysicalGroup(3, matriz, MATRIZ, "matriz")
            gmsh.model.addPhysicalGroup(3, particula, PARTICULA_1, "particula_1")
            gmsh.model.addPhysicalGroup(2, interface, INTERFACE_1, "interface_1")
            gmsh.model.addPhysicalGroup(2, externas, FACES_EXTERNAS, "faces_externas")
            _impor_malha_periodica(gmsh, lado_R)
            for superficie in interface:
                gmsh.model.mesh.setAlgorithm(2, superficie, 5)

            # Tamanho h_int na interface, crescendo até h_bulk a uma distância
            # ``alcance`` (como no 04). Longe da esfera a perturbação de θ cai
            # com (R/r)³, então de 1,5R a 4,5R o tamanho cresce até 4 h_bulk.
            # Sem isso, phi = 0,42 % (L = 10R) tinha 340 mil nós com R/8;
            # com isso tem 57 mil. Para phi >= 5 % quase nada muda. Todos os
            # tamanhos escalam com h_bulk, então a extrapolação continua valendo.
            campo = gmsh.model.mesh.field
            distancia = campo.add("Distance")
            campo.setNumbers(distancia, "SurfacesList", interface)
            perto = campo.add("Threshold")
            campo.setNumber(perto, "InField", distancia)
            campo.setNumber(perto, "SizeMin", h_int)
            campo.setNumber(perto, "SizeMax", 4 * h_bulk)
            campo.setNumber(perto, "DistMin", 0.0)
            # Mesma inclinação de h_int até h_bulk em ``alcance``.
            campo.setNumber(perto, "DistMax", alcance * (4 * h_bulk - h_int) / (h_bulk - h_int))
            longe = campo.add("Threshold")
            campo.setNumber(longe, "InField", distancia)
            campo.setNumber(longe, "SizeMin", h_bulk)
            campo.setNumber(longe, "SizeMax", 4 * h_bulk)
            campo.setNumber(longe, "DistMin", 1.5)
            campo.setNumber(longe, "DistMax", 4.5)
            minimo = campo.add("Min")
            campo.setNumbers(minimo, "FieldsList", [perto, longe])
            campo.setAsBackgroundMesh(minimo)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(h_bulk, h_int))
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 4 * h_bulk)
            gmsh.model.mesh.generate(3)
            dados = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3,
                                          partitioner=particionador)
        finally:
            gmsh.finalize()
    else:
        dados = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3,
                                      partitioner=particionador)
    dados.mesh.geometry.x[:] *= raio
    return dados


def escrever_comparacao(saida: Path, comparacao: Path, k_m: float) -> None:
    """Junta os 462 casos antigos com os novos, pelo par (phi, alpha)."""
    if not ANTIGO.exists():
        print(f"Arquivo antigo não encontrado: {ANTIGO}", flush=True)
        return
    with saida.open(encoding="utf-8") as arquivo:
        novos = {(round(float(l["phi_percent"]), 6), round(float(l["alpha"]), 9)): l
                 for l in csv.DictReader(arquivo)}
    linhas = []
    with ANTIGO.open(encoding="utf-8") as arquivo:
        for antigo in csv.DictReader(arquivo):
            chave = (round(float(antigo["phi_percent"]), 6),
                     round(float(antigo["R_K_m2K_W"]) * k_m / (float(antigo["R_nm"]) * 1e-9), 9))
            novo = novos.get(chave)
            if novo is None:
                continue
            linhas.append([antigo["phi_percent"], antigo["R_nm"], antigo["R_K_m2K_W"], novo["alpha"],
                           novo["k_extrapolado"], antigo["k_FEM_W_mK"], novo["k_HJ"],
                           novo["erro_HJ_percent"], antigo["erro_HJ_percent"]])
    with comparacao.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["phi_percent", "R_nm", "R_K", "alpha", "k_novo", "k_antigo", "k_HJ",
                           "erro_novo_percent", "erro_antigo_percent"])
        escritor.writerows(linhas)
    print(f"Comparação com o estudo antigo: {len(linhas)} casos em {comparacao.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phis", type=float, nargs="+", default=PHIS_PERCENT,
                        help="Frações volumétricas em %%.")
    parser.add_argument("--alfas", type=float, nargs="+", default=None,
                        help="alpha = R_K k_m / R (padrão: os 42 do estudo antigo).")
    parser.add_argument("--niveis", type=float, nargs=3, default=(0.25, 0.18, 0.125),
                        help="h_bulk/R dos três níveis, do mais grosso ao mais fino.")
    parser.add_argument("--h-interface-relativo", type=float, default=0.10)
    parser.add_argument("--saida", default=None, help="CSV de saída.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    base = Parametros()
    k_m, k_p, raio = base.k_matriz, base.k_particula, base.raio
    alfas = args.alfas or alfas_padrao(k_m)
    saida = Path(args.saida) if args.saida else PASTA / "uma_esfera.csv"

    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as arquivo:
            feitos = {round(float(l["phi_percent"]), 6) for l in csv.DictReader(arquivo)}
    elif comm.rank == 0:
        with saida.open("w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(COLUNAS)

    for phi_percent in args.phis:
        if round(phi_percent, 6) in feitos:
            continue
        phi = phi_percent / 100
        lado_R = (4 * pi / (3 * phi)) ** (1 / 3)
        geometria = SimpleNamespace(lado=lado_R * raio, raio=raio)
        inicio = time.time()
        valores = {alfa: [] for alfa in alfas}
        for h_bulk in args.niveis:
            dados = gerar_malha_uma_esfera(lado_R, raio, h_bulk, args.h_interface_relativo, 0.5)
            sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
            nos = dados.mesh.topology.index_map(0).size_global
            for alfa in alfas:
                caso = replace(base, fracao_volumetrica=phi, resistencia_kapitza=alfa * raio / k_m)
                resultado = calcular_condutividade(resolver_kapitza(caso, dados, sistema), caso)
                valores[alfa].append(resultado)
            if comm.rank == 0:
                print(f"phi={phi_percent:g}% h_bulk={h_bulk:g}: {nos} nós, "
                      f"{time.time() - inicio:.0f} s", flush=True)
        tempo = time.time() - inicio
        if comm.rank == 0:
            area_exata = 4 * pi * raio**2
            with saida.open("a", newline="", encoding="utf-8") as arquivo:
                escritor = csv.writer(arquivo)
                for alfa in alfas:
                    k1, k2, k3 = (v.k_efetivo_xx for v in valores[alfa])
                    k_ext, razao = extrapolar(k1, k2, k3)
                    k_hj = hasselman_johnson(phi, alfa, k_m, k_p)
                    fino = valores[alfa][-1]
                    escritor.writerow([
                        phi_percent, alfa, f"{lado_R:.6f}", f"{k1:.8f}", f"{k2:.8f}", f"{k3:.8f}",
                        f"{k_ext:.8f}", f"{razao:.4f}", f"{k_hj:.8f}",
                        f"{100 * (k_ext / k_hj - 1):.4f}", nos,
                        f"{100 * (fino.area_interface / area_exata - 1):.4f}",
                        f"{fino.salto_rms:.6e}", f"{tempo / len(alfas):.1f}"])
            piores = max(abs(extrapolar(*(v.k_efetivo_xx for v in valores[a]))[0]
                             / hasselman_johnson(phi, a, k_m, k_p) - 1) for a in alfas)
            print(f"==> phi={phi_percent:g}%: {len(alfas)} alphas em {tempo:.0f} s, "
                  f"maior desvio de HJ {100 * piores:.3f} %", flush=True)

    if comm.rank == 0 and args.alfas is None:
        escrever_comparacao(saida, saida.with_name("comparacao_462.csv"), k_m)


if __name__ == "__main__":
    main()
