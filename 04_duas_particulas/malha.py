"""Malha tetraédrica 3D da RVE, construída com Gmsh e importada no FEniCSx."""

from pathlib import Path
import os
from math import dist

from geometria import Geometria


MATRIZ = 1
PARTICULA_1 = 2
PARTICULA_2 = 3
INTERFACE_1 = 11
INTERFACE_2 = 12
FACES_EXTERNAS = 20


def _impor_malha_periodica(gmsh, lado: float) -> None:
    """Fazer as faces opostas do cubo terem nós coincidentes (x, y e z).

    Sem isto, as malhas das faces x=0 e x=L são independentes e o
    ``dolfinx_mpc`` precisa interpolar para impor a periodicidade, o que a
    viola levemente: num meio homogêneo (k_p = k_m) obtinha-se
    k_eff = 0,19965 em vez de 0,2 com h = R/2. Com nós coincidentes cada
    escravo tem um único mestre com coeficiente 1 e a periodicidade é exata.
    As esferas não tocam as faces, então cada face é uma única superfície.
    """
    eps = 1e-6 * lado
    for eixo in range(3):
        caixa_min = [-eps, -eps, -eps, lado + eps, lado + eps, lado + eps]
        caixa_min[3 + eixo] = eps
        faces_min = gmsh.model.getEntitiesInBoundingBox(*caixa_min, dim=2)
        for _, face_min in faces_min:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, face_min)
            deslocamento = [0.0, 0.0, 0.0]
            deslocamento[eixo] = lado
            busca = [xmin - eps + deslocamento[0], ymin - eps + deslocamento[1],
                     zmin - eps + deslocamento[2], xmax + eps + deslocamento[0],
                     ymax + eps + deslocamento[1], zmax + eps + deslocamento[2]]
            candidatas = gmsh.model.getEntitiesInBoundingBox(*busca, dim=2)
            correspondentes = []
            for _, face_max in candidatas:
                caixa = gmsh.model.getBoundingBox(2, face_max)
                esperado = (xmin + deslocamento[0], ymin + deslocamento[1], zmin + deslocamento[2],
                            xmax + deslocamento[0], ymax + deslocamento[1], zmax + deslocamento[2])
                if all(abs(a - b) < eps for a, b in zip(caixa, esperado)):
                    correspondentes.append(face_max)
            if len(correspondentes) != 1:
                raise RuntimeError(f"Face periódica sem par único no eixo {eixo}.")
            translacao = [1, 0, 0, deslocamento[0],
                          0, 1, 0, deslocamento[1],
                          0, 0, 1, deslocamento[2],
                          0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(2, correspondentes, [face_min], translacao)


def _configurar_refinamento_adaptativo(gmsh, geometria: Geometria, lado: float,
                                        interfaces: list[int], tamanho_bulk: float,
                                        tamanho_interface: float,
                                        tamanho_lacuna: float,
                                        alcance_refino: float) -> None:
    """Definir tamanhos locais com campos ``Distance``/``Threshold`` do Gmsh.

    A distância às superfícies esféricas refina a interface em ambos os lados.
    Esferas concêntricas (campos ``Ball``) centradas na lacuna entre as
    partículas dão elementos ainda menores ali, com tamanho que dobra a cada
    nível. Todos os comprimentos recebidos estão em metros; a geometria do
    Gmsh está normalizada pelo raio.
    """
    escala = geometria.raio
    h_bulk = tamanho_bulk / escala
    h_interface = tamanho_interface / escala
    h_lacuna = tamanho_lacuna / escala
    alcance = alcance_refino / escala

    distancia = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distancia, "SurfacesList", interfaces)
    limiar = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(limiar, "InField", distancia)
    gmsh.model.mesh.field.setNumber(limiar, "SizeMin", h_interface)
    gmsh.model.mesh.field.setNumber(limiar, "SizeMax", h_bulk)
    gmsh.model.mesh.field.setNumber(limiar, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(limiar, "DistMax", alcance)

    # Lacuna: esferas concêntricas no ponto médio entre as partículas. A
    # lacuna local a uma distância r do eixo vale ~s + r² (em unidades de R).
    # O nível k usa tamanho h_lac*2^k dentro do raio sqrt(2^k * s), onde a
    # lacuna é no máximo (1 + 2^k) s: com h_lac = s/4 há sempre >= 4
    # elementos através dela. Os níveis param quando o tamanho chega a h_int.
    # Uma caixa uniforme com h_lac cobria também o interior das esferas e
    # uma faixa transversal larga, e gerava ~7 vezes mais nós.
    centros = [[x / escala for x in centro] for centro in geometria.centros]
    meio = [(centros[0][i] + centros[1][i]) / 2 for i in range(3)]
    lacuna_relativa = dist(*centros) - 2.0
    campos = [limiar]
    tamanho, nivel = h_lacuna, 0
    while nivel == 0 or tamanho < h_interface:
        bola = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(bola, "Radius", (2**nivel * lacuna_relativa) ** 0.5)
        gmsh.model.mesh.field.setNumber(bola, "Thickness", 0.0)
        gmsh.model.mesh.field.setNumber(bola, "VIn", tamanho)
        gmsh.model.mesh.field.setNumber(bola, "VOut", h_bulk)
        for nome, valor in zip(("XCenter", "YCenter", "ZCenter"), meio):
            gmsh.model.mesh.field.setNumber(bola, nome, valor)
        campos.append(bola)
        tamanho, nivel = 2 * tamanho, nivel + 1

    minimo = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimo, "FieldsList", campos)
    gmsh.model.mesh.field.setAsBackgroundMesh(minimo)
    # Evita que outras fontes de tamanho alterem a lei definida acima.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


def gerar_malha(geometria: Geometria, tamanho_elemento: float | None = None,
                 arquivo_msh: str | Path | None = None,
                 numero_threads: int | None = None,
                 malha_adaptativa: bool = False,
                 tamanho_interface: float | None = None,
                 tamanho_lacuna: float | None = None,
                 alcance_refino: float | None = None):
    """Criar a malha com etiquetas de fases e interfaces.

    A operação ``fragment`` faz matriz e esferas compartilharem os nós da
    interface. Quando ``malha_adaptativa`` é ativa, o Gmsh usa elementos
    pequenos nas interfaces e na lacuna entre as esferas e elementos grandes
    no restante da RVE.
    """
    try:
        import gmsh
        from mpi4py import MPI
        from dolfinx.io import gmsh as gmshio
        from dolfinx.mesh import GhostMode, create_cell_partitioner
    except ModuleNotFoundError as erro:
        raise RuntimeError("Esta etapa requer Gmsh, mpi4py e FEniCSx.") from erro

    comm = MPI.COMM_WORLD
    # Integrais em facetas internas (dS, interface de Kapitza) exigem que as
    # facetas na divisa entre processos tenham as duas células disponíveis.
    particionador = create_cell_partitioner(GhostMode.shared_facet, 2)
    tamanho_elemento = tamanho_elemento or geometria.raio / 2
    if tamanho_elemento <= 0:
        raise ValueError("O tamanho característico dos elementos deve ser positivo.")
    tamanho_interface = tamanho_interface or geometria.raio / 10
    alcance_refino = alcance_refino or geometria.raio / 2
    # Quatro elementos na lacuna interna, no mínimo. A malha recebe somente
    # a geometria, então a lacuna é obtida diretamente dos centros.
    lacuna_interna = dist(geometria.centros[0], geometria.centros[1]) - 2 * geometria.raio
    if lacuna_interna <= 0:
        raise ValueError("As esferas se tocam ou se sobrepõem.")
    # Para a separação padrão s/R=0,5, o limite da interface (R/10) já
    # fornece cinco elementos através da lacuna.
    tamanho_lacuna = tamanho_lacuna or min(
        tamanho_interface, lacuna_interna / 4)
    if min(tamanho_interface, tamanho_lacuna, alcance_refino) <= 0:
        raise ValueError("Os tamanhos do refinamento adaptativo devem ser positivos.")
    numero_threads = numero_threads or (os.cpu_count() or 1)
    if numero_threads <= 0:
        raise ValueError("O número de threads deve ser positivo.")

    if comm.rank == 0:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("General.NumThreads", numero_threads)
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", numero_threads)
            # HXT é o gerador tetraédrico paralelo do Gmsh.
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            gmsh.model.add("duas_particulas_periodico")
            occ = gmsh.model.occ
            # O Gmsh aplica tolerâncias geométricas absolutas. Trabalhar
            # diretamente em metros (R = 5e-8) pode fazê-lo classificar a
            # geometria 3D como quase coplanar. Geramos em unidades de raio
            # e restauramos metros depois da importação no FEniCSx.
            escala = geometria.raio
            lado, raio = geometria.lado / escala, 1.0
            centros = tuple(tuple(x / escala for x in centro)
                            for centro in geometria.centros)
            caixa = occ.addBox(0, 0, 0, lado, lado, lado)
            esfera_1 = occ.addSphere(*centros[0], raio)
            esfera_2 = occ.addSphere(*centros[1], raio)
            _, mapas = occ.fragment([(3, caixa)], [(3, esfera_1), (3, esfera_2)])
            occ.synchronize()

            p1 = [tag for dim, tag in mapas[1] if dim == 3]
            p2 = [tag for dim, tag in mapas[2] if dim == 3]
            particulas = set(p1 + p2)
            matriz = [tag for _, tag in gmsh.model.getEntities(3) if tag not in particulas]
            if len(p1) != 1 or len(p2) != 1 or not matriz:
                raise RuntimeError("Não foi possível identificar os três volumes da RVE.")

            interface_1 = [tag for _, tag in gmsh.model.getBoundary(
                [(3, tag) for tag in p1], oriented=False, recursive=False)]
            interface_2 = [tag for _, tag in gmsh.model.getBoundary(
                [(3, tag) for tag in p2], oriented=False, recursive=False)]
            interfaces = set(interface_1 + interface_2)
            fronteira_matriz = gmsh.model.getBoundary(
                [(3, tag) for tag in matriz], oriented=False, recursive=False)
            externas = [tag for dim, tag in fronteira_matriz
                         if dim == 2 and tag not in interfaces]

            gmsh.model.addPhysicalGroup(3, matriz, MATRIZ, "matriz")
            gmsh.model.addPhysicalGroup(3, p1, PARTICULA_1, "particula_1")
            gmsh.model.addPhysicalGroup(3, p2, PARTICULA_2, "particula_2")
            gmsh.model.addPhysicalGroup(2, interface_1, INTERFACE_1, "interface_1")
            gmsh.model.addPhysicalGroup(2, interface_2, INTERFACE_2, "interface_2")
            gmsh.model.addPhysicalGroup(2, externas, FACES_EXTERNAS, "faces_externas")
            _impor_malha_periodica(gmsh, lado)
            # O Frontal-Delaunay (padrão) deixava, com s/R = 0,05, uma calota
            # da esfera 2 voltada para a lacuna com triângulos de ~0,6R
            # (flecha de 0,066R): a lacuna real ficava mais larga que s. O
            # Delaunay respeita o campo de tamanho nas duas esferas.
            for superficie in interfaces:
                gmsh.model.mesh.setAlgorithm(2, superficie, 5)
            if malha_adaptativa:
                _configurar_refinamento_adaptativo(
                    gmsh, geometria, lado, list(interfaces), tamanho_elemento,
                    tamanho_interface, tamanho_lacuna, alcance_refino)
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMin",
                min(tamanho_elemento, tamanho_interface, tamanho_lacuna) / escala)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", tamanho_elemento / escala)
            gmsh.model.mesh.generate(3)
            if arquivo_msh is not None:
                gmsh.write(str(arquivo_msh))
            dados = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3,
                                          partitioner=particionador)
        finally:
            gmsh.finalize()
    else:
        dados = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3,
                                          partitioner=particionador)
    dados.mesh.geometry.x[:] *= geometria.raio
    return dados
