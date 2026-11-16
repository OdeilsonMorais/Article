"""Malha periódica com esferas MACIÇAS de raios diferentes (células bidispersas).

Baseada na malha da pasta 11 (mesmas etiquetas, mesma periodicidade, mesmo
refino por Distance/Threshold e esferas concêntricas nas lacunas); a única
mudança é o raio por esfera. O Gmsh trabalha em unidades do MENOR raio.
O solver, a periodicidade e o pós-processamento da pasta 11 servem sem mudança
(não dependem do raio).
"""

import os
import sys
from dataclasses import dataclass
from itertools import combinations, product
from math import dist, pi
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent / "11_muitas_particulas"))
from malha import (FACES_EXTERNAS, INTERFACE_1, MATRIZ, MAXIMO_PARTICULAS, PARTICULA_1,  # noqa: E402
                   _impor_malha_periodica)


@dataclass(frozen=True)
class GeometriaBi:
    lado: float
    raios: tuple
    centros: tuple
    lacuna_minima: float

    @property
    def raio(self) -> float:  # escala de referência (o menor raio)
        return min(self.raios)

    @property
    def fracao_volumetrica(self) -> float:
        return sum(4 * pi * r**3 / 3 for r in self.raios) / self.lado**3

    raio_interno = 0.0
    oca = False


def geometria_bi(lado, raios, centros) -> GeometriaBi:
    centros = tuple(tuple(float(x) for x in c) for c in centros)
    raios = tuple(float(r) for r in raios)
    for c, r in zip(centros, raios):
        if min(min(x, lado - x) for x in c) <= r:
            raise ValueError("Uma esfera toca ou cruza uma face do cubo.")
    desl = [d for d in product((-lado, 0.0, lado), repeat=3)]
    menor = float("inf")
    for (i, a), (j, b) in product(enumerate(centros), repeat=2):
        for d in desl:
            if i == j and not any(d):
                continue
            s = dist(a, tuple(x + dx for x, dx in zip(b, d))) - raios[i] - raios[j]
            menor = min(menor, s)
    if menor <= 0:
        raise ValueError("Esferas se tocam (ou tocam imagens periódicas).")
    return GeometriaBi(lado, raios, centros, menor)


def _construir(gmsh, lado, centros, raios):
    occ = gmsh.model.occ
    caixa = occ.addBox(0, 0, 0, lado, lado, lado)
    esferas = [(3, occ.addSphere(*c, r)) for c, r in zip(centros, raios)]
    _, mapas = occ.fragment([(3, caixa)], esferas)
    occ.synchronize()
    vols = []
    for mapa in mapas[1:]:
        v = [t for d, t in mapa if d == 3]
        if len(v) != 1:
            raise RuntimeError("Não foi possível identificar uma esfera.")
        vols.append(v[0])
    matriz = [t for _, t in gmsh.model.getEntities(3) if t not in set(vols)]
    interfaces = []
    gmsh.model.addPhysicalGroup(3, matriz, MATRIZ, "matriz")
    for j, v in enumerate(vols):
        gmsh.model.addPhysicalGroup(3, [v], PARTICULA_1 + j)
        sup = [t for _, t in gmsh.model.getBoundary([(3, v)], oriented=False, recursive=False)]
        gmsh.model.addPhysicalGroup(2, sup, INTERFACE_1 + j)
        interfaces += sup
    fronteira = gmsh.model.getBoundary([(3, t) for t in matriz], oriented=False, recursive=False)
    externas = [t for d, t in fronteira if d == 2 and t not in set(interfaces)]
    gmsh.model.addPhysicalGroup(2, externas, FACES_EXTERNAS, "faces_externas")
    return interfaces


def gerar_malha(geometria: GeometriaBi, h_bulk, h_int, alcance=None, numero_threads=1, so_contar_nos=False):
    """h_bulk, h_int e alcance em metros; h_int é o mesmo em todas as esferas."""
    import gmsh
    if len(geometria.centros) > MAXIMO_PARTICULAS:
        raise ValueError("Esferas demais.")
    if not so_contar_nos:
        from mpi4py import MPI
        from dolfinx.io import gmsh as gmshio
        from dolfinx.mesh import GhostMode, create_cell_partitioner
        comm = MPI.COMM_WORLD
        part = create_cell_partitioner(GhostMode.shared_facet, 2)
        rank = comm.rank
    else:
        rank = 0
    e = geometria.raio
    alcance = alcance or e / 2
    hb, hi, al = h_bulk / e, h_int / e, alcance / e
    if rank == 0:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", numero_threads)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            gmsh.model.add("bidispersa")
            lado = geometria.lado / e
            centros = [[x / e for x in c] for c in geometria.centros]
            raios = [r / e for r in geometria.raios]
            interfaces = _construir(gmsh, lado, centros, raios)
            _impor_malha_periodica(gmsh, lado)
            for s in interfaces:
                gmsh.model.mesh.setAlgorithm(2, s, 5)
            f = gmsh.model.mesh.field
            d = f.add("Distance")
            f.setNumbers(d, "SurfacesList", interfaces)
            lim = f.add("Threshold")
            f.setNumber(lim, "InField", d)
            f.setNumber(lim, "SizeMin", hi)
            f.setNumber(lim, "SizeMax", 4 * hb)
            f.setNumber(lim, "DistMin", 0.0)
            f.setNumber(lim, "DistMax", al * (4 * hb - hi) / (hb - hi))
            longe = f.add("Threshold")
            f.setNumber(longe, "InField", d)
            f.setNumber(longe, "SizeMin", hb)
            f.setNumber(longe, "SizeMax", 4 * hb)
            f.setNumber(longe, "DistMin", 1.5)
            f.setNumber(longe, "DistMax", 4.5)
            campos = [lim, longe]
            # lacunas estreitas: esferas concêntricas, como na pasta 11
            for (a, ra), (b, rb) in combinations(zip(centros, raios), 2):
                s = dist(a, b) - ra - rb
                if s >= 4 * hi:
                    continue
                meio = [a[i] + (b[i] - a[i]) * (ra + s / 2) / (ra + rb + s) for i in range(3)]
                tam, nivel = min(hi, s / 4), 0
                while nivel == 0 or tam < hi:
                    bola = f.add("Ball")
                    f.setNumber(bola, "Radius", (2**nivel * s) ** 0.5)
                    f.setNumber(bola, "VIn", tam)
                    f.setNumber(bola, "VOut", hb)
                    for nome, val in zip(("XCenter", "YCenter", "ZCenter"), meio):
                        f.setNumber(bola, nome, val)
                    campos.append(bola)
                    tam, nivel = 2 * tam, nivel + 1
            mn = f.add("Min")
            f.setNumbers(mn, "FieldsList", campos)
            f.setAsBackgroundMesh(mn)
            for o in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"):
                gmsh.option.setNumber(o, 0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(hi, geometria.lacuna_minima / e / 4))
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 4 * hb)
            gmsh.model.mesh.generate(3)
            if so_contar_nos:
                return len(gmsh.model.mesh.getNodes()[0])
            dados = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3, partitioner=part)
        finally:
            gmsh.finalize()
    else:
        dados = gmshio.model_to_mesh(None, comm, rank=0, gdim=3, partitioner=part)
    dados.mesh.geometry.x[:] *= e
    return dados
