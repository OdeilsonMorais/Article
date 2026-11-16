"""Diagnóstico do sistema de Kapitza singular (MUMPS INFOG(1) = -10).

Rodar em série, dentro de 04_duas_particulas, com os mesmos parâmetros do caso
que falhou:  python3 diagnostico_kapitza.py --rk 1e-8 --threads 2
Requer o periodicidade.py novo (assinatura com cell_tags).
Não resolve o sistema; só monta e inspeciona.
"""

import argparse
from dataclasses import replace
from math import pi

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc

import solver
from geometria import preparar_geometria
from malha import INTERFACE_1, INTERFACE_2, gerar_malha
from parametros import Parametros
from periodicidade import preparar_periodicidade_kapitza

parser = argparse.ArgumentParser()
parser.add_argument("--rk", type=float, default=1e-8)
parser.add_argument("--threads", type=int, default=2)
args = parser.parse_args()

base = replace(Parametros(), resistencia_kapitza=args.rk)
geometria = preparar_geometria(base)
print(f"phi = {base.fracao_volumetrica}, s/R = {base.separacao_relativa}, "
      f"L/R = {geometria.lado / geometria.raio:.4f}")
dados = gerar_malha(geometria, numero_threads=args.threads, malha_adaptativa=True)
malha = dados.mesh

# 1) As interfaces são conformes? Cada faceta de interface deve ter 2 células.
malha.topology.create_connectivity(2, 3)
f2c = malha.topology.connectivity(2, 3)
for etiqueta in (INTERFACE_1, INTERFACE_2):
    facetas = dados.facet_tags.find(etiqueta)
    uma_celula = sum(1 for f in facetas if len(f2c.links(f)) == 1)
    print(f"interface {etiqueta}: {len(facetas)} facetas, {uma_celula} com só 1 célula")
# Os vértices de cada interface estão sobre a esfera da geometria?
x = malha.geometry.x
malha.topology.create_connectivity(2, 0)
f2v = malha.topology.connectivity(2, 0)
for etiqueta, centro in zip((INTERFACE_1, INTERFACE_2), geometria.centros):
    vertices = np.unique(np.concatenate([f2v.links(f) for f in dados.facet_tags.find(etiqueta)]))
    d = np.linalg.norm(x[vertices] - np.asarray(centro), axis=1) / geometria.raio
    print(f"vértices da interface {etiqueta}: {len(vertices)}, d/R de {d.min():.6f} a {d.max():.6f}")
dS = ufl.Measure("dS", domain=malha, subdomain_data=dados.facet_tags)
area = fem.assemble_scalar(fem.form(1.0 * (dS(INTERFACE_1) + dS(INTERFACE_2))))
print(f"área dS / 8πR² = {area / (8 * pi * geometria.raio**2):.6f}  (esperado ~1)")

# 2) Volume mínimo das células (lascas degeneradas na lacuna?).
Q = fem.functionspace(malha, ("DG", 0))
v = fem.assemble_vector(fem.form(ufl.TestFunction(Q) * ufl.dx)).array
h3 = (geometria.raio / 80) ** 3 / 6
print(f"volume das células: mín {v.min():.3e}, mediana {np.median(v):.3e} m³; "
      f"{np.sum(v < 1e-6 * h3)} células com volume < 1e-6 de um tetra R/80")


# 2b) Volumes etiquetados contra o exato.
dx = ufl.Measure("dx", domain=malha, subdomain_data=dados.cell_tags)
v_esfera = 4 / 3 * pi * geometria.raio**3
for etiqueta, nome in ((1, "matriz"), (2, "partícula 1"), (3, "partícula 2")):
    celulas = len(dados.cell_tags.find(etiqueta))
    volume = fem.assemble_scalar(fem.form(1.0 * dx(etiqueta)))
    print(f"{nome}: {celulas} células, volume / (4/3 πR³) = {volume / v_esfera:.6f}")

# 3) Montar o sistema reduzido e procurar linhas nulas / escalas.
def inspecionar(forma_a, forma_L, W, sistema):
    bcs = list(sistema.condicoes_dirichlet)
    A = fem_petsc.assemble_matrix(forma_a, bcs=bcs, diag=1.0)
    A.assemble()
    b = fem_petsc.assemble_vector(forma_L)
    A_g, _ = solver._reunir_no_processo_0(A, b, MPI.COMM_WORLD)
    n = A_g.shape[0]
    K, coluna = solver._matriz_transformacao(n, sistema.escravos, sistema.mestres)
    A_r = (K.T @ A_g @ K).tocsr()
    d = A_g.diagonal()
    ativos = d != 1.0
    print(f"DOFs: {n}; reduzido: {A_r.shape[0]}; inativos (diag = 1): {np.sum(~ativos)}")
    print(f"|diag| ativa: mín {np.abs(d[ativos]).min():.3e}, "
          f"mediana {np.median(np.abs(d[ativos])):.3e}, máx {np.abs(d[ativos]).max():.3e}")
    nulas = np.where(np.asarray(abs(A_r).sum(axis=1)).ravel() == 0)[0]
    print(f"linhas nulas no sistema reduzido: {len(nulas)}")
    dr = A_r.diagonal()
    print(f"diagonal nula ou negativa no reduzido: {np.sum(dr <= 0)}")

    # Onde estão as linhas nulas? Campo (theta_m / theta_p), esfera mais
    # próxima e distância ao centro em unidades de R.
    livres = np.where(coluna >= 0)[0]
    completos = livres[nulas]
    campo = np.full(n, -1)
    coords = np.zeros((n, 3))
    for i in range(2):
        Wi, mapa = W.sub(i).collapse()
        mapa = np.asarray(mapa)
        campo[mapa] = i
        coords[mapa] = Wi.tabulate_dof_coordinates()
    centros = np.asarray(geometria.centros)
    R = geometria.raio
    dist = np.linalg.norm(coords[completos][:, None, :] - centros[None], axis=2) / R
    esfera = dist.argmin(axis=1)
    dmin = dist.min(axis=1)
    for i, nome in enumerate(("theta_m", "theta_p")):
        for e in range(2):
            sel = (campo[completos] == i) & (esfera == e)
            if sel.any():
                print(f"  nulas {nome}, perto da esfera {e + 1}: {sel.sum()}; "
                      f"d/R mín {dmin[sel].min():.6f}, máx {dmin[sel].max():.6f}")
    raise SystemExit(0)


solver._resolver_reduzido = inspecionar
sistema = preparar_periodicidade_kapitza(malha, geometria, dados.cell_tags)
solver.resolver_kapitza(base, dados, sistema)
