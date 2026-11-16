"""Periodicidade da flutuação térmica ``theta`` em uma célula cúbica.

A malha é periódica (``malha._impor_malha_periodica``): cada vértice numa face
``x_i = L`` tem um vértice idêntico, transladado, em ``x_i = 0``. Para
elementos P1 os DOFs ficam nos vértices, então a periodicidade é uma
identificação exata ``theta(escravo) = theta(mestre)``, com coeficiente 1.
Arestas e cantos levam diretamente ao representante com todas as
coordenadas em ``[0, L)``, sem cadeias.

Não usamos ``dolfinx_mpc`` aqui: (i) na versão 0.11 ele não monta integrais
em facetas internas (``dS``), necessárias para Kapitza; (ii) seu
``create_periodic_constraint_geometrical`` interpola o ponto-imagem e gera
mestres espúrios com coeficientes ~1e-15, às vezes também escravos, o que
deixava o sistema reduzido singular.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SistemaPeriodico:
    """Espaço, pares periódicos, Dirichlet homogêneo e âncora.

    ``escravos`` e ``mestres``: índices GLOBAIS dos DOFs, iguais em todos os
    processos (só DOFs de fronteira, conjunto pequeno).

    ``condicoes_dirichlet``: vazio no contato perfeito; no caso de Kapitza
    (espaço misto) fixa em zero os DOFs inativos de cada campo
    (``theta_m`` dentro das partículas, ``theta_p`` fora delas).

    ``ancora``: índice global do DOF de ``theta`` (ou ``theta_m``) no vértice
    (0, 0, 0). Remove a constante livre; é imposta no sistema já reduzido.
    """

    espaco: object
    escravos: object
    mestres: object
    condicoes_dirichlet: tuple[object, ...]
    numero_restricoes: int
    ancora: int


def _importar():
    try:
        import numpy as np
        from petsc4py import PETSc
        from dolfinx import fem
    except ModuleNotFoundError as erro:
        raise RuntimeError("A periodicidade requer NumPy, petsc4py e FEniCSx.") from erro
    return np, PETSc, fem


def _indices_globais(espaco, locais):
    """Converter índices locais (não blocados, bs=1) de ``espaco`` em globais."""
    import numpy as np
    mapa = espaco.dofmap.index_map
    return np.asarray(mapa.local_to_global(np.asarray(locais, dtype=np.int32)), dtype=np.int64)


def _pares_periodicos(espaco, coordenadas, locais, lado: float):
    """Achar ``escravo -> mestre`` (globais) pelos DOFs das faces da célula.

    ``coordenadas[i]`` é a posição do DOF local ``locais[i]`` de ``espaco``.
    Só os DOFs próprios do processo entram; o resultado é reunido em todos.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    comm = espaco.mesh.comm
    tol = 1e-8 * lado
    mapa = espaco.dofmap.index_map
    proprios = locais < mapa.size_local
    locais, coordenadas = locais[proprios], coordenadas[proprios]
    na_borda = np.any((np.abs(coordenadas) < tol) | (np.abs(coordenadas - lado) < tol), axis=1)
    globais = _indices_globais(espaco, locais[na_borda])
    pontos = coordenadas[na_borda]
    globais = np.concatenate(comm.allgather(globais))
    pontos = np.concatenate(comm.allgather(pontos))

    eh_escravo = np.any(np.abs(pontos - lado) < tol, axis=1)
    imagem = np.where(np.abs(pontos - lado) < tol, 0.0, pontos)
    candidatos = np.where(~eh_escravo)[0]
    arvore = cKDTree(pontos[candidatos])
    distancia, posicao = arvore.query(imagem[eh_escravo])
    if len(distancia) and distancia.max() > tol:
        raise RuntimeError(
            "Malha não periódica: há nós em x_i = L sem par em x_i = 0 "
            f"(maior distância {distancia.max():.3e} m). Gere a malha com malha.gerar_malha."
        )
    escravos = globais[eh_escravo]
    mestres = globais[candidatos[posicao]]
    ordem = np.argsort(escravos)
    return escravos[ordem], mestres[ordem]


def _ancora_global(espaco, coordenadas, locais, lado: float) -> int:
    import numpy as np
    comm = espaco.mesh.comm
    tol = 1e-8 * lado
    proprios = locais < espaco.dofmap.index_map.size_local
    na_origem = np.all(np.abs(coordenadas) < tol, axis=1) & proprios
    encontrados = list(_indices_globais(espaco, locais[na_origem]))
    encontrados = sum(comm.allgather(encontrados), [])
    if len(encontrados) != 1:
        raise RuntimeError(f"Esperava 1 DOF em (0, 0, 0); encontrei {len(encontrados)}.")
    return int(encontrados[0])


def preparar_periodicidade(malha, lado: float, familia: str = "Lagrange", grau: int = 1) -> SistemaPeriodico:
    """Espaço P1 de ``theta`` com periodicidade exata (contato perfeito).

    A temperatura física é ``T(x) = -G_x*x[0] + theta(x)``.
    """
    np, PETSc, fem = _importar()
    if lado <= 0:
        raise ValueError("O lado da célula deve ser positivo.")
    if familia not in ("Lagrange", "P", "CG") or grau != 1:
        raise ValueError("A periodicidade por vértices exige Lagrange de grau 1.")
    V = fem.functionspace(malha, ("Lagrange", 1))
    coordenadas = V.tabulate_dof_coordinates()
    locais = np.arange(coordenadas.shape[0], dtype=np.int32)
    escravos, mestres = _pares_periodicos(V, coordenadas, locais, lado)
    if len(escravos) == 0:
        raise RuntimeError("Nenhuma restrição periódica foi criada.")
    ancora = _ancora_global(V, coordenadas, locais, lado)
    return SistemaPeriodico(V, escravos, mestres, (), len(escravos), ancora)


def _dofs_inativos(fem, np, subespaco, celulas):
    """DOFs (índices em W) do subespaço que não pertencem a nenhuma das células.

    Classificação topológica: um DOF é ativo se é vértice de alguma célula da
    fase. A marca é somada entre processos, porque com ``shared_facet`` um
    vértice pode ter suas células da fase só em outro processo.
    """
    from dolfinx import la
    colapsado, mapa = subespaco.collapse()
    mapa = np.asarray(mapa, dtype=np.int32).ravel()
    tdim = colapsado.mesh.topology.dim
    ativos = fem.locate_dofs_topological(colapsado, tdim, celulas)
    marca = fem.Function(colapsado)
    marca.x.array[ativos] = 1.0
    marca.x.scatter_reverse(la.InsertMode.add)
    marca.x.scatter_forward()
    return mapa[marca.x.array == 0.0]


def preparar_periodicidade_kapitza(malha, geometria, cell_tags) -> SistemaPeriodico:
    """Espaço misto P1 x P1 para a lei de Kapitza, com ``theta_m`` periódica.

    - Subespaço 0: ``theta_m``, ativo na matriz (inclusive nos nós da interface).
    - Subespaço 1: ``theta_p``, ativo nas partículas (inclusive nos nós da interface).

    Os nós da interface carregam os dois campos; a diferença entre eles é o
    salto de temperatura ``[T] = T_m - T_p``. Os DOFs fora da região de cada
    campo são fixados em zero (não aparecem em nenhuma integral). Só
    ``theta_m`` toca as faces da célula, portanto só ele é periódico.
    É a mesma formulação de ``03_rve_periodica/mapa_R_RK_phi``.

    A região de cada campo vem das etiquetas de volume da malha, não da
    distância aos centros das esferas. A classificação geométrica anterior
    deixava ativos DOFs de ``theta_p`` sem nenhuma célula de partícula
    (linhas nulas e sistema singular com s/R = 0,05).
    """
    np, PETSc, fem = _importar()
    from basix.ufl import element, mixed_element
    from malha import MATRIZ, PARTICULA_1, PARTICULA_2

    lado = geometria.lado
    P1 = element("Lagrange", malha.basix_cell(), 1, dtype=malha.geometry.x.dtype)
    W = fem.functionspace(malha, mixed_element([P1, P1]))
    if W.dofmap.index_map_bs != 1:
        raise RuntimeError("Esperava espaço misto sem blocagem (bs = 1).")

    W0, W1 = W.sub(0), W.sub(1)
    celulas_m = cell_tags.find(MATRIZ)
    celulas_p = np.concatenate([cell_tags.find(PARTICULA_1), cell_tags.find(PARTICULA_2)])
    zero = PETSc.ScalarType(0)
    inativos_m = _dofs_inativos(fem, np, W0, celulas_m)
    inativos_p = _dofs_inativos(fem, np, W1, celulas_p)
    bc_m_inativo = fem.dirichletbc(zero, inativos_m, W0)
    bc_p_inativo = fem.dirichletbc(zero, inativos_p, W1)

    # Coordenadas dos DOFs de theta_m, indexadas pelo DOF local em W.
    W0c, mapa0 = W0.collapse()
    coordenadas = W0c.tabulate_dof_coordinates()
    locais = np.asarray(mapa0, dtype=np.int32).ravel()
    escravos, mestres = _pares_periodicos(W, coordenadas, locais, lado)
    if len(escravos) == 0:
        raise RuntimeError("Nenhuma restrição periódica foi criada.")
    ancora = _ancora_global(W, coordenadas, locais, lado)
    return SistemaPeriodico(W, escravos, mestres, (bc_m_inativo, bc_p_inativo),
                            len(escravos), ancora)
