"""Problemas de condução estacionária para a RVE periódica.

Convenção: ``T(x) = -G·x + theta(x)``, com ``theta`` periódica. O fluxo local
é ``q = -k grad(T) = k (G - grad(theta))``.
"""

from dataclasses import dataclass

from kapitza import condutancia_interfacial

# Solver do sistema reduzido: "iterativo" (padrão: gradiente conjugado com
# multigrid algébrico, memória proporcional à malha) ou "lu" (MUMPS, exato,
# mas com memória que cresce mais que linearmente). Validado em 03/11/2026:
# par e tetraedro (s/R = 0,2, R_K = 1e-10 e 1e-8) iguais à LU nos 8 dígitos,
# 8 a 12 vezes mais rápido. Se o CG não convergir, cai para a LU com aviso.
SOLVER_LINEAR = "iterativo"
TOLERANCIA_ITERATIVO = 1e-10   # resíduo relativo (norma 2, não precondicionada)
from malha import (MATRIZ, etiquetas_de_interface, etiquetas_de_nucleo, etiquetas_de_particula,
                   etiquetas_presentes)


def medidas_por_fase(dados_malha):
    """Medidas ``dx`` da matriz, ``dx`` das partículas (cascas), ``dS`` das
    interfaces externas e ``dx`` dos núcleos de ar (None se maciças).

    As etiquetas vêm da malha, então o mesmo código serve para uma, duas ou N
    partículas, maciças ou ocas.
    """
    import ufl
    malha = dados_malha.mesh
    particulas = etiquetas_de_particula(dados_malha.cell_tags, malha.comm)
    nucleos = etiquetas_de_nucleo(dados_malha.cell_tags, malha.comm)
    interfaces = etiquetas_de_interface(dados_malha.facet_tags, malha.comm)
    if not particulas or not interfaces:
        raise RuntimeError("A malha não tem partículas ou interfaces etiquetadas.")
    dx = ufl.Measure("dx", domain=malha, subdomain_data=dados_malha.cell_tags)
    dS = ufl.Measure("dS", domain=malha, subdomain_data=dados_malha.facet_tags)
    # Pasta 11: uma única medida com a tupla de etiquetas (UFL >= 2024.2), em
    # vez da soma dx(1) + dx(2) + ..., que com dezenas de partículas gera um
    # termo por etiqueta.
    dx_p = dx(tuple(particulas))
    dS_K = dS(tuple(interfaces))
    dx_n = dx(tuple(nucleos)) if nucleos else None
    return dx(MATRIZ), dx_p, dS_K, dx_n


@dataclass(frozen=True)
class SolucaoContatoPerfeito:
    """Campos do caso de referência sem salto interfacial."""

    theta: object
    condutividade: object
    iteracoes: int | None

    def fluxo(self, gradiente: tuple[float, float, float]):
        """Vetor UFL do fluxo local ``q`` em W/m²."""
        import ufl
        G = ufl.as_vector(gradiente)
        return self.condutividade * (G - ufl.grad(self.theta))

    def medida_volume(self):
        import ufl
        return ufl.dx(domain=self.theta.function_space.mesh)


@dataclass(frozen=True)
class SolucaoKapitza:
    """Campos do caso com resistência de Kapitza (dois campos contínuos).

    ``theta_m`` vale na matriz e ``theta_p`` nas partículas. O salto de
    temperatura na interface é ``[T] = theta_m - theta_p`` (o termo ``-G_x x``
    é contínuo e não contribui para o salto).
    """

    theta: object          # função no espaço misto W
    theta_m: object
    theta_p: object
    k_matriz: float
    k_particula: float
    resistencia_kapitza: float
    dx_m: object           # medida da matriz
    dx_p: object           # medida de todas as partículas
    dS_K: object           # medida de todas as interfaces
    iteracoes: int | None
    k_nucleo: float | None = None   # pasta 09: núcleo de ar (None se maciças)
    dx_n: object = None             # medida dos núcleos

    def fluxo_por_fase(self, gradiente: tuple[float, float, float]):
        """Fluxos UFL na matriz e nas partículas, nessa ordem."""
        import ufl
        G = ufl.as_vector(gradiente)
        q_m = self.k_matriz * (G - ufl.grad(self.theta_m))
        q_p = self.k_particula * (G - ufl.grad(self.theta_p))
        return q_m, q_p

    def fluxo_nucleo(self, gradiente: tuple[float, float, float]):
        """Fluxo UFL no núcleo de ar (o núcleo usa o campo theta_p)."""
        import ufl
        G = ufl.as_vector(gradiente)
        return self.k_nucleo * (G - ufl.grad(self.theta_p))


def _condutividade_por_fase(parametros, dados_malha, fem):
    """Criar k DG0 a partir das etiquetas físicas dos volumes."""
    malha = dados_malha.mesh
    tdim = malha.topology.dim
    Q = fem.functionspace(malha, ("DG", 0))
    k = fem.Function(Q, name="condutividade")
    k.x.array[:] = parametros.k_matriz
    nucleos = set(etiquetas_de_nucleo(dados_malha.cell_tags, malha.comm))
    for etiqueta in etiquetas_presentes(dados_malha.cell_tags, MATRIZ, malha.comm):
        celulas = dados_malha.cell_tags.find(etiqueta)
        dofs = fem.locate_dofs_topological(Q, tdim, celulas)
        nucleo = etiqueta in nucleos
        k.x.array[dofs] = parametros.k_nucleo if nucleo else parametros.k_particula
    k.x.scatter_forward()
    return k


def resolver_contato_perfeito(parametros, dados_malha, sistema_periodico,
                              gradiente: tuple[float, float, float] | None = None) -> SolucaoContatoPerfeito:
    """Resolver a condução com contato perfeito em todas as interfaces.

    A forma resolvida é ``∫ k grad(theta)·grad(v) = ∫ k G e_x·grad(v)``.
    Com ``T = -G*x + theta``, isso corresponde a ``div(k grad(T)) = 0``.
    Este caso deliberadamente não usa ``R_K``: temperatura contínua é a
    referência que valida a formulação de Kapitza (limite R_K -> 0).
    """
    if parametros.resistencia_kapitza not in (None, 0.0):
        raise ValueError("O solver de contato perfeito exige R_K = None ou 0.")
    if dados_malha.cell_tags is None:
        raise RuntimeError("A malha não contém etiquetas de volume.")
    try:
        import ufl
        from petsc4py import PETSc
        from dolfinx import fem
    except ModuleNotFoundError as erro:
        raise RuntimeError("O solver requer UFL, PETSc e FEniCSx.") from erro

    malha = dados_malha.mesh
    k = _condutividade_por_fase(parametros, dados_malha, fem)
    V = sistema_periodico.espaco
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    gradiente = gradiente or (parametros.gradiente_x, 0.0, 0.0)
    if len(gradiente) != 3:
        raise ValueError("O gradiente macroscópico deve ter três componentes.")
    G = ufl.as_vector(tuple(PETSc.ScalarType(valor) for valor in gradiente))
    a = ufl.inner(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(k * G, ufl.grad(v)) * ufl.dx
    # Mesmo caminho numérico do caso de Kapitza (redução K^T A K e âncora no
    # sistema reduzido), para que o limite R_K -> 0 seja comparável.
    (resultado,), iteracoes = _resolver_reduzido(fem.form(a), [fem.form(L)], V, sistema_periodico)
    return SolucaoContatoPerfeito(resultado, k, iteracoes)


def resolver_kapitza(parametros, dados_malha, sistema_periodico,
                     gradiente: tuple[float, float, float] | None = None) -> SolucaoKapitza:
    """Um gradiente só; ver ``resolver_kapitza_gradientes``."""
    gradiente = gradiente or (parametros.gradiente_x, 0.0, 0.0)
    return resolver_kapitza_gradientes(parametros, dados_malha, sistema_periodico, [gradiente])[0]


def resolver_kapitza_gradientes(parametros, dados_malha, sistema_periodico,
                                gradientes) -> list[SolucaoKapitza]:
    """Resolver a RVE com salto de temperatura nas interfaces partícula-matriz.

    Formulação de dois campos contínuos (P1 x P1), a mesma validada em
    ``03_rve_periodica/mapa_R_RK_phi``::

        ∫_m k_m ∇θ_m·∇v_m + ∫_p k_p ∇θ_p·∇v_p + ∫_Γ h_K [θ][v]
            = ∫_m k_m G e_x·∇v_m + ∫_p k_p G e_x·∇v_p

    com ``[θ] = θ_m - θ_p`` e ``h_K = 1/R_K``. O termo de interface é a
    forma fraca da lei de Kapitza ``q_n = h_K [T]`` com fluxo contínuo.
    ``sistema_periodico`` deve vir de ``preparar_periodicidade_kapitza``.

    Vários gradientes (por exemplo x, y e z para o tensor) compartilham a
    mesma matriz: ela é montada e fatorada uma vez só, e cada gradiente é só
    um lado direito a mais.

    Versão anterior (DG + SIPG) removida: as integrais ``dS(0)`` não
    selecionavam faceta alguma, porque a malha do Gmsh só etiqueta as
    interfaces (11, 12) e as faces externas (20). Sem os termos de
    penalidade interior os elementos DG ficavam desacoplados.
    """
    if parametros.resistencia_kapitza is None:
        raise ValueError("Informe R_K para o solver de Kapitza.")
    if dados_malha.cell_tags is None or dados_malha.facet_tags is None:
        raise RuntimeError("A malha precisa das etiquetas de volume e interface.")
    try:
        import ufl
        from petsc4py import PETSc
        from dolfinx import fem
    except ModuleNotFoundError as erro:
        raise RuntimeError("O solver requer UFL, PETSc e FEniCSx.") from erro

    malha = dados_malha.mesh
    W = sistema_periodico.espaco
    if W.num_sub_spaces != 2:
        raise ValueError("Use preparar_periodicidade_kapitza: o solver exige o espaço misto P1 x P1.")

    dx_m, dx_p, dS_K, dx_n = medidas_por_fase(dados_malha)

    theta_m, theta_p = ufl.TrialFunctions(W)
    v_m, v_p = ufl.TestFunctions(W)
    k_m = fem.Constant(malha, PETSc.ScalarType(parametros.k_matriz))
    k_p = fem.Constant(malha, PETSc.ScalarType(parametros.k_particula))
    h_K = fem.Constant(malha, PETSc.ScalarType(condutancia_interfacial(parametros.resistencia_kapitza)))
    if not gradientes or any(len(gradiente) != 3 for gradiente in gradientes):
        raise ValueError("Cada gradiente macroscópico deve ter três componentes.")

    # Cada campo é contínuo, então avg(.) é apenas o valor no ponto da faceta.
    salto_u = ufl.avg(theta_m) - ufl.avg(theta_p)
    salto_v = ufl.avg(v_m) - ufl.avg(v_p)
    a = (k_m * ufl.inner(ufl.grad(theta_m), ufl.grad(v_m)) * dx_m
         + k_p * ufl.inner(ufl.grad(theta_p), ufl.grad(v_p)) * dx_p
         + h_K * salto_u * salto_v * dS_K)
    if dx_n is not None:
        # Núcleo de ar: mesmo campo theta_p da casca, temperatura contínua na
        # parede interna, condutividade k_nucleo.
        k_n = fem.Constant(malha, PETSc.ScalarType(parametros.k_nucleo))
        a = a + k_n * ufl.inner(ufl.grad(theta_p), ufl.grad(v_p)) * dx_n
    formas_L = []
    for gradiente in gradientes:
        G = ufl.as_vector(tuple(PETSc.ScalarType(valor) for valor in gradiente))
        L = (ufl.inner(k_m * G, ufl.grad(v_m)) * dx_m
             + ufl.inner(k_p * G, ufl.grad(v_p)) * dx_p)
        if dx_n is not None:
            L = L + ufl.inner(k_n * G, ufl.grad(v_p)) * dx_n
        formas_L.append(fem.form(L))

    thetas, iteracoes = _resolver_reduzido(fem.form(a), formas_L, W, sistema_periodico)
    solucoes = []
    for theta in thetas:
        t_m, t_p = theta.split()
        solucoes.append(SolucaoKapitza(theta, t_m, t_p, parametros.k_matriz, parametros.k_particula,
                                       parametros.resistencia_kapitza, dx_m, dx_p, dS_K, iteracoes,
                                       parametros.k_nucleo if dx_n is not None else None, dx_n))
    return solucoes


def _reunir_vetor_no_processo_0(b, comm):
    """Reunir um vetor PETSc distribuído como NumPy no processo 0."""
    import numpy as np
    inicio, fim = b.getOwnershipRange()
    partes = comm.gather((inicio, fim, b.array.copy()), root=0)
    if comm.rank != 0:
        return None
    vetor = np.zeros(b.getSize())
    for inicio, fim, valores in partes:
        vetor[inicio:fim] = valores
    return vetor


def _reunir_no_processo_0(A, b, comm):
    """Reunir matriz e vetor PETSc distribuídos como SciPy/NumPy no processo 0."""
    import numpy as np
    import scipy.sparse as sp

    inicio, fim = A.getOwnershipRange()
    ai, aj, av = A.getValuesCSR()
    partes = comm.gather((inicio, fim, ai.copy(), aj.copy(), av.copy(), b.array.copy()), root=0)
    if comm.rank != 0:
        return None, None
    n = A.getSize()[0]
    linhas, colunas, valores, vetor = [], [], [], np.zeros(n)
    for inicio, fim, ai, aj, av, bv in partes:
        linhas.append(np.repeat(np.arange(inicio, fim), np.diff(ai)))
        colunas.append(aj)
        valores.append(av)
        vetor[inicio:fim] = bv
    A_global = sp.csr_matrix((np.concatenate(valores),
                              (np.concatenate(linhas), np.concatenate(colunas))), shape=(n, n))
    return A_global, vetor


def _matriz_transformacao(n, escravos, mestres):
    """``K`` (n x n_reduzido) com ``u = K u_r``: identidade nos DOFs livres e
    ``u[escravo] = u_r[coluna(mestre)]`` nos escravos."""
    import numpy as np
    import scipy.sparse as sp

    livre = np.ones(n, dtype=bool)
    livre[escravos] = False
    coluna = np.full(n, -1, dtype=np.int64)
    coluna[livre] = np.arange(livre.sum())
    if np.any(coluna[mestres] < 0):
        raise RuntimeError("Um mestre periódico também é escravo.")
    coluna_de = coluna.copy()
    coluna_de[escravos] = coluna[mestres]
    K = sp.csr_matrix((np.ones(n), (np.arange(n), coluna_de)), shape=(n, int(livre.sum())))
    return K, coluna


def _resolver_reduzido(forma_a, formas_L, W, sistema_periodico):
    """Resolver ``K^T A K u_r = K^T b`` e reconstruir ``u = K u_r``.

    ``A`` e ``b`` são montados pelo DOLFINx, que aceita integrais em
    facetas internas (``dS``); ``K`` aplica a periodicidade exata da malha
    periódica (mesma estratégia de ``mapa_R_RK_phi_mpi.py``). A âncora
    ``theta(0) = 0`` é imposta no sistema JÁ reduzido: o vértice da origem
    é o mestre dos cantos, e impô-la antes somaria as equações dos cantos
    escravos à linha da âncora.

    ``formas_L`` é uma lista de lados direitos: a matriz é fatorada uma vez
    e cada um custa só uma substituição. Devolve uma função por lado direito.

    Funciona em série e com ``mpirun``: o sistema é reunido no processo 0,
    que o fatora (LU do PETSc, MUMPS se disponível), e a solução é
    redistribuída. Para malhas muito grandes, o próximo passo seria montar
    ``K`` como matriz PETSc distribuída e usar ``PtAP``.
    """
    import numpy as np
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc

    comm = W.mesh.comm
    bcs = list(sistema_periodico.condicoes_dirichlet)
    A = fem_petsc.assemble_matrix(forma_a, bcs=bcs, diag=1.0)
    A.assemble()
    vetores = []
    for forma_L in formas_L:
        b = fem_petsc.assemble_vector(forma_L)
        fem_petsc.apply_lifting(b, [forma_a], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, bcs)
        vetores.append(b)
    A_global, b_global = _reunir_no_processo_0(A, vetores[0], comm)
    b_globais = [b_global] + [_reunir_vetor_no_processo_0(b, comm) for b in vetores[1:]]
    A.destroy()
    for b in vetores:
        b.destroy()

    u_globais = None
    iteracoes = None
    if comm.rank == 0:
        n = A_global.shape[0]
        K, coluna = _matriz_transformacao(n, sistema_periodico.escravos, sistema_periodico.mestres)
        A_r = (K.T @ A_global @ K).tolil()
        # Escala típica da diagonal, para a âncora não piorar o condicionamento.
        escala = float(np.median(np.abs(A_global.diagonal()[A_global.diagonal() != 1.0]))) or 1.0
        r = int(coluna[sistema_periodico.ancora])
        A_r[r, :] = 0.0
        A_r[:, r] = 0.0
        A_r[r, r] = escala
        A_r = A_r.tocsr()
        A_r.sort_indices()
        M = PETSc.Mat().createAIJ(size=A_r.shape, csr=(A_r.indptr, A_r.indices, A_r.data),
                                  comm=MPI.COMM_SELF)
        M.assemble()
        x_r, rhs = M.createVecs()
        ksp = PETSc.KSP().create(MPI.COMM_SELF)
        ksp.setOperators(M)
        pc = ksp.getPC()
        if SOLVER_LINEAR == "iterativo":
            # A_r é simétrica e positiva definida (a âncora tira o modo constante).
            M.setOption(PETSc.Mat.Option.SPD, True)
            ksp.setType("cg")
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setTolerances(rtol=TOLERANCIA_ITERATIVO, atol=0.0, max_it=5000)
            try:
                pc.setType("hypre")
                pc.setHYPREType("boomeramg")
            except PETSc.Error:
                pc.setType("gamg")  # PETSc sem hypre
        elif SOLVER_LINEAR == "lu":
            ksp.setType("preonly")
            pc.setType("lu")
            try:
                pc.setFactorSolverType("mumps")
            except PETSc.Error:
                pass  # sem MUMPS: usa o LU nativo do PETSc
        else:
            raise ValueError(f"SOLVER_LINEAR deve ser 'lu' ou 'iterativo', não {SOLVER_LINEAR!r}.")
        ksp.setErrorIfNotConverged(True)
        u_globais = []
        for b_global in b_globais:
            b_r = np.asarray(K.T @ b_global).ravel()
            b_r[r] = 0.0
            rhs.array[:] = b_r
            try:
                ksp.solve(rhs, x_r)
            except PETSc.Error:
                if ksp.getType() != "cg":
                    raise
                print(f"Aviso: o CG não convergiu em {ksp.getIterationNumber()} iterações; "
                      "usando LU neste caso.", flush=True)
                ksp.setType("preonly")
                pc.setType("lu")
                try:
                    pc.setFactorSolverType("mumps")
                except PETSc.Error:
                    pass
                ksp.solve(rhs, x_r)  # LU: fatora no primeiro e reaproveita; CG: o setup do AMG também
            iteracoes = ksp.getIterationNumber()
            x = x_r.array
            referencia = abs(A_r).max() * np.abs(x).max() + np.abs(b_r).max() + 1e-300
            residuo = np.abs(A_r @ x - b_r).max() / referencia
            if not np.isfinite(residuo) or residuo > 1e-8:
                raise RuntimeError(f"Sistema reduzido mal resolvido (resíduo relativo {residuo:.2e}).")
            u_globais.append(np.asarray(K @ x).ravel())
        for objeto in (ksp, M, x_r, rhs):
            objeto.destroy()
    u_globais = comm.bcast(u_globais, root=0)
    iteracoes = comm.bcast(iteracoes, root=0)

    mapa = W.dofmap.index_map
    bs = W.dofmap.index_map_bs
    blocos = mapa.local_to_global(np.arange(mapa.size_local + mapa.num_ghosts, dtype=np.int32))
    globais = np.repeat(blocos, bs) * bs + np.tile(np.arange(bs), len(blocos))
    thetas = []
    for u_global in u_globais:
        theta = fem.Function(W, name="theta")
        theta.x.array[:len(globais)] = u_global[globais]
        theta.x.scatter_forward()
        thetas.append(theta)
    return thetas, iteracoes
