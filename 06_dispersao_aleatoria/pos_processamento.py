"""Fluxo médio e condutividade efetiva da RVE."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultadoHomogeneizacao:
    """Resposta média para um gradiente macroscópico imposto."""

    fluxo_medio: tuple[float, float, float]
    k_efetivo_xx: float | None
    volume: float
    salto_rms: float | None = None        # sqrt(<[T]²>_Γ) em K (só Kapitza)
    area_interface: float | None = None   # m² (só Kapitza)


def calcular_condutividade(solucao, parametros,
                           gradiente: tuple[float, float, float] | None = None) -> ResultadoHomogeneizacao:
    """Calcular ``<q>`` para o gradiente aplicado.

    Com ``theta`` periódica, o gradiente médio de temperatura é exatamente
    ``-G_x e_x`` (inclusive no caso de Kapitza, em que os saltos de ``theta``
    nas interfaces fazem parte desse balanço). O fluxo é integrado só no
    volume das fases. As componentes transversais servem de diagnóstico:
    na orientação longitudinal e transversal devem ser ~0 em relação a q_x.
    """
    try:
        from mpi4py import MPI
        import ufl
        from dolfinx import fem
    except ModuleNotFoundError as erro:
        raise RuntimeError("O pós-processamento requer mpi4py, UFL e FEniCSx.") from erro

    def integrar(forma):
        local = fem.assemble_scalar(fem.form(forma))
        return float(comm.allreduce(local, op=MPI.SUM))

    gradiente = gradiente or (parametros.gradiente_x, 0.0, 0.0)
    if len(gradiente) != 3:
        raise ValueError("O gradiente macroscópico deve ter três componentes.")
    salto_rms = area = None
    if hasattr(solucao, "fluxo_por_fase"):
        malha = solucao.theta.function_space.mesh
        comm = malha.comm
        dx_m, dx_p, dS_K = solucao.dx_m, solucao.dx_p, solucao.dS_K
        q_m, q_p = solucao.fluxo_por_fase(gradiente)
        volume = integrar(1.0 * dx_m) + integrar(1.0 * dx_p)
        componentes = [(integrar(q_m[i] * dx_m) + integrar(q_p[i] * dx_p)) for i in range(3)]
        salto = ufl.avg(solucao.theta_m) - ufl.avg(solucao.theta_p)
        area = integrar(1.0 * dS_K)
        if area > 0:
            salto_rms = (integrar(salto**2 * dS_K) / area) ** 0.5
    else:
        malha = solucao.theta.function_space.mesh
        comm = malha.comm
        dx = solucao.medida_volume()
        q = solucao.fluxo(gradiente)
        volume = integrar(1.0 * dx)
        componentes = [integrar(q[i] * dx) for i in range(3)]

    if volume <= 0:
        raise RuntimeError("O volume integrado da RVE deve ser positivo.")
    fluxo = tuple(c / volume for c in componentes)
    k_xx = fluxo[0] / gradiente[0] if gradiente[0] != 0 else None
    return ResultadoHomogeneizacao(fluxo, k_xx, volume, salto_rms, area)


def montar_tensor(resultados: tuple[ResultadoHomogeneizacao, ...],
                  gradientes: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, ...], ...]:
    """Montar ``K_eff`` a partir de três carregamentos cartesianos.

    A coluna ``j`` é ``<q>/G_j`` para o carregamento cuja única componente
    não nula é ``G_j``. Vale ``<q> = K_eff G`` na convenção adotada.
    """
    if len(resultados) != 3 or len(gradientes) != 3:
        raise ValueError("O tensor requer os três carregamentos cartesianos.")
    colunas = []
    for j, (resultado, G) in enumerate(zip(resultados, gradientes)):
        if len(G) != 3 or G[j] == 0 or any(G[i] != 0 for i in range(3) if i != j):
            raise ValueError("Cada carregamento deve ter somente G_j não nulo.")
        colunas.append(tuple(q / G[j] for q in resultado.fluxo_medio))
    return tuple(tuple(colunas[j][i] for j in range(3)) for i in range(3))
