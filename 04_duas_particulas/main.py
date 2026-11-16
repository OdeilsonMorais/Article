"""Entrada inicial: valida e apresenta a geometria, sem resolver a física."""

import argparse
import json
from dataclasses import asdict, replace

from geometria import preparar_geometria
from malha import gerar_malha
from parametros import Parametros
from periodicidade import preparar_periodicidade, preparar_periodicidade_kapitza
from pos_processamento import calcular_condutividade, montar_tensor
from solver import resolver_contato_perfeito, resolver_kapitza
from varreduras import gerar_casos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--varredura", action="store_true", help="Listar os dez casos iniciais.")
    parser.add_argument("--malha", action="store_true", help="Gerar a malha inicial via Gmsh/FEniCSx.")
    parser.add_argument("--periodicidade", action="store_true", help="Verificar o espaço periódico da flutuação térmica.")
    parser.add_argument("--solver-perfeito", action="store_true", help="Resolver o caso de referência sem Kapitza.")
    parser.add_argument("--solver-kapitza", action="store_true", help="Resolver o caso com resistência de Kapitza.")
    parser.add_argument("--tensor", action="store_true",
                        help="Com um solver, resolver os carregamentos em x, y e z e imprimir K_eff.")
    parser.add_argument("--rk", type=float, default=None, help="R_K em m² K/W, obrigatório para --solver-kapitza.")
    parser.add_argument("--threads", type=int, default=None, help="Núcleos usados pelo Gmsh na malha 3D.")
    parser.add_argument("--h-relativo", type=float, default=0.5,
                        help="Tamanho de elemento h/R; padrão: 0.5.")
    parser.add_argument("--malha-adaptativa", action="store_true",
                        help="Refinar interfaces e lacuna; --h-relativo define o tamanho longe delas.")
    parser.add_argument("--h-interface-relativo", type=float, default=0.1,
                        help="Tamanho de elemento nas interfaces, h_int/R; padrão: 0.1.")
    parser.add_argument("--h-lacuna-relativo", type=float, default=None,
                        help="Tamanho na lacuna, h_lac/R. Omitido: mínimo entre h_int e s/4.")
    parser.add_argument("--alcance-refino-relativo", type=float, default=0.5,
                        help="Alcance a partir das interfaces em R; padrão: 0.5.")
    args = parser.parse_args()
    base = Parametros()
    if args.h_relativo <= 0:
        parser.error("--h-relativo deve ser positivo.")
    if args.h_interface_relativo <= 0:
        parser.error("--h-interface-relativo deve ser positivo.")
    if args.h_lacuna_relativo is not None and args.h_lacuna_relativo <= 0:
        parser.error("--h-lacuna-relativo deve ser positivo.")
    if args.alcance_refino_relativo <= 0:
        parser.error("--alcance-refino-relativo deve ser positivo.")
    if args.solver_kapitza and (args.rk is None or args.rk <= 0):
        parser.error("--solver-kapitza requer --rk positivo em m² K/W.")
    if args.tensor and not (args.solver_perfeito or args.solver_kapitza):
        parser.error("--tensor requer --solver-perfeito ou --solver-kapitza.")
    if args.solver_perfeito and args.solver_kapitza:
        parser.error("Escolha somente um entre --solver-perfeito e --solver-kapitza.")
    if args.rk is not None:
        base = replace(base, resistencia_kapitza=args.rk)
    tamanho_elemento = args.h_relativo * base.raio
    argumentos_malha = {
        "tamanho_elemento": tamanho_elemento,
        "numero_threads": args.threads,
        "malha_adaptativa": args.malha_adaptativa,
        "tamanho_interface": args.h_interface_relativo * base.raio,
        "tamanho_lacuna": (None if args.h_lacuna_relativo is None
                            else args.h_lacuna_relativo * base.raio),
        "alcance_refino": args.alcance_refino_relativo * base.raio,
    }
    if args.solver_kapitza:
        geometria = preparar_geometria(base)
        dados = gerar_malha(geometria, **argumentos_malha)
        sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
        if args.tensor:
            gradientes = tuple(
                tuple(base.gradiente_x if i == j else 0.0 for i in range(3))
                for j in range(3))
            resultados = tuple(
                calcular_condutividade(resolver_kapitza(base, dados, sistema, gradiente), base, gradiente)
                for gradiente in gradientes)
            tensor = montar_tensor(resultados, gradientes)
            if dados.mesh.comm.rank == 0:
                print(f"Tensor Kapitza resolvido: R_K = {base.resistencia_kapitza:.8g} m² K/W")
                print("K_eff [W/(m K)]:")
                for linha in tensor:
                    print("  " + "  ".join(f"{valor:.8g}" for valor in linha))
                print("Fluxos médios por carregamento [W/m²]:")
                for eixo, resultado in zip("xyz", resultados):
                    print(f"  G_{eixo}: {resultado.fluxo_medio}")
            return
        solucao = resolver_kapitza(base, dados, sistema)
        resultado = calcular_condutividade(solucao, base)
        if dados.mesh.comm.rank == 0:
            print(f"Caso Kapitza resolvido: R_K = {base.resistencia_kapitza:.8g} m² K/W")
            print(f"Restrições periódicas: {sistema.numero_restricoes}")
            print(f"Iterações lineares: {solucao.iteracoes}")
            print("Fluxo médio [W/m²]: "
                  f"({resultado.fluxo_medio[0]:.8g}, {resultado.fluxo_medio[1]:.8g}, "
                  f"{resultado.fluxo_medio[2]:.8g})")
            print(f"k_eff,xx [W/(m K)]: {resultado.k_efetivo_xx:.8g}")
            print(f"Área das interfaces [m²]: {resultado.area_interface:.8g}")
            print(f"Salto RMS na interface [K]: {resultado.salto_rms:.8g}")
        return
    if args.solver_perfeito:
        geometria = preparar_geometria(base)
        dados = gerar_malha(geometria, **argumentos_malha)
        sistema = preparar_periodicidade(dados.mesh, geometria.lado)
        if args.tensor:
            gradientes = tuple(
                tuple(base.gradiente_x if i == j else 0.0 for i in range(3))
                for j in range(3))
            resultados = tuple(
                calcular_condutividade(resolver_contato_perfeito(base, dados, sistema, gradiente),
                                       base, gradiente)
                for gradiente in gradientes)
            tensor = montar_tensor(resultados, gradientes)
            if dados.mesh.comm.rank == 0:
                print("Tensor de contato perfeito resolvido.")
                print("K_eff [W/(m K)]:")
                for linha in tensor:
                    print("  " + "  ".join(f"{valor:.8g}" for valor in linha))
                print("Fluxos médios por carregamento [W/m²]:")
                for eixo, resultado in zip("xyz", resultados):
                    print(f"  G_{eixo}: {resultado.fluxo_medio}")
            return
        solucao = resolver_contato_perfeito(base, dados, sistema)
        resultado = calcular_condutividade(solucao, base)
        if dados.mesh.comm.rank == 0:
            print("Caso de contato perfeito resolvido.")
            print(f"Restrições periódicas: {sistema.numero_restricoes}")
            print(f"Iterações lineares: {solucao.iteracoes}")
            print("Fluxo médio [W/m²]: "
                  f"({resultado.fluxo_medio[0]:.8g}, {resultado.fluxo_medio[1]:.8g}, "
                  f"{resultado.fluxo_medio[2]:.8g})")
            print(f"k_eff,xx [W/(m K)]: {resultado.k_efetivo_xx:.8g}")
        return
    if args.periodicidade:
        geometria = preparar_geometria(base)
        dados = gerar_malha(geometria, **argumentos_malha)
        sistema = preparar_periodicidade(dados.mesh, geometria.lado)
        if dados.mesh.comm.rank == 0:
            print(f"Periodicidade criada: {sistema.numero_restricoes} restrições para theta.")
            print("Âncora aplicada: theta(0, 0, 0) = 0.")
        return
    if args.malha:
        dados = gerar_malha(preparar_geometria(base),
                            arquivo_msh="resultados/malha_inicial.msh",
                            **argumentos_malha)
        if dados.mesh.comm.rank == 0:
            print("Malha criada: resultados/malha_inicial.msh")
            print("Volumes: 1=matriz, 2=partícula 1, 3=partícula 2")
            print("Faces: 11=interface 1, 12=interface 2, 20=faces externas")
            if args.malha_adaptativa:
                h_lacuna = (args.h_lacuna_relativo if args.h_lacuna_relativo is not None
                             else min(args.h_interface_relativo,
                                      base.separacao_relativa / 4))
                print("Malha adaptativa: "
                      f"bulk h/R={args.h_relativo:g}, "
                      f"interface h/R={args.h_interface_relativo:g}, "
                      f"lacuna h/R={h_lacuna:g}")
            else:
                print(f"Tamanho de elemento uniforme: h/R = {args.h_relativo:g}")
        return
    casos = gerar_casos(base) if args.varredura else (base,)
    for parametros in casos:
        geometria = preparar_geometria(parametros)
        print(json.dumps({
            "estado": "preparacao_apenas",
            "parametros_SI": asdict(parametros),
            "geometria_SI": asdict(geometria),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
