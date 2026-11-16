"""Dispersões aleatórias com muitas esferas (N = 27, 64, ...): cópia de aleatorio.py.

Mesmo sorteio, mesma malha, mesmo Aitken e mesmo CSV de aleatorio.py, mas usa os
módulos da pasta 11 (malha.py com etiquetas em blocos de 1000, até 999 esferas)
no lugar dos da pasta 06 (máximo 9). k_p = 5 fixo, como na pasta 06.
Objetivo: mostrar que a média de k/k_HJ − 1 em φ = 5 e 10 % não depende de N
(8, 27, 64) e dar barras de erro ao coeficiente c da lei c·φ².

Rodar de 06_dispersao_aleatoria (a pasta 11 precisa estar ao lado):
    python3 estudos/aleatorio_n/aleatorio_n.py --n 27 --phi 0.05 0.1 --so-malha
    python3 estudos/aleatorio_n/aleatorio_n.py --n 27 --phi 0.05 0.1
    python3 estudos/aleatorio_n/aleatorio_n.py --n 64 --phi 0.05 0.1
Saída: estudos/aleatorio_n/aleatorio_n.csv (mesmas colunas de aleatorio.csv).

Texto original de aleatorio.py:

Para cada phi e cada semente, sorteia N centros na célula (adição sequencial
aleatória, sem sobreposição) e resolve o tensor de condutividade com os três
gradientes x, y e z, em três níveis de h_bulk com extrapolação de Aitken, como
em ``aglomerados.py``. Uma dispersão aleatória não tem simetria, então os três
eixos são sempre resolvidos; k_media = (k_xx + k_yy + k_zz)/3 é o valor que se
compara com Hasselman–Johnson no mesmo phi.

Para cada phi e R_K também roda a esfera isolada (semente = -1, N = 1), a
mesma referência dos aglomerados: o efeito da dispersão é
desvio(aleatória) - desvio(isolada).

Restrições do sorteio (limitações da malha, não da física):
  * (aqui: até 999 esferas, módulos da pasta 11);
  * nenhuma esfera cruza uma face do cubo: cada centro fica a pelo menos
    R + margem das faces (padrão margem = 0,2R). Assim a lacuna com qualquer
    imagem periódica é >= 2 margem = 0,4R = 4 h_int, que a malha resolve sem
    refino de lacuna (ela só refina lacunas internas). Isso cria uma camada
    sem centros junto às faces: a distribuição não é perfeitamente uniforme,
    e vale citar isso ao apresentar;
  * lacuna mínima entre esferas da célula >= --lacuna-minima (padrão 0,05R),
    para a malha da lacuna não explodir.

Rodar de 06_dispersao_aleatoria, em série, por exemplo
    python3 estudos/aleatorio/aleatorio.py --phi 0.01 0.025 0.05 0.1 \\
        --sementes 0 1 2 3 4 --rk 1e-10 1e-8
O CSV (estudos/aleatorio/aleatorio.csv) é gravado ao fim de cada caso, com
os centros (em unidades de R) para reproduzir a geometria; rodar de novo pula
os casos já feitos. Para ver um sorteio sem resolver nada:
    python3 estudos/aleatorio/aleatorio.py --phi 0.05 --sementes 3 --so-sortear
"""

import argparse
import csv
import json
import resource
import sys
import time
from dataclasses import replace
from math import dist, pi
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[2] / "11_muitas_particulas"  # módulos que aceitam até 999 esferas
if not RAIZ.exists():
    raise SystemExit(f"Não achei {RAIZ}: a pasta 11 precisa estar ao lado da 06.")
sys.path.insert(0, str(RAIZ))
K_PARTICULA = 5.0  # o mesmo da pasta 06 (o padrão de parametros.py da pasta 11 é 1,38)

import numpy as np  # noqa: E402

COLUNAS = ["N", "phi", "semente", "lado_R", "lacuna_interna_R", "lacuna_imagem_R", "R_K", "alfa",
           "k_xx", "k_yy", "k_zz", "k_media", "k_HJ", "desvio_HJ_percent", "anisotropia_percent",
           "razao_xx", "razao_yy", "razao_zz", "fora_diagonal_max", "nos_nivel_3", "tempo_s",
           "centros_R"]


def hasselman_johnson(phi: float, alfa: float, k_m: float, k_p: float) -> float:
    a = k_p * (1 + 2 * alfa) + 2 * k_m
    b = k_p * (1 - alfa) - k_m
    return k_m * (a + 2 * phi * b) / (a - phi * b)


def extrapolar(k1: float, k2: float, k3: float) -> tuple[float, float]:
    """Aitken (Δ²), igual ao de aglomerados.py e varredura_convergida.py."""
    d1, d2 = k2 - k1, k3 - k2
    if d1 == 0:
        return k3, float("nan")
    razao = d2 / d1
    if not 0 < razao < 1:
        return k3, razao
    return k3 - d2 * d2 / (d2 - d1), razao


def sortear_centros(n, phi, semente, margem=0.2, lacuna_minima=0.05, tentativas=200_000):
    """N centros em unidades de R, por adição sequencial aleatória.

    Devolve (lado, centros). Levanta ValueError se não couber depois de
    ``tentativas`` sorteios por esfera (phi alto demais para N e margem).
    """
    lado = (4 * pi * n / (3 * phi)) ** (1 / 3)
    baixo, alto = 1 + margem, lado - 1 - margem
    if alto <= baixo:
        raise ValueError(f"Célula pequena demais (L = {lado:.2f}R) para a margem {margem}R.")
    gerador = np.random.default_rng(semente)
    centros = []
    for _ in range(n):
        for _ in range(tentativas):
            candidato = tuple(gerador.uniform(baixo, alto, 3))
            if all(dist(candidato, c) >= 2 + lacuna_minima for c in centros):
                centros.append(candidato)
                break
        else:
            raise ValueError(f"Não coube a esfera {len(centros) + 1} de {n} com phi = {phi:g}.")
    return lado, centros


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phi", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1])
    parser.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rk", type=float, nargs="+", default=[1e-10, 1e-8])
    parser.add_argument("--n", type=int, default=27, help="Esferas por célula (até 999).")
    parser.add_argument("--margem", type=float, default=0.2, help="Distância mínima esfera-face, em R.")
    parser.add_argument("--lacuna-minima", type=float, default=0.05, help="Menor lacuna entre esferas, em R.")
    parser.add_argument("--niveis", type=float, nargs=3, default=(0.35, 0.25, 0.18),
                        help="h_bulk/R dos três níveis (Aitken).")
    parser.add_argument("--h-interface-relativo", type=float, default=0.10)
    parser.add_argument("--so-sortear", action="store_true", help="Só sortear e mostrar as lacunas.")
    parser.add_argument("--so-malha", action="store_true", help="Só gerar as malhas (Gmsh) e contar nós.")
    parser.add_argument("--sem-isolada", action="store_true",
                        help="Pula a esfera isolada (já está em aleatorio.csv).")
    parser.add_argument("--saida", default=str(PASTA / "aleatorio_n.csv"))
    args = parser.parse_args()

    if args.so_sortear:
        for phi in args.phi:
            for semente in args.sementes:
                lado, centros = sortear_centros(args.n, phi, semente, args.margem, args.lacuna_minima)
                lacunas = sorted(dist(a, b) - 2 for i, a in enumerate(centros) for b in centros[i + 1:])
                print(f"phi={phi:g} semente={semente}: L={lado:.3f}R, menores lacunas "
                      + ", ".join(f"{s:.3f}" for s in lacunas[:4]) + " (em R)")
        return

    if args.so_malha:
        from geometria import geometria_de_centros
        from malha import gerar_malha
        raio = 50e-9
        for phi in args.phi:
            semente = args.sementes[0]
            lado, centros = sortear_centros(args.n, phi, semente, args.margem, args.lacuna_minima)
            geometria = geometria_de_centros(lado * raio, raio, [tuple(x * raio for x in c) for c in centros])
            for h_bulk in args.niveis:
                inicio = time.time()
                nos = gerar_malha(geometria, tamanho_elemento=h_bulk * raio, numero_threads=1,
                                  malha_adaptativa=True, tamanho_interface=args.h_interface_relativo * raio,
                                  alcance_refino=raio / 2, engrossar_longe=True, so_contar_nos=True)
                pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
                print(f"N={args.n} phi={phi:g} semente {semente} h_bulk={h_bulk:g}R: {nos} nós "
                      f"({time.time() - inicio:.0f} s, pico {pico:.1f} GB no Gmsh)", flush=True)
        print("Referência: ~800 mil nós usaram ~7 GB no solver (pasta 11). Acima de ~900 mil, "
              "use --niveis 0.5 0.35 0.25.")
        return

    from mpi4py import MPI

    import solver
    from geometria import geometria_de_centros
    from malha import gerar_malha
    from parametros import Parametros
    from periodicidade import preparar_periodicidade_kapitza
    from pos_processamento import calcular_condutividade

    comm = MPI.COMM_WORLD
    saida = Path(args.saida)
    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as arquivo:
            feitos = {(int(l["N"]), round(float(l["phi"]), 6), int(l["semente"]), float(l["R_K"]))
                      for l in csv.DictReader(arquivo)}
    elif comm.rank == 0:
        saida.parent.mkdir(parents=True, exist_ok=True)
        with saida.open("w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(COLUNAS)

    gradientes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for phi in args.phi:
        base = Parametros(fracao_volumetrica=phi, k_particula=K_PARTICULA)
        raio = base.raio
        # A esfera isolada primeiro: é a referência de cada phi.
        casos = ([] if args.sem_isolada else [(1, -1)]) + [(args.n, semente) for semente in args.sementes]
        for n, semente in casos:
            rks = [rk for rk in args.rk if (n, round(phi, 6), semente, rk) not in feitos]
            if not rks:
                continue
            if semente < 0:
                lado = (4 * pi / (3 * phi)) ** (1 / 3)
                centros = [(lado / 2,) * 3]
            else:
                try:
                    lado, centros = sortear_centros(n, phi, semente, args.margem, args.lacuna_minima)
                except ValueError as erro:
                    if comm.rank == 0:
                        print(f"--- phi={phi:g} semente={semente} pulado: {erro}", flush=True)
                    continue
            geometria = geometria_de_centros(lado * raio, raio,
                                             [tuple(x * raio for x in c) for c in centros])
            inicio = time.time()
            # fluxos[rk][j] = [vetor fluxo médio nos três níveis] com G na direção j
            fluxos = {rk: {j: [] for j in range(3)} for rk in rks}
            for h_bulk in args.niveis:
                dados = gerar_malha(geometria, tamanho_elemento=h_bulk * raio, numero_threads=1,
                                    malha_adaptativa=True,
                                    tamanho_interface=args.h_interface_relativo * raio,
                                    alcance_refino=raio / 2, engrossar_longe=True)
                sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
                nos = dados.mesh.topology.index_map(0).size_global
                for rk in rks:
                    caso = replace(base, resistencia_kapitza=rk)
                    solucoes = solver.resolver_kapitza_gradientes(caso, dados, sistema, gradientes)
                    for j, (gradiente, solucao) in enumerate(zip(gradientes, solucoes)):
                        fluxo = calcular_condutividade(solucao, caso, gradiente).fluxo_medio
                        fluxos[rk][j].append(fluxo)
                    if comm.rank == 0:
                        diagonal = ", ".join(f"{fluxos[rk][j][-1][j]:.6f}" for j in range(3))
                        print(f"N={n} phi={phi:g} semente={semente} R_K={rk:g} h_bulk={h_bulk:g}: "
                              f"k_ii = ({diagonal}) ({nos} nós)", flush=True)
            tempo = (time.time() - inicio) / len(rks)
            pico_gb = comm.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                                  op=MPI.SUM, root=0)
            if comm.rank != 0:
                continue
            print(f"--- N={n} phi={phi:g} semente={semente}: {tempo * len(rks):.0f} s, "
                  f"pico de memória {pico_gb:.1f} GB", flush=True)
            with saida.open("a", newline="", encoding="utf-8") as arquivo:
                escritor = csv.writer(arquivo)
                for rk in rks:
                    k, razoes, fora = [], [], 0.0
                    for j in range(3):
                        k1, k2, k3 = (f[j] for f in fluxos[rk][j])
                        valor, razao = extrapolar(k1, k2, k3)
                        k.append(valor)
                        razoes.append(razao)
                        fora = max(fora, max(abs(fluxos[rk][j][-1][i]) for i in range(3) if i != j) / k3)
                    media = sum(k) / 3
                    alfa = rk * base.k_matriz / raio
                    k_hj = hasselman_johnson(geometria.fracao_volumetrica, alfa,
                                             base.k_matriz, base.k_particula)
                    escritor.writerow([
                        n, f"{geometria.fracao_volumetrica:.8f}", semente, f"{lado:.6f}",
                        f"{geometria.lacuna_interna / raio:.4f}" if n > 1 else "",
                        f"{geometria.lacuna_imagem_periodica / raio:.4f}", rk, f"{alfa:.6g}",
                        *(f"{x:.8f}" for x in k), f"{media:.8f}", f"{k_hj:.8f}",
                        f"{100 * (media / k_hj - 1):.4f}", f"{100 * (max(k) / min(k) - 1):.4f}",
                        *(f"{r:.4f}" for r in razoes), f"{fora:.2e}", nos, f"{tempo:.1f}",
                        json.dumps([[round(x, 5) for x in c] for c in centros])])
                    print(f"==> N={n} phi={phi:g} semente={semente} R_K={rk:g}: k_media={media:.6f}, "
                          f"desvio de HJ {100 * (media / k_hj - 1):+.3f} %, "
                          f"anisotropia {100 * (max(k) / min(k) - 1):.2f} %", flush=True)


if __name__ == "__main__":
    main()
