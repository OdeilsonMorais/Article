"""Muitas partículas (70–80) dispersas aleatoriamente, maciças ou ocas, a 5 % em massa.

A célula é dimensionada pela massa: 5 % em massa da tinta dá a fração de óxido
φ_sólido pela densidade de cada material; a partícula oca (parede t/R) ocupa
φ_externo = φ_sólido / (1 − (1 − t/R)³), e o lado da célula com N esferas de
raio R é

    L = R · (4π N / (3 φ_externo))^(1/3).

Ex.: N = 80, diâmetro 100 nm, γ-Al₂O₃ maciça (φ = 2,1 %): L ≈ 25R ≈ 1,26 µm.

Os centros são sorteados como na pasta 06 (adição sequencial aleatória, sem
sobreposição, sem cruzar as faces). Para cada caso resolve os três gradientes
em três níveis de h_bulk e extrapola com Aitken; compara k_média com HJ da
esfera oca equivalente no mesmo φ externo.

Rodar de 11_muitas_particulas, em série. Comece estimando a malha (só Gmsh):
    python3 estudos/muitas.py --material gama --so-malha
    python3 estudos/muitas.py --material gama --espessura 0.2 --so-malha
Depois resolva:
    python3 estudos/muitas.py --material gama
    python3 estudos/muitas.py --material sio2 --espessura 1 0.2
Resultados em estudos/muitas.csv (gravado a cada caso; rodar de novo pula o feito).
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
RAIZ = PASTA.parent  # 11_muitas_particulas
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

# nome: (rótulo, k W/mK, densidade g/cm³)
MATERIAIS = {"gama": ("γ-Al₂O₃", 5.0, 3.65), "sio2": ("SiO₂", 1.36, 2.20),
             "anatase": ("TiO₂ anatase", 8.5, 3.89), "quartzo": ("SiO₂ quartzo", 8.0, 2.65)}
COLUNAS = ["material", "k_p", "N", "diametro_nm", "fracao_massica", "phi_solido", "phi_externo",
           "t_R", "semente", "lado_R", "lado_um", "lacuna_minima_R", "R_K", "alfa",
           "k_xx", "k_yy", "k_zz", "k_media", "k_sobre_km", "k_HJ_oca", "desvio_HJ_percent",
           "anisotropia_percent", "razao_xx", "razao_yy", "razao_zz", "nos_nivel_3", "pico_GB",
           "tempo_s", "centros_R"]


def fracao_volumetrica(w, rho_p, rho_m):
    return (w / rho_p) / (w / rho_p + (1 - w) / rho_m)


def k_equivalente(k_p, k_n, c):
    return k_p * ((k_n + 2 * k_p) + 2 * c * (k_n - k_p)) / ((k_n + 2 * k_p) - c * (k_n - k_p))


def hasselman_johnson(phi, alfa, k_m, k_p):
    a = k_p * (1 + 2 * alfa) + 2 * k_m
    b = k_p * (1 - alfa) - k_m
    return k_m * (a + 2 * phi * b) / (a - phi * b)


def extrapolar(k1, k2, k3):
    d1, d2 = k2 - k1, k3 - k2
    if d1 == 0:
        return k3, float("nan")
    razao = d2 / d1
    if not 0 < razao < 1:
        return k3, razao
    return k3 - d2 * d2 / (d2 - d1), razao


def sortear_centros(n, phi, semente, margem=0.2, lacuna_minima=0.1, tentativas=200_000):
    """N centros em unidades de R (como na pasta 06). Devolve (lado, centros)."""
    lado = (4 * pi * n / (3 * phi)) ** (1 / 3)
    baixo, alto = 1 + margem, lado - 1 - margem
    if alto <= baixo:
        raise ValueError(f"Célula pequena demais (L = {lado:.2f}R).")
    gerador = np.random.default_rng(semente)
    centros = np.empty((0, 3))
    for _ in range(n):
        for _ in range(tentativas):
            candidato = gerador.uniform(baixo, alto, 3)
            if len(centros) == 0 or np.min(np.linalg.norm(centros - candidato, axis=1)) >= 2 + lacuna_minima:
                centros = np.vstack([centros, candidato])
                break
        else:
            raise ValueError(f"Não coube a esfera {len(centros) + 1} de {n} com φ = {phi:g}.")
    return lado, [tuple(c) for c in centros]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", choices=sorted(MATERIAIS), default="gama")
    parser.add_argument("--k-particula", type=float, default=None, help="sobrescreve o k do material")
    parser.add_argument("--n", type=int, default=80, help="número de partículas na célula")
    parser.add_argument("--diametro", type=float, default=100.0, help="diâmetro externo (nm)")
    parser.add_argument("--fracao-massica", type=float, default=0.05)
    parser.add_argument("--rho-tinta", type=float, default=1.5, help="g/cm³ (falta o valor da N2680)")
    parser.add_argument("--espessura", type=float, nargs="+", default=[1.0], help="t/R; 1 = maciça")
    parser.add_argument("--k-nucleo", type=float, default=0.026)
    parser.add_argument("--rk", type=float, nargs="+", default=[1e-8])
    parser.add_argument("--sementes", type=int, nargs="+", default=[0])
    parser.add_argument("--margem", type=float, default=0.2, help="distância mínima esfera-face (R)")
    parser.add_argument("--lacuna-minima", type=float, default=0.1, help="menor lacuna entre esferas (R)")
    parser.add_argument("--niveis", type=float, nargs=3, default=(0.5, 0.35, 0.25),
                        help="h_bulk/R dos três níveis (Aitken). 0,18 passa de 2 milhões de nós com 80 esferas.")
    parser.add_argument("--so-sortear", action="store_true", help="só sortear e mostrar a célula")
    parser.add_argument("--so-malha", action="store_true", help="só gerar as malhas (Gmsh) e contar nós")
    parser.add_argument("--saida", default=str(PASTA / "muitas.csv"))
    args = parser.parse_args()

    rotulo, k_p, rho = MATERIAIS[args.material]
    k_p = args.k_particula or k_p
    raio = args.diametro * 1e-9 / 2
    phi_sol = fracao_volumetrica(args.fracao_massica, rho, args.rho_tinta)
    k_m = 0.2

    from geometria import geometria_de_centros
    from malha import gerar_malha

    casos = []
    for t_R in args.espessura:
        if not 0 < t_R <= 1:
            raise SystemExit("t/R deve estar em (0, 1].")
        c = (1 - t_R) ** 3
        phi_ext = phi_sol / (1 - c)
        for semente in args.sementes:
            try:
                lado, centros = sortear_centros(args.n, phi_ext, semente, args.margem, args.lacuna_minima)
            except ValueError as erro:
                print(f"--- t/R={t_R:g} semente {semente} pulado: {erro} Tente --margem 0.1 "
                      "--lacuna-minima 0.05.", flush=True)
                continue
            lacunas = sorted(dist(a, b) - 2 for i, a in enumerate(centros) for b in centros[i + 1:])
            print(f"{rotulo} k={k_p:g}, t/R={t_R:g}, semente {semente}: 5 % em massa → φ_sólido = "
                  f"{phi_sol:.2%}, φ_externo = {phi_ext:.2%}; L = {lado:.2f}R = {lado * raio * 1e6:.3f} µm; "
                  f"menores lacunas {', '.join(f'{s:.2f}' for s in lacunas[:3])} R", flush=True)
            casos.append((t_R, phi_ext, semente, lado, centros))
    if args.so_sortear:
        return

    def geometria_do_caso(t_R, lado, centros):
        return geometria_de_centros(lado * raio, raio, [tuple(x * raio for x in p) for p in centros],
                                    (1 - t_R) * raio)

    def malha(geometria, t_R, h_bulk, contar=False):
        h_int = min(0.10, t_R / 3) if t_R < 1 else 0.10
        return gerar_malha(geometria, tamanho_elemento=h_bulk * raio, numero_threads=1,
                           malha_adaptativa=True, tamanho_interface=h_int * raio,
                           alcance_refino=raio / 2, engrossar_longe=True, so_contar_nos=contar)

    if args.so_malha:
        for t_R, phi_ext, semente, lado, centros in casos:
            geometria = geometria_do_caso(t_R, lado, centros)
            for h_bulk in args.niveis:
                inicio = time.time()
                nos = malha(geometria, t_R, h_bulk, contar=True)
                pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
                print(f"  t/R={t_R:g} h_bulk={h_bulk:g}R: {nos} nós ({time.time() - inicio:.0f} s, "
                      f"pico {pico:.1f} GB no Gmsh)", flush=True)
        print("Referência (pasta 06, CG+AMG): ~400 mil nós resolvem em ~2 min. "
              "Acima de ~2–3 milhões, 16 GB podem não bastar; use --niveis maiores.")
        return

    from mpi4py import MPI

    import solver
    from parametros import Parametros
    from periodicidade import preparar_periodicidade_kapitza
    from pos_processamento import calcular_condutividade

    comm = MPI.COMM_WORLD
    saida = Path(args.saida)
    feitos = set()
    if saida.exists():
        with saida.open(encoding="utf-8") as f:
            feitos = {(l["material"], float(l["k_p"]), int(l["N"]), float(l["diametro_nm"]),
                       round(float(l["t_R"]), 6), int(l["semente"]), float(l["R_K"]))
                      for l in csv.DictReader(f)}
    elif comm.rank == 0:
        with saida.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUNAS)

    gradientes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for t_R, phi_ext, semente, lado, centros in casos:
        rks = [rk for rk in args.rk if (args.material, k_p, args.n, args.diametro, round(t_R, 6),
                                        semente, rk) not in feitos]
        if not rks:
            continue
        geometria = geometria_do_caso(t_R, lado, centros)
        base = Parametros(raio=raio, fracao_volumetrica=phi_ext, k_particula=k_p, k_nucleo=args.k_nucleo)
        inicio = time.time()
        fluxos = {rk: {j: [] for j in range(3)} for rk in rks}
        for h_bulk in args.niveis:
            dados = malha(geometria, t_R, h_bulk)
            sistema = preparar_periodicidade_kapitza(dados.mesh, geometria, dados.cell_tags)
            nos = dados.mesh.topology.index_map(0).size_global
            for rk in rks:
                caso = replace(base, resistencia_kapitza=rk)
                solucoes = solver.resolver_kapitza_gradientes(caso, dados, sistema, gradientes)
                for j, (g, s) in enumerate(zip(gradientes, solucoes)):
                    fluxos[rk][j].append(calcular_condutividade(s, caso, g).fluxo_medio[j])
                if comm.rank == 0:
                    print(f"t/R={t_R:g} semente={semente} R_K={rk:g} h_bulk={h_bulk:g}: k_ii = ("
                          + ", ".join(f"{fluxos[rk][j][-1]:.6f}" for j in range(3)) + f") ({nos} nós, "
                          f"{time.time() - inicio:.0f} s)", flush=True)
        tempo = (time.time() - inicio) / len(rks)
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        if comm.rank != 0:
            continue
        with saida.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for rk in rks:
                k, razoes = zip(*(extrapolar(*fluxos[rk][j]) for j in range(3)))
                media = sum(k) / 3
                alfa = rk * k_m / raio
                k_hj = hasselman_johnson(phi_ext, alfa, k_m, k_equivalente(k_p, args.k_nucleo, (1 - t_R) ** 3))
                w.writerow([args.material, k_p, args.n, args.diametro, args.fracao_massica, f"{phi_sol:.6f}",
                            f"{phi_ext:.6f}", t_R, semente, f"{lado:.5f}", f"{lado * raio * 1e6:.5f}",
                            f"{geometria.lacuna_interna / raio:.4f}", rk, f"{alfa:.5g}",
                            *(f"{x:.8f}" for x in k), f"{media:.8f}", f"{media / k_m:.6f}", f"{k_hj:.8f}",
                            f"{100 * (media / k_hj - 1):.4f}", f"{100 * (max(k) / min(k) - 1):.4f}",
                            *(f"{r:.4f}" for r in razoes), nos, f"{pico:.1f}", f"{tempo:.0f}",
                            json.dumps([[round(x, 4) for x in p] for p in centros])])
                print(f"==> {rotulo} t/R={t_R:g} semente={semente} R_K={rk:g}: k/k_m = {media / k_m:.4f}, "
                      f"HJ {k_hj / k_m:.4f} ({100 * (media / k_hj - 1):+.2f} %), anisotropia "
                      f"{100 * (max(k) / min(k) - 1):.2f} %, pico {pico:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
