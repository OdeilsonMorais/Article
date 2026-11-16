"""B5 (FEM): células bidispersas, para conferir a regra de mistura de HJ.

Esferas maciças de dois raios (padrão 25 e 75 nm), com a mesma massa total
de partícula (5 % em massa) e uma fração --fracao-grandes do volume de sólido
nas grandes. Teoria (14_reforcos_teoria/b5_distribuicao.py):
    (k − k_m)/(k + 2k_m) = Σ φ_i β_i,  β_i com α_i = R_K k_m / R_i
Também roda a mesma célula com um raio só (--raios 25 ou --raios 75) para ter o
monodisperso no mesmo φ.

Rodar de 15_bidispersa (a pasta 11 precisa estar ao lado):
    python3 estudos/bidispersa.py --so-sortear
    python3 estudos/bidispersa.py --so-malha
    python3 estudos/bidispersa.py                       # SiO2, 25 + 75 nm, R_K 1e-10 e 1e-8
    python3 estudos/bidispersa.py --material gama
Saída: estudos/bidispersa.csv (retoma de onde parou).
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

import numpy as np

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parent
sys.path.insert(0, str(RAIZ))
from malha_bi import geometria_bi  # noqa: E402

MATERIAIS = {"sio2": (1.36, 2.20), "gama": (5.0, 3.65)}
COLUNAS = ["material", "k_p", "raios_nm", "n_por_raio", "fracao_grandes", "semente", "phi", "lado_nm", "R_K",
           "k_xx", "k_yy", "k_zz", "k_media", "k_sobre_km", "teoria_mistura", "teoria_so_menor",
           "teoria_so_maior", "desvio_mistura_percent", "anisotropia_percent", "nos_nivel_3", "tempo_s",
           "centros_nm"]


def beta(kp, a, km=0.2):
    return (kp * (1 - a) - km) / (kp * (1 + 2 * a) + 2 * km)


def mistura(phis, raios_m, kp, rk, km=0.2):
    s = sum(p * beta(kp, rk * km / r) for p, r in zip(phis, raios_m))
    return (1 + 2 * s) / (1 - s)


def sortear(raios, phi, semente, margem=0.2, lacuna=0.1, tentativas=200_000):
    """Centros (mesma unidade de raios), grandes primeiro; margem e lacuna em múltiplos do menor raio."""
    lado = (sum(4 * pi * r**3 / 3 for r in raios) / phi) ** (1 / 3)
    rmin = min(raios)
    g = np.random.default_rng(semente)
    ordem = sorted(range(len(raios)), key=lambda j: -raios[j])
    centros = [None] * len(raios)
    postos = []
    for j in ordem:
        r = raios[j]
        lo, hi = r + margem * rmin, lado - r - margem * rmin
        for _ in range(tentativas):
            c = tuple(g.uniform(lo, hi, 3))
            if all(dist(c, centros[k]) >= r + raios[k] + lacuna * rmin for k in postos):
                centros[j] = c
                postos.append(j)
                break
        else:
            raise ValueError(f"Não coube a esfera {len(postos) + 1} de {len(raios)} (φ = {phi:.3f}).")
    return lado, centros


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", choices=sorted(MATERIAIS), default="sio2")
    p.add_argument("--raios", type=float, nargs="+", default=[25.0, 75.0], help="nm; um ou dois valores")
    p.add_argument("--n-grandes", type=int, default=2, help="esferas do maior raio")
    p.add_argument("--n-mono", type=int, default=40, help="esferas quando --raios tem um valor só")
    p.add_argument("--fracao-grandes", type=float, default=0.5, help="fração do volume de sólido nas grandes")
    p.add_argument("--fracao-massica", type=float, default=0.05)
    p.add_argument("--rho-tinta", type=float, default=1.5)
    p.add_argument("--rk", type=float, nargs="+", default=[1e-10, 1e-8])
    p.add_argument("--sementes", type=int, nargs="+", default=[0])
    p.add_argument("--niveis", type=float, nargs=3, default=(0.7, 0.5, 0.35), help="h_bulk/R_menor")
    p.add_argument("--h-interface", type=float, default=0.1, help="h_int/R_menor")
    p.add_argument("--so-sortear", action="store_true")
    p.add_argument("--so-malha", action="store_true")
    p.add_argument("--saida", default=str(PASTA / "bidispersa.csv"))
    args = p.parse_args()

    kp, rho = MATERIAIS[args.material]
    w = args.fracao_massica
    phi = (w / rho) / (w / rho + (1 - w) / args.rho_tinta)
    raios_nm = sorted(args.raios)
    if len(raios_nm) == 1:
        lista = [raios_nm[0]] * args.n_mono
        contagem = [args.n_mono]
        f_g = 1.0
    else:
        r1, r2 = raios_nm
        n2 = args.n_grandes
        n1 = round(n2 * (1 - args.fracao_grandes) / args.fracao_grandes * (r2 / r1) ** 3)
        lista = [r1] * n1 + [r2] * n2
        contagem = [n1, n2]
        v1, v2 = n1 * r1**3, n2 * r2**3
        f_g = v2 / (v1 + v2)
    phis = [phi * (1 - f_g), phi * f_g] if len(raios_nm) == 2 else [phi]
    rotulo = "+".join(f"{r:g}" for r in raios_nm)
    print(f"{args.material}: k_p = {kp}, 5 % em massa → φ = {phi:.4f}; raios {rotulo} nm com {contagem} esferas "
          f"(fração de volume nas grandes {f_g:.3f})", flush=True)

    from malha_bi import gerar_malha
    casos = []
    for s in args.sementes:
        lado, centros = sortear(lista, phi, s)
        geo = geometria_bi(lado * 1e-9, [r * 1e-9 for r in lista], [[x * 1e-9 for x in c] for c in centros])
        print(f"  semente {s}: L = {lado:.0f} nm, menor lacuna {geo.lacuna_minima * 1e9:.2f} nm", flush=True)
        casos.append((s, lado, centros, geo))
    if args.so_sortear:
        return
    rmin = raios_nm[0] * 1e-9
    if args.so_malha:
        for s, lado, centros, geo in casos:
            for h in args.niveis:
                t0 = time.time()
                n = gerar_malha(geo, h * rmin, args.h_interface * rmin, so_contar_nos=True)
                pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
                print(f"  semente {s} h_bulk = {h:g} R_menor: {n} nós ({time.time() - t0:.0f} s, pico {pico:.1f} GB)",
                      flush=True)
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
            feitos = {(l["material"], l["raios_nm"], l["n_por_raio"], int(l["semente"]), float(l["R_K"]))
                      for l in csv.DictReader(f)}
    elif comm.rank == 0:
        with saida.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUNAS)
    chave_n = "+".join(str(c) for c in contagem)
    gradientes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for s, lado, centros, geo in casos:
        rks = [rk for rk in args.rk if (args.material, rotulo, chave_n, s, rk) not in feitos]
        if not rks:
            continue
        base = Parametros(raio=rmin, fracao_volumetrica=phi, k_particula=kp)
        t0 = time.time()
        k = {rk: {j: [] for j in range(3)} for rk in rks}
        for h in args.niveis:
            dados = gerar_malha(geo, h * rmin, args.h_interface * rmin)
            sistema = preparar_periodicidade_kapitza(dados.mesh, geo, dados.cell_tags)
            nos = dados.mesh.topology.index_map(0).size_global
            for rk in rks:
                caso = replace(base, resistencia_kapitza=rk)
                sols = solver.resolver_kapitza_gradientes(caso, dados, sistema, gradientes)
                for j, (G, so) in enumerate(zip(gradientes, sols)):
                    k[rk][j].append(calcular_condutividade(so, caso, G).fluxo_medio[j])
                if comm.rank == 0:
                    print(f"semente {s} R_K={rk:g} h={h:g}: k_ii = (" + ", ".join(f"{k[rk][j][-1]:.6f}" for j in range(3))
                          + f") ({nos} nós, {time.time() - t0:.0f} s)", flush=True)
        if comm.rank != 0:
            continue
        with saida.open("a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            for rk in rks:
                kk = []
                for j in range(3):
                    k1, k2, k3 = k[rk][j]
                    d1, d2 = k2 - k1, k3 - k2
                    kk.append(k3 - d2 * d2 / (d2 - d1) if d1 and 0 < d2 / d1 < 1 else k3)
                media = sum(kk) / 3
                teo = mistura(phis, [r * 1e-9 for r in raios_nm], kp, rk)
                so_menor = mistura([phi], [raios_nm[0] * 1e-9], kp, rk)
                so_maior = mistura([phi], [raios_nm[-1] * 1e-9], kp, rk)
                wcsv.writerow([args.material, kp, rotulo, chave_n, f"{f_g:.4f}", s, f"{phi:.6f}", f"{lado:.2f}", rk,
                               *(f"{x:.8f}" for x in kk), f"{media:.8f}", f"{media / 0.2:.6f}", f"{teo:.6f}",
                               f"{so_menor:.6f}", f"{so_maior:.6f}", f"{100 * (media / 0.2 / teo - 1):.4f}",
                               f"{100 * (max(kk) / min(kk) - 1):.4f}", nos, f"{(time.time() - t0) / len(rks):.0f}",
                               json.dumps([[round(x, 2) for x in c] for c in centros])])
                print(f"==> {rotulo} nm, semente {s}, R_K={rk:g}: k/k_m = {media / 0.2:.5f}; mistura {teo:.5f} "
                      f"({100 * (media / 0.2 / teo - 1):+.3f} %); só {raios_nm[0]:g} nm {so_menor:.5f}, "
                      f"só {raios_nm[-1]:g} nm {so_maior:.5f}", flush=True)


if __name__ == "__main__":
    main()
