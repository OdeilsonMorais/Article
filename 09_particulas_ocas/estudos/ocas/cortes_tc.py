"""Figura de campo para a espessura crítica: oca acima, em cima e abaixo de t_c.

Reaproveita calcular() e desenhar() de cortes_ocas.py (que não é alterado):
mesmas escalas de cor nas três colunas, T/(G R) com isotermas em cima e
|q|/(k_m G) com linhas de fluxo embaixo. A diferença é que os arquivos de
dados levam k_p no nome (cortes_ocas.py não leva, então misturaria ZIF-8 com
os casos antigos de k_p = 5) e o título fala de t_c.

Padrão: ZIF-8 (k_p = 0,33), φ externo 10 %, R_K = 1e-8, núcleo de ar,
t/R = 0,5, 0,3187 (t_c) e 0,15. Esperado: k/k_m = 1,0289, 1,0000 e 0,9516.
Sílica (k_p = 1,36, t_c/R = 0,068): --k-particula 1.36 --espessuras 0.2 0.068 0.03
(esperado 1,1018, 1,0000 e 0,9443; a parede de 0,03R dá malha bem maior).

Rodar de 09_particulas_ocas:
    python3 estudos/ocas/cortes_tc.py --calcular
    python3 estudos/ocas/cortes_tc.py            # só redesenha
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
import cortes_ocas as co  # noqa: E402


def nome(phi, rk, t_R, k_p, k_n):
    return f"tc_phi{phi:.4g}_rk{rk:g}_t{t_R:.4g}_kp{k_p:g}_kn{k_n:g}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calcular", action="store_true")
    p.add_argument("--phi", type=float, default=0.10, help="φ externo")
    p.add_argument("--rk", type=float, default=1e-8)
    p.add_argument("--espessuras", type=float, nargs="+", default=[0.5, 0.3187, 0.15])
    p.add_argument("--k-particula", type=float, default=0.33)
    p.add_argument("--k-nucleo", type=float, default=0.026)
    p.add_argument("--h-bulk", type=float, default=0.25)
    p.add_argument("--pontos", type=int, default=400)
    a = p.parse_args()

    pasta_dados = PASTA / "dados"
    casos = []
    for t in a.espessuras:
        arq = pasta_dados / f"{nome(a.phi, a.rk, t, a.k_particula, a.k_nucleo)}.npz"
        if a.calcular and not arq.exists():
            pasta_dados.mkdir(exist_ok=True)
            d = co.calcular(a.phi, a.rk, t, a.k_particula, a.k_nucleo, a.h_bulk, a.pontos)
            np.savez_compressed(arq, **d)
            print(f"t/R = {t:g}: k_xx/k_m = {d['k_xx'] / d['k_m']:.5f} ({d['nos']} nós) -> {arq.name}", flush=True)
        if not arq.exists():
            raise SystemExit(f"Falta {arq.name}: rode com --calcular.")
        casos.append(dict(np.load(arq)))

    # Troca só o título geral da figura de cortes_ocas.desenhar.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    original = Figure.suptitle
    titulo = (f"Espessura crítica · kₚ = {a.k_particula:g} W/mK, núcleo k = {a.k_nucleo:g} · "
              f"φ externo = {100 * a.phi:.3g} % · Rₖ = {a.rk:g} m²K/W · corte z = L/2, G ao longo de x").replace(".", ",")
    Figure.suptitle = lambda self, _t, **kw: original(self, titulo, **kw)
    try:
        esp = "_".join(f"{t:g}" for t in a.espessuras)
        co.desenhar(casos, PASTA / "figuras" / f"cortes_tc_kp{a.k_particula:g}_phi{a.phi:.4g}_rk{a.rk:g}_t{esp}.png")
    finally:
        Figure.suptitle = original


if __name__ == "__main__":
    main()
