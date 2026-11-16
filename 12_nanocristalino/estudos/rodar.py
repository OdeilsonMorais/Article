"""FEM das partículas nanocristalinas: uma esfera e muitas esferas.

Para cada cristalito d calcula k_nc(d) (nanocristal.py) e chama o
``11_muitas_particulas/estudos/muitas.py`` com esse k na partícula:
  --caso uma     uma esfera (N = 1) na célula, a 5 % em massa;
  --caso muitas  80 esferas aleatórias (a mesma semente da pasta 11).
O contorno de grão entra pelo k da partícula (escala do grão); o salto de
Kapitza epóxi/óxido (R_K) continua na superfície de cada partícula.

Rodar de 12_nanocristalino, em série:
    python3 estudos/rodar.py --caso uma
    python3 estudos/rodar.py --caso muitas
Opções: --material gama|alfa|anatase, --rgb 1e-9 (pode passar vários),
--cristalitos 1 2 3 5 10 15 25, --k0 5, --espessura 0.2 (esfera oca, t/R;
só entram cristalitos d <= espessura da parede), --diametro 100 (nm; para raio 100 nm use 200).
Resultados, um conjunto por material (e por --k0):
  estudos/nc_uma_<material>[_k0_X].csv, estudos/nc_muitas_<material>[_k0_X].csv
(uma linha por k_p) e estudos/nc_mapa_<material>[_k0_X].csv (k_p ↔ d).
Retoma de onde parou.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parent
sys.path.insert(0, str(RAIZ))
from nanocristal import MATERIAIS, com_k0, k_nanocristalino  # noqa: E402

MUITAS = RAIZ.parent / "11_muitas_particulas" / "estudos" / "muitas.py"
# muitas.py só conhece estes nomes de material (para a densidade); alfa usa a da gama
NOME_EM_MUITAS = {"gama": "gama", "alfa": "gama", "anatase": "anatase", "sio2": "sio2", "quartzo": "quartzo"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caso", choices=("uma", "muitas"), required=True)
    parser.add_argument("--material", choices=sorted(MATERIAIS), default="gama")
    parser.add_argument("--rgb", type=float, nargs="+", default=[1e-9], help="R_gb dos contornos (m²K/W)")
    parser.add_argument("--cristalitos", type=float, nargs="+", default=[1, 2, 3, 5, 10, 15, 25])
    parser.add_argument("--incluir-monocristal", action="store_true", help="também k0 (sem contornos)")
    parser.add_argument("--diametro", type=float, default=100.0)
    parser.add_argument("--rk", type=float, default=1e-8, help="Kapitza epóxi/óxido")
    parser.add_argument("--n", type=int, default=80, help="partículas no caso muitas")
    parser.add_argument("--k0", type=float, default=None, help="k do monocristal (ex.: 5); ℓ escala junto")
    parser.add_argument("--espessura", type=float, default=1.0,
                        help="t/R da parede (1 = maciça); núcleo de ar, mesma massa (muitas.py)")
    parser.add_argument("--margem", type=float, default=None, help="repassado ao muitas.py")
    parser.add_argument("--lacuna-minima", type=float, default=None, help="repassado ao muitas.py")
    args = parser.parse_args()
    if not MUITAS.exists():
        raise SystemExit(f"Não achei {MUITAS}: a pasta 11 precisa estar ao lado da 12.")

    ox = com_k0(MATERIAIS[args.material], args.k0)
    n = 1 if args.caso == "uma" else args.n
    sufixo = (f"_{args.material}" + (f"_k0_{args.k0:g}" if args.k0 else "")
              + (f"_oca_tR_{args.espessura:g}" if args.espessura < 1 else ""))
    parede_nm = args.espessura * args.diametro / 2
    if args.espessura < 1:
        fora = [d for d in args.cristalitos if d > parede_nm]
        args.cristalitos = [d for d in args.cristalitos if d <= parede_nm]
        print(f"Parede de {parede_nm:g} nm: cristalitos {args.cristalitos}"
              + (f" (fora, maiores que a parede: {fora})" if fora else ""), flush=True)
        if not args.cristalitos:
            raise SystemExit("Nenhum cristalito cabe na parede.")
    saida = PASTA / f"nc_{args.caso}{sufixo}.csv"
    mapa = PASTA / f"nc_mapa{sufixo}.csv"
    novo = not mapa.exists()
    with mapa.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["material", "R_gb", "d_nm", "k_p"])
        casos = [(rgb, d, round(k_nanocristalino(d, ox, rgb), 4)) for rgb in args.rgb for d in args.cristalitos]
        if args.incluir_monocristal:
            casos.append((0.0, float("inf"), ox.k0))
        for rgb, d, k in casos:
            w.writerow([args.material, rgb, d, k])
    for rgb, d, k in casos:
        print(f"\n### {ox.nome}, d = {d:g} nm, R_gb = {rgb:g}: k_p = {k:g} W/mK, N = {n}", flush=True)
        subprocess.run([sys.executable, str(MUITAS), "--material", NOME_EM_MUITAS[args.material],
                        "--k-particula", f"{k:g}", "--n", str(n), "--diametro", f"{args.diametro:g}",
                        "--rk", f"{args.rk:g}", "--espessura", f"{args.espessura:g}", "--saida", str(saida)]
                       + (["--margem", f"{args.margem:g}"] if args.margem is not None else [])
                       + (["--lacuna-minima", f"{args.lacuna_minima:g}"] if args.lacuna_minima is not None else []),
                       check=True)


if __name__ == "__main__":
    main()
