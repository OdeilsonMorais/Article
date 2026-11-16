"""Roda o FEM de ocas contra maciças para SiO₂ e γ-Al₂O₃, chamando ocas.py.

Duas etapas (--etapa):
  espessura    parede t/R = 1 (maciça), 0,5, 0,3, 0,2, 0,1 e 0,05, em
               φ externo 5 % e 20 % e na mesma massa (5 % em massa da tinta);
  concentracao φ externo 2, 5, 10, 15, 20 e 25 % para t/R = 1, 0,2, 0,1 e 0,05.
  (Uma esfera por célula vai até ~25 %; acima disso as esferas quase se tocam.)

R_K = 1e-8 (AMM/DMM da pasta 07). Os resultados vão para ocas_sio2.csv e
ocas_gama.csv; cada caso é gravado assim que termina, e rodar de novo pula o
que já existe, então dá para interromper com Ctrl+C e continuar depois.

Rodar de 09_particulas_ocas, em série:
    python3 estudos/ocas/rodar_materiais.py --etapa espessura
    python3 estudos/ocas/rodar_materiais.py --etapa concentracao
Depois:
    python3 estudos/ocas/teoria_materiais.py      # curvas × espessura
    python3 estudos/ocas/concentracao_materiais.py  # curvas × concentração
"""

import argparse
import subprocess
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from teoria_materiais import fracao_volumetrica  # noqa: E402

# (nome curto, k_p, densidade g/cm³, arquivo)
MATERIAIS = [("sio2", 1.36, 2.20, "ocas_sio2.csv"), ("gama", 5.0, 3.65, "ocas_gama.csv")]
ESPESSURAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
PAREDES_CONC = [1.0, 0.2, 0.1, 0.05]
CONCENTRACOES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]


def rodar(material, k_p, saida, modo, phi, espessuras, rk):
    cmd = [sys.executable, str(PASTA / "ocas.py"), "--modo", modo, "--phi", f"{phi:.6g}",
           "--k-particula", f"{k_p:g}", "--rk", f"{rk:g}", "--saida", str(PASTA / saida),
           "--espessuras", *[f"{t:g}" for t in espessuras]]
    print(f"\n### {material}: {modo} φ = {phi:.4g}, t/R = {espessuras}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--etapa", choices=("espessura", "concentracao"), required=True)
    parser.add_argument("--materiais", nargs="+", default=["sio2", "gama"])
    parser.add_argument("--rk", type=float, default=1e-8)
    parser.add_argument("--rho-tinta", type=float, default=1.5)
    args = parser.parse_args()
    for nome, k_p, rho, saida in MATERIAIS:
        if nome not in args.materiais:
            continue
        if args.etapa == "espessura":
            for phi in (0.05, 0.20):
                rodar(nome, k_p, saida, "mesmo_phi", phi, ESPESSURAS, args.rk)
            phi_sol = fracao_volumetrica(0.05, rho, args.rho_tinta)
            rodar(nome, k_p, saida, "mesma_massa", phi_sol, ESPESSURAS, args.rk)
        else:
            for phi in CONCENTRACOES:
                rodar(nome, k_p, saida, "mesmo_phi", phi, PAREDES_CONC, args.rk)


if __name__ == "__main__":
    main()
