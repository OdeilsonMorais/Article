"""R_K epóxi/sílica pelos modelos AMM e DMM, para comparar com a medida de Kothari et al. (2013).

Usa as funções de amm_dmm.py e a matriz de parametros_materiais.py (que não
são alterados) e acrescenta a sílica amorfa (e o quartzo, como referência
cristalina). Também mostra quanto R_K muda com as propriedades do epóxi
(v_T, v_L, ρ), que são a maior incerteza do DMM.

Sílica amorfa (vidro de sílica fundida): ρ = 2,20 g/cm³, v_L = 5,97 km/s,
v_T = 3,76 km/s (valores de catálogo bem estabelecidos), 3 átomos por SiO2.
Quartzo: K ≈ 37,8 GPa e G ≈ 44,3 GPa (média VRH), ρ = 2,65 g/cm³.
Kothari et al. (2013) mediram 4,0 a 6,3e-9 m²K/W para epóxi/sílica.

Rodar de 07_resistencia_interface:
    python3 estudos/silica/dmm_silica.py
Saída: resultados/rk_silica.csv
"""

import csv
import sys
from dataclasses import replace
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]
sys.path.insert(0, str(RAIZ))
import amm_dmm as am  # noqa: E402
import parametros_materiais as pm  # noqa: E402

SILICA = pm.Material(
    nome="SiO₂ amorfa", rho=2200.0, v_L=5970.0, v_T=3760.0, massa_atomica_media=60.08 / 3, k=1.38,
    fonte="Vidro de sílica fundida: ρ 2,20 g/cm³, v_L 5,97 km/s, v_T 3,76 km/s.", confianca="alta")
_vL, _vT = pm.velocidades(37.8, 44.3, 2650.0)
QUARTZO = pm.Material(
    nome="SiO₂ quartzo α", rho=2650.0, v_L=_vL, v_T=_vT, massa_atomica_media=60.08 / 3, k=8.0,
    fonte="Média VRH: K 37,8 GPa, G 44,3 GPa.", confianca="media")
MEDIDO = (4.0e-9, 6.3e-9)  # Kothari et al. 2013
R_NM = 50.0


def main() -> None:
    m1 = pm.MATRIZ
    linhas = []
    print(f"Matriz padrão: ρ = {m1.rho:.0f}, v_L = {m1.v_L:.0f}, v_T = {m1.v_T:.0f}\n")
    print(f"{'partícula':<16}{'T (K)':>8}{'AMM':>11}{'DMM':>11}{'α DMM':>9}{'R_c DMM (nm)':>14}")
    for m2 in (SILICA, QUARTZO):
        for T in (298.15, 353.15):
            ra = 1 / am.condutancia(m1, m2, T, "AMM")
            rd = 1 / am.condutancia(m1, m2, T, "DMM")
            rc = rd * m1.k / (1 - m1.k / m2.k)
            print(f"{m2.nome:<16}{T:>8.1f}{ra:>11.2e}{rd:>11.2e}{rd * m1.k / (R_NM * 1e-9):>9.4f}{rc * 1e9:>14.2f}")
            linhas.append(["padrao", m2.nome, T, m1.rho, m1.v_L, m1.v_T, f"{ra:.4e}", f"{rd:.4e}"])
    print(f"\nMedido (Kothari et al. 2013): {MEDIDO[0]:.1e} a {MEDIDO[1]:.1e} m²K/W\n")
    print("Sensibilidade ao epóxi (sílica amorfa, 298 K):")
    print(f"{'ρ':>6}{'v_L':>7}{'v_T':>7}{'AMM':>11}{'DMM':>11}")
    for rho in (1150.0, 1200.0, 1250.0):
        for vL in (2500.0, 2600.0, 2800.0):
            for vT in (1100.0, 1200.0, 1300.0):
                mm = replace(m1, rho=rho, v_L=vL, v_T=vT)
                ra = 1 / am.condutancia(mm, SILICA, 298.15, "AMM")
                rd = 1 / am.condutancia(mm, SILICA, 298.15, "DMM")
                linhas.append(["sensibilidade", SILICA.nome, 298.15, rho, vL, vT, f"{ra:.4e}", f"{rd:.4e}"])
                if rho == 1200.0 or (vL == 2600.0 and vT == 1200.0):
                    print(f"{rho:>6.0f}{vL:>7.0f}{vT:>7.0f}{ra:>11.2e}{rd:>11.2e}")
    sens = [float(l[7]) for l in linhas if l[0] == "sensibilidade"]
    sa = [float(l[6]) for l in linhas if l[0] == "sensibilidade"]
    print(f"\nFaixa DMM: {min(sens):.2e} a {max(sens):.2e}; AMM: {min(sa):.2e} a {max(sa):.2e} m²K/W")
    destino = RAIZ / "resultados" / "rk_silica.csv"
    destino.parent.mkdir(exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caso", "particula", "T_K", "rho_epoxi", "vL_epoxi", "vT_epoxi", "R_K_AMM", "R_K_DMM"])
        w.writerows(linhas)
    print(f"Resultados: {destino}")


if __name__ == "__main__":
    main()
