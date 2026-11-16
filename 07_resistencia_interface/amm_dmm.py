"""Resistência de Kapitza epóxi/óxido pelos modelos AMM e DMM, com fônons de Debye.

Lado 1 = matriz (epóxi), lado 2 = partícula. A condutância de interface é

    h = (1/4) Σ_j v_1j Γ_j ∫ ħω D_1j(ω) ∂n/∂T τ_j(ω) dω,      R_K = 1/h,

com D_1j = ω²/(2π² v_1j³) por polarização (1 longitudinal + 2 transversais),
n a ocupação de Bose–Einstein e o corte de Debye de cada ramo
ω_D,j = v_j (6π² n_at)^(1/3). Um fônon com ω acima do maior corte do lado 2
não tem para onde ir e não transmite.

  * AMM (descasamento acústico, Little 1959): reflexão especular, sem
    conversão de modo. Por ramo, transmissão de Fresnel acústica
        α(θ1) = 4 Z1 Z2 cosθ1 cosθ2 / (Z1 cosθ1 + Z2 cosθ2)²,  Z = ρ v,
    com sinθ2 = (v_2/v_1) sinθ1 e α = 0 além do ângulo crítico, e
    Γ_j = 2 ∫ α cosθ1 sinθ1 dθ1.
  * DMM (descasamento difuso, Swartz e Pohl 1989): o fônon esquece de onde
    veio; com Debye a transmissão não depende de ω nem do ângulo,
        τ = Σ_j v_2j⁻² / (Σ_j v_1j⁻² + Σ_j v_2j⁻²).

Os dois são limites idealizados. Para polímero/cerâmica eles costumam dar R_K
menores que medidas e dinâmica molecular, porque não veem a qualidade da
ligação química: leia o resultado como limite inferior e ordem de grandeza.

Também calcula, com Hasselman–Johnson, α = R_K k_m / R para o raio de cada
partícula e o raio crítico R_c = R_K k_m / (1 − k_m/k_p).

Rodar desta pasta: python3 amm_dmm.py   (só NumPy e Matplotlib)
Saídas em resultados/: rk_amm_dmm.csv (R_K contra T), resumo.csv e rk_amm_dmm.png.
"""

import csv
from pathlib import Path

import numpy as np

import parametros_materiais as pm

HBAR = 1.054571817e-34
KB = 1.380649e-23
NA = 6.02214076e23
PASTA = Path(__file__).resolve().parent / "resultados"


def densidade_atomica(m: pm.Material) -> float:
    return m.rho / (m.massa_atomica_media * 1e-3) * NA


def ramos(m: pm.Material):
    """[(velocidade, multiplicidade)]: 1 longitudinal + 2 transversais."""
    return [(m.v_L, 1), (m.v_T, 2)]


def cortes(m: pm.Material) -> dict:
    k_D = (6 * np.pi**2 * densidade_atomica(m)) ** (1 / 3)
    return {v: v * k_D for v, _ in ramos(m)}


def gamma_amm(rho1, v1, rho2, v2, pontos=4001) -> float:
    """Γ = 2 ∫ α(θ) cosθ sinθ dθ para um ramo, sem conversão de modo."""
    th1 = np.linspace(0, np.pi / 2, pontos)
    s2 = (v2 / v1) * np.sin(th1)
    ok = s2 < 1
    cos2 = np.sqrt(np.clip(1 - s2**2, 0, None))
    Z1, Z2 = rho1 * v1, rho2 * v2
    c1 = np.cos(th1)
    alfa = np.where(ok, 4 * Z1 * Z2 * c1 * cos2 / (Z1 * c1 + Z2 * cos2) ** 2, 0.0)
    return float(2 * np.trapezoid(alfa * c1 * np.sin(th1), th1))


def integral_espectral(v1, omega_max, T, pontos=4000) -> float:
    """∫ ħω D(ω) ∂n/∂T dω de 0 a omega_max, para um ramo com velocidade v1."""
    w = np.linspace(omega_max / pontos, omega_max, pontos)
    x = HBAR * w / (KB * T)
    dndT = (x / T) * np.exp(x) / np.expm1(x) ** 2
    D = w**2 / (2 * np.pi**2 * v1**3)
    integrando = HBAR * w * D * dndT
    return float(np.trapezoid(np.concatenate([[0.0], integrando]), np.concatenate([[0.0], w])))


def condutancia(m1: pm.Material, m2: pm.Material, T: float, modelo: str) -> float:
    """h (W/m² K) do lado 1 para o lado 2."""
    corte1 = cortes(m1)
    omega_max_2 = max(cortes(m2).values())
    if modelo == "DMM":
        s1 = sum(g / v**2 for v, g in ramos(m1))
        s2 = sum(g / v**2 for v, g in ramos(m2))
        tau = s2 / (s1 + s2)
    h = 0.0
    for (v1, g), (v2, _) in zip(ramos(m1), ramos(m2)):
        omega_max = min(corte1[v1], omega_max_2)
        if modelo == "AMM":
            fator = gamma_amm(m1.rho, v1, m2.rho, v2)
        else:
            fator = tau
        h += g * 0.25 * v1 * fator * integral_espectral(v1, omega_max, T)
    return h


def limite_radiacao(m1: pm.Material, T: float) -> float:
    """h com transmissão 1 (limite máximo de radiação de fônons do lado 1)."""
    corte1 = cortes(m1)
    return sum(g * 0.25 * v * integral_espectral(v, corte1[v], T) for v, g in ramos(m1))


def main() -> None:
    PASTA.mkdir(exist_ok=True)
    m1 = pm.MATRIZ
    temperaturas = np.linspace(200, 400, 41)
    linhas = []
    for chave, m2 in pm.PARTICULAS.items():
        for modelo in ("AMM", "DMM"):
            for T in temperaturas:
                linhas.append((chave, modelo, T, 1 / condutancia(m1, m2, T, modelo)))
    with (PASTA / "rk_amm_dmm.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["particula", "modelo", "T_K", "R_K_m2K_W"])
        escritor.writerows([(c, mo, f"{T:.2f}", f"{rk:.4e}") for c, mo, T, rk in linhas])

    print(f"Matriz: {m1.nome}, rho = {m1.rho:.0f} kg/m³, v_L = {m1.v_L:.0f}, v_T = {m1.v_T:.0f} m/s, "
          f"Θ_D(L, T) = " + ", ".join(f"{HBAR * w / KB:.0f}" for w in cortes(m1).values()) + " K")
    print(f"Limite de radiação de fônons do epóxi a 300 K: R_K ≥ {1 / limite_radiacao(m1, 300):.2e} m²K/W\n")
    resumo = []
    cab = (f"{'partícula':<16}{'T (K)':>8}{'R_K AMM':>12}{'R_K DMM':>12}{'R (nm)':>8}"
           f"{'α AMM':>9}{'α DMM':>9}{'R_c DMM (nm)':>14}")
    print(cab)
    for chave, m2 in pm.PARTICULAS.items():
        R = pm.RAIOS_NM[chave] * 1e-9
        for T in pm.TEMPERATURAS_K:
            rk_a = 1 / condutancia(m1, m2, T, "AMM")
            rk_d = 1 / condutancia(m1, m2, T, "DMM")
            alfa_a, alfa_d = rk_a * m1.k / R, rk_d * m1.k / R
            rc_d = rk_d * m1.k / (1 - m1.k / m2.k)
            print(f"{m2.nome:<16}{T:>8.1f}{rk_a:>12.2e}{rk_d:>12.2e}{R * 1e9:>8.0f}"
                  f"{alfa_a:>9.4f}{alfa_d:>9.4f}{rc_d * 1e9:>14.3f}")
            resumo.append([chave, m2.confianca, f"{T:.2f}", f"{rk_a:.4e}", f"{rk_d:.4e}",
                           f"{R * 1e9:.1f}", f"{alfa_a:.5f}", f"{alfa_d:.5f}", f"{rc_d * 1e9:.4f}"])
    with (PASTA / "resumo.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["particula", "confianca_parametros", "T_K", "R_K_AMM", "R_K_DMM", "R_nm",
                           "alfa_AMM", "alfa_DMM", "R_c_DMM_nm"])
        escritor.writerows(resumo)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cores = {"anatase": "#c2410c", "rutilo": "#e9a77f", "gama_alumina": "#2f5d9b", "alfa_alumina": "#8fa6c8"}
    fig, eixo = plt.subplots(figsize=(8, 6), dpi=200)
    for chave, m2 in pm.PARTICULAS.items():
        for modelo, estilo in (("AMM", "-"), ("DMM", "--")):
            rk = [r for c, mo, _, r in linhas if c == chave and mo == modelo]
            eixo.plot(temperaturas, rk, estilo, color=cores[chave], lw=2,
                      label=f"{m2.nome} ({modelo})")
    eixo.axhline(1e-8, color="#6b7385", lw=1, ls=":")
    eixo.text(402, 1e-8, " 10⁻⁸ (usado\n nas pastas 03–06)", va="center", fontsize=8, color="#6b7385")
    eixo.axvspan(298, 353, color="#f2c94c", alpha=0.15, lw=0)
    eixo.text(325.5, 0.98, "bancada\n25–80 °C", transform=eixo.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#6b7385")
    eixo.set_yscale("log")
    eixo.set_xlabel("T (K)")
    eixo.set_ylabel("Rₖ (m² K/W)")
    eixo.set_title("Epóxi/óxido: AMM (contínua) e DMM (tracejada), fônons de Debye")
    eixo.legend(fontsize=8, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    eixo.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PASTA / "rk_amm_dmm.png")
    print(f"\nResultados em {PASTA}")


if __name__ == "__main__":
    main()
