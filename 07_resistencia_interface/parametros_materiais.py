"""Parâmetros dos materiais: o único lugar onde ficam os números.

Os scripts desta pasta (amm_dmm.py e os que vierem) leem daqui. Para trocar
um valor (por exemplo, quando houver a ficha técnica da N2680 ou uma medida
de ultrassom do revestimento), mude só este arquivo.

Cada material tem:
  rho      densidade (kg/m³)
  v_L      velocidade do som longitudinal (m/s)
  v_T      velocidade do som transversal (m/s)
  massa_atomica_media   massa molar média por átomo (g/mol), para a densidade
           de átomos n = rho N_A / massa, que dá o corte de Debye
  k        condutividade térmica (W/m K) do material maciço (bulk)
  fonte    de onde vem cada número
  confianca  "alta" (medida bem estabelecida), "media" (calculado ou faixa
           larga na literatura), "baixa" (estimativa, confirmar antes de usar)

Quando a fonte dá módulos elásticos isotrópicos (compressibilidade K e
cisalhamento G, média de Voigt–Reuss–Hill), as velocidades vêm de
    v_L = sqrt((K + 4G/3)/rho),   v_T = sqrt(G/rho).
"""

from dataclasses import dataclass, field
from math import sqrt


@dataclass(frozen=True)
class Material:
    nome: str
    rho: float
    v_L: float
    v_T: float
    massa_atomica_media: float
    k: float
    fonte: str
    confianca: str
    notas: str = ""
    extras: dict = field(default_factory=dict)


def velocidades(K_GPa: float, G_GPa: float, rho: float) -> tuple[float, float]:
    """(v_L, v_T) em m/s a partir de K e G isotrópicos (GPa) e rho (kg/m³)."""
    K, G = K_GPa * 1e9, G_GPa * 1e9
    return sqrt((K + 4 * G / 3) / rho), sqrt(G / rho)


# --- Matriz -----------------------------------------------------------------
# Dissertação: tinta WEG N2680, resina DGEBA + endurecedor poliamida, cura de
# 7 dias em temperatura ambiente. A densidade e as velocidades abaixo são de
# epóxi DGEBA curado sem carga. A N2680 é uma tinta comercial e pode ter
# pigmentos e cargas: a densidade real dela está no boletim técnico da WEG.
EPOXI = Material(
    nome="epóxi (DGEBA curado)",
    rho=1200.0,
    v_L=2600.0,
    v_T=1200.0,
    massa_atomica_media=340.4 / 49,  # DGEBA C21H24O4: 49 átomos, 340,4 g/mol
    k=0.20,
    fonte="Valores típicos de epóxi DGEBA curado (rho 1,15–1,25 g/cm³; ultrassom: "
          "v_L 2,5–2,8 km/s, v_T 1,1–1,3 km/s). k = 0,2 W/m K é o valor usado nas pastas 03–06.",
    confianca="media",
    notas="Trocar rho pela densidade da N2680 (boletim técnico) e, se possível, v_L e v_T por "
          "medida de ultrassom do revestimento curado.",
)

# --- Partículas ---------------------------------------------------------------
# Dissertação: TiO2 sol-gel calcinado a 500 °C por choque térmico, fase
# anatase, cristalito 24 ± 4 nm (Williamson–Hall), partículas primárias de
# ~100 nm aglomeradas.
_vL, _vT = velocidades(179.0, 57.0, 3890.0)
ANATASE = Material(
    nome="TiO₂ anatase",
    rho=3890.0,
    v_L=_vL,
    v_T=_vT,
    massa_atomica_media=79.87 / 3,
    k=8.5,
    fonte="rho 3,89 g/cm³ (dissertação, seção 2.2.1). K ≈ 179 GPa (compressão hidrostática); "
          "G ≈ 57 GPa (média VRH de constantes elásticas calculadas por DFT, faixa 50–65 GPa).",
    confianca="media",
    notas="k de anatase maciça é incerto (5–9 W/m K na literatura); o cristalito de 24 nm "
          "reduz o k real da partícula. Esse é o kₚ que o DFT/BTE deve fornecer.",
)

_vL, _vT = velocidades(211.0, 112.0, 4250.0)
RUTILO = Material(
    nome="TiO₂ rutilo",
    rho=4250.0,
    v_L=_vL,
    v_T=_vT,
    massa_atomica_media=79.87 / 3,
    k=8.5,
    fonte="Constantes elásticas de monocristal medidas por ressonância ultrassônica "
          "(Isaak et al., Phys. Chem. Minerals 1998): K ≈ 211 GPa, G ≈ 112 GPa (VRH).",
    confianca="alta",
    notas="Referência: o rutilo não aparece na dissertação, mas é a fase estável.",
)

# Dissertação: Al2O3 por precipitação com molde de poliestireno de 100 nm,
# calcinado a 550 °C, fase gama, cristalito 3,7 ± 0,7 nm (Scherrer), poroso,
# 71 ± 13 nm no DLS.
_vL, _vT = velocidades(170.0, 100.0, 3650.0)
GAMA_ALUMINA = Material(
    nome="γ-Al₂O₃",
    rho=3650.0,
    v_L=_vL,
    v_T=_vT,
    massa_atomica_media=101.96 / 5,
    k=5.0,
    fonte="rho 3,6–3,7 g/cm³ (espinélio defeituoso). K ≈ 170 GPa e G ≈ 100 GPa: estimativa "
          "no meio da faixa de cálculos ab initio (K 150–190, G 85–115 GPa).",
    confianca="baixa",
    notas="Partícula porosa: a densidade e o k efetivos são menores que os da fase densa. "
          "k ≈ 5 W/m K é só uma ordem de grandeza para γ-alumina nanocristalina.",
)

_vL, _vT = velocidades(254.0, 163.0, 3987.0)
ALFA_ALUMINA = Material(
    nome="α-Al₂O₃ (safira)",
    rho=3987.0,
    v_L=_vL,
    v_T=_vT,
    massa_atomica_media=101.96 / 5,
    k=35.0,
    fonte="Constantes elásticas de monocristal (ressonância ultrassônica, Goto et al., "
          "J. Geophys. Res. 1989): K ≈ 254 GPa, G ≈ 163 GPa (VRH); k ≈ 35 W/m K a 300 K.",
    confianca="alta",
    notas="Referência: fase estável, não é a da dissertação.",
)

MATRIZ = EPOXI
PARTICULAS = {
    "anatase": ANATASE,
    "rutilo": RUTILO,
    "gama_alumina": GAMA_ALUMINA,
    "alfa_alumina": ALFA_ALUMINA,
}

# --- Geometria e condições do experimento (dissertação) ----------------------
RAIOS_NM = {
    "anatase": 50.0,       # partículas primárias de ~100 nm de diâmetro (MEV)
    "rutilo": 50.0,
    "gama_alumina": 35.0,  # 71 nm de diâmetro hidrodinâmico (DLS)
    "alfa_alumina": 35.0,
}
CRISTALITO_NM = {"anatase": 24.0, "gama_alumina": 3.7}
FRACAO_MASSICA = 0.05        # 5 % em massa da tinta total
TEMPERATURAS_K = (298.15, 353.15)  # ambiente e placa a 80 °C
