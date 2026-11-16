"""Condutividade de uma partícula nanocristalina em função do tamanho do cristalito d.

Duas reduções em relação ao monocristal (k0), as duas com o mesmo d:

1. Dentro do grão, os fônons batem nas bordas (espalhamento de contorno,
   regra de Matthiessen com um caminho livre médio cinzento ℓ):
       k_grão(d) = k0 / (1 + ℓ/d)

2. Entre grãos, cada contorno tem uma resistência de Kapitza R_gb (o "efeito
   de interface entre as paredes do cristalito"). Em média o calor atravessa um
   contorno a cada d, em série com o grão:
       k_nc(d) = k_grão / (1 + R_gb k_grão / d)
   Essa é a forma usada para cerâmicas nanocristalinas (ex.: YSZ, Yang et al.
   2002; Nan & Birringer 1998 dão o mesmo resultado a menos de um fator
   geométrico de ordem 1).

Os parâmetros são estimativas e estão marcados assim. ℓ vem da teoria
cinética cinzenta, ℓ = 3k0/(C v): ela subestima o caminho livre real (parte do
calor vai por fônons de caminho longo), então a redução intragrão aqui é
conservadora. R_gb de contornos em óxidos fica na faixa 1e-10 a 1e-8 m²K/W;
1e-9 é o valor central.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Oxido:
    nome: str
    k0: float          # W/mK, monocristal denso
    ell_nm: float      # caminho livre médio cinzento
    densidade: float   # g/cm³
    fonte: str


MATERIAIS = {
    "gama": Oxido("γ-Al₂O₃", 10.0, 2.0, 3.65,
                  "k0 ~10 é estimativa (espinélio com vacâncias, sem medida em material denso); "
                  "ℓ = 3k/(Cv) com C ≈ 3,0 MJ/m³K, v ≈ 6 km/s. Confiança baixa."),
    "alfa": Oxido("α-Al₂O₃", 35.0, 5.0, 3.98,
                  "k0 = 35 (safira, ambiente); ℓ = 3k/(Cv) ≈ 5 nm. Confiança média."),
    "sio2": Oxido("SiO₂ amorfa", 1.36, 0.6, 2.20,
                  "Sílica amorfa (vidro): não tem cristalitos de verdade, e o ℓ cinzento é ~0,6 nm "
                  "(3k/(Cv), C ≈ 1,65 MJ/m³K, v ≈ 4 km/s). Aqui d é o tamanho dos domínios "
                  "(partículas primárias de um agregado) e R_gb a resistência entre eles."),
    "quartzo": Oxido("SiO₂ quartzo", 8.0, 3.0, 2.65,
                     "Quartzo ~6 (⊥c) a 10 (∥c) W/mK; média 8 para grãos orientados ao acaso. "
                     "ℓ = 3k/(Cv) ≈ 3 nm. Confiança média."),
    "anatase": Oxido("TiO₂ anatase", 8.5, 2.0, 3.89,
                     "k0 ~8,5 (estimativa; rutilo 8–13); ℓ ≈ 2 nm. Confiança baixa."),
}


def com_k0(ox: Oxido, k0=None):
    """Mesmo óxido com outro k0; ℓ = 3k0/(Cv) escala junto com k0."""
    if k0 is None or k0 == ox.k0:
        return ox
    return replace(ox, k0=k0, ell_nm=ox.ell_nm * k0 / ox.k0)


def k_grao(d_nm, ox: Oxido):
    return ox.k0 / (1 + ox.ell_nm / d_nm)


def k_nanocristalino(d_nm, ox: Oxido, R_gb):
    kg = k_grao(d_nm, ox)
    return kg / (1 + R_gb * kg / (d_nm * 1e-9))


def fracao_volumetrica(w, rho_p, rho_m=1.5):
    return (w / rho_p) / (w / rho_p + (1 - w) / rho_m)


def hasselman_johnson(phi, alfa, k_m, k_p):
    a = k_p * (1 + 2 * alfa) + 2 * k_m
    b = k_p * (1 - alfa) - k_m
    return k_m * (a + 2 * phi * b) / (a - phi * b)
