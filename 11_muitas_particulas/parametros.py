"""Parâmetros em SI, sem dependência do FEniCS."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Parametros:
    raio: float = 50e-9
    fracao_volumetrica: float = 0.05
    k_matriz: float = 0.2
    k_particula: float = 1.38
    resistencia_kapitza: float | None = None
    separacao_relativa: float = 0.05
    orientacao: str = "longitudinal"
    gradiente_x: float = 1.0
    # Pasta 09: núcleo de ar. k do ar a ~300 K; numa cavidade de dezenas de
    # nm o gás está em regime de Knudsen e conduz menos (use 0,01 ou menos).
    k_nucleo: float = 0.026

    def validar(self) -> None:
        for nome in ("raio", "k_matriz", "k_particula", "separacao_relativa"):
            valor = getattr(self, nome)
            if not isfinite(valor) or valor <= 0:
                raise ValueError(f"{nome} deve ser finito e positivo.")
        if not isfinite(self.fracao_volumetrica) or not 0 < self.fracao_volumetrica < 1:
            raise ValueError("A fração volumétrica deve estar entre 0 e 1.")
        if not isfinite(self.k_nucleo) or self.k_nucleo <= 0:
            raise ValueError("k_nucleo deve ser finito e positivo.")
        if self.orientacao not in ("longitudinal", "transversal"):
            raise ValueError("Orientação inválida.")
        if not isfinite(self.gradiente_x) or self.gradiente_x == 0:
            raise ValueError("O gradiente imposto deve ser finito e não nulo.")
        rk = self.resistencia_kapitza
        if rk is not None and (not isfinite(rk) or rk < 0):
            raise ValueError("R_K deve ser finito e não negativo, ou None.")
