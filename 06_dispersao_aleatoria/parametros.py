"""Parâmetros em SI, sem dependência do FEniCS."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Parametros:
    raio: float = 50e-9
    fracao_volumetrica: float = 0.05
    k_matriz: float = 0.2
    k_particula: float = 5.0
    resistencia_kapitza: float | None = None
    separacao_relativa: float = 0.05
    orientacao: str = "longitudinal"
    gradiente_x: float = 1.0

    def validar(self) -> None:
        for nome in ("raio", "k_matriz", "k_particula", "separacao_relativa"):
            valor = getattr(self, nome)
            if not isfinite(valor) or valor <= 0:
                raise ValueError(f"{nome} deve ser finito e positivo.")
        if not isfinite(self.fracao_volumetrica) or not 0 < self.fracao_volumetrica < 1:
            raise ValueError("A fração volumétrica deve estar entre 0 e 1.")
        if self.orientacao not in ("longitudinal", "transversal"):
            raise ValueError("Orientação inválida.")
        if not isfinite(self.gradiente_x) or self.gradiente_x == 0:
            raise ValueError("O gradiente imposto deve ser finito e não nulo.")
        rk = self.resistencia_kapitza
        if rk is not None and (not isfinite(rk) or rk < 0):
            raise ValueError("R_K deve ser finito e não negativo, ou None.")
