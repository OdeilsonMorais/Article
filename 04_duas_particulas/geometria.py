"""Descrição analítica da célula; não gera malha."""

from dataclasses import dataclass
from math import pi

from parametros import Parametros


@dataclass(frozen=True)
class Geometria:
    lado: float
    raio: float
    centros: tuple[tuple[float, ...], tuple[float, ...]]
    lacuna_interna: float
    lacuna_imagem_periodica: float
    lacuna_minima: float


def preparar_geometria(parametros: Parametros) -> Geometria:
    parametros.validar()
    r = parametros.raio
    lado = (8 * pi * r**3 / (3 * parametros.fracao_volumetrica)) ** (1 / 3)
    distancia = r * (2 + parametros.separacao_relativa)
    lacuna_interna = distancia - 2 * r
    # A imagem periódica mais próxima da segunda esfera fica a L - d entre
    # centros; sua separação de superfície é L - d - 2R.
    lacuna_imagem = lado - distancia - 2 * r
    # O par e sua repetição ao longo do eixo devem permanecer separados.
    if lado <= 2 * r or lacuna_imagem <= 0:
        raise ValueError("As esferas tocam ou sobrepõem imagens periódicas; ajuste phi ou s/R.")
    eixo = 0 if parametros.orientacao == "longitudinal" else 1
    centros = []
    for sinal in (-1, 1):
        centro = [lado / 2] * 3
        centro[eixo] += sinal * distancia / 2
        centros.append(tuple(centro))
    return Geometria(lado, r, (centros[0], centros[1]), lacuna_interna,
                     lacuna_imagem, min(lacuna_interna, lacuna_imagem))
