"""Descrição analítica da célula; não gera malha."""

from dataclasses import dataclass
from itertools import combinations, product
from math import dist, pi

from parametros import Parametros


@dataclass(frozen=True)
class Geometria:
    """Cubo de lado ``lado`` com esferas iguais de raio ``raio``.

    ``centros`` tem um centro por partícula (duas no par, N num aglomerado).
    ``lacuna_interna`` é a menor distância entre superfícies dentro da célula;
    ``lacuna_imagem_periodica``, a menor distância entre uma esfera e a imagem
    periódica de outra (ou dela mesma).
    """

    lado: float
    raio: float
    centros: tuple[tuple[float, ...], ...]
    lacuna_interna: float
    lacuna_imagem_periodica: float
    lacuna_minima: float

    @property
    def fracao_volumetrica(self) -> float:
        return len(self.centros) * 4 * pi * self.raio**3 / (3 * self.lado**3)


def geometria_de_centros(lado: float, raio: float, centros) -> Geometria:
    """Validar N esferas dentro do cubo e calcular as lacunas.

    As esferas não podem cruzar as faces do cubo: a malha periódica supõe que
    cada face é uma única superfície plana.
    """
    centros = tuple(tuple(float(x) for x in centro) for centro in centros)
    if lado <= 0 or raio <= 0 or not centros:
        raise ValueError("Lado e raio devem ser positivos, e a lista de centros não pode ser vazia.")
    if any(len(centro) != 3 for centro in centros):
        raise ValueError("Cada centro deve ter três coordenadas.")
    if any(min(min(x, lado - x) for x in centro) <= raio for centro in centros):
        raise ValueError("Uma esfera toca ou cruza uma face do cubo; afaste-a das faces.")
    internas = [dist(a, b) - 2 * raio for a, b in combinations(centros, 2)]
    lacuna_interna = min(internas) if internas else float("inf")
    deslocamentos = [d for d in product((-lado, 0.0, lado), repeat=3) if any(d)]
    lacuna_imagem = min(
        dist(a, tuple(x + dx for x, dx in zip(b, d))) - 2 * raio
        for a in centros for b in centros for d in deslocamentos)
    if lacuna_interna <= 0 or lacuna_imagem <= 0:
        raise ValueError("As esferas se tocam ou tocam imagens periódicas; ajuste o lado ou os centros.")
    return Geometria(lado, raio, centros, lacuna_interna, lacuna_imagem,
                     min(lacuna_interna, lacuna_imagem))


def preparar_geometria(parametros: Parametros) -> Geometria:
    parametros.validar()
    r = parametros.raio
    lado = (8 * pi * r**3 / (3 * parametros.fracao_volumetrica)) ** (1 / 3)
    distancia = r * (2 + parametros.separacao_relativa)
    eixo = 0 if parametros.orientacao == "longitudinal" else 1
    centros = []
    for sinal in (-1, 1):
        centro = [lado / 2] * 3
        centro[eixo] += sinal * distancia / 2
        centros.append(tuple(centro))
    return geometria_de_centros(lado, r, centros)
