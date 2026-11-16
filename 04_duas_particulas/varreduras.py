"""Geração de casos; não dispara simulações."""

from dataclasses import replace
from collections.abc import Iterator

from parametros import Parametros


# A fração volumétrica permanece fixa; portanto L é igual em todos os casos.
# Com phi=0,05, s/R=1,5 deixa uma lacuna de apenas 0,013R até a imagem
# periódica da outra esfera. Ele não representa duas partículas afastadas e
# foi removido. O valor 0,1 completa a resolução na região de pequenas
# separações antes do caso limite 0,05.
SEPARACOES_RELATIVAS = (1.0, 0.5, 0.2, 0.1, 0.05)


def gerar_casos(base: Parametros) -> Iterator[Parametros]:
    for orientacao in ("longitudinal", "transversal"):
        for separacao in SEPARACOES_RELATIVAS:
            yield replace(base, orientacao=orientacao, separacao_relativa=separacao)
