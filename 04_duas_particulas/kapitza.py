"""Lei de Kapitza para uma interface térmica imperfeita."""


def condutancia_interfacial(resistencia: float) -> float:
    """Retornar a condutância por área ``h_K = 1/R_K`` em W/(m² K)."""
    if resistencia <= 0:
        raise ValueError("A formulação de Kapitza requer R_K estritamente positivo.")
    return 1.0 / resistencia
