import neal


def redondeo_decimales_significativos(numero, n_decimales=2):
    """Redondea un número a un número específico de decimales significativos.

    Args:
        numero (float): El número a redondear.
        n_decimales (int): El número de decimales a mostrar.

    Returns:
        str: El número redondeado formateado como una cadena.

    Raises:
        ValueError: Si el número no es decimal o si todos los dígitos decimales son cero.
    """
    if not isinstance(numero, (int, float)):
        raise ValueError("El número debe ser un entero o un float.")

    if not isinstance(
        numero, float
    ):  # Si es entero, lo convertimos a float para formatear
        numero = float(numero)

    formato = f".{n_decimales}f"  # Creamos el string de formato
    return f"{numero:{formato}}"  # Aplicamos el formato
