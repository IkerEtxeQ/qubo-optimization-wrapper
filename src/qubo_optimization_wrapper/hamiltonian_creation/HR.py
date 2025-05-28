from pyqubo import Binary, Placeholder
import sympy as sp
from IPython.display import display, Math
import math
from itertools import product


def construccion_HR(Omega, R):
    """
    Genera la expresión QUBO del Hamiltoniano de Regularización H^R y lo visualiza en LaTeX.
    Define los valores simbólicos de lambda dentro de la función.
    Adaptado para aceptar valores R_QUBO que pueden contener variables binarias de pyqubo.

    Args:
        NR (int): Número de términos en la suma.
        omegas (dict): Diccionario con los conjuntos Omega_k (clave: k, valor: lista de índices).
        Rs (dict): Diccionario con los valores de R_k (clave: k).

    Returns:
        tuple: (qubo_expression, H_R, term_expressions)
            qubo_expression (pyqubo.Express): Expresión lista para PyQUBO
            H_R (sympy.Expr): Expresión de SymPy del Hamiltoniano.
            term_expressions (list): Lista de expresiones de SymPy para cada término.
    """

    NR = len(Omega)  # Calcula el número de términos en la suma.

    # terminos_HR_continuo = HR_continuo(
    #     NR, Omega, R
    # )  # Genera la expresión simbólica del Hamiltoniano H^R
    HR_PYQUBO = HR_QUBO(NR, Omega, R)  # Genera la expresión QUBO del Hamiltoniano

    return HR_PYQUBO


def HR_QUBO(NR, Omega, R):
    qubo_expression = 0
    for k in range(1, NR + 1):
        X = {i: Binary(f"x_{i}") for i in Omega[k]}
        sum_omega = sum(X[i] for i in Omega[k])
        lambda_k = Placeholder(f"lambda_{k}")
        R_keys = list(R.keys())  # lista ordenada de claves
        clave = R_keys[k - 1]
        R2 = next(iter(R[clave]))
        qubo_expression += lambda_k * (sum_omega - R2) ** 2

    return qubo_expression


def HR_continuo(NR, Omega, R):
    # 0. Definición de los valores simbólicos de lambda
    lambdas = {k: sp.symbols(f"lambda_{k}") for k in range(1, NR + 1)}

    # 1. Construcción de la expresión simbólica (SymPy)
    x = sp.IndexedBase("x")  # Define x como una variable indexada (x_i)
    H_R = 0  # Inicializa el Hamiltoniano
    terminos_HR = []  # Lista para guardar las expresiones de cada término

    for k in range(1, NR + 1):
        sum_omega = sum(x[i] for i in Omega[k])  # Sumatoria de x_i para i en Omega_k
        term = lambdas[k] * (sum_omega - R[k]) ** 2
        H_R += term
        terminos_HR.append(term)

    # 2. Visualización en LaTeX
    print("Hamiltoniano de Restricciones:")
    display(Math(sp.latex(H_R)))

    return terminos_HR


def calcular_conjunto_R(valores_permitidos):
    """
    Calcula el conjunto R, evaluando la expresión QUBO para cada combinación binaria
    de las variables y, y agregando los resultados individuales al conjunto R.
    También guarda la expresión QUBO en el conjunto R_QUBO.

    Args:
        valores_permitidos: Lista de valores permitidos.

    Returns:
        Un conjunto (R) de resultados individuales de la evaluación de la expresión QUBO para cada combinación binaria.
        Un conjunto (R_QUBO) que contiene la expresión QUBO en términos de objetos Binary.
    """

    M = len(valores_permitidos)
    R = set()
    R_QUBO = set()

    if M == 1:
        R.add(valores_permitidos[0])
        R_QUBO.add(valores_permitidos[0])  # Agregar el valor permitido al conjunto QUBO
        return R, R_QUBO  # Return R_QUBO even for the base case

    if M > 1:
        equidistantes = True
        delta = valores_permitidos[1] - valores_permitidos[0]
        n = valores_permitidos[0]
        for i in range(2, M):
            if valores_permitidos[i] - valores_permitidos[i - 1] != delta:
                equidistantes = False
                break

        if equidistantes:
            if delta == 1 and M <= 3:
                if M == 2:
                    R.add(n + 0.5)
                    R_QUBO.add(n + 0.5)
                    return R, R_QUBO  # Return R_QUBO even for the base case
                if M == 3:
                    y = Binary("y_0")
                    R_QUBO.add(y + n + 0.5)
                    # Add the evaluations of the expression
                    R.add(n + 0.5)  # y=0
                    R.add(1 + n + 0.5)  # y=1
                    return R, R_QUBO  # Return R_QUBO even for the base case

            else:
                gamma = math.floor(math.log2(M))
                y = {i: Binary(f"y_{i}") for i in range(gamma + 1)}

                # Build the QUBO expression
                qubo_expression = delta * (M - 2**gamma) * y[gamma] + n
                for i in range(gamma):
                    qubo_expression += delta * (2**i * y[i])

                R_QUBO.add(qubo_expression)

                # Evaluate the QUBO expression for all possible binary combinations
                for combination in product([0, 1], repeat=len(y)):
                    c_values = {f"c_{i}": combination[i] for i in range(len(y))}
                    # Evaluate the expression with the binary values
                    result = delta * (M - 2**gamma) * c_values[f"c_{gamma}"] + n
                    for i in range(gamma):
                        result += delta * (2**i * c_values[f"c_{i}"])
                    R.add(result)  # Add the *individual* evaluation result

        else:
            num_y = M - 1
            y = {i: Binary(f"y_{i}") for i in range(num_y)}

            # Build QUBO expression
            qubo_expression = valores_permitidos[0]
            for i in range(2, M + 1):
                qubo_expression += (
                    valores_permitidos[i - 1] - valores_permitidos[0]
                ) * y[i - 2]
            R_QUBO.add(qubo_expression)
            # Evaluate for all binary combinations
            for combination in product([0, 1], repeat=len(y)):
                if sum(combination) == 0 or sum(combination) == 1:
                    c_values = {f"y_{i}": combination[i] for i in range(len(y))}
                    result = valores_permitidos[0]
                    for i in range(2, M + 1):
                        result += (
                            valores_permitidos[i - 1] - valores_permitidos[0]
                        ) * c_values[f"y_{i - 2}"]
                R.add(result)  # Add the *individual* evaluation result

    return R, R_QUBO


def calcular_conjunto_R_multi_valor(valores_permitidos):
    R_dict = {}
    R_QUBO_dict = {}

    for index, v_permitidos in valores_permitidos.items():
        R, R_QUBO = calcular_conjunto_R(v_permitidos)
        R_dict[index] = R
        R_QUBO_dict[index] = R_QUBO

    return R_dict, R_QUBO_dict
