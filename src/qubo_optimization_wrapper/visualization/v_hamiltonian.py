import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import itertools
import re


def asignar_valores_diagonales(Q, lineal, variables):
    """Asigna los valores lineales a la diagonal de la matriz Q.
    Q: Matriz cuadrada de ceros.
    lineal: Términos lineales del Hamiltoniano.
    variables: Lista de variables del Hamiltoniano.
    """
    # Crear la lista de valores lineales en función del índice de las variables
    diagonal_values = [
        lineal[var] for var in variables
    ]  # Ordena las variables antes de asignar valores

    # Comprobamos si el número de valores lineales coincide con el tamaño de la matriz Q
    if len(diagonal_values) == Q.shape[0]:
        np.fill_diagonal(Q, diagonal_values)
    else:
        raise ValueError(
            "El número de valores lineales no coincide con el tamaño de la matriz Q."
        )


def asignar_terminos_cuadraticos(Q, interacciones, var_index, eliminar_bajo_diagonal):
    """Asigna los términos cuadráticos a la matriz Q.
    Q: Matriz cuadrada de ceros.
    interacciones: Términos cuadráticos del Hamiltoniano.
    var_index: Diccionario que mapea las variables a sus índices en la matriz Q.
    eliminar_bajo_diagonal: Si es True, elimina los valores debajo de la diagonal.
    """
    for (var1, var2), coef in interacciones.items():
        i, j = var_index[var1], var_index[var2]
        if eliminar_bajo_diagonal:
            if i < j:
                Q[i, j] = coef
            else:
                Q[j, i] = coef
        else:
            Q[i, j] = Q[j, i] = coef


def imprimir_resultados_hamiltoniano(Q, lineal, interacciones, bqm, formatear_valor):
    """Imprime la matriz QUBO y los detalles del Hamiltoniano.

    Args:
        Q (numpy.ndarray): Matriz QUBO.
        lineal (dict): Términos lineales del Hamiltoniano.
        interacciones (dict): Términos cuadráticos del Hamiltoniano.
        bqm (dimod.BinaryQuadraticModel): Modelo BQM.
        formatear_valor (function): Función para formatear valores.
    """
    print("Matriz QUBO:")
    print("-------------------------")
    print(Q)
    print("")

    print("HAMILTONIANO DEL SISTEMA:")
    print("-------------------------")
    print(
        "Término lineal:",
        {var: formatear_valor(val) for var, val in sorted(lineal.items())},
    )
    print(
        "Términos cuadráticos:",
        {
            (var1, var2): formatear_valor(val)
            for (var1, var2), val in interacciones.items()
        },
    )
    print("Offset:", formatear_valor(bqm.offset))
    print("")


def visualizar_parábolas_HR(
    term_expressions, lambdas_valores, x_range=(-5, 5), num_points=100
):
    """
    Grafica los términos del Hamiltoniano, sustituyendo las x_i por una variable continua x,
    manteniendo el centro de la parábola.

    Args:
        term_expressions (list): Lista de expresiones de SymPy para cada término.
        lambdas_valores (dict): Diccionario con los valores numéricos de lambda_k (clave: "lambda_k").
        x_range (tuple): Rango de valores para x (min, max).
        num_points (int): Número de puntos para la graficación.
    """

    # Crea la figura y los ejes
    fig, ax = plt.subplots()

    x = sp.Symbol("x")  # Variable continua x
    xi = sp.IndexedBase("x")  # Indexed Base
    x_index_to_vary = 1  # Index of x_i to vary

    # Itera sobre cada término
    for k, term in enumerate(term_expressions, 1):
        print(f"\n---Visualizando Termino {k} con variable continua x---")
        print("Term (original):", term)

        # 0. Crear los símbolos lambda_k
        lambdas = {
            k: sp.symbols(f"lambda_{k}") for k in range(1, len(term_expressions) + 1)
        }

        # 1.  Sustituir los lambdas por sus valores numéricos
        # Convertir las claves de lambdas_valores a símbolos de SymPy
        lambda_subs = {
            lambdas[i]: lambdas_valores[f"lambda_{i}"]
            for i in range(1, len(lambdas) + 1)
        }

        term_sustituido_lambda = term.subs(lambda_subs)
        print("Term después de sustituir lambdas:", term_sustituido_lambda)

        # 2. Sustituir TODAS las x_i *EXCEPTO x_index_to_vary* por un valor constante (ej: 0)
        sustituciones = {}
        for sym in term_sustituido_lambda.free_symbols:
            if (
                isinstance(sym, sp.Indexed)
                and sym.base == xi
                and sym.indices[0] != x_index_to_vary
            ):
                sustituciones[sym] = 0  # Sustituir por 0

        term_con_xi_fijas = term_sustituido_lambda.subs(sustituciones)

        # 3. Sustituir la xi que no hemos fijado por la variable continua x
        term_con_x = term_con_xi_fijas.subs(xi[x_index_to_vary], x)
        print("Term después de sustituir x[i] por x:", term_con_x)

        # 4. Convertir la expresión de SymPy a una función numérica de NumPy
        try:
            f = sp.lambdify(x, term_con_x, modules=["numpy"])

        except Exception as e:
            print("Error in sp.lambdify:", e)
            continue

        # 5. Generar los valores de x y calcular los valores del término
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        try:
            term_values = f(x_values)
        except Exception as e:
            print("Error evaluating the function:", e)
            continue

        # 6. Graficar
        ax.plot(x_values, term_values, label=f"Term {k}")

    # Personaliza la gráfica
    ax.set_xlabel(f"x")
    ax.set_ylabel("Energía")
    ax.set_title("Términos de $H^R$")
    ax.legend()
    ax.grid(True)
    plt.show()


def show_QMatrix(compiled_hamiltonian, eliminar_bajo_diagonal):
    """Muestra la matriz QUBO y los detalles del Hamiltoniano.

    Args:
        H: Expresión del hamiltoniano dependiente de las variables binarias y coeficientes de lagrange (placeholders).
        lambda_dict (dict, optional): Diccionario que mapea los coeficientes de lagrange (Placeholder) a sus valores numéricos.
        eliminar_bajo_diagonal (bool): Indica si se eliminan los valores debajo de la diagonal.
    """
    bqm = compiled_hamiltonian.get_bqm()
    variables = sorted(bqm.variables)
    var_index = {var: i for i, var in enumerate(variables)}

    lineal = bqm.linear
    interacciones = bqm.quadratic

    n = len(bqm.variables)
    Q = np.zeros((n, n))

    asignar_valores_diagonales(Q, lineal, variables)
    asignar_terminos_cuadraticos(Q, interacciones, var_index, eliminar_bajo_diagonal)

    np.set_printoptions(precision=2, suppress=True)

    def formatear_valor(val):
        return (
            f"{val:.2f}" if isinstance(val, float) and val % 1 != 0 else f"{int(val)}"
        )

    imprimir_resultados_hamiltoniano(Q, lineal, interacciones, bqm, formatear_valor)


def visualize_energies(lambda_dict, hamiltonian, compiled_hamiltonian) -> None:
    """Visualiza las energías para cada solución con el Hamiltoniano dado usando un scatter plot.

    Args:
        hamiltonian: El Hamiltoniano de PyQUBO (expresión simbólica, sin compilar).
        lambda_dict (dict, optional): Diccionario que mapea los Placeholder a sus valores numéricos.
                                     Defaults to None.
    """

    all_solutions = generate_all_solutions(hamiltonian)
    energies = []
    for solution in all_solutions:
        energy = calculate_energy(compiled_hamiltonian, solution)
        energies.append(energy)

    # Crear etiquetas para las soluciones (solo los valores de x_i)
    solution_labels = []
    for solution in all_solutions:
        label = "".join(
            str(solution[var]) for var in sorted(solution.keys())
        )  # Combina los valores de x_i
        solution_labels.append(label)

    # Crear el gráfico de dispersión (scatter plot)
    plt.figure(figsize=(12, 6))  # Ajusta el tamaño de la figura
    x_values = range(
        len(all_solutions)
    )  # Crear valores para el eje x (índices de las soluciones)
    plt.scatter(x_values, energies)  # Usar plt.scatter en lugar de plt.bar

    # Añadir etiquetas y título
    plt.xlabel("Solución")  # Cambiar la etiqueta del eje x
    plt.ylabel("Energía")

    title = "Energías para todas las soluciones"
    if lambda_dict:
        lambda_str = ", ".join(
            [f"{k}={v}" for k, v in lambda_dict.items()]
        )  # Formatear los valores de lambda
        title += f" ({lambda_str})"
    plt.title(title)

    # Personalizar los ticks del eje x para mostrar las soluciones
    plt.xticks(x_values, solution_labels, rotation=45, ha="right")

    # Ajustar márgenes para evitar que las etiquetas se corten
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def generate_all_solutions(hamiltonian):
    """Genera todas las soluciones posibles para las variables binarias 'x_i' y 'y_i' en el Hamiltoniano.

    Args:
        H: El Hamiltoniano de PyQUBO (expresión simbólica).

    Returns:
        list: Una lista de diccionarios, donde cada diccionario es una solución.
    """
    H = hamiltonian
    # Buscar solo variables que empiecen con x_ o y_
    variable_names = sorted(
        set(re.findall(r"[xy]_\d+", str(H))),
        key=lambda x: (
            x[0],
            int(x.split("_")[1]),
        ),  # ordena por letra y luego por índice numérico
    )

    n_variables = len(variable_names)
    all_combinations = itertools.product([0, 1], repeat=n_variables)

    solutions = []
    for combination in all_combinations:
        solution = {variable_names[i]: combination[i] for i in range(n_variables)}
        solutions.append(solution)

    return solutions


def calculate_energy(compiled_hamiltonian, solution):
    """Calcula la energía del Hamiltoniano de PyQUBO para una solución dada.

    Args:
        H: El Hamiltoniano de PyQUBO (expresión simbólica sin compilar).
        solution (dict): Diccionario donde las claves son los nombres de las variables
                       (e.g., 'x0', 'x1') y los valores son 0 o 1.
        feed_dict (dict, optional): Diccionario que mapea los Placeholder a sus valores numéricos.
                                     Defaults to None.

    Returns:
        float: El valor de la energía.
    """

    bqm = compiled_hamiltonian.get_bqm()
    energy = bqm.energy(solution)

    return energy
