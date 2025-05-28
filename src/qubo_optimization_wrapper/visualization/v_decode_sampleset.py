import matplotlib.pyplot as plt
import itertools
import re
from qubo_optimization_wrapper.hamiltonian_creation.hamiltonian import Hamiltonian


def min_energy_result(decoded_samples):
    print("RESULTADOS SIMULATED ANNEALING:")
    print("-------------------------")
    best_sample = min(decoded_samples, key=lambda d: d.energy)
    print(best_sample)
    print("")


def visualize_energies(hamiltonian: Hamiltonian) -> None:
    """Visualiza las energías para cada solución con el Hamiltoniano dado usando un scatter plot.

    Args:
        hamiltonian: El Hamiltoniano de PyQUBO (expresión simbólica, sin compilar).
        lambda_dict (dict, optional): Diccionario que mapea los Placeholder a sus valores numéricos.
                                     Defaults to None.
    """
    lambda_dict = hamiltonian.get_lambda_dict()

    all_solutions = generate_all_solutions(hamiltonian)
    energies = []
    for solution in all_solutions:
        energy = calculate_energy(hamiltonian, solution)
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


def generate_all_solutions(hamiltonian: Hamiltonian):
    """Genera todas las soluciones posibles para las variables binarias 'x_i' y 'y_i' en el Hamiltoniano.

    Args:
        H: El Hamiltoniano de PyQUBO (expresión simbólica).

    Returns:
        list: Una lista de diccionarios, donde cada diccionario es una solución.
    """
    H = hamiltonian.get_hamiltonian()
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


def calculate_energy(hamiltonian: Hamiltonian, solution):
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

    bqm = hamiltonian.get_compiled_hamiltonian().get_bqm()
    energy = bqm.energy(solution)

    return energy
